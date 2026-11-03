#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from msmodelslim.core.const import DeviceType
from msmodelslim.core.train.protocol import ITrainHook
from msmodelslim.core.train.rank_worker import rank_worker_entry
from msmodelslim.core.train.task import CoreTrainTaskConfig
from msmodelslim.model import IModel
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger, get_root_logger, set_logger_level

_WORKER_LOG_LEVEL_ENV = "MSMODELSLIM_WORKER_LOG_LEVEL"
_KNOWN_LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})


def _child_rank_worker_entry(*worker_args) -> None:
    level = os.environ.get(_WORKER_LOG_LEVEL_ENV) or msmodelslim_config.env_vars.log_level
    set_logger_level(level)
    # Patch pyarrow before any datasets import so forked ranks skip duplicate
    # extension-type registration (Array2DExtensionType already defined).
    try:
        # dataset_build 为训练运行时依赖，按需懒加载。
        from msmodelslim.core.train.dataset_build import (  # pylint: disable=no-name-in-module
            _patch_pyarrow_skip_duplicate_extension_types,
        )

        _patch_pyarrow_skip_duplicate_extension_types()
    except Exception:
        get_logger().debug("pyarrow duplicate-extension patch skipped", exc_info=True)
    rank_worker_entry(*worker_args)


def _current_log_level_name() -> str:
    level_name = logging.getLevelName(get_root_logger().getEffectiveLevel())
    if isinstance(level_name, str) and level_name.lower() in _KNOWN_LOG_LEVELS:
        return level_name.lower()
    return msmodelslim_config.env_vars.log_level


class TrainLauncher:
    """Parent-process multi-rank launcher (Accelerate). Backend only — no pick/export."""

    def launch(
        self,
        cfg: CoreTrainTaskConfig,
        adapter: IModel,
        save_path: Path,
        hooks: Sequence[ITrainHook],
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ) -> None:
        # cfg / hooks 须在 setup 侧已完成校验与组装；此处原样下发，worker 不再二次组装。
        worker_args = (
            cfg,
            adapter,
            save_path,
            list(hooks),
        )
        num_processes = self._resolve_num_processes(device_indices)
        self._launch_workers(
            worker_args=worker_args,
            num_processes=num_processes,
            device=device,
            device_indices=device_indices,
        )
        get_logger().info(
            "TrainLauncher: all workers finished: num_processes=%s device=%s adapter=%s save_path=%s",
            num_processes,
            device,
            type(adapter).__name__,
            save_path,
        )

    def _launch_workers(
        self,
        worker_args: tuple,
        num_processes: int,
        device: DeviceType,
        device_indices: Optional[List[int]],
    ) -> None:
        try:
            from accelerate import debug_launcher, notebook_launcher
            from accelerate.utils import patch_environment
        except ImportError as exc:
            raise UnsupportedError(
                "accelerate is required to launch QAT workers",
                action="Please install accelerate before running QAT training.",
            ) from exc
        env_patch = self._build_launch_env(
            num_processes=num_processes,
            device=device,
            device_indices=device_indices,
        )
        with patch_environment(**env_patch):
            if device == DeviceType.CPU and num_processes > 1:
                debug_launcher(
                    _child_rank_worker_entry,
                    args=worker_args,
                    num_processes=num_processes,
                )
            else:
                master_port = self._pick_master_port()
                notebook_launcher(
                    _child_rank_worker_entry,
                    args=worker_args,
                    num_processes=num_processes,
                    mixed_precision="no",
                    use_port=master_port,
                )

    def _build_launch_env(
        self,
        num_processes: int,
        device: DeviceType,
        device_indices: Optional[List[int]],
    ) -> Dict[str, str]:
        env: Dict[str, str] = {
            _WORKER_LOG_LEVEL_ENV: _current_log_level_name(),
        }
        if num_processes > 1:
            env["ACCELERATE_USE_FSDP"] = "true"
            env["FSDP_VERSION"] = "2"
            env["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
            env["FSDP_ACTIVATION_CHECKPOINTING"] = "false"
            env["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
            env["FSDP_OFFLOAD_PARAMS"] = "false"
            env["FSDP_RESHARD_AFTER_FORWARD"] = "true"
            env["FSDP_STATE_DICT_TYPE"] = "SHARDED_STATE_DICT"
            # Storage/backward breadcrumbs: off by default; set MSMODELSLIM_QAT_DEBUG_STORAGE=1 to enable.
            env.setdefault("MSMODELSLIM_QAT_DEBUG_STORAGE", "0")

        if device == DeviceType.CPU:
            env["ACCELERATE_USE_CPU"] = "true"
        elif device_indices:
            env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_indices)

        get_logger().debug("TrainLauncher env: %s", {k: env[k] for k in sorted(env)})
        return env

    @staticmethod
    def _pick_master_port() -> str:
        requested = (os.environ.get("MASTER_PORT") or "").strip()
        if requested:
            if TrainLauncher._port_is_free(requested):
                return requested
            get_logger().warning(
                "MASTER_PORT=%s is already in use; picking a free port instead.",
                requested,
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            return str(sock.getsockname()[1])

    @staticmethod
    def _port_is_free(port: str) -> bool:
        try:
            port_i = int(port)
        except (TypeError, ValueError):
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port_i))
            except OSError:
                return False
        return True

    @staticmethod
    def _resolve_num_processes(device_indices: Optional[List[int]]) -> int:
        if device_indices is not None:
            if len(device_indices) < 1:
                raise SchemaValidateError(
                    "device_id list must contain at least one index",
                    action="Please pass --device_id 0 or a non-empty device list.",
                )
            return len(device_indices)
        return 1
