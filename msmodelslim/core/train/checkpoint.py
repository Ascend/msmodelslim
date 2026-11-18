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
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

# pylint: disable=no-name-in-module

# QAT checkpoint 收集与落盘：FSDP full-state-dict 收集（含 state_dict_type
# 临时切换与恢复）+ CheckpointSaverFormat 写出（qat_checkpoint.json +
# model.safetensors）。元数据（qparam_names / algorithm / 源模型路径）由构造注入。

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from msmodelslim.core.train.device import empty_device_cache, log_device_memory
from msmodelslim.core.quantization_format.checkpoint_format import CheckpointSaverFormat
from msmodelslim.core.quantization_format.interface import ExportContext
from msmodelslim.infra.io.default_json_writer_factory import DefaultJsonWriterFactory
from msmodelslim.infra.io.default_safetensors_writer_factory import (
    DefaultSafetensorsWriterFactory,
)
from msmodelslim.utils.logging import get_logger


class QATCheckpointSaver:
    """QAT checkpoint 保存器：全 rank 参与收集，仅主进程写盘，结束后释放显存。

    Disk pick of best vs latest is App ``_select_checkpoint``; in-memory
    end-of-train reload follows ``checkpointing.load_best_model_at_end``.
    """

    def __init__(
        self,
        *,
        qparam_names: Sequence[str],
        algorithm_cfg: Optional[Sequence[Dict[str, Any]]] = None,
        source_model_path: Optional[str] = None,
    ):
        self.qparam_names = list(qparam_names)
        self.algorithm_cfg: List[Dict[str, Any]] = [dict(item) for item in (algorithm_cfg or [])]
        self.source_model_path = source_model_path

    def save(
        self,
        *,
        model: Any,
        accelerator: Any,
        args: Any,
        output_dir: str,
        is_fsdp_enabled: bool = False,
    ) -> None:
        """收集 full state dict 并经 ``CheckpointSaverFormat`` 写出。"""
        os.makedirs(output_dir, exist_ok=True)
        log_device_memory("before_checkpoint_gather")
        state_dict: Optional[Dict[str, Any]] = None
        try:
            state_dict = self._gather_full_state_dict(
                model=model,
                accelerator=accelerator,
                is_fsdp_enabled=is_fsdp_enabled,
            )
            log_device_memory("after_checkpoint_gather")
            if getattr(args, "should_save", False):
                self._write_checkpoint(
                    output_dir,
                    state_dict,
                    accelerator=accelerator,
                    model=model,
                    args=args,
                )
        finally:
            if state_dict is not None:
                state_dict.clear()
                del state_dict
            empty_device_cache()
            log_device_memory("after_checkpoint_release")

    def _gather_full_state_dict(
        self,
        *,
        model: Any,
        accelerator: Any,
        is_fsdp_enabled: bool,
    ) -> Dict[str, Any]:
        """Gather the full state dict of the model (FSDP-aware)."""
        if not is_fsdp_enabled:
            return model.state_dict()

        plugin = getattr(accelerator.state, "fsdp_plugin", None)
        if plugin is None:
            return accelerator.get_state_dict(model)

        current_type = getattr(plugin, "state_dict_type", None)
        already_full = current_type is not None and "FULL_STATE_DICT" in str(current_type).upper()
        if already_full:
            return accelerator.get_state_dict(model)

        try:
            plugin.set_state_dict_type("FULL_STATE_DICT")
        except Exception:  # pragma: no cover - accelerate version drift
            get_logger().exception(
                "QATCheckpointSaver: failed to switch FSDP state_dict_type to FULL_STATE_DICT "
                "(current=%s); gathering with current type.",
                current_type,
            )
            return accelerator.get_state_dict(model)

        try:
            return accelerator.get_state_dict(model)
        finally:
            try:
                plugin.set_state_dict_type(current_type)
            except Exception:  # pragma: no cover - accelerate version drift
                get_logger().exception(
                    "QATCheckpointSaver: failed to restore FSDP state_dict_type=%s",
                    current_type,
                )

    def _write_checkpoint(
        self,
        output_dir: str,
        state_dict: Dict[str, Any],
        *,
        accelerator: Any,
        model: Any,
        args: Any,
    ) -> None:
        """Write ``qat_checkpoint.json`` + ``model.safetensors`` via CheckpointSaverFormat."""
        unwrapped = accelerator.unwrap_model(model, keep_torch_compile=False)
        source = self.source_model_path
        ctx = ExportContext(
            save_directory=Path(output_dir),
            source_model_path=Path(source) if source else None,
            rank=int(getattr(args, "process_index", 0) or 0),
            world_size=int(getattr(args, "world_size", 1) or 1),
            state_dict=state_dict,
        )
        fmt = CheckpointSaverFormat(
            ctx,
            DefaultSafetensorsWriterFactory(),
            DefaultJsonWriterFactory(),
            qparam_names=list(self.qparam_names),
            algorithm=list(self.algorithm_cfg),
        )
        fmt.prepare_export()
        fmt.process_module_tensors("", unwrapped)
        fmt.finalize_export(unwrapped)
        get_logger().info(
            "QATCheckpointSaver saved checkpoint via CheckpointSaverFormat: %s (tensors=%d)",
            output_dir,
            len(state_dict),
        )
