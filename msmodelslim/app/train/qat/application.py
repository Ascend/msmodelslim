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

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Type

from msmodelslim.app.train.proxy.task import TrainTask, TrainTaskConfig
from msmodelslim.app.train.qat.task import QATTask, QATTaskConfig
from msmodelslim.core.const import DeviceType
from msmodelslim.core.train.train_launcher import TrainLauncher
from msmodelslim.format.checkpoint_format.constants import STRUCTURE_FILE
from msmodelslim.model import IModelFactory
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.exception_decorator import exception_catcher
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.validation.conversion import (
    convert_to_readable_dir,
    convert_to_writable_dir,
)

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
_TRAINER_STATE_FILE = "trainer_state.json"


def _last_eval_loss(ckpt_dir: Path) -> Optional[float]:
    """Return the latest ``eval_loss`` from ``trainer_state.json`` log_history."""
    state_path = ckpt_dir / _TRAINER_STATE_FILE
    if not state_path.is_file():
        return None
    try:
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        get_logger().warning(
            "QATApplication: failed to read trainer_state at %s; skip eval_loss",
            state_path,
        )
        return None
    if not isinstance(state, dict):
        return None
    for entry in reversed(state.get("log_history") or []):
        if not isinstance(entry, dict) or "eval_loss" not in entry:
            continue
        try:
            return float(entry["eval_loss"])
        except (TypeError, ValueError):
            continue
    return None


def _select_checkpoint(save_path: Path) -> Optional[Path]:
    """Pick deploy ckpt among ``checkpoint-{step}/``.

    1. 扫描 ``checkpoint-(\\d+)`` 且含 QAT structure 文件的目录；
    2. 对每个候选，取 ``trainer_state.json`` 中最新一条 ``eval_loss``；
    3. 在有 ``eval_loss`` 的候选中选最低者（相同 loss 取更大 step）；
    4. 若全部没有 ``eval_loss``，回退到最大 step。
    """
    root = Path(save_path)
    if not root.is_dir():
        return None

    # (step, path, latest_eval_loss | None)
    candidates: List[Tuple[int, Path, Optional[float]]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_DIR_RE.match(child.name)
        if match is None or not (child / STRUCTURE_FILE).is_file():
            continue
        step = int(match.group(1))
        candidates.append((step, child, _last_eval_loss(child)))

    if not candidates:
        return None

    scored = [(step, path, loss) for step, path, loss in candidates if loss is not None]
    if scored:
        # 最低 eval_loss；并列时取更大 step
        best_step, best_path, best_loss = min(scored, key=lambda item: (item[2], -item[0]))
        get_logger().info(
            "QATApplication: selected checkpoint by eval_loss=%s step=%s path=%s (candidates=%s)",
            best_loss,
            best_step,
            best_path,
            [(step, loss) for step, _path, loss in candidates],
        )
        return best_path

    # 全无 eval → 回退最大 step
    best_step, best_path, _ = max(candidates, key=lambda item: item[0])
    get_logger().warning(
        "QATApplication: no eval_loss in any checkpoint; deploy uses latest step=%s path=%s",
        best_step,
        best_path,
    )
    return best_path


@logger_setter(prefix="msmodelslim.app.train.qat")
class QATApplication:
    """QAT orchestration: task -> train -> deploy."""

    def __init__(
        self,
        config: QATTaskConfig,
        *,
        model_factory: IModelFactory,
        train_launcher: TrainLauncher,
    ) -> None:
        self.config = config
        self.model_factory = model_factory
        self.train_launcher = train_launcher
        self.qat_task: Optional[QATTask] = None

    @exception_catcher
    def run(
        self,
        proxy_task: TrainTask,
        *,
        model_path: Path,
        save_path: Path,
        model_type: str,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ) -> None:
        model_path = convert_to_readable_dir(model_path, param_name="model_path")
        save_path = convert_to_writable_dir(save_path, param_name="save_path")
        # 将YAML配置变为可执行的QATTask
        self.build_task(proxy_task)
        # 执行训练
        self.do_train(
            model_path=model_path,
            save_path=save_path,
            model_type=model_type,
            device=device,
            device_indices=device_indices,
        )
        # 执行部署
        self.do_deploy(
            save_path=save_path,
            device=device,
            device_indices=device_indices,
        )
        get_logger().info("QATApplication finished successfully")

    def build_task(self, proxy_task: TrainTask) -> QATTask:
        """根据已校验的 ``QATTaskConfig`` 构建 ``QATTask`` 实例。"""
        proxy_task.config = self.config
        self.qat_task = QATTask.build_from(self.config)
        get_logger().info(
            "QATTask ready: algorithms=%s",
            [a.type for a in self.config.algorithm],
        )
        return self.qat_task

    def do_train(
        self,
        *,
        model_path: Path,
        save_path: Path,
        model_type: str,
        device: DeviceType,
        device_indices: Optional[List[int]],
    ) -> None:
        cfg = self.qat_task.config
        cfg.source_model_path = str(model_path)
        adapter = self.model_factory.create(
            model_type=model_type,
            model_path=model_path,
            trust_remote_code=False,
        )
        get_logger().info("===========QAT TRAIN===========")
        self.train_launcher.launch(
            cfg=cfg,
            adapter=adapter,
            save_path=save_path,
            hooks=self.qat_task.hooks,
            device=device,
            device_indices=device_indices,
        )

    def do_deploy(
        self,
        *,
        save_path: Path,
        device: DeviceType,
        device_indices: Optional[List[int]],
    ) -> None:
        if self.qat_task is None:
            raise SchemaValidateError(
                "QATTask missing before deploy",
                action="Please run build_task / do_train first.",
            )
        best = _select_checkpoint(save_path)
        if best is None:
            get_logger().warning("QATApplication: no checkpoint selected; skip deploy")
            return

        deploy_save_path = Path(save_path) / "deploy"
        deploy_save_path.mkdir(parents=True, exist_ok=True)
        convert_cfg = self.qat_task.build_convert_config(
            checkpoint_path=best,
            deploy_save_path=deploy_save_path,
            device=device,
            device_indices=device_indices,
        )

        from msmodelslim.core.quant_service.modelslim_convert.factory import (
            create_convert_application,
        )

        get_logger().info(
            "===========QAT DEPLOY=========== checkpoint=%s dst_format=%s out=%s modules=%s",
            best,
            convert_cfg.dst_format,
            deploy_save_path,
            len(convert_cfg.module_rules),
        )
        create_convert_application().run(convert_cfg)
        get_logger().info("QAT deploy finished: %s", deploy_save_path)


def get_plugin() -> Tuple[Type[TrainTaskConfig], Type[QATApplication]]:
    """
    Plugin:train_task entry — 由 config.ini entry_point 加载并注册。

    Returns:
        (QATTaskConfig, QATApplication)：配置类 + Application 类（需提供 ``run``）
    """
    return QATTaskConfig, QATApplication
