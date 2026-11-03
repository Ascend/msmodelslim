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

from pathlib import Path
from typing import Any, List, Optional

from msmodelslim.app.train.proxy.task import TrainTask, TrainTaskConfig
from msmodelslim.core.const import DeviceType
from msmodelslim.core.train.train_launcher import TrainLauncher
from msmodelslim.model import IModelFactory
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.exception_decorator import exception_catcher
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.plugin import TypedFactory
from msmodelslim.utils.security import yaml_safe_load
from msmodelslim.utils.validation.conversion import convert_to_readable_file


_TRAIN_APP_FACTORY = TypedFactory[Any](config_base_class=TrainTaskConfig)


@logger_setter(prefix="msmodelslim.app.train.proxy")
class TrainProxyApplication:
    """CLI train entry: load flat YAML, dispatch by ``type`` via train_task plugins."""

    def __init__(
        self,
        model_factory: IModelFactory,
        train_launcher: TrainLauncher,
    ) -> None:
        self.model_factory = model_factory
        self.train_launcher = train_launcher

    @exception_catcher
    def run(
        self,
        config_path: Path,
        model_path: Optional[Path] = None,
        save_path: Optional[Path] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
        model_type: Optional[str] = None,
    ) -> None:
        config_path = convert_to_readable_file(config_path, param_name="config")
        payload = yaml_safe_load(str(config_path))
        if not isinstance(payload, dict):
            raise SchemaValidateError("train YAML root must be a mapping")

        config = TrainTaskConfig.model_validate(payload)
        proxy_task = TrainTask(config)
        get_logger().info(
            "TrainProxyApplication: type=%s config=%s",
            config.type,
            config_path,
        )
        self._dispatch(
            proxy_task,
            model_path=model_path,
            save_path=save_path,
            device=device,
            device_indices=device_indices,
            model_type=model_type,
        )

    def _dispatch(
        self,
        proxy_task: TrainTask,
        *,
        model_path: Optional[Path],
        save_path: Optional[Path],
        device: DeviceType,
        device_indices: Optional[List[int]],
        model_type: Optional[str],
    ) -> None:
        app = _TRAIN_APP_FACTORY.create(
            proxy_task.config,
            model_factory=self.model_factory,
            train_launcher=self.train_launcher,
        )
        app.run(
            proxy_task,
            model_path=Path(model_path),
            save_path=Path(save_path),
            model_type=model_type,
            device=device,
            device_indices=device_indices,
        )
