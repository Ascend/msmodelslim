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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List

from pydantic import ConfigDict, Field

from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from msmodelslim.utils.plugin import TypedConfig

QUANT_SERVICE_PLUGIN_GROUP = "msmodelslim.quant_service.plugins"

QUANT_TASK_PLUGIN_GROUP = "msmodelslim.quant_task.plugins"


@TypedConfig.plugin_entry(entry_point_group=QUANT_SERVICE_PLUGIN_GROUP)
class QuantServiceConfig(TypedConfig):
    apiversion: TypedConfig.TypeField


# --- BaseQuantConfig（QuantConfig）：任务级量化配置，用于 quantize(quant_config, ...) ---
@TypedConfig.plugin_entry(entry_point_group=QUANT_TASK_PLUGIN_GROUP)
class BaseQuantConfig(TypedConfig):
    """量化任务配置：apiversion + spec，用于 quantize() 入参。与 QuantServiceConfig 区分。

    - 在【本基类】上 model_validate({apiversion, spec, ...}) 时，_validate_plugin
      按 apiversion 懒加载注册的后端子类并以其强 schema 校验（判别即校验），
      apiversion 未知时抛 UnsupportedError（判别即探测）；
    - 子类（如 PracticeConfig、各后端 *QuantConfig）自身 model_validate 不触发判别
      （_entry_point_group 不在子类 __dict__ 中，机制防递归），弱/强语义各自保持。
    """

    apiversion: TypedConfig.TypeField = Field(
        default="Unknown",
        description="API 版本（任务类型），决定 spec 的结构：`modelslim_v1`、`multimodal_vlm_modelslim_v1`、"
        "`multimodal_sd_modelslim_v1`、`modelslim_convert`；YAML 中必须显式指定，默认值 `Unknown` 仅为代码内部占位，不可直接使用。",
    )
    spec: object = Field(default_factory=dict, description="任务规格，结构随 apiversion 而定。")

    model_config = ConfigDict(extra="allow")


class IQuantService(ABC):
    @abstractmethod
    def quantize(
        self,
        quant_config: BaseQuantConfig,
        model_adapter: IModel,
        save_path: Optional[Path] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ) -> None: ...
