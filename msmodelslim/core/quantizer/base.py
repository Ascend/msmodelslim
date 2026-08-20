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

from abc import abstractmethod

import torch
from pydantic import BaseModel, Field
from pydantic import validate_call
from torch import nn
from typing_extensions import Self, Optional, Dict, Any, Tuple

from msmodelslim.utils.exception import SpecError
from msmodelslim.ir.qal.qbase import QStorage, QParam, QScheme, QScope, QDType
from msmodelslim.utils.distributed.task_scheduler import DTSMixin
from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext
from msmodelslim.ir.qal.qregistry import QABCRegistry


class QConfig(BaseModel):
    """描述单个张量（权重或激活）的量化方式。

    `QConfig` 是量化配置的公共组成单元，字段直接内联在所属配置下（如
    `LinearQConfig.qconfig.act` / `qconfig.weight`、AWQ 的 `weight_qconfig`）。
    它由量化数据类型、量化粒度、是否对称与参数估计算法四要素决定量化行为，
    其中 `dtype`/`scope`/`symmetric` 组合出一个量化方案（scheme），
    再由 `method` 选定该方案下的参数估计算法实现。
    """

    dtype: QDType = Field(
        description="量化数据类型，如 `int8`、`int4`、`mxfp8`、`mxfp4`、`fp8_e4m3`；`float` 表示该张量不量化。"
    )
    scope: QScope = Field(
        description="量化粒度，即 scale/zero_point 的计算范围：`per_tensor`（整张量一个尺度）、"
        "`per_channel`（按通道）、`per_group`/`per_block`（按分组或固定块）、`per_token`（按 token）、"
        "`per_head`（按注意力头）、`dual_scale`（双尺度）等；合法取值组合取决于 `dtype` 与量化器实现。"
    )
    symmetric: bool = Field(
        description="是否对称量化。对称量化只保存 scale；非对称量化额外保存 zero_point，"
        "可用性取决于 `dtype`/`scope` 组合。"
    )
    method: str = Field(
        description="量化参数估计算法，如 `minmax`、`mse_round`、`histogram`、`ssz`、`none` 等；"
        "可用取值取决于 `dtype`/`scope`/`symmetric` 组合，`none` 表示不估计参数（配合 `float` 使用）。"
    )
    ext: Dict[str, Any] = Field(
        default_factory=dict,
        exclude_if=lambda v: not v,
        description="量化器扩展参数，随 `method` 与量化器实现而定（如 gptq 的 `percdamp`/`group_size`）；"
        "空对象表示无扩展参数。",
    )

    def to_scheme(self):
        return QScheme(QScope(self.scope), QDType(self.dtype), self.symmetric)


@QABCRegistry.register_abc(dispatch_key=Tuple[QScheme, str])
class AutoActQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.sync = False  # 默认不启用同步操作

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_config(cls, config: QConfig) -> Self:
        return QABCRegistry.create(AutoActQuantizer, (config.to_scheme(), config.method), *(config,))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_q_param(self) -> QParam:
        """
        获取量化参数
        """
        pass

    def support_distributed(self) -> bool:
        """
        判断是否支持分布式

        Returns:
            bool: 是否支持分布式，默认为True
        """
        return True

    def is_data_free(self) -> bool:
        """
        判断是否data free场景

        Returns:
            bool: 是否是否data free场景，默认为False
        """
        return False

    def enable_sync(self):
        """
        启用同步操作
        子类可以重写此方法以实现更复杂的同步逻辑
        """
        self.sync = True

    def validate_ext_config(self):
        """
        扩展参数校验
        """
        pass


@QABCRegistry.register_abc(dispatch_key=Tuple[QScheme, str])
class AutoWeightQuantizer(nn.Module, DTSMixin):
    def __init__(self):
        super().__init__()
        self.sync = False  # 默认不启用同步操作

    def distributed_sync(self, record: TaskExecutionRecord, sync_ctx: TaskSyncContext) -> None:
        """默认分布式同步：触发 forward 在各 rank 独立重算（仅 data-free 量化器适用）。"""
        if not self.is_data_free():
            raise SpecError(
                "distributed_sync with None input requires a data-free weight quantizer. "
                "Non-data-free quantizers require calibration data.",
                action="distributed_sync with None is only supported for data-free quantizers. "
                "Non-data-free quantizers must provide calibration data via the normal forward path.",
            )
        with torch.no_grad():
            _ = self.forward(None)

    @classmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def from_config(cls, config: QConfig) -> Self:
        return QABCRegistry.create(AutoWeightQuantizer, (config.to_scheme(), config.method), *(config,))

    @abstractmethod
    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        """
        初始化权重Tensor

        Args:
            bias: 偏移
            weight: 权重

        Returns:
            None
        """

        pass

    @abstractmethod
    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对权重进行量化和反量化

        Args:
            x: 激活值

        Returns:
            torch.Tensor: 量化然后反量化后的激活值，与init_weight所提供的权重shape/dtype相同
        """

        pass

    @abstractmethod
    def get_q_storage(self) -> QStorage:
        """
        获取量化后的权重
        """
        pass

    @abstractmethod
    def get_q_param(self) -> QParam:
        """
        获取量化参数
        """
        pass

    def support_distributed(self) -> bool:
        """
        判断是否支持分布式

        Returns:
            bool: 是否支持分布式，默认为True
        """
        return True

    def is_data_free(self) -> bool:
        """
        判断是否data free场景

        Returns:
            bool: 是否是否data free场景，默认为True
        """
        return True

    def enable_sync(self):
        """
        启用同步操作
        子类可以重写此方法以实现更复杂的同步逻辑
        """
        self.sync = True

    def validate_ext_config(self):
        """
        扩展参数校验
        """
        pass
