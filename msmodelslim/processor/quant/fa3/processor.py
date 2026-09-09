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

from typing import Optional, Literal, List, Union, Annotated

import torch
import torch.distributed as dist
from pydantic import BaseModel, Field, ConfigDict, model_validator, AfterValidator
from torch import nn

from msmodelslim.ir.api import calculate_qparam
from msmodelslim.ir.qal import QScope, QDType, QABCRegistry, QParam
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim import ir as qir
from msmodelslim.core.observer.minmax import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.core.observer.recall_window import RecallWindowObserver, RecallWindowObserverConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.distributed.dist_helper import DistHelper
from msmodelslim.utils.validation.pydantic import validate_str_length
from .interface import FA3QuantAdapterInterface, FA3QuantPlaceHolder

FA3_BRANCHES = ("fa_q", "fa_k", "fa_v")


class FA3AttentionDetails(BaseModel):
    """FA3 注意力分支级量化配置，分别指定 Q/K/V 的量化方式。"""

    fa_q: QConfig = Field(default=None, description="Query 分支的量化配置，见《QConfig 配置说明》")
    fa_k: QConfig = Field(default=None, description="Key 分支的量化配置，见《QConfig 配置说明》")
    fa_v: QConfig = Field(default=None, description="Value 分支的量化配置，见《QConfig 配置说明》")

    model_config = ConfigDict(extra="forbid")


class FA3QuantProcessorConfig(AutoProcessorConfig):
    """FA3（FlashAttention-3）量化处理器配置。

    位于 `spec.process[]`，由 `type: fa3_quant` 分派；对 FA3 注意力模块应用统一量化
    （`qconfig`）或 Q/K/V 分支级量化（`details`），两者互斥。
    """

    type: Literal["fa3_quant"] = Field(default="fa3_quant", description="处理器类型，固定为 `fa3_quant`。")
    qconfig: Optional[QConfig] = Field(
        default=None, description="统一量化配置；不提供时默认使用 INT8 per-head symmetric，见《QConfig 配置说明》"
    )
    include: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=lambda: ["*"], description="包含的模块名称模式，默认 `*` 匹配全部模块"
    )
    exclude: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=lambda: [], description="排除的模块名称模式，优先级高于 `include`"
    )
    details: Optional[FA3AttentionDetails] = Field(
        default=None, description="Q/K/V 分支级量化配置（FA3AttentionDetails），见《FA3AttentionDetails 配置说明》"
    )
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_exclusivity(self) -> "FA3QuantProcessorConfig":
        """校验 qconfig 与 details 互斥：两者都提供时报错；两者都未提供时默认 INT8 per-head symmetric。"""
        # 如果没有提供qconfig，使用默认的INT8 per-head symmetric配置
        if self.qconfig is None and not self.details:
            self.qconfig = QConfig(dtype=QDType.INT8, scope=QScope.PER_HEAD, symmetric=True, method="minmax")
            return self

        if self.qconfig is not None and self.details:
            raise ValueError("FA3 quantization supports only one of the qconfig and details configurations.")

        return self


class _FA3PerHeadObserver(nn.Module):
    """监测器：复用 MsMinMaxObserver 的按维度统计，得到 per-head min/max。"""

    def __init__(self, ratio: float = 1.0, name: str = ""):
        super().__init__()
        self._observer = RecallWindowObserver(RecallWindowObserverConfig(ratio=ratio, dim=-1, keepdim=True))
        self._dist_helper = None
        self._name = name

    @property
    def min_val(self) -> Optional[torch.Tensor]:
        return self._observer.get_min()

    @property
    def max_val(self) -> Optional[torch.Tensor]:
        return self._observer.get_max()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对 (B, H, S, D) 按 [0, 2, 3] 归约，保留 H 维；keepdim=True 得到形如 (1, H, 1, 1)
        samples = x.contiguous().view(x.shape[1], -1)
        # 只有 shared 模块才需要在 forward 时同步
        sync = self._dist_helper is not None and self._dist_helper.is_shared(self._name)
        self._observer.update(samples, sync=sync)
        return x

    def set_dist_helper(self, dist_helper: DistHelper):
        """设置分布式辅助类"""
        self._dist_helper = dist_helper


class _FA3PerChannelObserver(nn.Module):
    """监测器：与参考分支 DynamicCacheQuantizer 对齐，head 并入通道做 per-channel 静态统计。

    输入 (B, H, S, D) → (B*S, H*D)，沿行维归约得到 (H*D,) 的 per-(head,dim) min/max。
    """

    def __init__(self, name: str = ""):
        super().__init__()
        minmax_config = MinMaxObserverConfig(dim=[0], keepdim=False, aggregation_type="max")
        self._observer = MsMinMaxObserver(minmax_config)
        self._dist_helper = None
        self._name = name

    @property
    def min_val(self) -> Optional[torch.Tensor]:
        impl = self._observer._impl
        return getattr(impl, "min_val", None)

    @property
    def max_val(self) -> Optional[torch.Tensor]:
        impl = self._observer._impl
        return getattr(impl, "max_val", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, S, D) → (B, S, H, D) → (B*S, H*D)，沿 N 归约得到 channel-wise min/max
        x_t = x.transpose(-2, -3)
        samples = x_t.reshape(-1, x_t.shape[-1] * x_t.shape[-2])
        sync = self._dist_helper is not None and self._dist_helper.is_shared(self._name)
        self._observer.update(samples, sync=sync)
        return x

    def set_dist_helper(self, dist_helper: DistHelper):
        self._dist_helper = dist_helper


@QABCRegistry.register(dispatch_key=FA3QuantProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.fa3_quant")
class FA3QuantProcessor(AutoSessionProcessor):
    def __init__(
        self,
        model: nn.Module,
        config: FA3QuantProcessorConfig,
        adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        if not isinstance(adapter, FA3QuantAdapterInterface):
            raise UnsupportedError(
                f"Adapter {adapter.__class__.__name__} does not implement FA3QuantAdapterInterface",
                action="Please implement FA3QuantAdapterInterface",
            )
        self.adapter = adapter
        self.include = ConfigSet(config.include)
        self.exclude = ConfigSet(config.exclude)
        self.dist_helper: Optional[DistHelper] = None

    def _resolve_branch_qconfig(self, fa_prefix: str) -> Optional[QConfig]:
        if self.config.details:
            return getattr(self.config.details, fa_prefix, None)
        return self.config.qconfig

    def check_scope_condition(self, target_scope: QScope):
        if self.config.qconfig is not None:
            return self.config.qconfig.scope == target_scope
        elif self.config.details:
            active_branches = [
                cfg for branch in FA3_BRANCHES if (cfg := getattr(self.config.details, branch)) is not None
            ]
            return all(cfg.scope == target_scope for cfg in active_branches)
        return False

    def is_data_free(self) -> bool:
        return self.check_scope_condition(QScope.PER_TOKEN) or self.check_scope_condition(QScope.PER_BLOCK)

    def support_distributed(self) -> bool:
        return True

    def preprocess(self, request: BatchProcessRequest) -> None:
        # 1) 调用适配器接口注入占位模块（如果提供）
        # 期望适配器实现方法：install_fa3_placeholders(module, should_inject) -> None
        try:
            self.adapter.inject_fa3_placeholders(
                request.name,
                request.module,
                lambda module_name: (module_name in self.include and module_name not in self.exclude),
            )
        except Exception as e:
            get_logger().warning("install fa3 placeholders at %s failed: %s", request.name, e)

        # 2) 将占位模块替换为监测器（仅静态分支需要校准数据，动态分支保留透传占位符）
        for name, submodule in request.module.named_modules(prefix=request.name):
            if not isinstance(submodule, FA3QuantPlaceHolder):
                continue

            fa_prefix = name.rsplit('.', 1)[-1]
            qconfig = self._resolve_branch_qconfig(fa_prefix)
            if qconfig is None:
                # 未配置分支：保留占位符（纯透传，零开销）
                continue
            if qconfig.scope not in (QScope.PER_HEAD, QScope.PER_CHANNEL):
                # PER_TOKEN / PER_BLOCK 为动态量化，数据无关，无需校准 observer；
                # 保留占位符可避免校准阶段对 Q/K 做无谓统计（如 recall_window 排序）
                continue
            if qconfig.scope == QScope.PER_CHANNEL:
                observer = _FA3PerChannelObserver(name=name)
            else:
                observer = _FA3PerHeadObserver(ratio=submodule.get_ratio(), name=name)
            self.model.set_submodule(name, observer)

        # 3) 设置分布式辅助类
        if dist.is_initialized():
            self.dist_helper = DistHelper(request.module, prefix=request.name)
            for _, submodule in request.module.named_modules(prefix=request.name):
                if not isinstance(submodule, (_FA3PerHeadObserver, _FA3PerChannelObserver)):
                    continue
                submodule.set_dist_helper(self.dist_helper)

    def postprocess(self, request: BatchProcessRequest) -> None:
        # 遍历所有 observer/占位符：静态 observer 先同步统计量再建 IR；动态占位符直接建动态 IR
        for name, submodule in request.module.named_modules(prefix=request.name):
            if not isinstance(submodule, (FA3QuantPlaceHolder, _FA3PerHeadObserver, _FA3PerChannelObserver)):
                continue

            fa_prefix = name.rsplit('.', 1)[-1]
            qconfig = self._resolve_branch_qconfig(fa_prefix)

            if qconfig is None:
                get_logger().debug("No config for %s, skipping", name)
                continue

            is_observer = isinstance(submodule, (_FA3PerHeadObserver, _FA3PerChannelObserver))

            if qconfig.scope in (QScope.PER_TOKEN, QScope.PER_BLOCK):
                # 动态量化：数据无关，占位符（或残留 observer）直接替换为动态 IR
                if qconfig.scope == QScope.PER_TOKEN:
                    self._process_per_token(qconfig, name)
                else:
                    self._process_per_block(qconfig, name)
                continue

            if not is_observer:
                # 静态分支 preprocess 已替换为 observer；仍为占位符说明未走到注入路径
                get_logger().debug("No calibration observer for %s, skipping", name)
                continue

            if qconfig.scope == QScope.PER_HEAD:
                self._process_per_head(qconfig, name, submodule)
            elif qconfig.scope == QScope.PER_CHANNEL:
                self._process_per_channel(qconfig, name, submodule)
            else:
                raise UnsupportedError(
                    f"fa3 quantization does not support following configuration:{qconfig}",
                    action="Please check configuration in .yaml file",
                )
        # 清理 dist_helper
        self.dist_helper = None

    def _process_per_head(self, fa_config: Union[QConfig, FA3AttentionDetails], name: str, submodule: nn.Module):
        # per-head 需要从 observer 获取统计数据
        if submodule.min_val is None:
            raise UnsupportedError(
                f"FA3 quantization at {name} collected no calibration data",
                action="Please ensure a calibration run covers this attention path before postprocess",
            )
        # 形状 (1, H, 1, 1) → (H,)
        min_v = submodule.min_val.squeeze()
        max_v = submodule.max_val.squeeze()
        q_param = calculate_qparam(
            min_val=min_v,
            max_val=max_v,
            q_dtype=fa_config.dtype,
            q_scope=fa_config.scope,
            symmetric=fa_config.symmetric,
        )
        fa_quantizer = qir.AutoFakeQuantActivation.create(q_param)
        self.model.set_submodule(name, fa_quantizer)

    def _process_per_channel(self, fa_config: Union[QConfig, FA3AttentionDetails], name: str, submodule: nn.Module):
        if submodule.min_val is None or submodule.max_val is None:
            raise UnsupportedError(
                f"FA3 per-channel quantization at {name} collected no calibration data",
                action="Please ensure a calibration run covers this attention path before postprocess",
            )
        min_v = submodule.min_val
        max_v = submodule.max_val
        # MX / 对称量化用 absmax 作为 shared_exp 输入
        amax = torch.maximum(min_v.abs(), max_v.abs())
        q_param = calculate_qparam(
            min_val=-amax,
            max_val=amax,
            q_dtype=fa_config.dtype,
            q_scope=fa_config.scope,
            symmetric=fa_config.symmetric,
        )
        fa_quantizer = qir.AutoFakeQuantActivation.create(q_param)
        self.model.set_submodule(name, fa_quantizer)

    def _process_per_token(self, fa_config: Union[QConfig, FA3AttentionDetails], name: str):
        # 创建空的QParam，per-token在forward中动态计算
        q_param = QParam(scheme=fa_config.to_scheme())
        fa_quantizer = qir.AutoFakeQuantActivation.create(q_param)
        self.model.set_submodule(name, fa_quantizer)

    def _process_per_block(self, fa_config: Union[QConfig, FA3AttentionDetails], name: str):
        # 创建空的QParam，per-block在forward中动态计算
        q_param = QParam(scheme=fa_config.to_scheme())
        fa_quantizer = qir.AutoFakeQuantActivation.create(q_param)
        self.model.set_submodule(name, fa_quantizer)
