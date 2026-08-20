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

from typing import Any, Literal, Annotated, Optional, List

from torch import nn
from pydantic import AfterValidator, Field

from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.norm_bias import RMSNormBias
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.validation.pydantic import validate_normalized_value, validate_str_length

from ..common import (
    SmoothQuantConfig,
    SmoothQuantContext,
    SubgraphRegistry,
    StatKey,
)
from ..smooth_base import BaseSmoothProcessor
from ..iter_smooth import IterStatsCollector
from .api import smooth_quant
from .interface import SmoothQuantInterface


class SmoothQuantProcessorConfig(AutoProcessorConfig):
    """SmoothQuant 平滑量化处理器配置。

    位于 `spec.process[]`，由 `type: smooth_quant` 分派；把激活的离群值按 `alpha`
    迁移到权重上，使激活便于量化，通常与线性量化配合使用。
    """

    type: Literal["smooth_quant"] = Field(default="smooth_quant", description="处理器类型，固定为 `smooth_quant`。")
    alpha: Annotated[float, AfterValidator(validate_normalized_value)] = Field(
        default=0.5, description="平滑迁移强度（0~1），越大表示把越多的激活离群值迁移到权重。"
    )
    symmetric: bool = Field(default=True, description="是否对称量化；影响平滑后量化的对称性。")
    include: Optional[List[Annotated[str, AfterValidator(validate_str_length())]]] = Field(
        default=None, description="包含的模块名称模式；不设置表示全部匹配。"
    )
    exclude: Optional[List[Annotated[str, AfterValidator(validate_str_length())]]] = Field(
        default=None, description="排除的模块名称模式，优先级高于 `include`。"
    )


@QABCRegistry.register(dispatch_key=SmoothQuantProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.smooth_quant")
class SmoothQuantProcessor(BaseSmoothProcessor):
    """SmoothQuant Processor - 仅支持 norm-linear 子图类型"""

    ENABLE_SUBGRAPH_TYPE = ["norm-linear"]

    def __init__(self, model: nn.Module, config: SmoothQuantProcessorConfig, adapter: object, **kwargs):
        super().__init__(model, config, adapter)
        self._validate_parameters()
        self.stats_collector = IterStatsCollector(symmetric=config.symmetric)

        # 初始化分布式辅助类（延迟到preprocess时创建，因为需要prefix信息）
        self.dist_helper = None

    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        subgraph_type = SubgraphRegistry.get_name(type(subgraph_obj))
        if subgraph_type not in IterStatsCollector.ASYM_SUPPORT_SUBGRAPH_TYPES:
            shift_value = False
            get_logger().debug("Non-asym subgraph (%s), setting shift=False", subgraph_type)
        else:
            shift_value = not self.config.symmetric
            get_logger().debug("Asym-capable subgraph (%s), setting shift=%s", subgraph_type, shift_value)

        smooth_quant_cfg = SmoothQuantConfig(
            alpha=self.config.alpha,
            shift=shift_value,
        )
        smooth_context = self._build_smooth_context(linear_names)
        if smooth_context is None:
            get_logger().warning(
                "No statistics collected for %s subgraph, skipping. This may happen for unused MOE experts.",
                subgraph_type,
            )
            return

        smooth_quant(subgraph_obj, smooth_quant_cfg, smooth_context)
        get_logger().info("Successfully applied SmoothQuant to %s subgraph (shift=%s)", subgraph_type, shift_value)

    def preprocess(self, request: BatchProcessRequest) -> None:
        super().preprocess(request)
        self._replace_norm_modules()
        get_logger().debug("Processed %d subgraphs for submodule %s", len(self.adapter_config), request.name)

    def _validate_parameters(self) -> None:
        """SmoothQuant 使用固定的子图类型，无需验证相关配置"""
        pass

    def _filter_adapter_configs_by_config(self, adapter_configs, config, scope):
        """重写过滤方法，使用固定的 ENABLE_SUBGRAPH_TYPE"""
        result = []
        layer_prefix = f"{scope}." if scope != "" else ""
        include = ConfigSet(config.include) if config.include else ConfigSet(["*"])
        exclude = ConfigSet(config.exclude) if config.exclude else ConfigSet([])

        for adapter_config in adapter_configs:
            if adapter_config.subgraph_type not in self.ENABLE_SUBGRAPH_TYPE:
                continue
            if not adapter_config.mapping:
                continue

            source_name = adapter_config.mapping.source
            if not source_name.startswith(layer_prefix):
                continue
            if source_name not in include:
                continue
            if source_name in exclude:
                continue

            result.append(adapter_config)

        return result

    def _build_smooth_context(self, linear_names: List[str]) -> Optional[SmoothQuantContext]:
        a_smooth_scale = None
        shift = None

        if not linear_names:
            get_logger().warning(
                "No linear modules provided while building SmoothQuantContext; skipping smooth application."
            )
            return None
        # 仅用第一个linear的激活统计信息
        linear_name = linear_names[0]

        # 获取激活统计信息
        if linear_name in self.stats_collector.act_stats:
            stats = self.stats_collector.act_stats[linear_name]

            # 获取 smooth_scale
            if StatKey.STAT_KEY_SMOOTH_SCALE in stats:
                a_smooth_scale = stats[StatKey.STAT_KEY_SMOOTH_SCALE]
            else:
                a_smooth_scale = None

            # 获取 shift
            if StatKey.STAT_KEY_SHIFT in stats:
                shift = stats[StatKey.STAT_KEY_SHIFT]
            else:
                shift = None
        else:
            get_logger().warning("Linear name %s not in act_stats", linear_name)
            return None

        # 检查是否成功获取到激活平滑尺度
        if a_smooth_scale is None:
            # 返回 None 而不是抛出异常，让调用者决定如何处理
            get_logger().debug(
                "Failed to get activation smooth scale from linear name {linear_name}. "
                "This may happen for unused subgraphs (e.g., unactivated MOE experts)."
            )
            return None
        # 创建 SmoothQuantContext
        smooth_context = SmoothQuantContext(version=1, a_smooth_scale=a_smooth_scale, shift=shift)

        return smooth_context

    def _replace_norm_modules(self) -> None:
        for adapter_config in self.adapter_config:
            if adapter_config.subgraph_type != "norm-linear":
                continue
            norm_name = adapter_config.mapping.source
            norm_module = self.model.get_submodule(norm_name) if norm_name else None
            if not norm_name or norm_module is None:
                continue
            try:
                if not hasattr(norm_module, 'weight'):
                    get_logger().warning("Norm module %s does not have weight attribute", norm_name)
                    continue
                norm_bias = RMSNormBias(norm_module.weight.shape[-1])
                norm_bias.weight.data.copy_(norm_module.weight.data)
                norm_bias.weight.data = norm_bias.weight.data.type(norm_module.weight.data.dtype)
                if hasattr(norm_module, 'bias') and norm_module.bias is not None:
                    norm_bias.bias.data.copy_(norm_module.bias.data)
                    norm_bias.bias.data = norm_bias.bias.data.type(norm_module.weight.data.dtype)
                norm_bias.to(norm_module.weight.data.device)
                self.model.set_submodule(norm_name, norm_bias)
                get_logger().debug("%s: %s -> %s", norm_name, type(norm_module), type(norm_bias))
            except Exception as e:
                get_logger().warning("Failed to replace norm module %s: %s", norm_name, e)

    def _validate_adapter_interface(self, adapter: object) -> None:
        """Validate that the adapter implements SmoothQuantInterface."""
        if not isinstance(adapter, SmoothQuantInterface):
            get_logger().warning(
                '%s does not implement SmoothQuantInterface. Fallback to default model adapter logic (hook-based auto-detect). '
                'To use model-specific config, ensure %s inherits from SmoothQuantInterface and implements the methods defined by the interface',
                adapter.__class__.__name__,
                adapter.__class__.__name__,
            )
            self.is_defalut_adapter = True
