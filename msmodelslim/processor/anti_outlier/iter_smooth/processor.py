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

from typing import Callable, Any, Literal, Annotated, Optional, List, Dict

import torch.distributed as dist
from torch import nn
from pydantic import AfterValidator, Field

from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.norm_bias import RMSNormBias
from msmodelslim.ir.rms_norm import RMSNorm
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.core.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.utils.distributed import DistHelper
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.ir.non_fusion_smooth_quant_ir import NonFusionSmoothQuantHookIR
from msmodelslim.processor.anti_outlier.common.subgraph_type import NonFusionSubgraph
import msmodelslim.utils.validation.pydantic as pydtc

from ..common import (
    IterSmoothConfig,
    IterSmoothContext,
    StatsCollector,
    SubgraphRegistry,
    StatKey,
)
from ..smooth_base import BaseSmoothProcessor
from .api import iter_smooth
from .interface import IterSmoothInterface


class IterSmoothProcessorConfig(AutoProcessorConfig):
    """迭代平滑（IterativeSmooth）处理器配置。

    位于 `spec.process[]`，由 `type: iter_smooth` 分派；通过迭代更新平滑系数
    把激活离群值迁移到权重，直到满足收敛条件。
    """

    type: Literal["iter_smooth"] = Field(default="iter_smooth", description="处理器类型，固定为 `iter_smooth`。")
    alpha: Annotated[float, AfterValidator(pydtc.validate_normalized_value)] = Field(
        default=0.9, description="平滑迁移强度（0~1），越大迁移越多离群值到权重。"
    )
    scale_min: Annotated[float, AfterValidator(pydtc.validate_normalized_value)] = Field(
        default=1e-5, description="平滑缩放因子的最小值下限，防止数值下溢。"
    )
    symmetric: bool = Field(default=True, description="是否对称量化后续的权重/激活。")
    enable_subgraph_type: Annotated[list, AfterValidator(pydtc.is_string_list)] = Field(
        default_factory=lambda: ["norm-linear", "linear-linear", "ov", "up-down"],
        description="应用迭代平滑的子图类型列表，默认 `norm-linear`、`linear-linear`、`ov`、`up-down`。",
    )
    include: Optional[List[Annotated[str, AfterValidator(pydtc.validate_str_length())]]] = Field(
        default=None, description="包含的模块名称模式；不设置表示全部匹配。"
    )
    exclude: Optional[List[Annotated[str, AfterValidator(pydtc.validate_str_length())]]] = Field(
        default=None, description="排除的模块名称模式，优先级高于 `include`。"
    )


class IterStatsCollector(StatsCollector):
    ASYM_SUPPORT_SUBGRAPH_TYPES = ["norm-linear"]

    def __init__(self, symmetric: bool):
        super().__init__()
        self.symmetric = symmetric
        self.dist_helper: Optional[DistHelper] = None
        self.minmax_observers: Dict[str, MsMinMaxObserver] = {}
        self.channel_max_observers: Dict[str, MsMinMaxObserver] = {}

    def set_dist_helper(self, dist_helper: Optional[DistHelper]):
        """设置分布式辅助类"""
        self.dist_helper = dist_helper

    def create_hook(self, name: str, subgraph_type: str = None) -> Callable:
        def stats_hook(module: nn.Linear, input_tensor: tuple, output: Any) -> None:
            if not input_tensor or not isinstance(input_tensor, tuple):
                get_logger().warning("Input tensor is empty for module %s", name)
                return

            tensor = input_tensor[0]
            if name not in self.act_stats:
                self.act_stats[name] = {}
                self.act_stats[name][StatKey.TENSOR] = tensor.cpu()

            hidden_dim = tensor.shape[-1]
            tensor = tensor.reshape(-1, hidden_dim).detach()

            stats_dict = self.act_stats[name]

            if name not in self.minmax_observers:
                observer_config = MinMaxObserverConfig(dim=0, keepdim=False)
                self.minmax_observers[name] = MsMinMaxObserver(observer_config)

            # 根据模块是否共享决定是否同步
            sync = self.dist_helper.is_shared(name) if self.dist_helper is not None else False
            self.minmax_observers[name].update(tensor, sync=sync)
            coming_min, coming_max = self.minmax_observers[name].get_min_max()

            stats_dict[StatKey.STAT_KEY_MAX] = coming_max
            stats_dict[StatKey.STAT_KEY_MIN] = coming_min

            stats_dict[StatKey.STAT_KEY_SHIFT] = (coming_max + coming_min) / 2

            if name not in self.channel_max_observers:
                observer_config = MinMaxObserverConfig(dim=0, keepdim=False)
                self.channel_max_observers[name] = MsMinMaxObserver(observer_config)

            # 根据symmetric/asymmetric模式计算channel_max
            if not self.symmetric and subgraph_type in self.ASYM_SUPPORT_SUBGRAPH_TYPES:
                # asymmetric模式：计算shift后的绝对值最大值
                shifted_tensor = (tensor - stats_dict[StatKey.STAT_KEY_SHIFT]).abs()
                self.channel_max_observers[name].update(shifted_tensor, sync=sync)
            else:
                # symmetric模式：计算绝对值最大值
                abs_tensor = tensor.abs()
                self.channel_max_observers[name].update(abs_tensor, sync=sync)

            _, channel_max = self.channel_max_observers[name].get_min_max()
            stats_dict[StatKey.STAT_KEY_SMOOTH_SCALE] = channel_max

        return stats_hook

    def clear_stats(self) -> None:
        """清除统计信息和observer"""
        super().clear_stats()
        # 重置所有observer
        for observer in self.minmax_observers.values():
            observer.reset()
        for observer in self.channel_max_observers.values():
            observer.reset()
        self.minmax_observers.clear()
        self.channel_max_observers.clear()


@QABCRegistry.register(dispatch_key=IterSmoothProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.iter_smooth")
class IterSmoothProcessor(BaseSmoothProcessor):
    def __init__(self, model: nn.Module, config: IterSmoothProcessorConfig, adapter: object, **kwargs):
        super().__init__(model, config, adapter)
        self.config = config
        self._validate_parameters()
        self.stats_collector = IterStatsCollector(symmetric=config.symmetric)

        # 初始化分布式辅助类
        self.dist_helper = None

        if not config.symmetric:
            supported_types = ', '.join(IterStatsCollector.ASYM_SUPPORT_SUBGRAPH_TYPES)
            get_logger().warning(
                "Detected asymmetric IterSmooth; currently only supports %s subgraph types", supported_types
            )

    def support_distributed(self) -> bool:
        return True

    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        subgraph_type = SubgraphRegistry.get_name(type(subgraph_obj))
        if subgraph_type not in IterStatsCollector.ASYM_SUPPORT_SUBGRAPH_TYPES:
            shift_value = False
            get_logger().debug("Non-asym subgraph (%s), setting shift=False", subgraph_type)
        else:
            shift_value = not self.config.symmetric
            get_logger().debug("Asym-capable subgraph (%s), setting shift=%s", subgraph_type, shift_value)

        iter_smooth_cfg = IterSmoothConfig(alpha=self.config.alpha, shift=shift_value, scale_min=self.config.scale_min)
        smooth_context = self._build_smooth_context(linear_names)
        if smooth_context is None:
            get_logger().warning(
                "No statistics collected for %s subgraph, skipping. This may happen for unused MOE experts.",
                subgraph_type,
            )
            return

        scales = iter_smooth(subgraph_obj, iter_smooth_cfg, smooth_context)
        if scales is not None and isinstance(subgraph_obj, NonFusionSubgraph):
            for linear_module in subgraph_obj.linears:
                hook_ir = NonFusionSmoothQuantHookIR(scales)
                hook_handle = linear_module.register_forward_pre_hook(hook_ir)
                hook_ir.set_hook_handle(hook_handle)

        get_logger().info("Successfully applied IterSmooth to %s subgraph (shift=%s)", subgraph_type, shift_value)

    def preprocess(self, request: BatchProcessRequest) -> None:
        # 在preprocess时创建DistHelper，传入prefix信息
        if dist.is_initialized():
            self.dist_helper = DistHelper(request.module, prefix=request.name)
            self.stats_collector.set_dist_helper(self.dist_helper)

        super().preprocess(request)
        self._replace_norm_modules()
        get_logger().debug("Processed %d subgraphs for submodule %s", len(self.adapter_config), request.name)

    def postprocess(self, request: BatchProcessRequest) -> None:
        super().postprocess(request)
        # 清理分布式辅助类
        self.stats_collector.set_dist_helper(None)
        self.dist_helper = None

    def _build_smooth_context(self, linear_names: List[str]) -> Optional[IterSmoothContext]:
        a_smooth_scale = None
        shift = None

        if not linear_names:
            get_logger().warning(
                "No linear modules provided while building IterSmoothContext; skipping smooth application."
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
        # 创建 IterSmoothContext
        smooth_context = IterSmoothContext(version=1, a_smooth_scale=a_smooth_scale, shift=shift)

        return smooth_context

    def _replace_norm_modules(self) -> None:
        for adapter_config in self.adapter_config:
            if adapter_config.subgraph_type != "norm-linear":
                continue
            norm_name = adapter_config.mapping.source
            norm_module = self.model.get_submodule(norm_name) if norm_name else None
            if not norm_name or norm_module is None:
                continue
            if not hasattr(norm_module, 'weight'):
                get_logger().warning("Norm module %s does not have weight attribute", norm_name)
                continue
            try:
                hidden_size = norm_module.weight.shape[-1]
                need_bias = not self.config.symmetric or hasattr(norm_module, 'bias')

                norm_bias = RMSNormBias(hidden_size) if need_bias else RMSNorm(hidden_size)
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
        """Validate that the adapter implements IterSmoothInterface."""
        if not isinstance(adapter, IterSmoothInterface):
            get_logger().warning(
                '%s does not implement IterSmoothInterface. Fallback to default model adapter logic (hook-based auto-detect). '
                'To use model-specific config, ensure %s inherits from IterSmoothInterface and implements the methods defined by the interface',
                adapter.__class__.__name__,
                adapter.__class__.__name__,
            )
            self.is_defalut_adapter = True
