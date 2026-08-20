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

import torch
import torch.distributed as dist
from torch import nn

from typing import Callable, Any, Literal, Annotated, List, Optional, Dict
from tqdm import tqdm

from pydantic import AfterValidator, Field, model_validator
from pydantic_core import PydanticUndefined
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.utils.distributed.task_scheduler import DistributedTaskScheduler
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.core.observer import MsMinMaxObserver, MinMaxObserverConfig
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.distributed.dist_ops import sync_gather_tensor_lists
from msmodelslim.utils.distributed import DistHelper, sync_base_operation
import msmodelslim.utils.validation.pydantic as pydtc
from msmodelslim.ir.non_fusion_smooth_quant_ir import NonFusionSmoothQuantHookIR
from msmodelslim.processor.anti_outlier.common.subgraph_type import NonFusionSubgraph
from ..common import (
    FlexSmoothQuantConfig,
    FlexAWQSSZConfig,
    FlexSmoothQuantContext,
    FlexAWQSSZContext,
    StatsCollector,
    SubgraphRegistry,
    StatKey,
)
from ..smooth_base import BaseSmoothProcessor
from .api import flex_smooth_quant, flex_awq_ssz
from .interface import FlexSmoothQuantInterface


class FlexSmoothBaseProcessorConfig(AutoProcessorConfig):
    """Flex 平滑类处理器的公共基类配置，定义 alpha/beta 平滑强度与子图类型等公共字段。"""

    type: Literal["_abstract_flex_smooth_base_"] = "_abstract_flex_smooth_base_"

    alpha: Annotated[float, AfterValidator(pydtc.validate_normalized_value)] = Field(
        default=None, description="激活→权重的平滑迁移强度（0~1），越大迁移越多离群值到权重。"
    )
    beta: Annotated[float, AfterValidator(pydtc.validate_normalized_value)] = Field(
        default=None, description="额外的平滑调优系数（0~1），配合 alpha 使用。"
    )
    enable_subgraph_type: Annotated[list, AfterValidator(pydtc.is_string_list)] = Field(
        default_factory=lambda: ["norm-linear", "linear-linear", "ov", "up-down"],
        description="应用平滑的子图类型列表，默认 `norm-linear`、`linear-linear`、`ov`、`up-down`。",
    )
    include: Optional[List[Annotated[str, AfterValidator(pydtc.validate_str_length())]]] = Field(
        default=None, description="包含的模块名称模式；不设置表示全部匹配。"
    )
    exclude: Optional[List[Annotated[str, AfterValidator(pydtc.validate_str_length())]]] = Field(
        default=None, description="排除的模块名称模式，优先级高于 `include`。"
    )


class FlexSmoothQuantProcessorConfig(FlexSmoothBaseProcessorConfig):
    """FlexSmoothQuant 平滑量化处理器配置。

    位于 `spec.process[]`，由 `type: flex_smooth_quant` 分派；在 SmoothQuant 基础上
    增加 `beta` 调优，并支持子图级平滑。
    """

    type: Literal["flex_smooth_quant"] = Field(
        default="flex_smooth_quant", description="处理器类型，固定为 `flex_smooth_quant`。"
    )


class FlexAWQSSZProcessorConfig(FlexSmoothBaseProcessorConfig):
    """FlexAWQSSZ 平滑+AWQ 处理器配置。

    位于 `spec.process[]`，由 `type: flex_awq_ssz` 分派；结合平滑（alpha/beta）与
    AWQ 风格仅权重量化，`qconfig` 指定激活与权重的量化方式。
    """

    qconfig: LinearQConfig = Field(description="激活与权重的量化配置，见《LinearQConfig 配置说明》。")
    type: Literal["flex_awq_ssz"] = Field(default="flex_awq_ssz", description="处理器类型，固定为 `flex_awq_ssz`。")

    @model_validator(mode="before")
    @classmethod
    def check_qconfig_missing(cls, values: dict) -> dict:
        """模型级前置校验：拦截 qconfig 缺失的场景"""
        if "qconfig" not in values or values["qconfig"] is PydanticUndefined:
            raise SchemaValidateError(
                "qconfig is a required parameter for flex_awq_ssz processor",
                action=(
                    "Please provide qconfig parameter in the YAML configuration, "
                    "including act and weight quantization settings"
                ),
            )
        return values


class FlexStatsCollector(StatsCollector):
    """
    Flex smooth statistics collector
    """

    def __init__(self):
        super().__init__()
        self.dist_helper: Optional[DistHelper] = None
        # 为每个模块名称创建observer，用于收集channel_max统计
        self.observers: Dict[str, MsMinMaxObserver] = {}

    def set_dist_helper(self, dist_helper: Optional[DistHelper]):
        """设置分布式辅助类"""
        self.dist_helper = dist_helper

    def create_hook(self, name: str, subgraph_type: str = None) -> Callable:
        def stats_hook(module: nn.Linear, input_tensor: tuple, output: Any) -> None:
            # 有的路由专家可能采集不到激活，需要跳过
            if not input_tensor or not isinstance(input_tensor, tuple) or input_tensor[0].numel() == 0:
                get_logger().warning("Input tensor is empty for module %s", name)
                return

            tensor = input_tensor[0]
            hidden_dim = tensor.shape[-1]
            tensor = tensor.reshape(-1, hidden_dim).detach()

            if name not in self.act_stats:
                self.act_stats[name] = {}
            module_stats = self.act_stats[name]

            # 收集tensor用于后续算法
            cpu_tensor = tensor.to("cpu").reshape(-1, tensor.shape[-1])
            if StatKey.TENSOR not in module_stats:
                module_stats[StatKey.TENSOR] = [cpu_tensor]
            else:
                module_stats[StatKey.TENSOR].append(cpu_tensor)

            if name not in self.observers:
                observer_config = MinMaxObserverConfig(dim=0, keepdim=False)
                self.observers[name] = MsMinMaxObserver(observer_config)
            abs_tensor = tensor.abs()
            self.observers[name].update(abs_tensor)
            _, channel_max = self.observers[name].get_min_max()
            module_stats[StatKey.STAT_KEY_SMOOTH_SCALE] = channel_max

        return stats_hook

    def sync_act_stats(self, on_cpu: bool = False) -> None:
        """
        在各 rank 间汇总 ``act_stats[name][StatKey.TENSOR]`` 列表（原地写回），
        并对 ``StatKey.STAT_KEY_SMOOTH_SCALE``（channel_max）做跨 rank **逐元素取大**
        （``all_reduce(MAX)``），与原先在 hook 内对 observer 做 ``sync`` 时的语义一致。

        仅处理「所有 rank 上都存在非空 TENSOR 列表」的模块名，以保证 collective 对齐。
        rank 0 上进度通过 tqdm  postfix 单行刷新；结束后仅打一条汇总 info 日志。
        """
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        world_size = dist.get_world_size()
        my_tensor_keys = {k for k, st in self.act_stats.items() if st.get(StatKey.TENSOR)}
        gathered_key_sets: List[Optional[set]] = [None] * world_size
        dist.all_gather_object(gathered_key_sets, my_tensor_keys)
        key_sets = [s for s in gathered_key_sets if s is not None]
        if not key_sets:
            return
        keys_to_sync = sorted(set.intersection(*key_sets))
        if not keys_to_sync:
            get_logger().info("sync_act_stats: no module keys with non-empty TENSOR on all ranks; skip gather.")
            return
        show_bar = dist.get_rank() == 0
        bar_desc = f"sync_act_stats (on_cpu={on_cpu})"
        pbar = tqdm(
            keys_to_sync,
            desc=bar_desc,
            disable=not show_bar,
            unit="module",
        )
        for name in pbar:
            # 同步 StatKey.TENSOR
            local_tensors = self.act_stats[name][StatKey.TENSOR]
            merged = sync_gather_tensor_lists(local_tensors, on_cpu=on_cpu)
            self.act_stats[name][StatKey.TENSOR] = merged

            # 同步 StatKey.STAT_KEY_SMOOTH_SCALE
            sm = self.act_stats[name].get(StatKey.STAT_KEY_SMOOTH_SCALE)
            if not isinstance(sm, torch.Tensor):
                raise SchemaValidateError(
                    f"sync_act_stats: missing STAT_KEY_SMOOTH_SCALE for {name!r} while TENSOR is present; "
                    "cannot run collective max across ranks.",
                    action="Ensure hooks record smooth scale before sync_act_stats.",
                )
            # 各 rank 本地 channel_max 在 CPU float 上做 MAX 规约，再写回原 dtype/设备
            buf = sm.detach().to(dtype=torch.float32).contiguous()
            if buf.device.type == "cpu":
                if hasattr(torch, "npu") and torch.npu.is_available():
                    buf = buf.to(f"npu:{torch.npu.current_device()}")
            sync_base_operation(buf, op="max")
            self.act_stats[name][StatKey.STAT_KEY_SMOOTH_SCALE] = buf.to(device=sm.device, dtype=sm.dtype)

            # 更新进度条
            if show_bar:
                disp = name if len(name) <= 56 else f"{name[:53]}..."
                pbar.set_postfix_str(f"{disp} | n={len(merged)}", refresh=True)
        if show_bar:
            get_logger().info(
                "sync_act_stats (on_cpu=%s): finished %d module(s)",
                on_cpu,
                len(keys_to_sync),
            )

    def clear_stats(self) -> None:
        super().clear_stats()
        for observer in self.observers.values():
            observer.reset()
        self.observers.clear()


class FlexSmoothBaseProcessor(BaseSmoothProcessor):
    def __init__(self, model: nn.Module, config: FlexSmoothBaseProcessorConfig, adapter: object, **kwargs):
        super().__init__(model, config, adapter)
        self.config = config
        self._validate_parameters()
        self.stats_collector = FlexStatsCollector()
        self.sorted_configs = None
        self.dist_helper = None

    def support_distributed(self) -> bool:
        return True

    def preprocess(self, request: BatchProcessRequest) -> None:
        # 在preprocess时创建DistHelper，传入prefix信息
        if dist.is_initialized():
            self.dist_helper = DistHelper(request.module, prefix=request.name)
            self.stats_collector.set_dist_helper(self.dist_helper)

        super().preprocess(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        # 同步激活统计信息，必须在super().postprocess(request)清理资源之前执行
        self.stats_collector.sync_act_stats(on_cpu=False)
        super().postprocess(request)
        # 清理分布式辅助类
        self.stats_collector.set_dist_helper(None)
        self.dist_helper = None

    def _validate_adapter_interface(self, adapter: object) -> None:
        """Validate that the adapter implements FlexSmoothQuantInterface."""
        if not isinstance(adapter, FlexSmoothQuantInterface):
            get_logger().warning(
                '%s does not implement FlexSmoothQuantInterface. Fallback to default model adapter logic (hook-based auto-detect). '
                'To use model-specific config, ensure %s inherits from FlexSmoothQuantInterface and implements the methods defined by the interface',
                adapter.__class__.__name__,
                adapter.__class__.__name__,
            )
            self.is_defalut_adapter = True

    def _worker_fn(self, idx) -> None:
        adapter_config = self.sorted_configs[idx - 1]
        priority = self.SUBGRAPH_PRIORITY.get(adapter_config.subgraph_type, 999)
        module_name = (
            adapter_config.mapping.source if adapter_config.mapping.source else adapter_config.mapping.targets[0]
        )
        get_logger().debug("  %d. %s (priority: %d) - %s", idx, adapter_config.subgraph_type, priority, module_name)
        self._process_single_subgraph(adapter_config)

    def _process_subgraphs_by_priority(self) -> None:
        """Process subgraphs in priority order"""
        get_logger().debug("Starting smoothing application")
        self.sorted_configs = sorted(
            self.adapter_config, key=lambda x: self.SUBGRAPH_PRIORITY.get(x.subgraph_type, 999)
        )

        if not self.sorted_configs:
            get_logger().warning("No subgraphs to process for current layer.")
            return

        with DistributedTaskScheduler(self.model) as scheduler:
            for idx, adapter_config in enumerate(self.sorted_configs, start=1):
                m = adapter_config.mapping
                is_non_fusion = m.source is None and m.targets is not None
                has_non_shared_module = False
                if self.dist_helper is not None:
                    module_names = []
                    if m.source is not None:
                        module_names.append(m.source)
                    if m.targets is not None:
                        module_names.extend(m.targets)
                    has_non_shared_module = any(not self.dist_helper.is_shared(name) for name in module_names)
                scheduler.submit(
                    fn=self._worker_fn,
                    args=(idx,),
                    dependencies=([m.source] if m.source else []) + list(m.targets),
                    parallel=not (is_non_fusion or has_non_shared_module),
                )

            scheduler.run()

        self.sorted_configs = None


@QABCRegistry.register(dispatch_key=FlexSmoothQuantProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.flex_smooth_quant")
class FlexSmoothQuantProcessor(FlexSmoothBaseProcessor):
    """FlexSmoothQuant Processor"""

    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        """Apply FlexSmoothQuant algorithm"""
        subgraph_type = SubgraphRegistry.get_name(type(subgraph_obj))
        config = FlexSmoothQuantConfig(
            alpha=self.config.alpha, beta=self.config.beta, extra_config=getattr(subgraph_obj, 'extra_config', None)
        )
        smooth_context = self._build_smooth_context(linear_names)
        if smooth_context is None:
            get_logger().warning(
                "No statistics collected for %s subgraph, skipping. This may happen for unused MOE experts.",
                subgraph_type,
            )
            return
        scales = flex_smooth_quant(subgraph_obj, config, smooth_context)
        if scales is not None and isinstance(subgraph_obj, NonFusionSubgraph):
            for linear_module in subgraph_obj.linears:
                hook_ir = NonFusionSmoothQuantHookIR(scales)
                hook_handle = linear_module.register_forward_pre_hook(hook_ir)
                hook_ir.set_hook_handle(hook_handle)
        get_logger().info("Successfully applied FlexSmoothQuant to %s subgraph", subgraph_type)

    def _build_smooth_context(self, linear_names: List[str]) -> Optional[FlexSmoothQuantContext]:
        a_smooth_scale = None
        tensors = None
        if not linear_names:
            get_logger().warning(
                "No linear modules provided while building FlexSmoothQuantContext; skipping smooth application."
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

            # 获取 tensors
            if StatKey.TENSOR in stats:
                tensors = stats[StatKey.TENSOR]
            else:
                tensors = None
        else:
            get_logger().warning("Linear name %s not in act_stats", linear_name)
            return None

        # 检查是否成功获取到激活平滑尺度
        if a_smooth_scale is None or tensors is None:
            # 返回 None 而不是抛出异常，让调用者决定如何处理
            get_logger().debug(
                "Failed to get activation smooth scale from linear name %s. "
                "This may happen for unused subgraphs (e.g., unactivated MOE experts).",
                linear_name,
            )
            return None

        # 创建 FlexSmoothQuantContext
        smooth_context = FlexSmoothQuantContext(
            version=1,
            a_smooth_scale=a_smooth_scale,
            tensors=tensors,
        )

        return smooth_context


@QABCRegistry.register(dispatch_key=FlexAWQSSZProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.flex_awq_ssz")
class FlexAWQSSZProcessor(FlexSmoothBaseProcessor):
    """FlexAWQSSZ Processor"""

    def apply_smooth_algorithm(self, subgraph_obj: Any, linear_names: List[str]) -> None:
        """Apply FlexAWQSSZ algorithm"""
        subgraph_type = SubgraphRegistry.get_name(type(subgraph_obj))
        config = FlexAWQSSZConfig(alpha=self.config.alpha, beta=self.config.beta, qconfig=self.config.qconfig)
        smooth_context = self._build_smooth_context(linear_names)
        if smooth_context is None:
            get_logger().warning(
                "No statistics collected for %s subgraph, skipping. This may happen for unused MOE experts.",
                subgraph_type,
            )
            return
        flex_awq_ssz(subgraph_obj, config, smooth_context)
        get_logger().info("Successfully applied FlexAWQSSZ to %s subgraph", subgraph_type)

    def _build_smooth_context(self, linear_names: List[str]) -> Optional[FlexAWQSSZContext]:
        tensors = None
        if not linear_names:
            get_logger().warning(
                "No linear modules provided while building FlexSmoothQuantContext; skipping smooth application."
            )
            return None
        # 仅用第一个linear的激活统计信息
        linear_name = linear_names[0]

        # 获取激活统计信息
        if linear_name in self.stats_collector.act_stats:
            stats = self.stats_collector.act_stats[linear_name]

            # 获取 tensors
            if StatKey.TENSOR in stats:
                tensors = stats[StatKey.TENSOR]
            else:
                tensors = None
        else:
            get_logger().warning("Linear name %s not in act_stats", linear_name)
            return None

        # 检查是否成功获取到激活平滑尺度
        if tensors is None:
            # 返回 None 而不是抛出异常，让调用者决定如何处理
            get_logger().debug(
                "Failed to get activation tensors from linear name %s. "
                "This may happen for unused subgraphs (e.g., unactivated MOE experts).",
                linear_name,
            )
            return None

        # 创建 FlexAWQSSZContext
        smooth_context = FlexAWQSSZContext(version=1, tensors=tensors)

        return smooth_context
