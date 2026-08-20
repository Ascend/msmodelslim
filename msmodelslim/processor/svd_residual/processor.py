# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

from typing import List, Optional, Literal, Tuple

from msmodelslim.utils.seed import seed_all
from pydantic import Field, ConfigDict
import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger, logger_setter

# 常量定义
DEFAULT_SEED = 42
SVD_LOWRANK_L1_PARAM_NAME = "svd_lowrank_l1"
SVD_LOWRANK_L2_PARAM_NAME = "svd_lowrank_l2"


class SVDResidualProcessorConfig(AutoProcessorConfig):
    """SVD 残差（低秩补偿）处理器配置。

    位于 `spec.process[]`，由 `type: svd_res` 分派；对匹配模块做低秩分解，以残差
    补偿低比特量化带来的精度损失。
    """

    type: Literal["svd_res"] = Field(default="svd_res", description="处理器类型，固定为 `svd_res`。")
    rank: int = Field(default=32, gt=0, description="低秩分解的秩，必须大于0")
    include: List[str] = Field(default_factory=lambda: ["*"], description="包含的模块名称模式，默认 `*` 匹配全部模块")
    exclude: List[str] = Field(default_factory=lambda: [], description="排除的模块名称模式，优先级高于 `include`")

    model_config = ConfigDict(extra="forbid")


def _warning_unmatched_pattern(name: str, config_set: ConfigSet) -> None:
    """警告未匹配的模式"""
    unmatched_keys = [key for key in config_set.unmatched_keys() if key != "*"]
    if unmatched_keys:
        get_logger().warning(
            "These %s patterns are not matched any module, please ensure this is as expected: %s",
            name,
            unmatched_keys,
        )


@QABCRegistry.register(dispatch_key=SVDResidualProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.svd_residual")
class SVDResidualProcessor(AutoSessionProcessor):
    """SVD残差处理器：对Linear层进行低秩分解，将权重替换为残差"""

    def __init__(
        self,
        model: nn.Module,
        config: SVDResidualProcessorConfig,
        adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self.include = ConfigSet(config.include)
        self.exclude = ConfigSet(config.exclude)

    def is_data_free(self) -> bool:
        """只处理低秩分解无需校准集"""
        return True

    def post_run(self) -> None:
        _warning_unmatched_pattern("include", self.include)
        _warning_unmatched_pattern("exclude", self.exclude)

    def process(self, request: BatchProcessRequest) -> None:
        """处理请求：对模块执行SVD分解"""
        self.decompose(request.name, request.module)

    def postprocess(self, request: BatchProcessRequest) -> None:
        self.set_hook_ir(request.module)

    def decompose(self, prefix: str, module: nn.Module) -> None:
        """
        对指定模块内的Linear层执行SVD分解，并更新权重为残差

        Args:
            prefix: 模块前缀名称
            module: 待分解的模块
        """
        seed_all(seed=DEFAULT_SEED, mode=True)

        for name, submodule in module.named_modules(prefix=prefix):
            if self._should_process_module(name, submodule):
                self._decompose_linear_layer(submodule)

    @torch.no_grad()
    def set_hook_ir(self, block: torch.nn.Module) -> None:
        """
        为已分解的Linear层设置前向钩子

        Args:
            block: 待处理的模块块
        """
        for _, child in list(block.named_children()):
            if isinstance(child, nn.Linear):
                self._register_hook_for_linear(child)
            else:
                self.set_hook_ir(child)

    def _should_process_module(self, name: str, submodule: nn.Module) -> bool:
        """判断是否应该处理该模块"""
        if not isinstance(submodule, nn.Linear):
            return False
        if name not in self.include:
            return False
        if name in self.exclude:
            return False
        return True

    def _decompose_linear_layer(self, linear_layer: nn.Linear) -> None:
        """
        对单个Linear层执行SVD分解

        Args:
            linear_layer: 待分解的Linear层
        """
        # 提取权重并记录原始属性
        original_weight = linear_layer.weight
        original_dtype = original_weight.dtype
        original_device = original_weight.device

        # 执行SVD分解
        svd_lowrank_l1, svd_lowrank_l2 = self._perform_svd_decomposition(original_weight)

        # 计算重构矩阵
        reconstructed = self._reconstruct_weight_from_svd_params(svd_lowrank_l1, svd_lowrank_l2)
        reconstructed = reconstructed.to(device=original_device, dtype=original_dtype)

        # 计算残差并更新权重：residual = original_weight - reconstructed
        residual = original_weight - reconstructed
        linear_layer.weight.data = residual

        # 注册svd_lowrank_l1和svd_lowrank_l2参数
        svd_lowrank_l1 = svd_lowrank_l1.to(device=original_device, dtype=original_dtype)
        svd_lowrank_l2 = svd_lowrank_l2.to(device=original_device, dtype=original_dtype)

        self._register_or_update_svd_params(linear_layer, svd_lowrank_l1, svd_lowrank_l2)

    def _register_or_update_svd_params(
        self, linear_layer: nn.Linear, svd_lowrank_l1: torch.Tensor, svd_lowrank_l2: torch.Tensor
    ) -> None:
        """注册或更新Linear层上的SVD低秩参数。"""
        # 注册或更新svd_lowrank_l1参数
        if hasattr(linear_layer, SVD_LOWRANK_L1_PARAM_NAME):
            param = getattr(linear_layer, SVD_LOWRANK_L1_PARAM_NAME)
            param.data = svd_lowrank_l1
        else:
            linear_layer.register_parameter(
                SVD_LOWRANK_L1_PARAM_NAME, nn.Parameter(svd_lowrank_l1, requires_grad=False)
            )

        # 注册或更新svd_lowrank_l2参数
        if hasattr(linear_layer, SVD_LOWRANK_L2_PARAM_NAME):
            param = getattr(linear_layer, SVD_LOWRANK_L2_PARAM_NAME)
            param.data = svd_lowrank_l2
        else:
            linear_layer.register_parameter(
                SVD_LOWRANK_L2_PARAM_NAME, nn.Parameter(svd_lowrank_l2, requires_grad=False)
            )

    def _reconstruct_weight_from_svd_params(
        self, svd_lowrank_l1: torch.Tensor, svd_lowrank_l2: torch.Tensor
    ) -> torch.Tensor:
        """
        根据SVD低秩参数重构近似权重矩阵。
            - svd_lowrank_l1  = V^T，形状 [rank, in_dim]
            - svd_lowrank_l2 = U * S，形状 [out_dim, rank]
            - reconstructed = (US@V^T)
        """
        return svd_lowrank_l2 @ svd_lowrank_l1

    def _perform_svd_decomposition(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行SVD低秩分解

        Args:
            weight: 权重张量

        Returns:
            (svd_lowrank_l1, svd_lowrank_l2): 分解后的两个低秩矩阵
                - svd_lowrank_l1: V^T，形状 [rank, in_dim]
                - svd_lowrank_l2: U * S，形状 [out_dim, rank]
        """
        # 转换为Float32以确保SVD计算的稳定性
        weight_float = weight.float()
        rank = self.config.rank

        max_rank = min(weight_float.shape[0], weight_float.shape[1])
        if rank > max_rank:
            raise SchemaValidateError(
                "rank (%s) must not exceed min(out_features, in_features) = %s "
                "for weight shape %s" % (rank, max_rank, list(weight_float.shape)),
                action="Please set rank <= %s" % max_rank,
            )

        # 使用svd_lowrank进行低秩分解（适合大矩阵）
        # 返回值：U [out_dim, rank], S [rank], V [in_dim, rank]
        with torch.amp.autocast(device_type=weight_float.device.type, enabled=False):
            U, S, V = torch.svd_lowrank(weight_float, q=rank)

        # 构建低秩矩阵
        svd_lowrank_l2 = U[:, :rank] * S[:rank]  # [out_dim, rank]
        svd_lowrank_l1 = V[:, :rank].t()  #  [rank, in_dim]

        return svd_lowrank_l1, svd_lowrank_l2

    def _register_hook_for_linear(self, linear_layer: nn.Linear) -> None:
        """
        为Linear层注册SVD残差钩子

        Args:
            linear_layer: Linear层
        """
        svd_lowrank_l1 = getattr(linear_layer, SVD_LOWRANK_L1_PARAM_NAME, None)
        svd_lowrank_l2 = getattr(linear_layer, SVD_LOWRANK_L2_PARAM_NAME, None)

        # 仅当同时存在有效的svd_lowrank_l1和svd_lowrank_l2时才注册hook
        if svd_lowrank_l1 is None or svd_lowrank_l2 is None:
            return

        hook_ir = qir.SVDResidualHookIR(svd_lowrank_l1, svd_lowrank_l2)
        hook_handle = linear_layer.register_forward_pre_hook(hook_ir)
        hook_ir.set_hook_handle(hook_handle)
