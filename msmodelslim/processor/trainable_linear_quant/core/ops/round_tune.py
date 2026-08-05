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

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
from pydantic import Field
from torch import nn

from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.plugin import register_plugin

from msmodelslim.ir.qal import QDType
from msmodelslim.processor.trainable_linear_quant.core.kernels.base import FakeQuantStage
from msmodelslim.processor.trainable_linear_quant.core.wrapper import TrainableLinearQuantWrapper
from msmodelslim.processor.trainable_linear_quant.core.kernels.context import (
    FakeQuantContext,
    kernel_family_from_config,
)
from msmodelslim.processor.trainable_linear_quant.core.kernels.int import IntFakeQuantContext, int_max_bound
from msmodelslim.processor.trainable_linear_quant.core.kernels.mxfp import (
    MxFakeQuantContext,
    _mx_format,
    _reshape_blocks_last_dim,
)

from .base import LinearTLQOp, TLQOpConfig, format_tensor_dbg


class RoundTuneOpConfig(TLQOpConfig):
    type: Literal["round_tune"] = Field(default="round_tune", description="插件类型：round_tune")
    lr: Optional[float] = Field(default=None, gt=0.0)


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def _clamp_v(v: Any, q_dtype: QDType, max_bound: int) -> Any:
    if not isinstance(v, torch.Tensor):
        return v
    if q_dtype == QDType.INT4:
        return torch.clamp(v, min=-0.5, max=0.5)
    if q_dtype == QDType.INT8:
        return torch.clamp(v, min=-1.0, max=1.0)
    _ = max_bound
    return v


def _align_v_to_normed(v: Any, normed: torch.Tensor, weight: torch.Tensor, config) -> Any:
    """将 ``value`` 对齐到 MX blocked ``normed`` 形状（与 auto-round ``quant_mx`` 一致）。"""
    if not isinstance(v, torch.Tensor):
        return v
    if v.shape == normed.shape:
        return v
    mx_fmt = _mx_format(config)
    blocked, _, _ = _reshape_blocks_last_dim(v.to(device=normed.device, dtype=normed.dtype), mx_fmt.block_size)
    if blocked.shape == normed.shape:
        return blocked
    return v


class RoundTuneOp(LinearTLQOp):
    """对齐 auto-round round tuning：INT 为整型 grid；MXFP 为 ``normed + v``（``quant_mx``）。"""

    PARAM_KEY = "value"

    def __init__(
        self,
        config: RoundTuneOpConfig,
        layer_path: str,
        wrapper: TrainableLinearQuantWrapper,
    ) -> None:
        super().__init__(config, layer_path=layer_path, wrapper=wrapper)
        w_kernel = wrapper.w_kernel
        self._kernel_family = kernel_family_from_config(w_kernel.config)
        if self._kernel_family not in ("int", "mxfp"):
            raise UnsupportedError(f"RoundTuneOp does not support dtype {w_kernel.config.dtype!r}")
        self._value: Optional[nn.Parameter] = None
        self._value_init_dbg = "none"

    def bind(self) -> None:
        weight = self.target_modules[self.layer_path].orig_layer.weight
        dev = weight.device
        self._value = nn.Parameter(torch.zeros(weight.shape, dtype=torch.float32, device=dev), requires_grad=True)
        self._value_init_dbg = format_tensor_dbg(self._value, include_abs_mean=True, include_nnz=True)
        get_logger().info(
            "[round_tune] bind %s family=%s v_init(%s)",
            self.layer_path,
            self._kernel_family,
            self._value_init_dbg,
        )
        wrapper = self.target_modules[self.layer_path]
        wrapper.register_train_op(self, side="weight")
        wrapper.w_kernel.add_listener(self._on_stage, op_id=self.op_id)

    def unbind(self) -> None:
        # load_best_params 已在 Processor finalize 前调用；此处打印最优快照数值便于对比初值。
        get_logger().info(
            "[round_tune] finalize %s family=%s v_init(%s) v(%s)",
            self.layer_path,
            self._kernel_family,
            self._value_init_dbg,
            format_tensor_dbg(self._value, include_abs_mean=True, include_nnz=True),
        )
        self.target_modules[self.layer_path].w_kernel.remove_listeners(self.op_id)

    @property
    def train_params(self) -> dict[str, torch.Tensor]:
        return {self.PARAM_KEY: self._value}

    def _on_stage(self, stage: FakeQuantStage, ctx: FakeQuantContext) -> None:
        if stage != FakeQuantStage.PRE_ROUND:
            return
        if ctx.normed is None:
            raise RuntimeError("RoundTuneOp requires ctx.normed")
        if isinstance(ctx, IntFakeQuantContext):
            self._on_pre_round_int(ctx)
        elif isinstance(ctx, MxFakeQuantContext):
            self._on_pre_round_mx(ctx)

    def _on_pre_round_int(self, ctx: IntFakeQuantContext) -> None:
        if ctx.q_param is None:
            raise RuntimeError("RoundTuneOp requires q_param")

        q_dtype = ctx.q_param.scheme.dtype
        max_bound = int_max_bound(ctx, ctx.config)
        v: Any = ctx.train_tensors.get(self.PARAM_KEY, 0)
        v = _clamp_v(v, q_dtype, max_bound)

        normed = ctx.normed
        if isinstance(v, torch.Tensor) or v not in (0, 0.0, None):
            normed = normed + v

        quantized = _round_ste(normed)
        min_bound = -max_bound - 1
        quantized = quantized.clamp(min=min_bound, max=max_bound)
        ctx.quantized = quantized

    def _on_pre_round_mx(self, ctx: MxFakeQuantContext) -> None:
        weight = self.target_modules[self.layer_path].orig_layer.weight
        v: Any = ctx.train_tensors.get(self.PARAM_KEY, 0)
        v = _align_v_to_normed(v, ctx.normed, weight, ctx.config)

        normed = ctx.normed
        if isinstance(v, torch.Tensor) or v not in (0, 0.0, None):
            normed = normed + v

        mx_fmt = _mx_format(ctx.config)
        ctx.normed = torch.clamp(normed, min=-mx_fmt.max_norm, max=mx_fmt.max_norm)


def _get_round_tune_tlq_op_plugin():
    return RoundTuneOpConfig, RoundTuneOp


register_plugin(_get_round_tune_tlq_op_plugin)
