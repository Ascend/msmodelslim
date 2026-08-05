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

from typing import Literal, Optional

import torch
from pydantic import Field
from torch import nn

from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.plugin import register_plugin

from msmodelslim.processor.trainable_linear_quant.core.kernels.base import FakeQuantStage
from msmodelslim.processor.trainable_linear_quant.core.wrapper import TrainableLinearQuantWrapper
from msmodelslim.processor.trainable_linear_quant.core.kernels.context import (
    FakeQuantContext,
    kernel_family_from_config,
)
from msmodelslim.processor.trainable_linear_quant.core.kernels.int import IntFakeQuantContext
from msmodelslim.processor.trainable_linear_quant.core.kernels.mxfp import MxFakeQuantContext

from .base import LinearTLQOp, TLQOpConfig, format_tensor_dbg


class MinmaxTuneOpConfig(TLQOpConfig):
    type: Literal["minmax_tune"] = Field(default="minmax_tune", description="插件类型：minmax_tune")
    lr: Optional[float] = Field(default=None, gt=0.0)


class MinmaxTuneOp(LinearTLQOp):
    """对齐 auto-round minmax tuning：INT 调对称范围；MXFP 仅 ``max_scale`` 缩放 block max。"""

    MIN_KEY = "min_scale"
    MAX_KEY = "max_scale"

    def __init__(
        self,
        config: MinmaxTuneOpConfig,
        layer_path: str,
        wrapper: TrainableLinearQuantWrapper,
    ) -> None:
        super().__init__(config, layer_path=layer_path, wrapper=wrapper)
        w_kernel = wrapper.w_kernel
        self._kernel_family = kernel_family_from_config(w_kernel.config)
        if self._kernel_family not in ("int", "mxfp"):
            raise UnsupportedError(f"MinmaxTuneOp does not support dtype {w_kernel.config.dtype!r}")
        self._min_scale: Optional[nn.Parameter] = None
        self._max_scale: Optional[nn.Parameter] = None
        self._min_scale_init_dbg = "none"
        self._max_scale_init_dbg = "none"

    def bind(self) -> None:
        weight = self.target_modules[self.layer_path].orig_layer.weight
        shape = (1,) if weight.dim() < 2 else (weight.shape[0],)
        dev = weight.device
        self._min_scale = None
        if self._kernel_family == "int":
            self._min_scale = nn.Parameter(torch.ones(shape, dtype=torch.float32, device=dev), requires_grad=True)
        self._max_scale = nn.Parameter(torch.ones(shape, dtype=torch.float32, device=dev), requires_grad=True)
        self._min_scale_init_dbg = format_tensor_dbg(self._min_scale)
        self._max_scale_init_dbg = format_tensor_dbg(self._max_scale)
        get_logger().info(
            "[minmax_tune] bind %s family=%s min_scale_init(%s) max_scale_init(%s)",
            self.layer_path,
            self._kernel_family,
            self._min_scale_init_dbg,
            self._max_scale_init_dbg,
        )
        wrapper = self.target_modules[self.layer_path]
        wrapper.register_train_op(self, side="weight")
        wrapper.w_kernel.add_listener(self._on_stage, op_id=self.op_id)

    def unbind(self) -> None:
        # load_best_params 已在 Processor finalize 前调用；此处打印最优快照数值便于对比初值。
        get_logger().info(
            "[minmax_tune] finalize %s family=%s min_scale_init(%s) min_scale(%s) max_scale_init(%s) max_scale(%s)",
            self.layer_path,
            self._kernel_family,
            self._min_scale_init_dbg,
            format_tensor_dbg(self._min_scale),
            self._max_scale_init_dbg,
            format_tensor_dbg(self._max_scale),
        )
        self.target_modules[self.layer_path].w_kernel.remove_listeners(self.op_id)

    @property
    def train_params(self) -> dict[str, torch.Tensor]:
        params = {self.MAX_KEY: self._max_scale}
        if self._min_scale is not None:
            params[self.MIN_KEY] = self._min_scale
        return params

    def _on_stage(self, stage: FakeQuantStage, ctx: FakeQuantContext) -> None:
        if stage != FakeQuantStage.BEFORE_QPARAM:
            return
        if isinstance(ctx, IntFakeQuantContext):
            self._on_before_qparam_int(ctx)
        elif isinstance(ctx, MxFakeQuantContext):
            self._on_before_qparam_mx(ctx)

    def _on_before_qparam_int(self, ctx: IntFakeQuantContext) -> None:
        if ctx.min_val is None or ctx.max_val is None:
            return
        min_scale = self._clamped_scale(ctx, self.MIN_KEY, default=1.0)
        max_scale = self._clamped_scale(ctx, self.MAX_KEY, default=1.0)
        ctx.min_abs = -(ctx.min_val * min_scale)
        ctx.max_abs = ctx.max_val * max_scale

    def _on_before_qparam_mx(self, ctx: MxFakeQuantContext) -> None:
        if ctx.block_max is None:
            return
        max_scale = self._clamped_scale(ctx, self.MAX_KEY, default=1.0)
        if isinstance(max_scale, torch.Tensor):
            target_shape = [-1] + [1] * (ctx.block_max.dim() - 1)
            ctx.block_max = ctx.block_max * max_scale.reshape(target_shape).to(ctx.block_max.device)
        else:
            ctx.block_max = ctx.block_max * float(max_scale)

    @staticmethod
    def _clamped_scale(ctx: FakeQuantContext, key: str, default: float) -> torch.Tensor | float:
        if key not in ctx.train_tensors:
            return default
        t = ctx.train_tensors[key]
        if isinstance(t, torch.nn.Parameter):
            t.data.clamp_(0.0, 1.0)
            return t
        if isinstance(t, torch.Tensor):
            return t.clamp(0.0, 1.0)
        return t


def _get_minmax_tune_tlq_op_plugin():
    return MinmaxTuneOpConfig, MinmaxTuneOp


register_plugin(_get_minmax_tune_tlq_op_plugin)
