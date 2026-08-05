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

# TrainableLinearQuantWrapper：持有 x/w 伪量化 Kernel，不含 Op 逻辑。

from __future__ import annotations

from typing import Callable, Dict, List, Literal, TYPE_CHECKING

import torch
from torch import nn
from torch.functional import F

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType

from .kernels import create_tlq_kernel

if TYPE_CHECKING:
    from msmodelslim.processor.trainable_linear_quant.core.ops.base import TLQOp

TrainTensorSide = Literal["act", "weight"]


def _needs_act_fake_quant(act_qconfig: QConfig) -> bool:
    if QDType(act_qconfig.dtype) == QDType.FLOAT:
        return False
    if act_qconfig.method == "none":
        return False
    return True


def _apply_pre_hook_to_tensor(hook: object, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """执行 ``orig_layer`` 上的 pre-hook，兼容返回 ``Tensor`` 或 ``(Tensor,)``。"""
    hook_result = hook(module, (x,))
    if hook_result is None:
        return x
    if isinstance(hook_result, tuple):
        return hook_result[0]
    return hook_result


class TrainableLinearQuantWrapper(nn.Module):
    """可训练 Linear 包装：通过 ``w_kernel`` / ``x_kernel`` 执行伪量化。"""

    def __init__(
        self,
        orig_layer: nn.Module,
        linear_qconfig: LinearQConfig,
        train_with_act_quant: bool = False,
    ) -> None:
        super().__init__()
        self.orig_layer = orig_layer
        self.config = linear_qconfig
        self.layer_path = ""
        self.train_with_act_quant = train_with_act_quant
        self._act_train_ops: List[TLQOp] = []
        self._weight_train_ops: List[TLQOp] = []
        self._forward_act_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = []

        self.w_kernel = create_tlq_kernel(linear_qconfig.weight)
        self.x_kernel = None
        if train_with_act_quant and _needs_act_fake_quant(linear_qconfig.act):
            self.x_kernel = create_tlq_kernel(linear_qconfig.act)

    def register_train_op(self, op: TLQOp, side: TrainTensorSide | Literal["both"] = "both") -> None:
        if side in ("act", "both"):
            if op not in self._act_train_ops:
                self._act_train_ops.append(op)
        if side in ("weight", "both"):
            if op not in self._weight_train_ops:
                self._weight_train_ops.append(op)

    def register_forward_act_transform(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """注册前向激活变换（如 SmoothQuant ``×s``），与 act fake-quant 开关无关。"""
        if fn not in self._forward_act_transforms:
            self._forward_act_transforms.append(fn)

    def unregister_forward_act_transform(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        try:
            self._forward_act_transforms.remove(fn)
        except ValueError:
            pass

    def _apply_forward_act_transforms(self, x: torch.Tensor) -> torch.Tensor:
        for fn in self._forward_act_transforms:
            x = fn(x)
        return x

    def _collect_train_params(self, side: TrainTensorSide) -> Dict[str, torch.Tensor]:
        """合并本侧已注册 Op 的 ``train_params``，供 ``fake_quantize`` 的 ``train_tensors`` 使用。"""
        ops = self._act_train_ops if side == "act" else self._weight_train_ops
        merged: Dict[str, torch.Tensor] = {}
        for op in ops:
            merged.update(op.train_params)
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Snapshot hooks so registration/removal during apply cannot break iteration.
        pre_hooks = list(self.orig_layer._forward_pre_hooks.values())
        for hook in pre_hooks:
            x = _apply_pre_hook_to_tensor(hook, self.orig_layer, x)

        x = self._apply_forward_act_transforms(x)

        train_tensors_act = self._collect_train_params("act")
        if self.x_kernel is not None:
            x = self.x_kernel.fake_quantize(x, train_tensors_act).tensor

        weight = self.w_kernel.fake_quantize(
            self.orig_layer.weight,
            self._collect_train_params("weight"),
        ).tensor

        return F.linear(x, weight, self.orig_layer.bias)

    def unwrapper(self) -> nn.Module:
        """用 Op 已 commit 的 ``train_params`` 做 weight 伪量化并写回 ``orig_layer``。"""
        result = self.w_kernel.fake_quantize(
            self.orig_layer.weight,
            self._collect_train_params("weight"),
            return_quantized_weight=True,
        )
        layer = self.orig_layer
        layer.weight.data.copy_(result.tensor)
        layer.scale = result.scale
        layer.zp = result.offset
        layer.weight_qconfig = self.config.weight
        layer.act_qconfig = self.config.act
        return layer


__all__ = ["TrainableLinearQuantWrapper"]
