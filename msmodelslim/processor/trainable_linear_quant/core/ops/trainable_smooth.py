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

# 可训练 Smooth Op（source/target 安装规格 + 可训练 scale）。

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
from pydantic import Field, field_validator
from torch import nn

from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    Subgraph,
    UpDownSubgraph,
)
from msmodelslim.ir.non_fusion_smooth_quant_ir import NonFusionSmoothQuantHookIR
from msmodelslim.processor.anti_outlier.common.scale_computation import (
    prepare_mqga_parameters,
    reduce_scales_for_mqga_mean,
)
from msmodelslim.processor.anti_outlier.common.subgraph_fusion import apply_smooth_scale_shift
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.plugin import register_plugin
from msmodelslim.processor.trainable_linear_quant.core.kernels.base import FakeQuantStage
from msmodelslim.processor.trainable_linear_quant.core.kernels.context import FakeQuantPipelineContext
from msmodelslim.processor.trainable_linear_quant.core.wrapper import TrainableLinearQuantWrapper
from .base import SubgraphTLQOp, TLQOpConfig, format_tensor_dbg

SMOOTH_SCALE_KEY = "smooth_scale"
SMOOTH_SCALE_MIN = 0.1
SMOOTH_SCALE_MAX = 10.0

SMOOTH_SUPPORTED_SUBGRAPH_TYPES = (
    "norm-linear",
    "linear-linear",
    "ov",
    "up-down",
    "non-fusion",
)


def smooth_source_module(subgraph: Subgraph) -> Optional[nn.Module]:
    """Smooth act 融合用的 source 模块（norm / 上游 linear / v_proj / up_proj）。"""
    if isinstance(subgraph, NormLinearSubgraph):
        return subgraph.norm
    if isinstance(subgraph, LinearLinearSubgraph):
        return subgraph.linear1
    if isinstance(subgraph, OVSubgraph):
        return subgraph.v_proj
    if isinstance(subgraph, UpDownSubgraph):
        return subgraph.up_proj
    if isinstance(subgraph, NonFusionSubgraph):
        return None
    raise UnsupportedError(f"smooth_source_module unsupported for {type(subgraph).__name__}")


def _resolve_smooth_fusion_layer(module: nn.Module) -> nn.Module:
    """Smooth finalize 写回真实层；source 可能仍为 TLQ wrapper（unbind 早于 block 摘除）。"""
    if isinstance(module, TrainableLinearQuantWrapper):
        return module.orig_layer
    return module


def get_non_fusion_smooth_hook_scales(module: nn.Module) -> Optional[torch.Tensor]:
    """返回 ``module`` 上首个 ``NonFusionSmoothQuantHookIR`` 的 scale，若无则 ``None``。"""
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, NonFusionSmoothQuantHookIR):
            return hook.scales
    return None


@torch.no_grad()
def remove_non_fusion_smooth_hooks(module: nn.Module) -> int:
    """移除 ``NonFusionSmoothQuantHookIR``；调用方须保证 ``s`` 已写入 ``module.weight``。"""
    removed = 0
    for hook in list(module._forward_pre_hooks.values()):
        if isinstance(hook, NonFusionSmoothQuantHookIR):
            hook.remove_hook()
            removed += 1
    return removed


def _register_non_fusion_smooth_hook(layer: nn.Module, scales: torch.Tensor) -> None:
    hook_ir = NonFusionSmoothQuantHookIR(scales.clone())
    hook_handle = layer.register_forward_pre_hook(hook_ir)
    hook_ir.set_hook_handle(hook_handle)


class TrainableSmoothOpConfig(TLQOpConfig):
    type: Literal["trainable_smooth"] = Field(default="trainable_smooth", description="插件类型：trainable_smooth")
    lr: Optional[float] = Field(default=None, gt=0.0)
    enable_subgraph_type: List[str] = Field(
        default_factory=lambda: list(SMOOTH_SUPPORTED_SUBGRAPH_TYPES),
        description="启用的 Smooth 子图类型，须为 SMOOTH_SUPPORTED_SUBGRAPH_TYPES 子集",
    )
    include: Optional[List[str]] = Field(default=None, description="子图入口 include 通配")
    exclude: Optional[List[str]] = Field(default=None, description="子图入口 exclude 通配")

    @field_validator("enable_subgraph_type")
    @classmethod
    def _validate_subgraph_types(cls, v: List[str]) -> List[str]:
        allowed = set(SMOOTH_SUPPORTED_SUBGRAPH_TYPES)
        unknown = [t for t in v if t not in allowed]
        if unknown:
            raise ValueError(
                f"enable_subgraph_type contains unsupported types {unknown!r}; expected subset of {sorted(allowed)}"
            )
        return v


class TrainableSmoothOp(SubgraphTLQOp):
    """可训练 Smooth：训练期 ``W/s``（weight listener）+ ``×s``（forward act transform）；finalize 写入 source 或 hook。"""

    PARAM_KEY = SMOOTH_SCALE_KEY
    SUPPORTED_SUBGRAPH_TYPES = SMOOTH_SUPPORTED_SUBGRAPH_TYPES

    def __init__(
        self,
        config: TrainableSmoothOpConfig,
        subgraph: Subgraph,
        target_modules: Dict[str, TrainableLinearQuantWrapper],
    ) -> None:
        super().__init__(config, subgraph=subgraph, target_modules=target_modules)
        if self.subgraph_type not in SMOOTH_SUPPORTED_SUBGRAPH_TYPES:
            raise UnsupportedError(f"TrainableSmoothOp unsupported subgraph_type: {self.subgraph_type!r}")
        if self.subgraph_type != "non-fusion" and smooth_source_module(self.subgraph) is None:
            raise UnsupportedError("TrainableSmoothOp requires a smooth source module")
        self._smooth_scale: Optional[nn.Parameter] = None

    def _effective_scale(self, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """约束到 ``[SMOOTH_SCALE_MIN, SMOOTH_SCALE_MAX]``（Parameter 就地投影）。"""
        raw = self._smooth_scale if scale is None else scale
        if isinstance(raw, nn.Parameter):
            raw.data.clamp_(SMOOTH_SCALE_MIN, SMOOTH_SCALE_MAX)
            return raw
        return raw.clamp(SMOOTH_SCALE_MIN, SMOOTH_SCALE_MAX)

    def bind(self) -> None:
        first = next(iter(self.target_modules.values()))
        in_features = first.orig_layer.weight.shape[1]
        dev = first.orig_layer.weight.device
        self._smooth_scale = nn.Parameter(
            torch.ones(in_features, dtype=torch.float32, device=dev),
            requires_grad=True,
        )
        for wrapper in self.target_modules.values():
            wrapper.register_train_op(self, side="both")
            wrapper.w_kernel.add_listener(self._on_weight_quant_input, op_id=self.op_id)
            wrapper.register_forward_act_transform(self.apply_forward_act_smooth)
        if isinstance(self.subgraph, NonFusionSubgraph):
            names = self.subgraph.linear_names or []
            for idx, linear in enumerate(self.subgraph.linears):
                layer = _resolve_smooth_fusion_layer(linear)
                name = names[idx] if idx < len(names) and names[idx] else self.op_id
                s_o = get_non_fusion_smooth_hook_scales(layer)
                get_logger().info(
                    "[trainable_smooth] bind %s s_o(%s) s_t_init(%s)",
                    name,
                    format_tensor_dbg(s_o),
                    format_tensor_dbg(self._smooth_scale),
                )

    def _listener_smooth_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """1D smooth scale；OV/GQA 时按 head 数展开。``scale`` 须已是有效区间内的值。"""
        flat = scale.reshape(-1)
        if self.subgraph_type != "ov" or not isinstance(self.subgraph, OVSubgraph):
            return flat
        n_attn = self.subgraph.num_attention_heads
        n_kv = self.subgraph.key_value_heads
        if n_attn == n_kv:
            return flat
        ratio, _ = prepare_mqga_parameters(n_attn, n_kv)
        expanded, _ = reduce_scales_for_mqga_mean(flat, ratio, n_attn)
        return expanded

    def _smooth_factor(
        self,
        scale: torch.Tensor,
        tensor: torch.Tensor,
        for_weight: bool,
    ) -> torch.Tensor:
        s = self._listener_smooth_scale(scale).to(dtype=tensor.dtype, device=tensor.device)
        if for_weight:
            return s.unsqueeze(0)
        return s.view(*((1,) * (tensor.dim() - 1)), -1)

    @torch.no_grad()
    def _fuse_source_activation(self, scale: torch.Tensor) -> None:
        """finalize：activation ×s 等价写入 source（norm 或 linear）。"""
        source = smooth_source_module(self.subgraph)
        if source is None:
            raise UnsupportedError("TrainableSmoothOp missing smooth source module")
        layer = _resolve_smooth_fusion_layer(source)
        inv = (1.0 / scale.reshape(-1)).to(dtype=layer.weight.dtype, device=layer.weight.device)
        if getattr(layer, "bias", None) is not None:
            layer.bias.mul_(inv.squeeze())
        if self.subgraph_type == "norm-linear":
            apply_smooth_scale_shift(layer, inv.squeeze(), None, None)
            return
        if self.subgraph_type not in SMOOTH_SUPPORTED_SUBGRAPH_TYPES:
            raise UnsupportedError(f"unsupported smooth subgraph_type: {self.subgraph_type!r}")
        if not isinstance(layer, nn.Linear):
            raise UnsupportedError(f"linear act fusion expects nn.Linear source, got {type(layer).__name__}")
        apply_smooth_scale_shift(
            layer,
            inv.view(-1, 1),
            None,
            getattr(layer, "name", None),
        )

    @torch.no_grad()
    def _fuse_non_fusion_activation(self, scale: torch.Tensor) -> None:
        """finalize：non-fusion 激活侧 smooth hook 与训练图对齐。

        - 无 prior hook（仅 trainable_smooth）：注册 ``s_t``，配合 ``Q(W/s_t)``。
        - 有 prior hook ``s_o``（OASQ 等）：合并为 ``s_o/s_t``，等价训练期 ``(x/s_o)×s_t``。
        """
        if not isinstance(self.subgraph, NonFusionSubgraph):
            raise UnsupportedError("expected NonFusionSubgraph for non-fusion finalize")
        s_t = scale.reshape(-1)
        names = self.subgraph.linear_names or []
        for idx, linear in enumerate(self.subgraph.linears):
            layer = _resolve_smooth_fusion_layer(linear)
            name = names[idx] if idx < len(names) and names[idx] else self.op_id
            prior_s = get_non_fusion_smooth_hook_scales(layer)
            if prior_s is not None:
                merged = (prior_s.to(device=s_t.device, dtype=s_t.dtype) / s_t).clamp(min=SMOOTH_SCALE_MIN)
                get_logger().info(
                    "[trainable_smooth] finalize %s s_o(%s) s_t(%s) s(%s)",
                    name,
                    format_tensor_dbg(prior_s),
                    format_tensor_dbg(s_t),
                    format_tensor_dbg(merged),
                )
                remove_non_fusion_smooth_hooks(layer)
                _register_non_fusion_smooth_hook(layer, merged.detach().clone())
                continue
            get_logger().info(
                "[trainable_smooth] finalize %s s_o(none) s_t(%s) s(%s)",
                name,
                format_tensor_dbg(s_t),
                format_tensor_dbg(s_t),
            )
            remove_non_fusion_smooth_hooks(layer)
            _register_non_fusion_smooth_hook(layer, s_t.detach().clone())

    @torch.no_grad()
    def unbind(self) -> None:
        for wrapper in self.target_modules.values():
            wrapper.w_kernel.remove_listeners(self.op_id)
            wrapper.unregister_forward_act_transform(self.apply_forward_act_smooth)
        ref = next(iter(self.target_modules.values())).orig_layer
        scale = self._effective_scale().to(dtype=ref.weight.dtype, device=ref.weight.device)
        if self.subgraph_type == "non-fusion":
            self._fuse_non_fusion_activation(scale)
        else:
            self._fuse_source_activation(scale)

    @property
    def train_params(self) -> dict[str, torch.Tensor]:
        return {self.PARAM_KEY: self._smooth_scale}

    def _get_scale_tensor(self, ctx: FakeQuantPipelineContext) -> torch.Tensor:
        return self._effective_scale(ctx.train_tensors[self.PARAM_KEY])

    def apply_forward_act_smooth(self, x: torch.Tensor) -> torch.Tensor:
        """前向激活 SmoothQuant 迁移 ``×s_t``（与 ``W/s_t`` 成对，不依赖 act fake-quant）。"""
        factor = self._smooth_factor(self._effective_scale(), x, for_weight=False)
        return x * factor

    def _on_weight_quant_input(self, stage: FakeQuantStage, ctx: FakeQuantPipelineContext) -> None:
        if stage != FakeQuantStage.QUANT_INPUT:
            return
        factor = self._smooth_factor(self._get_scale_tensor(ctx), ctx.working_tensor, for_weight=True)
        ctx.set_working_tensor(ctx.working_tensor / factor)


def _get_trainable_smooth_tlq_op_plugin():
    return TrainableSmoothOpConfig, TrainableSmoothOp


register_plugin(_get_trainable_smooth_tlq_op_plugin)
