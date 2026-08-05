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

# Block finalize: load best params, unwrap wrappers, export fake-quant IR.

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.ir.qal import QParam, QStorage, QScheme
from msmodelslim.processor.trainable_linear_quant.core.wrapper import TrainableLinearQuantWrapper
from msmodelslim.processor.trainable_linear_quant.pipeline.runtime.context import BlockTLQContext
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger


def _resolve_per_block_size(w_q_scheme: QScheme, group_size: int) -> int:
    """Return block/group size for scale expansion, or -1 if not applicable."""
    try:
        return int(w_q_scheme.dtype.mx_finfo.block_size)
    except SchemaValidateError:
        pass
    if group_size > 0:
        return group_size
    return -1


def _is_mx_scheme(w_q_scheme: QScheme) -> bool:
    try:
        return int(w_q_scheme.dtype.mx_finfo.block_size) > 0
    except SchemaValidateError:
        return False


def _export_scale_offset_for_fake_quant(
    scale: torch.Tensor,
    offset: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    block_size: int,
    is_mx: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reshape scale/offset for ``AutoFakeQuantLinear``.

    MX FakeQuant IR runs ``reshape_to_blocks`` on weight → ``[out, n_blocks, block_size]``.
    Scale must broadcast with that layout, so MX exports ``[out, n_blocks, 1]``.
    Non-MX group/channel paths keep the previous ``[out, n_blocks]`` / 1D layout.
    """
    if block_size is not None and block_size > 0 and is_mx:
        n_blocks = (in_features + block_size - 1) // block_size
        if scale.numel() == out_features:
            scale = scale.reshape(out_features, 1, 1).expand(out_features, n_blocks, 1).contiguous()
            offset = offset.reshape(out_features, 1, 1).expand(out_features, n_blocks, 1).contiguous()
        else:
            scale = scale.reshape(out_features, n_blocks, 1)
            offset = offset.reshape(out_features, n_blocks, 1)
        return scale, offset

    if block_size is not None and block_size > 0 and scale.numel() == out_features:
        n_blocks = (in_features + block_size - 1) // block_size
        scale = scale.reshape(out_features, 1).expand(out_features, n_blocks).contiguous()
        offset = offset.reshape(out_features, 1).expand(out_features, n_blocks).contiguous()
        return scale, offset

    scale = scale.reshape(out_features, -1).squeeze(1)
    offset = offset.reshape(out_features, -1).squeeze(1)
    return scale, offset


def create_fake_quantizer(orig_layer: torch.nn.Linear) -> qir.AutoFakeQuantLinear:
    w_q_scheme: QScheme = orig_layer.weight_qconfig.to_scheme()
    group_size = orig_layer.weight_qconfig.ext.get("group_size", -1)

    weight = orig_layer.weight
    out_features, in_features = weight.shape[0], weight.shape[1]
    block_size = _resolve_per_block_size(w_q_scheme, group_size)
    scale, offset = _export_scale_offset_for_fake_quant(
        torch.as_tensor(orig_layer.scale),
        torch.as_tensor(orig_layer.zp),
        out_features=out_features,
        in_features=in_features,
        block_size=block_size,
        is_mx=_is_mx_scheme(w_q_scheme),
    )

    get_logger().debug(
        "Creating fake quantizer: block_size=%s, scale_shape=%s, offset_shape=%s",
        block_size,
        scale.shape,
        offset.shape,
    )

    w_q_param = QParam(
        scheme=w_q_scheme,
        ext={
            "scale": scale,
            "offset": offset,
            "group_size": group_size,
            "axes": -1,
        },
    )
    x_q_scheme: QScheme = orig_layer.act_qconfig.to_scheme()
    x_q_param = QParam(scheme=x_q_scheme, ext={"axes": -1})
    w_q = QStorage(dtype=w_q_scheme.dtype, value=weight)

    get_logger().debug(
        "Fake quantizer parameters: weight_scheme=%s, activation_scheme=%s",
        w_q_scheme,
        x_q_scheme,
    )

    return qir.AutoFakeQuantLinear.create(x_q_param, w_q_param, w_q, orig_layer.bias)


def _apply_hook_ir_to_fake_quantizer(
    orig_layer: nn.Linear,
    fake_quantizer: qir.AutoFakeQuantLinear,
) -> Tuple[qir.AutoFakeQuantLinear, int]:
    """Apply HookIR wrappers from a snapshot of ``_forward_pre_hooks``.

    Snapshot avoids iteration issues if hooks mutate the dict; individual hook
    failures are logged and skipped so one bad hook does not abort finalize.
    """
    hook_count = 0
    result = fake_quantizer
    for hook_id, hook in list(orig_layer._forward_pre_hooks.items()):
        if not isinstance(hook, qir.HookIR):
            continue
        try:
            result = hook.wrapper_module(result)
            hook_count += 1
        except Exception as exc:
            get_logger().warning(
                "Failed to apply HookIR %s (%s) on layer: %s",
                hook_id,
                type(hook).__name__,
                exc,
            )
    return result, hook_count


@torch.no_grad()
def finalize_block(
    block_name: str,
    block: nn.Module,
    ctx: BlockTLQContext,
    model: nn.Module,
) -> int:
    """Load best params, unwrap wrappers, and export fake-quant IR for one block."""
    if not ctx.ops:
        get_logger().info("block %s: no TLQ ops to finalize", block_name)
        return 0

    ops = list(ctx.require_ops())
    for op in ops:
        op.load_best_params()

    unwrapped: List[Tuple[str, nn.Linear]] = []
    for layer_name, m in block.named_modules(prefix=block_name):
        if not isinstance(m, TrainableLinearQuantWrapper):
            continue
        get_logger().debug("Unwrapping layer '%s'", layer_name)
        m.unwrapper()
        unwrapped.append((layer_name, m.orig_layer))

    for op in ops:
        op.unbind()

    unwrapped_count = 0
    for layer_name, orig_layer in unwrapped:
        with torch.device(device=ctx.device):
            fake_quantizer = create_fake_quantizer(orig_layer)
            fake_quantizer, hook_count = _apply_hook_ir_to_fake_quantizer(
                orig_layer,
                fake_quantizer,
            )
            if hook_count > 0:
                get_logger().debug(
                    "Applied %d hooks to layer '%s'",
                    hook_count,
                    layer_name,
                )
            model.set_submodule(layer_name, fake_quantizer)
            unwrapped_count += 1

    get_logger().info("Fused and unwrapped %d layers", unwrapped_count)
    return unwrapped_count


__all__ = ["create_fake_quantizer", "finalize_block"]
