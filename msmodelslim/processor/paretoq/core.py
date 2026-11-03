# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# pylint: disable=redefined-builtin,consider-using-in,abstract-method,arguments-differ

"""ParetoQ quantization core: LSQ/SEQ ops + weight fake-quant API."""

import math

import torch


def _owned_snapshot(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into storage FSDP2 cannot free via ``resize_(0)`` after forward.

    ``clone()`` may still alias the all-gather buffer on Ascend; ``empty`` +
    ``copy_`` allocates a new block. Call only from ``autograd.Function.forward``.
    """
    local = tensor.to_local() if type(tensor).__name__ == "DTensor" else tensor
    snap = torch.empty(local.shape, device=local.device, dtype=local.dtype)
    snap.copy_(local)
    return snap


class LsqBinaryTernaryExtension(torch.autograd.Function):
    """Modified from Learned Step-size Quantization (https://arxiv.org/abs/1902.08153)."""

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            qn = -1
            qp = 1
        else:
            qn = -(2 ** (num_bits - 1))
            qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)
        # Own storage: FSDP2 may free the all-gathered weight before backward.
        input = _owned_snapshot(input)
        alpha = _owned_snapshot(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not qp else 1.0 / math.sqrt(input.numel() * qp)
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, qn, qp, layerwise
        with torch.no_grad():
            if num_bits == 1:
                q_w = input.sign()
            else:
                q_w = (input / alpha).round().clamp(qn, qp)
            return q_w * alpha

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, qn, qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < qn).float()
        indicate_big = (q_w > qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            grad_alpha_raw = (
                (indicate_small * qn + indicate_big * qp + indicate_middle * (-q_w + q_w.round()))
                * grad_output
                * grad_scale
            )
            if layerwise:
                grad_alpha = grad_alpha_raw.sum().unsqueeze(dim=0)
            else:
                grad_alpha = torch.sum(grad_alpha_raw, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class StretchedElasticQuant(torch.autograd.Function):
    """Modified from Learned Step-size Quantization (https://arxiv.org/abs/1902.08153)."""

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            qn = -1
            qp = 1
        else:
            qn = -(2 ** (num_bits - 1))
            qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)
        input = _owned_snapshot(input)
        alpha = _owned_snapshot(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not qp else 1.0 / math.sqrt(input.numel() * qp)
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        qp = (n_levels - shift) / n_levels
        qn = -qp
        ctx.other = grad_scale, qn, qp, layerwise
        with torch.no_grad():
            if num_bits == 1:
                q_w = input.sign()
            else:
                q_w = (
                    torch.round(torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift) + shift
                ) / n_levels
            return q_w * alpha

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, qn, qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            quantized_q_w = (torch.round(torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels
            grad_alpha_raw = (
                (indicate_small * qn + indicate_big * qp + indicate_middle * (-q_w + quantized_q_w))
                * grad_output
                * grad_scale
            )
            if layerwise:
                grad_alpha = grad_alpha_raw.sum().unsqueeze(dim=0)
            else:
                grad_alpha = torch.sum(grad_alpha_raw, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


WEIGHT_CLIP_INIT_AMAX = "amax"
WEIGHT_CLIP_INIT_MEAN2_SQRT_QP = "mean2_sqrt_qp"
WEIGHT_CLIP_INIT_CHOICES = (WEIGHT_CLIP_INIT_AMAX, WEIGHT_CLIP_INIT_MEAN2_SQRT_QP)


def fake_quant_weight(
    real_weights: torch.Tensor,
    weight_clip_val: torch.Tensor,
    w_bits: int,
    weight_layerwise: bool,
) -> torch.Tensor:
    """Apply ParetoQ / LSQ fake-quantization on a 2D weight tensor (out_features, in_features)."""
    if w_bits >= 16:
        return real_weights
    if w_bits == 2 or w_bits == 0:
        return StretchedElasticQuant.apply(real_weights, weight_clip_val, w_bits, weight_layerwise).to(
            real_weights.dtype
        )
    if w_bits <= 4:
        return LsqBinaryTernaryExtension.apply(real_weights, weight_clip_val, w_bits, weight_layerwise).to(
            real_weights.dtype
        )
    raise NotImplementedError(f"ParetoQ fake quant for w_bits={w_bits} is not implemented")


def paretoq_weight_qp(w_bits: int) -> int:
    """Positive quant upper bound Qp, aligned with ops.LsqBinaryTernaryExtension."""
    if w_bits >= 16 or w_bits == 1 or w_bits == 0:
        return 1
    return 2 ** (w_bits - 1) - 1


def _init_weight_clip_amax(weight: torch.Tensor, w_bits: int) -> torch.Tensor:
    """Per-output-row LSQ step-size init: amax / Qp so max |w| maps to Qp."""
    clip = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    return clip / paretoq_weight_qp(w_bits)


def _init_weight_clip_mean2_sqrt_qp(weight: torch.Tensor, w_bits: int) -> torch.Tensor:
    """Per-output-row init: 2 * mean(|w|) / sqrt(Qp)."""
    qp = paretoq_weight_qp(w_bits)
    clip = 2.0 * weight.abs().mean(dim=1, keepdim=True) / math.sqrt(qp)
    return clip.clamp(min=1e-5)


def init_weight_clip_from_weight(
    weight: torch.Tensor,
    w_bits: int,
    init_method: str = WEIGHT_CLIP_INIT_AMAX,
) -> torch.Tensor:
    """Initialize per-output-row weight_clip_val (LSQ alpha)."""
    if init_method == WEIGHT_CLIP_INIT_AMAX:
        fn = _init_weight_clip_amax
    elif init_method == WEIGHT_CLIP_INIT_MEAN2_SQRT_QP:
        fn = _init_weight_clip_mean2_sqrt_qp
    else:
        raise ValueError(f"Unknown weight_clip init_method={init_method!r}; choices: {WEIGHT_CLIP_INIT_CHOICES}")
    with torch.no_grad():
        return fn(weight, w_bits).clone()
