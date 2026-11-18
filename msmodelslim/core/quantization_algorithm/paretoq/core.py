# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# pylint: disable=redefined-builtin,consider-using-in,abstract-method,arguments-differ

"""ParetoQ quantization core: LSQ/SEQ ops + weight fake-quant API."""

import math

import torch


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

        # Own storage: FSDP2 may free the all-gathered weight before backward.

        grad_scale = 1.0 / math.sqrt(input.numel()) if not qp else 1.0 / math.sqrt(input.numel() * qp)
        with torch.no_grad():
            # Match yang/Megatron exactly: the fp32 eps tensor promotes alpha to
            # fp32, so ratio/rounding/multiply run in fp32; the fake-quant
            # result is cast back to the parameter dtype (bf16 in QAT).
            eps = torch.tensor(0.00001, device=alpha.device).float()
            alpha_f = torch.where(alpha > eps, alpha, eps)
            ratio = input.div(alpha_f)
            if num_bits == 1:
                q_w = ratio.sign()
            else:
                q_w = ratio.round().clamp(qn, qp)
            # Save UNROUNDED ratio: backward needs (round(ratio) - ratio) for
            # the LSQ rounding-error gradient; saving rounded q_w makes it 0.
            # ratio is a fresh tensor from div(), not an FSDP2 all-gather view,
            # so no _owned_snapshot needed.
            ctx.save_for_backward(ratio)
            ctx.other = grad_scale, qn, qp, layerwise
            return (q_w * alpha_f).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        ratio = ctx.saved_tensors[0]  # unrounded input/alpha (fp32)
        grad_scale, qn, qp, layerwise = ctx.other
        indicate_small = (ratio < qn).float()
        indicate_big = (ratio > qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        # Same expression as yang/Megatron: pass grad where in range, zero outside.
        grad_input = indicate_middle * grad_output

        if ctx.num_bits == 1:
            # For 1-bit, grad_alpha uses sign(ratio) == input.sign()
            sign_ratio = ratio.sign()
            if layerwise:
                grad_alpha = (sign_ratio * grad_output * grad_scale).sum().unsqueeze(dim=0)
            else:
                grad_alpha = sign_ratio * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            # (small*qn + big*qp + middle*(round(q_w) - q_w)) * grad_output * grad_scale
            grad_alpha_raw = (
                (indicate_small * qn + indicate_big * qp + indicate_middle * (-ratio + ratio.round()))
                * grad_output
                * grad_scale
            )
            if layerwise:
                grad_alpha = grad_alpha_raw.sum().unsqueeze(dim=0)
            else:
                grad_alpha = torch.sum(grad_alpha_raw, dim=-1, keepdim=True)

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

        grad_scale = 1.0 / math.sqrt(input.numel()) if not qp else 1.0 / math.sqrt(input.numel() * qp)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        qp = (n_levels - shift) / n_levels
        qn = -qp
        ctx.other = grad_scale, qn, qp, layerwise, clip_val, n_levels, shift
        with torch.no_grad():
            # Same fp32 promotion + dtype cast as the LSQ path / yang.
            eps = torch.tensor(0.00001, device=alpha.device).float()
            alpha_f = torch.where(alpha > eps, alpha, eps)
            ratio = input.div(alpha_f)
            if num_bits == 1:
                q_w = ratio.sign()
            else:
                q_w = ratio.clamp(-clip_val, clip_val).mul_(n_levels).sub_(shift)
                q_w.round_().add_(shift).div_(n_levels)
            # Save UNROUNDED ratio: backward needs quantized_q_w - ratio for
            # the LSQ gradient; saving already-quantized q_w makes it 0.
            ctx.save_for_backward(ratio)
            return (q_w * alpha_f).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        ratio = ctx.saved_tensors[0]  # unrounded input/alpha
        grad_scale, qn, qp, layerwise, clip_val, n_levels, shift = ctx.other
        indicate_small = (ratio < -clip_val).float()
        indicate_big = (ratio > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        grad_input = indicate_middle * grad_output
        if ctx.num_bits == 1:
            # For 1-bit, grad_alpha uses sign(ratio) == input.sign()
            sign_ratio = ratio.sign()
            if layerwise:
                grad_alpha = (sign_ratio * grad_output * grad_scale).sum().unsqueeze(dim=0)
            else:
                grad_alpha = sign_ratio * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            quantized_q_w = ratio.clamp(-clip_val, clip_val).mul_(n_levels).sub_(shift)
            quantized_q_w.round_().add_(shift).div_(n_levels)
            grad_alpha_raw = quantized_q_w.sub_(ratio)  # quantized_q_w - ratio (nonzero!)
            grad_alpha_raw.mul_(indicate_middle)
            grad_alpha_raw.add_(indicate_small * qn)
            grad_alpha_raw.add_(indicate_big * qp)
            grad_alpha_raw.mul_(grad_output).mul_(grad_scale)
            if layerwise:
                grad_alpha = grad_alpha_raw.sum().unsqueeze(dim=0)
            else:
                grad_alpha = torch.sum(grad_alpha_raw, dim=-1, keepdim=True)

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
