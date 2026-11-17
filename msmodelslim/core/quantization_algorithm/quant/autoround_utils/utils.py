#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2023 Intel Corporation
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

SHARED_CACHE_KEYS = ("position_ids", "cache_position", "position_embeddings")
SPECIAL_SHARED_CACHE_KEYS = {
    "Gemma3ForConditionalGeneration": ("position_embeddings_global", "position_embeddings_local")
}
SPECIAL_SHARED_CACHE_KEYS["MiniMaxText01ForCausalLM"] = ("slope_rate",)


def get_shared_keys(model):
    """
    Retrieves shared keys from the model's state dictionary.

    Args:
        model (torch.nn.Module): The model to retrieve shared keys from.

    Returns:
        tuple: tuple of shared keys.
    """
    shared_keys = SHARED_CACHE_KEYS
    shared_keys += SPECIAL_SHARED_CACHE_KEYS.get(model.__class__.__name__, ())
    return shared_keys


#####################量化相关#####################
QUANT_FUNC_WITH_DTYPE = {}


def register_dtype(names):
    """Class decorator to register a EXPORT subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        cls (class): The subclass of register.
        name: A string. Define the export type.

    Returns:
        cls: The class of register.
    """

    def register(dtype):
        if isinstance(names, (tuple, list)):
            for name in names:
                QUANT_FUNC_WITH_DTYPE[name] = dtype
        else:
            QUANT_FUNC_WITH_DTYPE[names] = dtype

        return dtype

    return register


def get_quant_func(dtype, bits, sym):
    """Retrieve the quantization function based on data type, bit width, and symmetry.

    This function returns the appropriate quantization function from the QUANT_FUNC_WITH_DTYPE
    dictionary based on the provided data type (`dtype`), bit width (`bits`), and whether
    the quantization is symmetric (`sym`). If the function does not exist, raise ValueError.

    Args:
        dtype (str): The data type for the quantization (e.g., 'int', 'mxfp4').
        bits (int): The bit width for the quantization (e.g., 2,4,8).
        sym (bool): A flag indicating whether the quantization is symmetric (True) or asymmetric (False).

    Returns:
        function: The quantization function corresponding to the specified parameters.
    """
    key = dtype
    if key in QUANT_FUNC_WITH_DTYPE:
        return QUANT_FUNC_WITH_DTYPE[key], key

    if sym:
        key = dtype + "_sym"
    else:
        key = dtype + "_asym"

    if key in QUANT_FUNC_WITH_DTYPE:
        return QUANT_FUNC_WITH_DTYPE[key], key

    raise ValueError(f"{dtype} is not supported")


@register_dtype("int_asym")
def quant_tensor_asym(
    tensor,
    bits=4,
    group_size=-1,
    w_corr=0,
    min_scale=1.0,
    max_scale=1.0,
    robust_quantile=0.998,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    output_qdq=True,
    **kwargs,
):
    """Quantize and de-quantize tensor asymmetrically. full range, credict goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        w_corr: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """

    robust_quantile = 1.0

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** (bits - 1) - 1

    if tensor_min is None or tensor_max is None:
        q_lo = torch.quantile(tensor.to(torch.float32), 1 - robust_quantile, dim=-1)
        q_hi = torch.quantile(tensor.to(torch.float32), robust_quantile, dim=-1)

        wmin_tmp = torch.clamp(q_lo, max=0)
        wmax_tmp = torch.clamp(q_hi, min=0)

    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max

    wmin_abs = -(wmin_tmp * min_scale)
    wmax_abs = wmax_tmp * max_scale
    max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.max(wmax_abs, wmin_abs)
    scale = (max_v / maxq).to(scale_dtype)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    zp = torch.full_like(scale, maxq + 1)

    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(tensor / scale + w_corr)

    q = torch.clamp(int_w + zp, 0, 2**bits - 1)
    if not output_qdq:
        q = revert_tensor_by_pad(q, orig_shape=orig_shape, pad_len=pad_len)
        return q, scale, zp

    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp


@register_dtype("int_sym")
def quant_tensor_sym(
    tensor,
    bits=4,
    group_size=-1,
    w_corr=0,
    min_scale=1.0,
    max_scale=1.0,
    robust_quantile=0.998,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=torch.finfo(torch.float32).eps,
    output_qdq=True,
    use_quantile=False,
    **kwargs,
):
    """Quantize and de-quantize tensor asymmetrically. full range, credict goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        w_corr: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** (bits - 1) - 1

    if tensor_min is None or tensor_max is None:
        if use_quantile and bits == 4:
            q_lo = torch.quantile(tensor.to(torch.float32), 1 - robust_quantile, dim=-1)
            q_hi = torch.quantile(tensor.to(torch.float32), robust_quantile, dim=-1)
            wmin_tmp = torch.clamp(q_lo, max=0)
            wmax_tmp = torch.clamp(q_hi, min=0)
        else:
            wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
            wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)

    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max

    wmin_abs = -(wmin_tmp * min_scale)
    wmax_abs = wmax_tmp * max_scale
    max_v = torch.max(wmax_abs, wmin_abs)
    scale = (max_v / maxq).to(scale_dtype)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))

    scale = scale.unsqueeze(dim=-1)
    zp = torch.full_like(scale, 0)

    if isinstance(w_corr, torch.Tensor) and bits == 4:
        W_corr = torch.clamp(w_corr, min=-0.5, max=0.5)
    elif isinstance(w_corr, torch.Tensor) and bits == 8:
        W_corr = torch.clamp(w_corr, min=-1.0, max=1.0)
    else:
        W_corr = w_corr

    int_w = round_ste(tensor / scale + W_corr)

    q = torch.clamp(int_w, -maxq - 1, maxq)
    if not output_qdq:
        q = revert_tensor_by_pad(q, orig_shape=orig_shape, pad_len=pad_len)
        return q, scale, zp

    qdq_result = (scale * q).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp


def reshape_pad_tensor_by_group_size(data: torch.Tensor, group_size: int):
    """Reshapes and pads the tensor to ensure that it can be quantized in groups of `group_size`.

    This function adjusts the
    input tensor's shape so that its last dimension is a multiple
    of the specified `group_size`. If padding is required, it adds padding to the tensor
    to achieve this. If the tensor's last dimension is already divisible by `group_size`,
    no padding is applied.

    Args:
        data (torch.Tensor): The input tensor to be reshaped and padded.
        group_size (int): The size of the groups that the tensor should be reshaped into.

    Returns:
        torch.Tensor: The reshaped and padded tensor, if necessary.
        tuple: The original shape of the input tensor.
        int: The padding length applied to the tensor. Returns 0 if no padding is applied.
    """
    orig_shape = data.shape
    pad_len = 0
    if group_size == 0:
        data = data.reshape(1, -1)
        return data, orig_shape, pad_len
    if len(data.shape) > 2:
        data = data.reshape(-1, orig_shape[-1])
    if group_size == -1 or data.shape[1] < group_size:
        return data, orig_shape, pad_len
    elif data.shape[1] % group_size == 0:
        data = data.reshape(-1, group_size)
        return data, orig_shape, pad_len
    else:
        pad_len = (data.shape[1] + group_size - 1) // group_size * group_size - data.shape[1]
        data_new = torch.nn.functional.pad(data, (0, pad_len))
        data_new = data_new.reshape(-1, group_size)
        return data_new, orig_shape, pad_len


def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for rounding.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.round() - x).detach() + x


def revert_tensor_by_pad(data: torch.Tensor, orig_shape: tuple, pad_len: int):
    """Reverts the tensor to its original shape by removing padding.

    This function removes the padding added during reshaping and returns the tensor to
    its original shape.

    Args:
        data (torch.Tensor): The reshaped and possibly padded tensor.
        orig_shape (tuple): The original shape of the tensor before reshaping.
        pad_len (int): The length of the padding to be removed.

    Returns:
        torch.Tensor: The tensor restored to its original shape.
    """
    if pad_len == 0:
        return data.reshape(orig_shape)
    else:
        if len(orig_shape) > 2:
            tmp_shape = torch.prod(torch.tensor(orig_shape[:-1])).item()
        else:
            tmp_shape = orig_shape[0]
        data_new = data.reshape(tmp_shape, -1)
        data_new = data_new[:, :-pad_len]
        data_new = data_new.reshape(orig_shape)
        return data_new
