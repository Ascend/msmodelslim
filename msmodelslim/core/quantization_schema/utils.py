"""
Copyright (c) Microsoft Corporation.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.
Licensed under the MIT License.

Name:    mx_ops.py

Pytorch methods for MX quantization.

Usage Notes:
 - Use the "Exposed Methods" below to implement autograd functions
 - Use autograd functions to then implement torch.nn.Module(s)
 - Do *not* use methods in this file in Modules, they have no defined
   backwards pass and will block gradient computation.
 - Avoid importing internal function if at all possible.

Exposed Methods:
    quantize_mx_op - quantizes a tensor to MX format.

Internal Methods:
    _safe_lshift, _safe_rshift - fp16 compatible shifts
    _shared_exponents - Returns MX shared exponent for the passed tensor
    _reshape_to_blocks - tiles a tensor by splitting one dim into two
    _undo_reshape_to_blocks - undos the above reshaping
    _quantize_mx - quantizes a tensor to MX format
"""

import torch.nn.functional as F

from msmodelslim.utils.exception import SchemaValidateError


def reshape_to_blocks(a, axes, block_size):
    if axes is None:
        raise SchemaValidateError("axes required in order to determine which dimension toapply block size to")
    if block_size <= 0:
        raise SchemaValidateError("block_size must be >0 got {block_size}")
    # 1. Fix axes to be positive and sort them
    ndim = a.dim()
    axes = sorted([(ax + ndim if ax < 0 else ax) for ax in axes])
    if any(ax < 0 or ax >= ndim for ax in axes):
        raise SchemaValidateError("All elements must be greater than or equal to 0")

    orig_shape = list(a.shape)
    # 2. computing padding
    pad_param = [0] * (2 * ndim)
    do_padding = False
    for ax in axes:
        remainder = orig_shape[ax] % block_size
        if remainder != 0:
            pad_amount = block_size - remainder
            pad_param[2 * (ndim - 1 - ax) + 1] = pad_amount
            do_padding = True

    if do_padding:
        a = F.pad(a, pad_param, mode="constant", value=0)

    padded_shape = list(a.shape)

    # 3. Calculates the reshaped shape.
    new_shape = []
    curr_ax = 0
    for i, size in enumerate(padded_shape):
        if i in axes:
            new_shape.extend([size // block_size, block_size])
            curr_ax += 2
        else:
            new_shape.append(size)
            curr_ax += 1
    return a.view(new_shape), axes, tuple(orig_shape), tuple(padded_shape)


def undo_reshape_to_blocks(a, padded_shape, orig_shape, axes=None):
    # To compatible with the original code in A5, the parameter 'axes' is reserved although it is not used.
    a = a.view(padded_shape)
    # Undo padding
    if list(padded_shape) != list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        a = a[tuple(slices)]
    return a


def pad_and_reshape(a, orig_shape, axes, hifx_finfo):
    if axes not in (-1, len(orig_shape) - 1):
        raise SchemaValidateError("HiF4 current implementation only supports the last dimension as qdim")
    dim0 = orig_shape[-1]
    blk_size_total = hifx_finfo.block_size * hifx_finfo.blk_outer_size
    padc = (blk_size_total - dim0 % blk_size_total) % blk_size_total
    pads = []
    if axes >= 0:
        axes = axes - len(orig_shape)
    if padc > 0:
        pads = [0] * (-axes * 2 - 1) + [padc]
        a = F.pad(a, pads, value=0.0)
    a = a.unflatten(axes, (-1, 8, 2, 4))
    return a, axes, padc, dim0


def undo_pad_and_reshape(a, axes, **kwargs):
    if axes != -1:
        raise SchemaValidateError("HiF4 current implementation only supports the last dimension as qdim")
    a = a.flatten(axes - 3, axes)
    padc = kwargs.get("padc", 0)
    if padc > 0:
        # If the padc parameter exists,  padded_shape/orig_shape/dim0 parameter is used to restore the shape.
        required_params = ['padded_shape', 'orig_shape', 'dim0']
        missing = [param for param in required_params if param not in kwargs]
        if missing:
            raise SchemaValidateError(
                f"The parameters:{required_params} \
                    need to be provided when padc is bigger than 0"
            )
        else:
            slices = [slice(0, kwargs["padded_shape"][i]) for i in range(len(kwargs["orig_shape"]) + axes)] + [
                slice(0, kwargs["dim0"])
            ]
            a = a[tuple(slices)]
    return a
