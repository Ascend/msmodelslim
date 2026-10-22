#!/usr/bin/env python
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

共用原语。量化公式在各数据格式文件里，不放在这里。

common.py
  ├─ _elem_cfg_from_bits / elem_cfg     e4m3 / e5m2 / e2m1 的 ebits、mbits、emax、max_norm
  ├─ _RoundSTE / round_ste              round 直通
  ├─ _FloorSTE / floor_ste              floor 直通
  ├─ _ClampSTE / clamp_ste              clamp 直通（量化默认开，ste_clamp=False 关）
  ├─ reduce_minmax / reduce_amax        按 axis 归约
  ├─ view_groups / restore_groups / grouped_apply   只做 per-group 切组
  ├─ _reshape_to_blocks / _undo_reshape_to_blocks
  ├─ view_blocks / restore_blocks       OCP MX §6.3 切块 / 还原
  │    align_block_param / blocked_apply
  ├─ pack_e8m0 / unpack_e8m0 / as_shared_exp
  └─ cast_q                            默认 fp32；其他 dtype 断开计算图后转换
"""

import torch
from torch.autograd import Function

_MX_BLOCK_SIZE = 32


def _elem_cfg_from_bits(ebits: int, mbits: int, emax_offset: int, *, e4m3: bool = False) -> dict:
    """OCP MX / OFP8 元素格式参数。

    emax = 2^(ebits - 1) + emax_offset
    e4m3 保留 2 个 NaN 码点：max_norm = 2^emax * 1.75
    其余：max_norm = 2^emax * (2^(mbits-1) - 1) / 2^(mbits-2)
    """
    emax = 2 ** (ebits - 1) + emax_offset
    if e4m3:
        max_norm = (2**emax) * 1.75
    else:
        max_norm = (2**emax) * float(2 ** (mbits - 1) - 1) / (2 ** (mbits - 2))
    return {"ebits": ebits, "mbits": mbits, "emax": emax, "max_norm": float(max_norm)}


# e4m3: ebits=4, mbits=5, emax_offset=0 → emax=8,  max_norm=448
# e5m2: ebits=5, mbits=4, emax_offset=-1 → emax=15, max_norm=57344
# e2m1: ebits=2, mbits=3, emax_offset=0 → emax=2,  max_norm=6
_ELEM_CFG = {
    "e4m3": _elem_cfg_from_bits(4, 5, 0, e4m3=True),
    "e5m2": _elem_cfg_from_bits(5, 4, -1),
    "e2m1": _elem_cfg_from_bits(2, 3, 0),
}


class _RoundSTE(Function):  # pylint: disable=abstract-method
    """OCP / IEEE roundTiesToEven 的直通估计：前向 round，反向恒等。"""

    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ,unused-argument
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ,unused-argument
        return grad_output


class _FloorSTE(Function):  # pylint: disable=abstract-method
    """floor 的直通估计：前向 floor，反向恒等。"""

    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ,unused-argument
        return x.floor()

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ,unused-argument
        return grad_output


class _ClampSTE(Function):  # pylint: disable=abstract-method
    """clamp 的直通估计：前向 clamp，反向恒等。"""

    @staticmethod
    def forward(ctx, x, min_v, max_v):  # pylint: disable=arguments-differ
        return torch.clamp(x, min=min_v, max=max_v)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ,unused-argument
        return grad_output, None, None


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(x)


def floor_ste(x: torch.Tensor) -> torch.Tensor:
    return _FloorSTE.apply(x)


def clamp_ste(x: torch.Tensor, min=None, max=None) -> torch.Tensor:  # pylint: disable=redefined-builtin
    """前向与 torch.clamp 相同，反向恒等。"""
    min_v = x.new_tensor(float("-inf") if min is None else float(min))
    max_v = x.new_tensor(float("inf") if max is None else float(max))
    return _ClampSTE.apply(x, min_v, max_v)


def elem_cfg(fmt: str) -> dict:
    """按元素格式名返回 ebits / mbits / emax / max_norm。

    fmt 是 OCP 元素编码名：FP8 / MXFP8 接口默认 "e4m3"，可选 "e5m2"；
    MXFP4 内部固定 "e2m1"，对外无 fmt 参数。不是存储 dtype。
    """
    if fmt not in _ELEM_CFG:
        raise ValueError(f"unsupported element format {fmt!r}, expected one of {sorted(_ELEM_CFG)}")
    return dict(_ELEM_CFG[fmt])


def reduce_minmax(x: torch.Tensor, axis=None):
    """沿 axis 以外的维归约 min/max；axis is None 时全张量一个标量。"""
    x = x.to(torch.float32)
    if axis is None:
        return x.amin(), x.amax()
    axis = axis if axis >= 0 else axis + x.ndim
    dims = [i for i in range(x.ndim) if i != axis]
    if not dims:
        return x, x
    return x.amin(dim=dims, keepdim=True), x.amax(dim=dims, keepdim=True)


def reduce_amax(x: torch.Tensor, axis=None) -> torch.Tensor:
    """沿 axis 以外的维归约 absmax；axis is None 时全张量一个标量。"""
    x = x.to(torch.float32).abs()
    if axis is None:
        return x.amax()
    axis = axis if axis >= 0 else axis + x.ndim
    dims = [i for i in range(x.ndim) if i != axis]
    if not dims:
        return x
    return x.amax(dim=dims, keepdim=True)


def view_groups(x: torch.Tensor, axis: int, group_size: int):
    """把 axis 维切成 (n_groups, group_size)。不能整除则报错，不补齐。"""
    orig_shape = tuple(x.shape)
    axis = axis if axis >= 0 else axis + x.ndim
    length = x.shape[axis]
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if length % group_size != 0:
        raise ValueError(f"axis {axis} length {length} is not divisible by group_size {group_size}")
    x = x.movedim(axis, -1)
    leading = tuple(x.shape[:-1])
    x = x.reshape(*leading, length // group_size, group_size)
    return x, {"orig_shape": orig_shape, "axis": axis, "leading": leading}


def restore_groups(x: torch.Tensor, meta: dict) -> torch.Tensor:
    x = x.reshape(*meta["leading"], -1)
    x = x.movedim(-1, meta["axis"])
    return x.reshape(meta["orig_shape"])


def _reshape_to_blocks(a: torch.Tensor, axes, block_size: int):
    """OCP MX Spec §6.3 / Algorithm 1：沿 axes 切成 block_size 块，不能整除则 pad。

    1. 每个切块轴后插一维
    2. 该轴不能整除 block_size 时在末尾 pad
    3. 把该轴拆成 (n_blocks, block_size)；短于一块则 (1, 原长)
    """
    if axes is None:
        raise ValueError("axes required to determine which dimension to apply block size to")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    axes = [(x + a.ndim if x < 0 else x) for x in axes]
    if not all(x >= 0 for x in axes):
        raise ValueError("all axes must be >= 0 after normalization")
    axes = sorted(axes)

    for i, axis in enumerate(axes):
        a = torch.unsqueeze(a, dim=axis + i + 1)

    # 插维后原 axis 挪到 axis+i，后续 pad / reshape / undo 都用移位后的下标。
    shifted = [axis + i for i, axis in enumerate(axes)]
    orig_shape = a.size()
    pad = [0, 0] * len(orig_shape)
    do_padding = False
    for axis in shifted:
        pre_pad_size = orig_shape[axis]
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True
    if do_padding:
        a = torch.nn.functional.pad(a, list(reversed(pad)), mode="constant")

    padded_shape = a.size()
    reshape = list(padded_shape)
    for axis in shifted:
        if reshape[axis] >= block_size:
            if reshape[axis] % block_size != 0:
                raise ValueError(f"dim {axis} size {reshape[axis]} is not divisible by block_size {block_size}")
            reshape[axis + 1] = block_size
            reshape[axis] = reshape[axis] // block_size
        else:
            reshape[axis + 1] = reshape[axis]
            reshape[axis] = 1
    return a.view(reshape), shifted, orig_shape, padded_shape


def _undo_reshape_to_blocks(a: torch.Tensor, padded_shape, orig_shape, axes) -> torch.Tensor:
    """还原 _reshape_to_blocks：先 view 回 padded，再裁 pad，再挤掉插入维。"""
    a = a.view(padded_shape)
    if list(padded_shape) != list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        a = a[slices]
    for axis in reversed(axes):
        a = torch.squeeze(a, dim=axis + 1)
    return a


def view_blocks(x: torch.Tensor, axis: int, block_size: int = _MX_BLOCK_SIZE):
    """OCP MX k=32 切块，长度不够则 pad。"""
    axes = [axis] if isinstance(axis, int) else list(axis)
    blocked, axes, orig_shape, padded_shape = _reshape_to_blocks(x, axes, block_size)
    return blocked, {
        "axes": axes,
        "orig_shape": orig_shape,
        "padded_shape": padded_shape,
    }


def restore_blocks(x: torch.Tensor, meta: dict) -> torch.Tensor:
    return _undo_reshape_to_blocks(x, meta["padded_shape"], meta["orig_shape"], meta["axes"])


def grouped_apply(fn, x, axis, group_size, *params):
    """沿 axis 按 group_size 切组，参数广播到切组后的 x，计算后还原。

    只服务 per-group。per-tensor / per-channel 在各自函数里直接算，不进这里。
    """
    x_v, meta = view_groups(x, axis, group_size)
    aligned = []
    for param in params:
        param = param.to(dtype=torch.float32, device=x_v.device)
        try:
            aligned.append(param.expand_as(x_v))
        except RuntimeError as exc:
            raise ValueError(
                f"param shape {tuple(param.shape)} cannot broadcast to grouped x {tuple(x_v.shape)}"
            ) from exc
    return restore_groups(fn(x_v, *aligned), meta)


def pack_e8m0(shared_exp: torch.Tensor) -> torch.Tensor:
    """E8M0 落盘：shared_exp ∈ [-127, 127] → uint8 = shared_exp + 127。0xFF 保留为 NaN，不写出。"""
    return (shared_exp.to(torch.float32) + 127.0).round().clamp(0, 254).to(torch.uint8)


def unpack_e8m0(raw: torch.Tensor) -> torch.Tensor:
    """E8M0 还原：uint8 → float32 shared_exp。"""
    return raw.to(torch.float32) - 127.0


def as_shared_exp(shared_exp: torch.Tensor) -> torch.Tensor:
    """uint8 视为 E8M0；其余视为已经是共享指数。"""
    if shared_exp.dtype == torch.uint8:
        return unpack_e8m0(shared_exp)
    return shared_exp.to(torch.float32)


def align_block_param(param: torch.Tensor, x_blocked: torch.Tensor) -> torch.Tensor:
    param = as_shared_exp(param).to(device=x_blocked.device)
    try:
        return param.expand_as(x_blocked)
    except RuntimeError:
        return torch.broadcast_to(param, x_blocked.shape)


def cast_q(q: torch.Tensor, dtype=None) -> torch.Tensor:
    """量化结果默认 float32，只有 float32 留在计算图里。

    dtype 为 None 或 torch.float32 时原样返回。其他 torch.dtype 先断开计算图再逐元素转换，不打包。
    """
    if dtype is None or dtype == torch.float32:
        return q
    return q.detach().to(dtype)


def blocked_apply(fn, x, axis, *params, block_size: int = _MX_BLOCK_SIZE):
    blocked, meta = view_blocks(x, axis, block_size)
    aligned = tuple(align_block_param(p, blocked) for p in params)
    return restore_blocks(fn(blocked, *aligned), meta)
