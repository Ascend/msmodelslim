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

MXFP8 数据格式 ops（OCP MX v1.0）。一个类 Mxfp8，仅对称。
MX 本身就是块粒度，接口名不带 per_block，对称性放在末尾。
calc_qparam 返回 uint8 E8M0（raw = shared_exp + 127）。
quantize 默认 float32。
fmt 选块内元素编码，默认 "e4m3"（emax=8，max_norm=448），可选 "e5m2"（emax=15，max_norm=57344）。
不改 E8M0；e_shared = floor(log2(amax)) - emax。dtype 决定 q 的存储类型。

mxfp8_ops.py
  └─ Mxfp8
       calc_qparam_sym / quantize_sym / dequantize_sym / quantize_dynamic_sym
"""

import torch

from .common import (
    as_shared_exp,
    blocked_apply,
    cast_q,
    clamp_ste,
    elem_cfg,
    floor_ste,
    round_ste,
    view_blocks,
)

_MXFP8_STORAGE = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}
_FP32_MIN_NORMAL = 2 ** (1 - 127)
_E8M0_EXP_MIN = -127
_E8M0_EXP_MAX = 127


def _round_element(a: torch.Tensor, private_exp: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """OCP MXFP8 roundTiesToEven。

    min_exp = 2 - 2^(ebits-1)
    bits_   = mbits - 2
    a ← round(a / 2^private_exp * 2^bits_) / 2^bits_ * 2^private_exp
    """
    min_exp = -(2 ** (ebits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    bits_ = mbits - 2
    a = a / (2**private_exp) * (2**bits_)
    a = round_ste(a)
    a = a / (2**bits_) * (2**private_exp)
    return a


class Mxfp8:
    """MXFP8。仅对称。fmt 默认 e4m3，只选块内元素，不改 E8M0。"""

    @staticmethod
    def calc_qparam_sym(x, axis=-1, fmt: str = "e4m3"):
        """MXFP8 动态共享指数。

        x: 浮点张量，任意形状。axis 为分块轴，块长固定 32，不够则内部 pad。
        fmt: 块内元素编码，默认 "e4m3"（emax=8）；可选 "e5m2"（emax=15）。
        决定 e_shared = floor(log2(amax)) - emax。不是存储类型。

        返回: 切块布局下 keepdim 的 uint8 E8M0（raw = shared_exp + 127），可落盘。

        OCP MX Spec §6.3 / E8M0：
        shared_exp = clamp(floor(log2(max_abs)) - emax, -127, 127)
        max_abs==0 时加 fp32 最小正规数，避免 log2(0)。
        raw=0 即 shared_exp=-127 为合法有限值；0xFF 为 NaN，用 clamp 不把下溢写成 NaN。
        """
        if fmt not in _MXFP8_STORAGE:
            raise ValueError(f"unsupported MXFP8 fmt {fmt!r}, expected e4m3 or e5m2")
        blocked, meta = view_blocks(x, axis)
        block_dims = [a + 1 for a in meta["axes"]]
        emax = elem_cfg(fmt)["emax"]
        max_abs = blocked.abs().amax(dim=block_dims, keepdim=True)
        shared = floor_ste(torch.log2(max_abs + _FP32_MIN_NORMAL * (max_abs == 0).to(max_abs.dtype)))
        se = (shared - emax).clamp(min=_E8M0_EXP_MIN, max=_E8M0_EXP_MAX)
        return (se + 127.0).round().clamp(0, 254).to(torch.uint8)

    @staticmethod
    def quantize_sym(x, shared_exp_t, axis=-1, fmt: str = "e4m3", dtype=None, ste_clamp=True):
        """MXFP8 对称静态量化。

        x: 浮点张量。shared_exp_t: uint8 E8M0 或 float32 共享指数，须能广播到切块后的 x。
        fmt: 同 calc_qparam，默认 "e4m3"；决定元素网格与饱和上界。须与算指数时一致。
        dtype: 默认 float32，只有 float32 能反传。低精度按 fmt 传 float8_e4m3fn / float8_e5m2。
        ste_clamp: 默认 True，clamp 反向恒等；False 用 torch.clamp 真导数。

        返回: (q, shared_exp_t)。q 与 x 同形状（pad 已裁掉）；shared_exp 原样带回。

        OCP MXFP8：
        x'          = x / 2^shared_exp     （用 ldexp，避免 2^shared_exp 次正规导致除 0）
        private_exp = floor(log2(|x'|))
        x_q         = roundTiesToEven(x')
        x_q         = clamp(x_q, ±max_norm)
        """
        if fmt not in _MXFP8_STORAGE:
            raise ValueError(f"unsupported MXFP8 fmt {fmt!r}, expected e4m3 or e5m2")
        cfg = elem_cfg(fmt)
        se = as_shared_exp(shared_exp_t)
        clip = clamp_ste if ste_clamp else torch.clamp

        def quantize(t, exp):
            inp = torch.ldexp(t.to(torch.float32), (-exp).to(torch.int32))
            zero_mask = inp == 0
            private_exp = floor_ste(torch.log2(torch.abs(inp) + zero_mask.to(inp.dtype)))
            inp = _round_element(inp, private_exp, cfg["ebits"], cfg["mbits"])
            return clip(inp, min=-cfg["max_norm"], max=cfg["max_norm"])

        q = blocked_apply(quantize, x, axis, se)
        return cast_q(q, dtype), shared_exp_t

    @staticmethod
    def dequantize_sym(q, shared_exp_t, axis=-1):
        """MXFP8 对称反量化。q 为 float8 或 float32 网格。返回与 q 同形状的 float32。

        OCP：x = q * 2^shared_exp
        """
        return blocked_apply(
            lambda t, exp: t.to(torch.float32) * (2 ** exp.to(torch.float32)),
            q.to(torch.float32),
            axis,
            as_shared_exp(shared_exp_t),
        )

    @staticmethod
    def quantize_dynamic_sym(x, axis=-1, fmt: str = "e4m3", dtype=None, ste_clamp=True):
        """MXFP8 对称动态量化。返回 (q, shared_exp)，shared_exp 为 uint8 E8M0 且已 detach。

        fmt 默认 "e4m3"，含义同 calc_qparam / quantize。
        """
        se = Mxfp8.calc_qparam_sym(x, axis=axis, fmt=fmt).detach()
        return Mxfp8.quantize_sym(x, se, axis=axis, fmt=fmt, dtype=dtype, ste_clamp=ste_clamp)


mxfp8_calc_qparam_sym = Mxfp8.calc_qparam_sym
mxfp8_quantize_sym = Mxfp8.quantize_sym
mxfp8_dequantize_sym = Mxfp8.dequantize_sym
mxfp8_quantize_dynamic_sym = Mxfp8.quantize_dynamic_sym
