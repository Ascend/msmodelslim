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

MXFP4 数据格式 ops（OCP MX v1.0）。一个类 Mxfp4，仅对称。
不打包：默认 float32 的 E2M1 网格，dtype=torch.int8 为同形 8bit。
calc_qparam 返回 uint8 E8M0。
元素固定 E2M1，无 fmt 参数（与 MXFP8 可选 e4m3 / e5m2 不同）。

mxfp4_ops.py
  └─ Mxfp4
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

_FMT = "e2m1"
_FP32_MIN_NORMAL = 2 ** (1 - 127)
_E8M0_EXP_MIN = -127
_E8M0_EXP_MAX = 127


def _round_element(a: torch.Tensor, private_exp: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """OCP MXFP4 roundTiesToEven。

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


class Mxfp4:
    """MXFP4。仅对称。输出是 E2M1 网格，不打包。"""

    @staticmethod
    def calc_qparam_sym(x, axis=-1):
        """MXFP4 动态共享指数。x 为浮点张量；axis 为分块轴，块长固定 32。

        返回: 切块布局下 keepdim 的 uint8 E8M0（raw = shared_exp + 127），可落盘。
        与 MXFP8 同一套编码：clamp 到 [0, 254]，不把下溢写成 NaN。

        OCP MX Spec §6.3 / E8M0，元素格式 E2M1：
        shared_exp = clamp(floor(log2(max_abs)) - emax, -127, 127)
        """
        blocked, meta = view_blocks(x, axis)
        block_dims = [a + 1 for a in meta["axes"]]
        emax = elem_cfg(_FMT)["emax"]
        max_abs = blocked.abs().amax(dim=block_dims, keepdim=True)
        shared = floor_ste(torch.log2(max_abs + _FP32_MIN_NORMAL * (max_abs == 0).to(max_abs.dtype)))
        se = (shared - emax).clamp(min=_E8M0_EXP_MIN, max=_E8M0_EXP_MAX)
        return (se + 127.0).round().clamp(0, 254).to(torch.uint8)

    @staticmethod
    def quantize_sym(x, shared_exp_t, axis=-1, dtype=None, ste_clamp=True):
        """MXFP4 对称静态量化。返回 (q, shared_exp_t)，shared_exp 原样带回。

        q 默认 float32 的 E2M1 网格，形状与 x 相同。dtype=torch.int8 为同形 8bit，不打包。
        只有 float32 能反传。
        ste_clamp: 默认 True，clamp 反向恒等；False 用 torch.clamp 真导数。

        OCP MXFP4：
        x'          = x / 2^shared_exp     （用 ldexp，避免 2^shared_exp 次正规导致除 0）
        private_exp = floor(log2(|x'|))
        x_q         = roundTiesToEven(x')
        x_q         = clamp(x_q, ±max_norm)
        """
        cfg = elem_cfg(_FMT)
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
        """MXFP4 对称反量化。q 为 float32 或 int8 网格。返回 float32。

        OCP：x = q * 2^shared_exp。shared_exp 可为 uint8 E8M0。
        """
        return blocked_apply(
            lambda t, exp: t.to(torch.float32) * (2 ** exp.to(torch.float32)),
            q.to(torch.float32),
            axis,
            as_shared_exp(shared_exp_t),
        )

    @staticmethod
    def quantize_dynamic_sym(x, axis=-1, dtype=None, ste_clamp=True):
        """MXFP4 对称动态量化。返回 (q, shared_exp)，shared_exp 为 uint8 E8M0 且已 detach。"""
        se = Mxfp4.calc_qparam_sym(x, axis=axis).detach()
        return Mxfp4.quantize_sym(x, se, axis=axis, dtype=dtype, ste_clamp=ste_clamp)


mxfp4_calc_qparam_sym = Mxfp4.calc_qparam_sym
mxfp4_quantize_sym = Mxfp4.quantize_sym
mxfp4_dequantize_sym = Mxfp4.dequantize_sym
mxfp4_quantize_dynamic_sym = Mxfp4.quantize_dynamic_sym
