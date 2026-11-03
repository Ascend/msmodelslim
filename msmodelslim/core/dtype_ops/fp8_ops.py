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

FP8 数据格式 ops（OCP OFP8）。一个类 Fp8，仅对称。
公式写在 *_per_channel_sym，per_tensor 复用它。
quantize 默认 float32；dtype 选 float8_e4m3fn / float8_e5m2。
fmt 选元素编码，默认 "e4m3"（E4M3，max_norm=448），可选 "e5m2"（E5M2，max_norm=57344）。
fmt 决定 scale 与量化网格；dtype 决定 q 的存储类型。calc_qparam 与 quantize 须同一 fmt。

fp8_ops.py
  └─ Fp8
       calc_qparam / quantize / dequantize / quantize_dynamic
       后缀 per_tensor_sym / per_channel_sym
"""

import torch

from .common import cast_q, clamp_ste, elem_cfg, floor_ste, reduce_amax, round_ste

_FP8_STORAGE = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}
_FP32_MIN_NORMAL = 2 ** (1 - 127)


def _round_element(a: torch.Tensor, private_exp: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """OCP OFP8 roundTiesToEven。

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


class Fp8:
    """FP8。仅对称。公式在 *_per_channel_sym。fmt 默认 e4m3。"""

    @staticmethod
    def calc_qparam_per_channel_sym(x, axis, fmt="e4m3"):
        """FP8 对称动态 scale。

        x: 浮点张量，dtype 为 float32 / float16 / bfloat16，任意形状。
        axis: 沿该维保留独立 scale；per_tensor 复用时传 None。
        fmt: 元素编码，默认 "e4m3"（max_norm=448）；可选 "e5m2"（max_norm=57344）。
        决定 scale = amax / max_norm。不是存储类型。

        返回: scale = max(amax, eps) / max_norm，float32，已能广播到 x。
        eps 取 fp32 最小正规数。
        """
        if fmt not in _FP8_STORAGE:
            raise ValueError(f"unsupported FP8 fmt {fmt!r}, expected e4m3 or e5m2")
        amax = reduce_amax(x, axis).to(torch.float32).abs()
        return amax.clamp(min=_FP32_MIN_NORMAL) / elem_cfg(fmt)["max_norm"]

    @staticmethod
    def calc_qparam_per_tensor_sym(x, fmt="e4m3"):
        return Fp8.calc_qparam_per_channel_sym(x, None, fmt=fmt)

    @staticmethod
    def quantize_per_channel_sym(x, scale, axis, fmt="e4m3", dtype=None, ste_clamp=True):
        """FP8 对称静态量化。

        x: 浮点张量；scale 须能广播到 x。
        axis: 仅调用约定，核心按已广播的 scale 计算，无粒度分支。
        fmt: 同 calc_qparam，默认 "e4m3"；决定舍入网格与饱和上界。须与算 scale 时一致。
        dtype: 默认 float32，只有 float32 能反传。低精度按 fmt 传 float8_e4m3fn / float8_e5m2。
        ste_clamp: 默认 True，clamp 反向恒等；False 用 torch.clamp 真导数。

        返回: (q, scale)。q 与 x 同形状；scale 原样带回。

        OCP OFP8：
        x'          = x / scale
        private_exp = floor(log2(|x'|))
        x_q         = roundTiesToEven(x')
        x_q         = clamp(x_q, ±max_norm)
        """
        del axis
        if fmt not in _FP8_STORAGE:
            raise ValueError(f"unsupported FP8 fmt {fmt!r}, expected e4m3 or e5m2")
        cfg = elem_cfg(fmt)
        inp = x.to(torch.float32) / scale.to(torch.float32)
        zero_mask = inp == 0
        private_exp = floor_ste(torch.log2(torch.abs(inp) + zero_mask.to(inp.dtype)))
        inp = _round_element(inp, private_exp, cfg["ebits"], cfg["mbits"])
        clip = clamp_ste if ste_clamp else torch.clamp
        q = clip(inp, min=-cfg["max_norm"], max=cfg["max_norm"])
        return cast_q(q, dtype), scale

    @staticmethod
    def quantize_per_tensor_sym(x, scale, fmt="e4m3", dtype=None, ste_clamp=True):
        return Fp8.quantize_per_channel_sym(x, scale, None, fmt=fmt, dtype=dtype, ste_clamp=ste_clamp)

    @staticmethod
    def dequantize_per_channel_sym(q, scale, axis):
        """FP8 对称反量化。q 为 float8 或 float32 网格。返回 float32。

        OCP：x = q * scale
        """
        del axis
        return q.to(torch.float32) * scale.to(torch.float32)

    @staticmethod
    def dequantize_per_tensor_sym(q, scale):
        return Fp8.dequantize_per_channel_sym(q, scale, None)

    @staticmethod
    def quantize_dynamic_per_channel_sym(x, axis, fmt="e4m3", dtype=None, ste_clamp=True):
        """FP8 对称动态量化。返回 (q, scale)，scale 已 detach。

        fmt 默认 "e4m3"，含义同 calc_qparam / quantize。
        """
        scale = Fp8.calc_qparam_per_channel_sym(x, axis, fmt=fmt).detach()
        return Fp8.quantize_per_channel_sym(x, scale, axis, fmt=fmt, dtype=dtype, ste_clamp=ste_clamp)

    @staticmethod
    def quantize_dynamic_per_tensor_sym(x, fmt="e4m3", dtype=None, ste_clamp=True):
        return Fp8.quantize_dynamic_per_channel_sym(x, None, fmt=fmt, dtype=dtype, ste_clamp=ste_clamp)


fp8_calc_qparam_per_tensor_sym = Fp8.calc_qparam_per_tensor_sym
fp8_calc_qparam_per_channel_sym = Fp8.calc_qparam_per_channel_sym
fp8_quantize_per_tensor_sym = Fp8.quantize_per_tensor_sym
fp8_quantize_per_channel_sym = Fp8.quantize_per_channel_sym
fp8_dequantize_per_tensor_sym = Fp8.dequantize_per_tensor_sym
fp8_dequantize_per_channel_sym = Fp8.dequantize_per_channel_sym
fp8_quantize_dynamic_per_tensor_sym = Fp8.quantize_dynamic_per_tensor_sym
fp8_quantize_dynamic_per_channel_sym = Fp8.quantize_dynamic_per_channel_sym
