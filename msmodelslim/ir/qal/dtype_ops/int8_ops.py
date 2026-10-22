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

INT8 数据格式 ops。一个类 Int8，对称与非对称都是它的方法。
元素公式不带粒度参数。per_tensor / per_channel 在原张量上直接算，per_group 经 grouped_apply。
quantize 默认 float32；只有 float32 能反传。dtype 选择低精度输出。

int8_ops.py
  └─ Int8
       *_per_tensor_sym / *_per_channel_sym / *_per_group_sym
       *_per_tensor_asym / *_per_channel_asym / *_per_group_asym
"""

import torch

from .common import cast_q, clamp_ste, grouped_apply, reduce_minmax, round_ste, view_groups

# OCP INT8（有符号 b=8）：
#   qmin = -2^(b-1) = -128
#   qmax =  2^(b-1) - 1 = 127
#   levels = 2^b - 1 = 255
_QMIN = -128
_QMAX = 127
_LEVELS = 255


def _eps(device):
    return torch.tensor(torch.finfo(torch.float32).eps, device=device, dtype=torch.float32)


def _contain_zero(min_val, max_val):
    min_val = min_val.to(torch.float32)
    max_val = max_val.to(torch.float32)
    return (
        torch.minimum(min_val, torch.zeros_like(min_val)),
        torch.maximum(max_val, torch.zeros_like(max_val)),
    )


def _minmax_reduced(x, axis):
    return _contain_zero(*reduce_minmax(x, axis))


def _minmax_per_group(x, axis, group_size):
    x_v, _meta = view_groups(x, axis, group_size)
    return _contain_zero(x_v.amin(dim=-1, keepdim=True), x_v.amax(dim=-1, keepdim=True))


def _sym_scale(min_val, max_val):
    amax = torch.maximum(-min_val, max_val)
    return torch.maximum(amax / float(_QMAX), _eps(min_val.device))


def _asym_params(min_val, max_val):
    scale = torch.maximum(max_val / float(_LEVELS) - min_val / float(_LEVELS), _eps(min_val.device))
    offset = round_ste(-min_val / scale) + float(_QMIN)
    return scale, offset


def _q_sym(x, scale, clip):
    return clip(
        round_ste(x.to(torch.float32) / scale.to(torch.float32)),
        min=float(_QMIN),
        max=float(_QMAX),
    )


def _q_asym(x, scale, offset, clip):
    return clip(
        round_ste(x.to(torch.float32) / scale.to(torch.float32) + offset.to(torch.float32)),
        min=float(_QMIN),
        max=float(_QMAX),
    )


def _dq_sym(q, scale):
    return q.to(torch.float32) * scale.to(torch.float32)


def _dq_asym(q, scale, offset):
    return (q.to(torch.float32) - offset.to(torch.float32)) * scale.to(torch.float32)


def _clip(ste_clamp):
    return clamp_ste if ste_clamp else torch.clamp


class Int8:
    """INT8。对称与非对称、三种粒度都在这个类里。元素公式在 _q_sym / _q_asym。"""

    @staticmethod
    def calc_qparam_per_group_sym(x, axis, group_size):
        """INT8 对称 per-group scale。axis 维须整除 group_size。

        返回: float32 scale，形状为切组后最后一维 keepdim。

        OCP 有符号 INT8 对称：
        amax  = max(-min_val, max_val)，区间强制包含 0
        scale = max(amax, eps) / 127
        """
        return _sym_scale(*_minmax_per_group(x, axis, group_size))

    @staticmethod
    def calc_qparam_per_channel_sym(x, axis):
        """INT8 对称 per-channel scale。沿 axis 以外的维归约，keepdim。"""
        return _sym_scale(*_minmax_reduced(x, axis))

    @staticmethod
    def calc_qparam_per_tensor_sym(x):
        """INT8 对称 per-tensor scale。全张量一个标量。"""
        return _sym_scale(*_minmax_reduced(x, None))

    @staticmethod
    def quantize_per_group_sym(x, scale, axis, group_size, dtype=None, ste_clamp=True):
        """INT8 对称 per-group 静态量化。返回 (q, scale)，scale 原样带回。

        q 默认 float32，只有 float32 能反传。dtype 换成 torch.int8 等低精度，不打包。
        ste_clamp: 默认 True，clamp 反向恒等；False 用 torch.clamp 真导数。
        OCP：q = clamp(roundTiesToEven(x / scale), -128, 127)
        """
        q = grouped_apply(
            lambda t, s: _q_sym(t, s, _clip(ste_clamp)),
            x,
            axis,
            group_size,
            scale,
        )
        return cast_q(q, dtype), scale

    @staticmethod
    def quantize_per_channel_sym(x, scale, axis, dtype=None, ste_clamp=True):
        """INT8 对称 per-channel 静态量化。scale 须能广播到 x。axis 与 calc 对齐。"""
        del axis
        return cast_q(_q_sym(x, scale, _clip(ste_clamp)), dtype), scale

    @staticmethod
    def quantize_per_tensor_sym(x, scale, dtype=None, ste_clamp=True):
        """INT8 对称 per-tensor 静态量化。"""
        return cast_q(_q_sym(x, scale, _clip(ste_clamp)), dtype), scale

    @staticmethod
    def dequantize_per_group_sym(q, scale, axis, group_size):
        """INT8 对称 per-group 反量化。返回与 q 同形状的 float32。OCP：x = q * scale"""
        return grouped_apply(_dq_sym, q, axis, group_size, scale)

    @staticmethod
    def dequantize_per_channel_sym(q, scale, axis):
        """INT8 对称 per-channel 反量化。"""
        del axis
        return _dq_sym(q, scale)

    @staticmethod
    def dequantize_per_tensor_sym(q, scale):
        """INT8 对称 per-tensor 反量化。"""
        return _dq_sym(q, scale)

    @staticmethod
    def quantize_dynamic_per_group_sym(x, axis, group_size, dtype=None, ste_clamp=True):
        """INT8 对称 per-group 动态量化。返回 (q, scale)，scale 已 detach。"""
        scale = Int8.calc_qparam_per_group_sym(x, axis, group_size).detach()
        return Int8.quantize_per_group_sym(x, scale, axis, group_size, dtype, ste_clamp)

    @staticmethod
    def quantize_dynamic_per_channel_sym(x, axis, dtype=None, ste_clamp=True):
        """INT8 对称 per-channel 动态量化。scale 已 detach。"""
        scale = Int8.calc_qparam_per_channel_sym(x, axis).detach()
        return Int8.quantize_per_channel_sym(x, scale, axis, dtype, ste_clamp)

    @staticmethod
    def quantize_dynamic_per_tensor_sym(x, dtype=None, ste_clamp=True):
        """INT8 对称 per-tensor 动态量化。scale 已 detach。"""
        scale = Int8.calc_qparam_per_tensor_sym(x).detach()
        return Int8.quantize_per_tensor_sym(x, scale, dtype, ste_clamp)

    @staticmethod
    def calc_qparam_per_group_asym(x, axis, group_size):
        """INT8 非对称 per-group 参数。返回 (scale, offset)。

        OCP 有符号 INT8 非对称：
        scale = max((max_val - min_val) / 255, eps)，区间强制包含 0
        zp    = roundTiesToEven(-min_val / scale) + qmin
        """
        return _asym_params(*_minmax_per_group(x, axis, group_size))

    @staticmethod
    def calc_qparam_per_channel_asym(x, axis):
        """INT8 非对称 per-channel 参数。返回 (scale, offset)。"""
        return _asym_params(*_minmax_reduced(x, axis))

    @staticmethod
    def calc_qparam_per_tensor_asym(x):
        """INT8 非对称 per-tensor 参数。返回 (scale, offset)。"""
        return _asym_params(*_minmax_reduced(x, None))

    @staticmethod
    def quantize_per_group_asym(x, scale, offset, axis, group_size, dtype=None, ste_clamp=True):
        """INT8 非对称 per-group 静态量化。返回 (q, scale, offset)。

        q 默认 float32。OCP：q = clamp(roundTiesToEven(x / scale + zp), -128, 127)
        """
        q = grouped_apply(
            lambda t, s, zp: _q_asym(t, s, zp, _clip(ste_clamp)),
            x,
            axis,
            group_size,
            scale,
            offset,
        )
        return cast_q(q, dtype), scale, offset

    @staticmethod
    def quantize_per_channel_asym(x, scale, offset, axis, dtype=None, ste_clamp=True):
        """INT8 非对称 per-channel 静态量化。"""
        del axis
        return cast_q(_q_asym(x, scale, offset, _clip(ste_clamp)), dtype), scale, offset

    @staticmethod
    def quantize_per_tensor_asym(x, scale, offset, dtype=None, ste_clamp=True):
        """INT8 非对称 per-tensor 静态量化。"""
        return cast_q(_q_asym(x, scale, offset, _clip(ste_clamp)), dtype), scale, offset

    @staticmethod
    def dequantize_per_group_asym(q, scale, offset, axis, group_size):
        """INT8 非对称 per-group 反量化。OCP：x = (q - zp) * scale"""
        return grouped_apply(_dq_asym, q, axis, group_size, scale, offset)

    @staticmethod
    def dequantize_per_channel_asym(q, scale, offset, axis):
        """INT8 非对称 per-channel 反量化。"""
        del axis
        return _dq_asym(q, scale, offset)

    @staticmethod
    def dequantize_per_tensor_asym(q, scale, offset):
        """INT8 非对称 per-tensor 反量化。"""
        return _dq_asym(q, scale, offset)

    @staticmethod
    def quantize_dynamic_per_group_asym(x, axis, group_size, dtype=None, ste_clamp=True):
        """INT8 非对称 per-group 动态量化。参数已 detach。"""
        scale, offset = Int8.calc_qparam_per_group_asym(x, axis, group_size)
        return Int8.quantize_per_group_asym(x, scale.detach(), offset.detach(), axis, group_size, dtype, ste_clamp)

    @staticmethod
    def quantize_dynamic_per_channel_asym(x, axis, dtype=None, ste_clamp=True):
        """INT8 非对称 per-channel 动态量化。参数已 detach。"""
        scale, offset = Int8.calc_qparam_per_channel_asym(x, axis)
        return Int8.quantize_per_channel_asym(x, scale.detach(), offset.detach(), axis, dtype, ste_clamp)

    @staticmethod
    def quantize_dynamic_per_tensor_asym(x, dtype=None, ste_clamp=True):
        """INT8 非对称 per-tensor 动态量化。参数已 detach。"""
        scale, offset = Int8.calc_qparam_per_tensor_asym(x)
        return Int8.quantize_per_tensor_asym(x, scale.detach(), offset.detach(), dtype, ste_clamp)


int8_calc_qparam_per_tensor_sym = Int8.calc_qparam_per_tensor_sym
int8_calc_qparam_per_channel_sym = Int8.calc_qparam_per_channel_sym
int8_calc_qparam_per_group_sym = Int8.calc_qparam_per_group_sym
int8_quantize_per_tensor_sym = Int8.quantize_per_tensor_sym
int8_quantize_per_channel_sym = Int8.quantize_per_channel_sym
int8_quantize_per_group_sym = Int8.quantize_per_group_sym
int8_dequantize_per_tensor_sym = Int8.dequantize_per_tensor_sym
int8_dequantize_per_channel_sym = Int8.dequantize_per_channel_sym
int8_dequantize_per_group_sym = Int8.dequantize_per_group_sym
int8_quantize_dynamic_per_tensor_sym = Int8.quantize_dynamic_per_tensor_sym
int8_quantize_dynamic_per_channel_sym = Int8.quantize_dynamic_per_channel_sym
int8_quantize_dynamic_per_group_sym = Int8.quantize_dynamic_per_group_sym
int8_calc_qparam_per_tensor_asym = Int8.calc_qparam_per_tensor_asym
int8_calc_qparam_per_channel_asym = Int8.calc_qparam_per_channel_asym
int8_calc_qparam_per_group_asym = Int8.calc_qparam_per_group_asym
int8_quantize_per_tensor_asym = Int8.quantize_per_tensor_asym
int8_quantize_per_channel_asym = Int8.quantize_per_channel_asym
int8_quantize_per_group_asym = Int8.quantize_per_group_asym
int8_dequantize_per_tensor_asym = Int8.dequantize_per_tensor_asym
int8_dequantize_per_channel_asym = Int8.dequantize_per_channel_asym
int8_dequantize_per_group_asym = Int8.dequantize_per_group_asym
int8_quantize_dynamic_per_tensor_asym = Int8.quantize_dynamic_per_tensor_asym
int8_quantize_dynamic_per_channel_asym = Int8.quantize_dynamic_per_channel_asym
int8_quantize_dynamic_per_group_asym = Int8.quantize_dynamic_per_group_asym
