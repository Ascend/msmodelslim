#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import torch
from msmodelslim.utils.exception import UnexpectedError, UnsupportedError


def round_ste(x: torch.Tensor):
    """实现直通估计器（STE）的四舍五入操作"""
    return (x.round() - x).detach() + x


def get_qmin_qmax(bits, sym):
    """根据位宽和对称性获取量化范围的最小值和最大值"""
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max - 1
    else:
        q_max, q_min = torch.tensor(2**bits - 1), 0
    return q_max, q_min


def get_maxq(bits, sym):
    """获取量化最大值（仅用于符号量化）"""
    if sym:
        return torch.tensor(2 ** (bits - 1) - 1)
    else:
        return torch.tensor(2**bits - 1)


def sym_quant(x, scale, bits, is_signed=True):
    """对称量化：x -> q"""
    scale = scale.to(x.device)
    q_min = -(2 ** (bits - 1)) if is_signed else 0
    q_max = 2 ** (bits - 1) - 1 if is_signed else 2**bits - 1
    q = torch.clamp(round_ste(x / scale), q_min, q_max)
    return q, scale


def sym_dequant(q, scale):
    """对称反量化：q -> x"""
    return scale * q


def sym_quant_dequant(x, scale, bits, is_signed=True):
    """对称量化-反量化（端到端）"""
    return sym_dequant(*sym_quant(x, scale, bits, is_signed))


def asym_quant(x, scale, zero, bits, is_signed=True):
    """非对称量化：x -> q"""
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q_min = -(2 ** (bits - 1)) if is_signed else 0
    q_max = 2 ** (bits - 1) - 1 if is_signed else 2**bits - 1
    q = torch.clamp(round_ste(x / scale) + zero, q_min, q_max)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    """非对称反量化：q -> x"""
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, bits, is_signed=True):
    """非对称量化-反量化（端到端）"""
    return asym_dequant(*asym_quant(x, scale, zero, bits, is_signed))


class ActivationQuantizer(torch.nn.Module):
    """激活量化器，支持对称/非对称量化"""

    def __init__(self, bits, sym=False, lac=False, groupsize=-1, clip_ratio=None, per_tensor=False, is_signed=True):
        super().__init__()
        self.bits = bits
        self.q_max, self.q_min = get_qmin_qmax(bits, sym)
        self.sym = sym
        self.lac = lac
        self._clip_ratio = clip_ratio
        self.groupsize = groupsize
        if self.groupsize > 0:
            raise UnsupportedError("Activating per-group quantization is not yet supported.")

        if self.lac:
            init_value = 4.0
            self.sigmoid = torch.nn.Sigmoid()
            self.clip_factor = torch.nn.Parameter(torch.ones(1) * init_value, requires_grad=True)
        self.enable = True

        self.is_signed = is_signed
        self.scale, self.zero = None, None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(bits={self.bits}, sym={self.sym}, lac={self.lac}, is_signed={self.is_signed})"
        )

    def reparameterize(self):
        """将可学习的剪裁因子转为缓冲区"""
        if self.lac:
            clip_factor = self.clip_factor
            del self.clip_factor
            self.register_buffer('clip_factor', clip_factor)

    def forward(self, x, quantize=True):
        """前向传播：根据模式决定是否量化"""
        if self.bits == 16 or not self.enable:
            return x
        if not quantize:
            return x
        return self.fake_quant(x)

    def fake_quant(self, x):
        """应用伪量化"""
        x_dtype = x.dtype
        scale, zero = self.get_scale_zero(x)
        if self.sym:
            return sym_quant_dequant(x, scale, self.bits, self.is_signed).to(x_dtype)
        else:
            return asym_quant_dequant(x, scale, zero, self.bits, self.is_signed).to(x_dtype)

    def get_clip_ratio(self):
        """获取剪裁比例（LAC 模式下使用 Sigmoid）"""
        return self.sigmoid(self.clip_factor) if self.lac else self._clip_ratio

    def get_scale_zero(self, x):
        """根据输入计算量化参数（scale 和 zero）"""
        q_max = self.q_max.to(x)
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))
        xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)

        if self.lac:
            clip_ratio = self.sigmoid(self.clip_factor)
            xmax = xmax * clip_ratio
            xmin = xmin * clip_ratio
        elif self._clip_ratio is not None:
            xmax = xmax * self._clip_ratio
            xmin = xmin * self._clip_ratio

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            scale = xmax / q_max
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = torch.zeros_like(scale)
        else:
            scale = (xmax - xmin).clamp(min=1e-5) / q_max
            zero = torch.round(-xmin / scale)
            if self.is_signed:
                zero = zero - 2 ** (self.bits - 1)
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = zero.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        return scale, zero


class WeightQuantizer(torch.nn.Module):
    """权重量化器，支持每通道/全局、对称/非对称、可学习剪裁"""

    def __init__(self, in_size, out_size, bits=8, perchannel=False, sym=True, lwc=False, is_signed=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.enable = True
        self.enable_find = True
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.maxq = get_maxq(bits, sym)
        self.lwc = lwc
        self.is_signed = is_signed

        if self.lwc:
            init_value = 4.0
            self.clip_factor_w_max = torch.nn.Parameter(torch.ones(out_size, 1) * init_value, requires_grad=True)
            self.clip_factor_w_min = torch.nn.Parameter(torch.ones(out_size, 1) * init_value, requires_grad=True)
            self.sigmoid = torch.nn.Sigmoid()
        self.scale = None
        self.zero = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(bits={self.bits}, sym={self.sym}, lwc={self.lwc}, is_signed={self.is_signed})"
        )

    def reparameterize(self):
        """将可学习的权重剪裁因子转为缓冲区"""
        if self.lwc:
            clip_factor_w_max = self.clip_factor_w_max
            clip_factor_w_min = self.clip_factor_w_min
            del self.clip_factor_w_max
            del self.clip_factor_w_min
            self.register_buffer('clip_factor_w_max', clip_factor_w_max)
            self.register_buffer('clip_factor_w_min', clip_factor_w_min)

    def apply_wclip(self, weight):
        """应用可学习权重剪裁（LWC）"""
        wmin, wmax = weight.min(1, keepdim=True)[0], weight.max(1, keepdim=True)[0]
        wmax *= self.sigmoid(self.clip_factor_w_max)
        wmin *= self.sigmoid(self.clip_factor_w_min)
        return torch.clamp(weight, min=wmin, max=wmax)

    def find_params(self, x):
        """根据输入权重计算 scale 和 zero（支持 per-channel）"""
        if self.bits == 16 or not self.enable:
            return
        dev = x.device
        self.maxq = get_maxq(self.bits, self.sym).to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)
            if self.is_signed:
                self.zero = self.zero - 2 ** (self.bits - 1)

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)

    def quantize(self, x, y=None):
        """执行量化（支持动态参数查找）"""
        if self.enable and self.bits < 16:
            x_dtype = x.dtype
            if self.enable_find:
                self.find_params(x)
            if not self.ready():
                raise UnexpectedError("WeightQuantizer is not ready. Please call find_params first.")
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.bits, self.is_signed).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.bits, self.is_signed).to(x_dtype)
        return x

    def forward(self, x, y=None, quantize=True):
        """前向传播：根据 quantize 标志决定是否量化"""
        return self.quantize(x, y) if quantize else x

    def ready(self):
        """检查量化参数是否已正确初始化"""
        return torch.all(self.scale != 0)

    def get_fake_quant_weight(self, x):
        """对权重应用伪量化并禁用后续查找（用于 reparameterize）"""
        x = self.quantize(x)
        self.enable = False
        return x
