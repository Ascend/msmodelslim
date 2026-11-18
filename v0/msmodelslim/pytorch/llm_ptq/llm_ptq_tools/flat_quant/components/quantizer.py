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

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import StatMinMaxObserver, linear_quantization_params


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def get_qmin_qmax(bits, sym):
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = - q_max - 1
    else:
        q_max, q_min = torch.tensor(2 ** bits - 1), 0
    return q_max, q_min


def get_maxq(bits, sym):
    if sym:
        return torch.tensor(2 ** (bits - 1) - 1)
    else:
        return torch.tensor(2 ** bits - 1)


def sym_quant(x, scale, bits, is_signed=True):
    scale = scale.to(x.device)
    if is_signed:
        q_min, q_max = - 2 ** (bits - 1), 2 ** (bits - 1) - 1
    else:
        q_min, q_max = 0, 2 ** bits - 1

    q = torch.clamp(round_ste(x / scale), q_min, q_max)

    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, bits, is_signed=True):
    return sym_dequant(*sym_quant(x, scale, bits, is_signed))


def asym_quant(x, scale, zero, bits, is_signed=True):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    if is_signed:
        q_min, q_max = - 2 ** (bits - 1), 2 ** (bits - 1) - 1
    else:
        q_min, q_max = 0, 2 ** bits - 1

    q = torch.clamp(round_ste(x / scale) + zero, q_min, q_max)

    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, bits, is_signed=True):
    return asym_dequant(*asym_quant(x, scale, zero, bits, is_signed))


class ActivationQuantizer(torch.nn.Module):
    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''
    def __init__(self, bits, sym=False, lac=False, groupsize=-1, clip_ratio=None, per_tensor=False, is_signed=True):
        super(ActivationQuantizer, self).__init__()
        self.bits = bits
        self.q_max, self.q_min = get_qmin_qmax(bits, sym)
        self.sym = sym
        self.lac = lac
        self._clip_ratio = clip_ratio
        self.groupsize = groupsize
        if self.groupsize > 0:
            raise NotImplementedError("Not support per-group quantization for activation yet.")
        if self.lac:
            init_value = 4.
            self.sigmoid = torch.nn.Sigmoid()
            self.clip_factor = torch.nn.Parameter(torch.ones((1, )) * init_value, requires_grad=True)
        
        self.enable = True
        self.per_tensor = per_tensor
        if per_tensor:
            self.observer = StatMinMaxObserver(
                bits,
                sym,
                True,
            )
        self.is_signed = is_signed
        self.scale, self.zero = None, None

    def __repr__(self):
        res = f"{self.__class__.__name__}(bits={self.bits}, "
        res += f"sym={self.sym}, "
        res += f"lac={self.lac}, "
        res += f"per_tensor={self.per_tensor}, "
        res += f"is_signed={self.is_signed})"
        return res

    def reparameterize(self):
        if self.lac:
            clip_factor = self.clip_factor
            del self.clip_factor
            self.register_buffer('clip_factor', clip_factor)

    def forward(self, x, quantize=True):
        if self.bits == 16 or (not self.enable):
            return x
        if not quantize:
            if self.per_tensor:
                self.observer.update(x)
            return x
        fq_x = self.fake_quant(x)
        return fq_x

    def fake_quant(self, x):
        x_dtype = x.dtype
        scale, zero = self.get_scale_zero(x)
        if self.sym:
            return sym_quant_dequant(x, scale, self.bits, is_signed=self.is_signed).to(x_dtype)
        else:
            return asym_quant_dequant(x, scale, zero, self.bits, is_signed=self.is_signed).to(x_dtype)

    def get_clip_ratio(self):
        if self.lac:
            return self.sigmoid(self.clip_factor)
        else:
            return self._clip_ratio

    def get_scale_zero(self, x):
        if self.per_tensor:
            if self.scale is None or self.zero is None:
                if x is None:
                    x_min, x_max = self.observer.get_min_max("cpu")
                else:
                    x_min, x_max = self.observer.get_min_max(x.device)
                self.scale, self.zero = linear_quantization_params(
                    bit=self.bits,
                    x_min=x_min,
                    x_max=x_max,
                    q_signed=self.is_signed,
                    sym=self.sym
                )
            return self.scale, self.zero
        
        q_max = self.q_max.to(x)
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))
        xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)

        if self.lac:
            xmax = xmax * self.sigmoid(self.clip_factor)
            xmin = xmin * self.sigmoid(self.clip_factor)
        elif self._clip_ratio is not None:
            xmax = xmax * self._clip_ratio
            xmin = xmin * self._clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            scale = (xmax / q_max)
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

            
            zero = torch.zeros_like(scale)
        else:
            scale = (xmax - xmin) / q_max
            zero = torch.round(-xmin / scale)
            if self.is_signed:
                zero = zero - 2 ** (self.bits - 1)
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)   
            zero = zero.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        return scale, zero


class WeightQuantizer(torch.nn.Module):
    def __init__(self, in_size, 
                        out_size, 
                        bits=8, 
                        perchannel=False, 
                        sym=True, 
                        lwc=False, 
                        is_signed=True):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(out_size, 1))
        self.register_buffer('zero', torch.zeros(out_size, 1))
        self.in_size = in_size
        self.out_size = out_size
        self.enable = True
        self.enable_find = True
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.maxq = get_maxq(self.bits, self.sym)
        self.lwc = lwc
        self.is_signed = is_signed

        if self.lwc:
            init_value = 4.
            self.clip_factor_w_max = torch.nn.Parameter(torch.ones((out_size, 1)) * init_value, requires_grad=True)
            self.clip_factor_w_min = torch.nn.Parameter(torch.ones((out_size, 1)) * init_value, requires_grad=True)
            self.sigmoid = torch.nn.Sigmoid()

    def __repr__(self):
        res = f"{self.__class__.__name__}(bits={self.bits}, sym={self.sym}, lwc={self.lwc}, is_signed={self.is_signed})"
        return res

    def reparameterize(self):
        if self.lwc:
            clip_factor_w_max = self.clip_factor_w_max
            clip_factor_w_min = self.clip_factor_w_min
            del self.clip_factor_w_max
            del self.clip_factor_w_min
            self.register_buffer('clip_factor_w_max', clip_factor_w_max)
            self.register_buffer('clip_factor_w_min', clip_factor_w_min)

    def apply_wclip(self, weight):
        wmin, wmax = weight.min(1, keepdim=True)[0], weight.max(1, keepdim=True)[0]
        wmax *= self.sigmoid(self.clip_factor_w_max)
        wmin *= self.sigmoid(self.clip_factor_w_min)
        weight = torch.clamp(weight, min=wmin, max=wmax)
        return weight

    def enable_find_params(self, enable=True):
        self.enable_find = enable

    def get_scale_zero(self, x):
        if self.scale is None or self.zero is None:
            self.find_params(x)
        return self.scale, self.zero

    def find_params(self, x):
        if self.bits == 16 or (not self.enable):
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
        return

    def quantize(self, x, y=None):
        if self.enable and self.bits < 16:
            x_dtype = x.dtype
            if self.enable_find:
                self.find_params(x)
            if not self.ready():
                raise ValueError("WeightQuantizer is not ready, please call find_params first.")
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.bits, self.is_signed).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.bits, self.is_signed).to(x_dtype)
        return x
    
    def forward(self, x, y=None, quantize=True):
        if quantize:
            return self.quantize(x, y)
        else:
            return x

    def enable_quant(self, enable=True):
        self.enable = enable

    def ready(self):
        return torch.all(self.scale != 0)

    def get_fake_quant_weight(self, x):
        x = self.quantize(x)
        self.enable = False
        return x

