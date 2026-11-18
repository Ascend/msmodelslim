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

from __future__ import absolute_import, division, print_function
import time
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from msmodelslim.pytorch.quant.ptq_tools.ptq_kia.quant_funcs import (
    StatMinMaxObserver,
    HistogramObserver,
    activation_quant_opt,
    fake_quantize,
    linear_quantization_params,
    is_signed_check,
    init_weight_quant_normal,
    weight_quant_opt,
)  # squant algorithm api
from msmodelslim import logger


class Quantizer(nn.Module):
    """ Quantizer for quantize the tensor"""

    def __init__(self,
                 bit=8,
                 is_signed=True,
                 is_enable=False,
                 is_input=False,
                 cfg=None):
        super(Quantizer, self).__init__()
        self.register_buffer('bit', torch.tensor(1))
        self.bit = torch.tensor(bit).to(cfg.device)
        self.is_signed = is_signed
        self.is_enable = is_enable
        self.is_enable_input = is_enable
        self.is_input = is_input
        self.is_sym = cfg.a_sym if is_input else cfg.w_sym

        # keep accuracy, [bool, int] or bool
        self.admm = cfg.keep_acc['admm']
        self.easy_quant = cfg.keep_acc['easy_quant']
        self.round_opt = cfg.keep_acc['round_opt']

        # initial data_free hyper-params
        self.dfree_mode = "dfree"
        self.dfree_k = True
        self.dfree_c = True
        self.percent = 1
        self.is_sigma = False
        if cfg.sigma > 0:
            self.percent = cfg.sigma
            self.is_sigma = True

        if is_input:
            self.init_act_and_observer(cfg)

        if cfg.quant_mode == 1:
            self.weight_dfree = False
        else:
            self.weight_dfree = True

        self.input_scale = None
        self.input_offset = None
        self.weight_scale = None
        self.weight_offset = None
        self.name = None
        self.has_zero = True
        self.quant_weight_tensor = None
        self.int_weight_tensor = None
        self.x_min = torch.tensor(1.0)
        self.x_max = torch.tensor(1.0)
        self.has_init_quant_para = False
        
        self.tensor_sum = None
        self.tensor_sum_cov = None

    def init_act_and_observer(self, cfg):
        if cfg.quant_mode in [1, 2]:
            self.act_dfree = False
        if cfg.act_method == 1:
            self.observer = \
                StatMinMaxObserver(self.bit, self.is_signed,
                                   self.is_sym, method="quantile")
        elif cfg.act_method == 2:
            self.observer = \
                HistogramObserver(qscheme=torch.per_tensor_affine)
        else:
            self.act_dfree = True


    def disable_input_quantization(self):
        self.is_enable_input = False

    def enable_quantization(self, name):
        self.name = name
        self.is_enable = True

    def disable_quantization(self, name):
        self.name = name
        self.is_enable = False

    def update_signed(self, tensor):
        self.is_signed = is_signed_check(tensor)

    @torch.no_grad()
    def _init_data_free_param(self, data):
        """ Initial data-free quantizaton"""
        if not self.has_init_quant_para:
            logger.info("QUANT %r bit: %r", self.bit.item(), self.name)
            self.update_signed(data)

            # determine the data_free mode
            if self.dfree_mode == "dfree-e":
                self.dfree_k = False
                self.dfree_c = False
                self.dfree_mode = "dfree"
            elif self.dfree_mode == "dfree-k":
                self.dfree_c = False
                self.dfree_mode = "dfree"
            elif self.dfree_mode == "dfree-c":
                self.dfree_k = False
                self.dfree_mode = "dfree"

            if self.dfree_mode == "dfree":
                if not self.is_input:
                    start = time.perf_counter()
                    self.int_weight_tensor, \
                    self.quant_weight_tensor, \
                    self.weight_scale, \
                    self.weight_offset = weight_quant_opt(
                        data,
                        self.bit,
                        self.is_sym,
                        self.dfree_mode,
                        self.dfree_k,
                        self.dfree_c,
                        admm=self.admm,
                    )
                    elapsed = (time.perf_counter() - start)
                    logger.info("Quantization time:%s ms", elapsed * 1000)
                else:
                    # Min-max activation quantization for data-free PTQ
                    self.x_min, self.x_max = activation_quant_opt(
                        data,
                        self.bit,
                        self.is_signed,
                        self.is_sym,
                        self.is_sigma,
                        self.percent,
                        easy_quant=self.easy_quant,
                    )
            else:
                raise RuntimeError("Unsupported mode:%r" % self.dfree_mode)

        self.has_init_quant_para = True

    def new_quant_tensor(self, data):
        _, tensor = fake_quantize(
            data, self.input_scale, self.input_offset, self.bit
        )
        return tensor

    def tensor_forward(self, tensor):
        if not self.is_enable:
            return tensor
        if self.is_input:
            if not self.is_enable_input:
                return tensor
        # value check of forward tensor 
        if tensor.numel() == 0:
            logger.info("the numel of input tensor is zero, disable quantization:%r", self.name)
            self.disable_quantization(self.name)
            return tensor
        with torch.no_grad():
            # for weight
            if not self.is_input:
                if self.weight_dfree:
                    self._init_data_free_param(tensor)
                else:
                    # get the scale and offset
                    self._init_weight_quant_normal(tensor)
                return self._weight_forward(tensor)
            # for activation
            else:
                if not self.act_dfree:
                    # update the scale and offset and return dequant_tensor
                    self._update_input_observer(tensor)
                    return self._observer_forward(tensor)
                return self._forward(tensor)

    def stop_calibration(self):
        self.observer = None

    def _forward(self, data):
        self.input_scale, \
        self.input_offset = linear_quantization_params(
            self.bit, self.x_min, self.x_max, q_signed=True, sym=self.is_sym
        )
        return self.new_quant_tensor(data)

    def _update_input_observer(self, data):
        if self.observer is not None:
            self.observer.update(data)

    def _observer_forward(self, data):
        if self.observer is not None:
            self.input_scale, self.input_offset = self.observer.get_scale_offset()
        return self.new_quant_tensor(data)

    def _init_weight_quant_normal(self,
                                  weight,
                                  integral_zero_point=True):
        if self.has_init_quant_para:
            return

        start = time.perf_counter()
        if 'low_mem' in inspect.signature(init_weight_quant_normal).parameters:
            _, _, self.weight_scale, self.weight_offset = init_weight_quant_normal(
                weight,
                self.bit,
                self.is_sym,
                self.is_signed,
                integral_zero_point,
                admm=self.admm,
                round_opt=self.round_opt,
                force_per_channel=True,
                low_mem=True,
            )
        else:
            _, _, self.weight_scale, self.weight_offset = init_weight_quant_normal(
                weight,
                self.bit,
                self.is_sym,
                self.is_signed,
                integral_zero_point,
                admm=self.admm,
                round_opt=self.round_opt,
                force_per_channel=True,
            )
        elapsed = (time.perf_counter() - start)
        logger.info("Quantization time:%s ms", elapsed * 1000)
        self.has_init_quant_para = True

    def _weight_forward(self, data):
        _, dequant_tensor = fake_quantize(data, self.weight_scale, self.weight_offset, self.bit, self.round_opt)
        return dequant_tensor


class TensorQuantizer(Quantizer):
    """
    Class to quantize given tensor
    """
    def __init__(self, **kwargs):
        super(TensorQuantizer, self).__init__(**kwargs)

    def forward(self, tensor):
        return self.tensor_forward(tensor)


class LinearQuantizer(nn.Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, cfg=None):
        """
        cfg: quantizaton configuration
        """
        super(LinearQuantizer, self).__init__()
        self.quant_input = TensorQuantizer(
            bit=cfg.a_bit, is_signed=cfg.a_signed, is_enable=True,
            is_input=True, cfg=cfg)
        self.quant_weight = TensorQuantizer(
            bit=cfg.w_bit, is_signed=cfg.w_signed, is_enable=True,
            cfg=cfg)
        self.in_features = None
        self.out_features = None
        self.weight = None
        self.bias = None

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        try:
            self.bias = linear.bias
        except AttributeError:
            self.bias = None
        finally:
            pass

    def forward(self, input_data, *args, **kwargs):
        input_quant = self.quant_input(input_data)
        weight_quant = self.quant_weight(self.weight)
        return F.linear(input_quant, weight_quant, self.bias)


class Conv2dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, cfg=None):
        """
        cfg: quantizaton configuration
        """
        super(Conv2dQuantizer, self).__init__()
        self.quant_input = TensorQuantizer(
            bit=cfg.a_bit, is_signed=cfg.a_signed, is_enable=True,
            is_input=True, cfg=cfg
        )
        self.quant_weight = TensorQuantizer(
            bit=cfg.w_bit, is_signed=cfg.w_signed, is_enable=True,
            cfg=cfg
        )
        self.in_channels = None
        self.out_channels = None
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.dilation = None
        self.groups = None
        self.weight = None
        self.bias = None
        self.padding_mode = None
        self.transposed = None
        self.reversed_padding_repeated_twice = None

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        try:
            self.padding_mode = conv.padding_mode
            self.reversed_padding_repeated_twice = self.get_reversed_padding()
            self.transposed = conv.transposed
        except AttributeError as e:
            logger.error(e)
            self.transposed = None
            self.padding_mode = 'zeros'
            self.reversed_padding_repeated_twice = None
        self.weight = conv.weight
        try:
            self.bias = conv.bias
        except AttributeError:
            self.bias = None
        finally:
            pass

    def reverse_repeat_tuple(self, t, n):
        """Reverse the order of `t` and repeat each element for `n` times.

        This can be used to translate padding arg used by Conv and Pooling modules
        to the ones used by `F.pad`.
        """
        return tuple(x for x in reversed(t) for _ in range(n))

    def get_reversed_padding(self):
        if isinstance(self.padding, str):
            reversed_padding_repeated_twice = [0, 0] * len(self.kernel_size)
            if self.padding == 'same':
                for d, k, i in zip(self.dilation, self.kernel_size,
                                   range(len(self.kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    reversed_padding_repeated_twice[2 * i] = left_pad
                    reversed_padding_repeated_twice[2 * i + 1] = (
                            total_padding - left_pad)
        else:
            reversed_padding_repeated_twice = self.reverse_repeat_tuple(self.padding, 2)
        return reversed_padding_repeated_twice

    def forward(self, input_data, *args, **kwargs):
        input_quant = self.quant_input(input_data)
        weight_quant = self.quant_weight(self.weight)
        if self.padding_mode != "zeros":
            input_quant = F.pad(input_quant, self.reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding

        return F.conv2d(
            input_quant, weight_quant, self.bias, self.stride, padding, self.dilation, self.groups
        )