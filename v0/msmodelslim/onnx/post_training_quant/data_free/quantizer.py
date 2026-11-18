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
import numpy as np

from msmodelslim.onnx.post_training_quant.data_free.flip import Flip
from msmodelslim.onnx.post_training_quant.data_free.quantize_funcs import get_quant_param
from msmodelslim.onnx.post_training_quant.data_free.quantize_funcs import linear_quantize, squant_func


class Quantizer:
    def __init__(self, name, bit=8):
        self.name = name
        self.bit = bit
        self.scale = None
        self.offset = None

    def __call__(self, data: np.ndarray):
        self.quantize(data)

    def quantize(self, data: np.ndarray):
        raise NotImplementedError("Please implement the method in subclass.")


class ActivationQuantizer(Quantizer):
    """
    Activation Quantizer for quantize activation
    """
    def __init__(self, name, bit=8, is_signed=True, sigma=25):
        super(ActivationQuantizer, self).__init__(name, bit)
        self.is_signed = is_signed
        self.percent = 1
        self.is_sigma = False
        if sigma > 0:
            self.percent = sigma
            self.is_sigma = True
        self.x_max = None
        self.x_min = None

    def get_sigma(self, data: np.ndarray):
        if self.is_signed:
            return data.std()
        return data[data > 0].std()

    def undate_signed(self, data: np.ndarray):
        if data.min() < 0:
            self.is_signed = True
        else:
            self.is_signed = False

    def quantize(self, data: np.ndarray):
        self.undate_signed(data)
        x_max = data.max()
        alpha = self.percent * abs(data).max()
        if self.is_sigma:
            sigma = self.get_sigma(data)
            alpha = self.percent * sigma
            if self.is_signed:
                # We also consider the signed activation. Other framworks will skip this tensor.
                alpha = self.percent * sigma / 1.25

            # For a higher bit-width, using a wider range still will not cause accuracy loss.
            if self.bit < 6:
                # For small bit, need clip.
                alpha = min(alpha, x_max)
        if self.is_signed:
            self.x_min = -alpha
        else:
            self.x_min = np.zeros_like(alpha)
        self.x_max = alpha
        self.scale, self.offset = get_quant_param(self.bit, self.x_min, self.x_max)


class WeightQuantizer(Quantizer):
    """
    Weight quantizer for quantize weight
    """
    def __init__(self, name, bit=8, mode="squant", is_per_channel=True):
        super(WeightQuantizer, self).__init__(name, bit)
        self.mode = mode
        self.squant_k = True
        self.squant_c = True
        if self.mode == "squant-e":
            self.squant_k = False
            self.squant_c = False
            self.mode = "squant"
        elif self.mode == "squant-k":
            self.squant_c = False
            self.mode = "squant"
        elif self.mode == "squant-c":
            self.squant_k = False
            self.mode = "squant"
        self.quant_weight = None
        self.is_per_channel = is_per_channel

    def quantize(self, data: np.ndarray):
        if self.is_per_channel:
            x_max = data.reshape((data.shape[0], -1)).max(1)
            x_max = np.expand_dims(x_max, 1)
            x_min = data.reshape((data.shape[0], -1)).min(1)
            x_min = np.expand_dims(x_min, 1)
        else:
            x_max = data.max()
            x_min = data.min()

        self.scale, self.offset = get_quant_param(self.bit, x_min, x_max, sym=True)
        quant_weight = linear_quantize(data, self.scale, self.offset)

        q_min = -2 ** (self.bit - 1)
        q_max = 2 ** (self.bit - 1) - 1

        if self.mode == "squant":
            quant_weight = self.adaptive_round(quant_weight, q_min, q_max)
        else:
            quant_weight = quant_weight.round()

        quant_weight = np.clip(quant_weight, q_min, q_max)
        self.quant_weight = quant_weight

    def adaptive_round(self, data: np.ndarray, t_min=None, t_max=None):
        # Get the rounding integer and fraction.
        rounding_number = data.round()
        rounding_error = rounding_number - data

        flip_up = Flip(data=data, round_data=rounding_number, round_err=rounding_error, t_max=t_max)

        flip_down = Flip(data=data, round_data=rounding_number, round_err=rounding_error,
                         t_min=t_min, is_flip_up=False)

        conver_shape = data.reshape((data.shape[0], data.shape[1], -1)).shape
        if conver_shape[2] == 1:
            self.squant_k = False

        if self.squant_k:
            rounding_error_sum = rounding_error.reshape(conver_shape).sum(-1)

            flip_up.order = np.argsort(-flip_up.priority.reshape(conver_shape))
            flip_up.priority *= 0.0

            flip_down.order = np.argsort(-flip_down.priority.reshape(conver_shape))
            flip_down.priority *= 0.0

            squant_func(
                rounding_error_sum,
                rounding_number.reshape(conver_shape),
                rounding_error.reshape(conver_shape),
                flip_up,
                flip_down,
            )

        if self.squant_c:
            conver_shape = (1, data.shape[0], -1)
            rounding_error_sum = rounding_error.reshape(conver_shape).sum(-1)
            flip_up.order = np.argsort(-flip_up.priority.reshape(conver_shape))
            flip_up.order = np.argsort(-flip_down.priority.reshape(conver_shape))

            squant_func(
                rounding_error_sum,
                rounding_number.reshape(conver_shape),
                rounding_error.reshape(conver_shape),
                flip_up,
                flip_down
            )

        rounding_number = np.clip(rounding_number, t_min, t_max)
        return rounding_number
