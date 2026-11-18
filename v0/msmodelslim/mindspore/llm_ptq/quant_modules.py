# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import sys

import numpy as np
import mindspore as ms
import mindspore.nn as nn

import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor, ops
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from msmodelslim.mindspore.llm_ptq.quant_funcs import fake_quantize, linear_quantization_params


class StatMinMaxObserver(nn.Cell):
    """Min-Max Observer"""

    def __init__(self):
        super(StatMinMaxObserver, self).__init__()
        self._stat_max = P.ReduceMax()
        self._stat_min = P.ReduceMin()
        self._min = Parameter(
            Tensor(np.array([0]), ms.dtype.float32), name="float_min")
        self._max = Parameter(
            Tensor(np.array([0]), ms.dtype.float32), name="float_max")
        
        self.sample_num = Parameter(initializer(0, (1,), mstype.int32), name="sample_num")

    def construct(self, x):

        new_min = self._stat_min(x)
        new_max = self._stat_max(x)
        new_min = (new_min - self._min) / (self.sample_num + 1) + self._min
        new_max = (new_max - self._max) / (self.sample_num + 1) + self._max
        new_sample_num = self.sample_num + 1

        ops.assign(self._min, new_min)
        ops.assign(self._max, new_max)
        ops.assign(self.sample_num, new_sample_num)

        return self._min, self._max


class Quantizer(nn.Cell):
    """ Quantizer for quantize the tensor"""

    def __init__(self, is_fake_quant=False):
        super(Quantizer, self).__init__()

        self.is_fake_quant = is_fake_quant
        self.observer = StatMinMaxObserver()

        self.input_scale = Parameter(
            initializer(1.0, (1,)), name="input_scale")
        self.input_offset = Parameter(
            initializer(1.0, (1,)), name="input_offset")
        self.x_max = Parameter(initializer(1.0, (1,)), name="x_max")
        self.x_min = Parameter(initializer(1.0, (1,)), name="x_min")


    def construct(self, tensor, y=None):
        if not self.is_fake_quant:
            x_min_, x_max_ = self.observer(tensor)
            ops.assign(self.x_min, x_min_)
            ops.assign(self.x_max, x_max_)
            input_scale_, input_offset_ = linear_quantization_params(
                8, self.x_min, self.x_max
            )
            ops.assign(self.input_scale, input_scale_)
            ops.assign(self.input_offset, input_offset_)

        _, data_dequant = fake_quantize(tensor, self.input_scale, self.input_offset)
        return data_dequant


class LinearSparseQuantizer(nn.Cell):
    """
    Class to quantize given linear cell
    """

    def __init__(self, cell=None, is_fake_quant=False):
        """
        cfg: quantizaton configuration
        """
        super(LinearSparseQuantizer, self).__init__()
        self.cell = cell
        self.is_fake_quant = is_fake_quant
        self.quant_input = Quantizer(is_fake_quant=is_fake_quant)

    def construct(self, x):
        x = self.quant_input(x)
        x = self.cell(x)
        return x
