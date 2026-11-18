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
"""

from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
from mindspore import ops
from mindspore.ops.operations import _quant_ops as Q
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer


class SimulatedQuant(Cell):
    r"""
    Ifmr activation calibration layer.
    Compute the scale and offset for activation calibration.

    Args:
        cell: The cube layer need to be quantized
        scale_init: The initializer of Parameter scale
        offset_init: The initializer of Parameter offset
        num_bins: The number of bins used by HistogramFixedWidth primitive
        min_percentile: The min_percentile of IFMR alogrithm
        max_percentile:The max_percentile of IFMR alogrithm
        search_start: The search start ratio of best max of IFMR
        search_end: The search end ratio of best max of IFMR
        search_step: the step of [search_start, search_end]
        with_offset: algorithm IFMR whether with offset
    return:
        The compute result of input cell
    """

    def __init__(
        self,
        cell,
        cell_name,
        scale_init,
        offset_init,
        num_bins=128,
        min_percentile=0.999999,
        max_percentile=0.999999,
        search_start=0.7,
        search_end=1.3,
        search_step=0.01,
        with_offset=True,
    ):
        super().__init__()
        self.compute_cell = cell
        self.compute_cell.weight.name = cell_name + '.weight'
        if self.compute_cell.has_bias:
            self.compute_cell.bias.name = cell_name + '.bias'
        self.compute_cell.original_weight.name = cell_name + '.original_weight'
        self.hist = P.HistogramFixedWidth(num_bins)
        self.reduce_min = P.ReduceMin(keep_dims=False)
        self.reduce_max = P.ReduceMax(keep_dims=False)
        self.concat = P.Concat(axis=0)
        self.cumsum = P.CumSum()
        self.assign = P.Assign()
        self.reshape = P.Reshape()
        self.dtype = P.DType()
        self.cast = P.Cast()

        self.scale_featuremap = Parameter(
            initializer(scale_init, scale_init.shape), name=(cell_name + '.scale_featuremap')
        )
        self.offset_featuremap = Parameter(
            initializer(offset_init, offset_init.shape), name=(cell_name + '.offset_featuremap')
        )

        self.ifmr = Q.IFMR(min_percentile, max_percentile, [search_start, search_end], search_step, with_offset)

    def construct(self, input_x):
        """define the construct of IFMR"""
        if self._phase == 'train':
            # calibration
            min_val = self.reduce_min(input_x)
            max_val = self.reduce_max(input_x)
            min_val = self.reshape(min_val, (1,))
            max_val = self.reshape(max_val, (1,))
            range_tensor = self.concat((min_val, max_val))
            hist = self.hist(input_x, range_tensor)
            cdf = self.cumsum(hist, -1)
            scale, offset = self.ifmr(input_x, min_val, max_val, cdf)
            self.scale_featuremap = self.assign(self.scale_featuremap, scale)
            self.offset_featuremap = self.assign(self.offset_featuremap, offset)
        elif self._phase == 'predict':
            # acumulate acc
            input_dtype = self.dtype(input_x)
            _round = ops.Round()
            input_x = _round(input_x / self.scale_featuremap + self.offset_featuremap).clip(-128, 127)
            input_x = (input_x - self.offset_featuremap) * self.scale_featuremap
            input_x = self.cast(input_x, input_dtype)
        return self.compute_cell(input_x)
