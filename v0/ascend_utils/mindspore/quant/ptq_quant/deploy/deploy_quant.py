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
# pylint: disable=ungrouped-imports

import logging

from mindspore.nn import Cell
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops

from ascend_utils.mindspore.quant.ptq_quant.deploy.deploy_utils import compute_weight_bias
from ascend_utils.mindspore.quant.ptq_quant.deploy.deploy_utils import compute_fused_deq_scale


class DeployQuant(Cell):
    def __init__(self, simulated_quant_cell):
        super().__init__()
        scale_input = float(simulated_quant_cell.scale_featuremap.data.asnumpy()[0])
        offset_input = int(simulated_quant_cell.offset_featuremap.data.asnumpy()[0])

        tmp_s_w = getattr(simulated_quant_cell.name_cells()['compute_cell'], 'scale_weight')
        scale_weight = np.squeeze(tmp_s_w.asnumpy())
        tmp_o_w = getattr(simulated_quant_cell.name_cells()['compute_cell'], 'offset_weight')
        offset_weight = np.squeeze(tmp_o_w.asnumpy())

        weight_tensor, bias_tensor = compute_weight_bias(
            simulated_quant_cell, scale_input, offset_input, scale_weight, offset_weight
        )
        deq_scale_tmp = np.squeeze(scale_weight) * scale_input
        self.deq_scale = Tensor(deq_scale_tmp, mstype.float16)

        fused_deq_scale = compute_fused_deq_scale(scale_input, scale_weight, offset_weight)
        self.weight = Parameter(weight_tensor, name='weight')
        self.weight_offset = Parameter(np.zeros(shape=weight_tensor.shape, dtype=np.int8), name='weight_offset')
        self.fused_deq_scale = fused_deq_scale

        self.has_bias = bias_tensor is not None
        if self.has_bias:
            self.bias = Parameter(bias_tensor, name='bias')
            self.bias_add = simulated_quant_cell.compute_cell.bias_add
        else:
            self.bias = None

        try:
            self.quant = _inner_ops.Quant(float(1 / scale_input), float(offset_input))
        except ZeroDivisionError as ex:
            logging.error('scale_input cannot be zero. %s', str(ex))
            raise ex
        self.dequant = _inner_ops.Dequant()

        self.cast = P.Cast()
        self.sub = P.Sub()
        self.dtype = P.DType()
        self.shape_op = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, input_x):
        raise ValueError("Please implement the construct in sub class.")
