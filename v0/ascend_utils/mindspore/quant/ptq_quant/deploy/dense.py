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

from mindspore.ops import operations as P
from mindspore import ops

from ascend_utils.mindspore.quant.ptq_quant.deploy.deploy_quant import DeployQuant


class DenseDeployQuant(DeployQuant):
    """
    convert Dense to quantize op, that can be exported to air model.
    """

    def __init__(self, simulated_quant_cell):
        super().__init__(simulated_quant_cell)
        self.activation = simulated_quant_cell.compute_cell.activation
        self.activation_flag = simulated_quant_cell.compute_cell.activation_flag
        self.op_core = P.MatMul()

    def construct(self, input_x):
        input_dtype = self.dtype(input_x)
        quant_x = self.quant(input_x)
        x_shape = self.shape_op(input_x)
        if len(x_shape) != 2:
            quant_x = self.reshape(quant_x, (-1, x_shape[-1]))
        quant_y = self.op_core(quant_x, self.weight)
        if self.has_bias:
            if quant_y.dtype != self.bias.dtype:
                cast = ops.Cast()
                quant_y = cast(quant_y, self.bias.dtype)
            quant_y = self.bias_add(quant_y, self.bias)
        dequant_y = self.dequant(quant_y, self.fused_deq_scale)
        output = self.cast(dequant_y, input_dtype)
        if self.activation_flag:
            output = self.activation(output)
        if len(x_shape) != 2:
            output = self.reshape(output, x_shape[:-1] + (-1,))
        return output
