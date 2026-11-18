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

from ascend_utils.mindspore.quant.ptq_quant.deploy.deploy_quant import DeployQuant


class Conv2dDeployQuant(DeployQuant):
    """
    convert Conv2d to quantize op, that can be exported to air model.
    """

    def __init__(self, simulated_quant_cell):
        super().__init__(simulated_quant_cell)
        self.op_core = simulated_quant_cell.compute_cell.conv2d

    def construct(self, input_x):
        input_dtype = self.dtype(input_x)
        quant_x = self.quant(input_x)
        if self.has_bias:
            weight = self.sub(self.weight, self.weight_offset)
            quant_y = self.op_core(quant_x, weight)
            quant_y = self.bias_add(quant_y, self.bias)
        else:
            quant_y = self.op_core(quant_x, self.weight)
        dequant_y = self.dequant(quant_y, self.fused_deq_scale)
        output = self.cast(dequant_y, input_dtype)
        return output
