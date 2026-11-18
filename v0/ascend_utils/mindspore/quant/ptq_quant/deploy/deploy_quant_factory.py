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

from ascend_utils.mindspore.quant.ptq_quant.deploy.deploy_utils import get_op_type
from ascend_utils.mindspore.quant.ptq_quant.deploy.conv2d import Conv2dDeployQuant
from ascend_utils.mindspore.quant.ptq_quant.deploy.conv1d import Conv1dDeployQuant
from ascend_utils.mindspore.quant.ptq_quant.deploy.linear import LinearDeployQuant
from ascend_utils.mindspore.quant.ptq_quant.deploy.dense import DenseDeployQuant


class DeployQuantFactory:
    @staticmethod
    def creat_deploy_quant_op(simulated_quant_cell):
        quant_op_type = get_op_type(simulated_quant_cell.compute_cell)
        if quant_op_type == "Conv2d":
            return Conv2dDeployQuant(simulated_quant_cell)
        elif quant_op_type == "Conv1d":
            return Conv1dDeployQuant(simulated_quant_cell)
        elif quant_op_type == "Dense":
            return DenseDeployQuant(simulated_quant_cell)
        else:
            return LinearDeployQuant(simulated_quant_cell)
