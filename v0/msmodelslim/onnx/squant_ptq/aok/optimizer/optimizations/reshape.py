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
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations.single_transform_optimization import \
    SingleTransformOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations import reshape


class DoubleReshapeOptimization(SingleTransformOptimization):

    def __init__(self):
        super(DoubleReshapeOptimization, self).__init__(reshape.DoubleReshapeTransformation)

    def get_simple_name(self) -> str:
        return 'DoubleReshape'


class SimplifyShapeOptimization(SingleTransformOptimization):

    def __init__(self):
        super(SimplifyShapeOptimization, self).__init__(reshape.SimplifyShapeTransformation)

    def get_simple_name(self) -> str:
        return 'SimplifyShape'
