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
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations import AbstractOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations.abstract_transformation import AbstractTransformation
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations.single_transform_optimization import \
    SingleTransformOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations import gelu


class FastClipOptimization(AbstractOptimization):

    def get_simple_name(self) -> str:
        return 'FastClip'

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        return [
            gelu.ChangeClipTransformationV1(op_version=op_version),
            gelu.ChangeClipTransformationV2(op_version=op_version),
            gelu.ChangeClipTransformationV3(op_version=op_version)
        ]


class GeluErf2SigmoidOptimization(SingleTransformOptimization):

    def __init__(self):
        super(GeluErf2SigmoidOptimization, self).__init__(gelu.GeluErf2SigmoidTransformation)

    def get_simple_name(self) -> str:
        return 'GeluErf2Sigmoid'


class GeluTanh2SigmoidOptimization(SingleTransformOptimization):

    def __init__(self):
        super(GeluTanh2SigmoidOptimization, self).__init__(gelu.GeluTanh2SigmoidTransformation)

    def get_simple_name(self) -> str:
        return 'GeluTanh2Sigmoid'


class GeluErf2TanhOptimization(SingleTransformOptimization):

    def __init__(self):
        super(GeluErf2TanhOptimization, self).__init__(gelu.GeluErf2TanhTransformation)

    def get_simple_name(self) -> str:
        return 'GeluErf2Tanh'


class GeluErf2FastGeluOptimization(SingleTransformOptimization):

    def __init__(self):
        super(GeluErf2FastGeluOptimization, self).__init__(gelu.GeluErf2FastGeluTransformation)

    def get_simple_name(self) -> str:
        return 'GeluErf2FastGelu'


class ReplaceLeakyReluOptimization(SingleTransformOptimization):

    def __init__(self):
        super(ReplaceLeakyReluOptimization, self).__init__(gelu.ReplaceLeakyReluTransformation)

    def get_simple_name(self) -> str:
        return 'ReplaceLeakyRelu'
