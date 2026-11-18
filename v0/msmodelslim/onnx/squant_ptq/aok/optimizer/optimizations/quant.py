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
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations import quant


class ChangeGAPCONVOptimization(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ChangeGAPCONVOptimization, self).__init__(quant.ChangeGAPCONVTransformation)

    def get_simple_name(self) -> str:
        return 'ChangeGAPConv'


class ReplaceAscendQuantOptimizationV1(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceAscendQuantOptimizationV1, self).__init__(quant.ChangeAscendQuantTransformation)

    def get_simple_name(self) -> str:
        return 'ReplaceAscendQuantV1'


class ReplaceAscendQuantOptimizationV2(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceAscendQuantOptimizationV2, self).__init__(quant.ChangeAscendQuantTransformationV2)

    def get_simple_name(self) -> str:
        return 'ReplaceAscendQuantV2'


class ReplaceConcatQuantOptimizationV1(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceConcatQuantOptimizationV1, self).__init__(quant.ReplaceConcatQuantTransformationV1)

    def get_simple_name(self) -> str:
        return 'ReplaceConcatQuantV1'


class ReplaceConcatQuantOptimizationV2(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceConcatQuantOptimizationV2, self).__init__(quant.ReplaceConcatQuantTransformationV2)

    def get_simple_name(self) -> str:
        return 'ReplaceConcatQuantV2'


class ReplaceConcatQuantOptimizationV3(AbstractOptimization):

    def get_simple_name(self) -> str:
        return 'ReplaceConcatQuantV3'

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        return [
            quant.UpdateAscendQuantTransformation(op_version),
            quant.ReplaceConcatQuantTransformationV3(op_version),
            quant.ChangeAscendQuantTransformation(op_version),
        ]


class ReplaceConcatQuantOptimizationV4(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceConcatQuantOptimizationV4, self).__init__(quant.ReplaceConcatQuantTransformationV4)

    def get_simple_name(self) -> str:
        return 'ReplaceConcatQuantV4'


class ReplaceConcatQuantOptimizationV5(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceConcatQuantOptimizationV5, self).__init__(quant.ReplaceConcatQuantTransformationV5)

    def get_simple_name(self) -> str:
        return 'ReplaceConcatQuantV5'


class ReplaceConcatQuantOptimizationV6(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceConcatQuantOptimizationV6, self).__init__(quant.ReplaceConcatQuantTransformationV6)

    def get_simple_name(self) -> str:
        return 'ReplaceConcatQuantV6'


class ReplaceConcatQuantOptimizationV7(AbstractOptimization):

    def get_simple_name(self) -> str:
        return 'ReplaceConcatQuantV7'

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        return [
            quant.UpdateAscendQuantTransformation(op_version),
            quant.ReplaceConcatQuantTransformationV7(op_version),
            quant.ChangeAscendQuantTransformation(op_version),
        ]


class ReplaceConcatQuantOptimizationV8(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceConcatQuantOptimizationV8, self).__init__(quant.ReplaceConcatQuantTransformationV8)

    def get_simple_name(self) -> str:
        return 'ReplaceConcatQuantV8'


class ReplaceConcatQuantOptimizationV9(AbstractOptimization):

    def get_simple_name(self) -> str:
        return 'ReplaceConcatQuantV9'

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        return [
            quant.UpdateAscendQuantTransformation(op_version),
            quant.ReplaceConcatQuantTransformationV9(op_version),
            quant.ReplaceConcatQuantTransformationV10(op_version),
            quant.ChangeAscendQuantTransformation(op_version),
        ]


class ReplaceResizeQuantOptimization(SingleTransformOptimization):

    def __init__(self) -> None:
        super(ReplaceResizeQuantOptimization, self).__init__(quant.ReplaceResizeQuantTransformation)

    def get_simple_name(self) -> str:
        return 'ReplaceResizeQuant'


class SimplifyShapeOptimizationV2(SingleTransformOptimization):

    def __init__(self) -> None:
        super(SimplifyShapeOptimizationV2, self).__init__(quant.SimplifyShapeV2Transformation)

    def get_simple_name(self) -> str:
        return 'SimplifyShapeV2'
