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
import sys
import inspect

from abc import ABC, abstractmethod
from importlib import import_module
from inspect import isabstract, isclass, ismodule
from pathlib import Path
from pkgutil import iter_modules
from types import ModuleType
from typing import Type, List, Tuple, TypeVar

from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations import (
    AbstractOptimization,
    gemm_matmul,
    conv,
    gelu,
    misc,
    quant,
    reshape,
)
# G.IMP.02 is deliberately violated because we don't want to enumerate all the many optimizations here.
# We accept that names of optimization classes must not be preceded with '_'


class AbstractArchitecture(ABC):
    # Below are mask values of the optimizations filters
    # 'opt_filter_mask' argument, which is passed to the 'get_optimizations' method, defines a mode, which is
    # a bitwise disjunction of the mask values that correspond to the optimizations to be taken for the architecture.
    OPT_FILTER_MASK_NONE: int = 0
    OPT_FILTER_MASK_FP: int = 1
    OPT_FILTER_MASK_QUANT: int = 2
    OPT_FILTER_MASK_ALL: int = OPT_FILTER_MASK_FP | OPT_FILTER_MASK_QUANT

    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_optimizations(self, opt_filter_mask: int, logger) -> [AbstractOptimization]:
        pass


_PACKAGE_DIR = str(Path(__file__).resolve().parent)
for (_, _, _) in iter_modules([_PACKAGE_DIR]):
    module = import_module(f'{__name__}')
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isclass(attribute) and issubclass(attribute, AbstractArchitecture):
            globals()[attribute_name] = attribute

supported_architectures = [
    'bert', 'roberta', 'efficientnet', 'mobilenetv2',
    'mobilenetv3', 'shufflenetv2', 'swin',
    'seresnet', 'densenet', 'u2net',
    'yolov5', 'yolov5s1', 'yolov7', 'deeplab'
]


TOpt = TypeVar('TOpt', bound=AbstractOptimization)


class AbstractOptMaskArchitecture(AbstractArchitecture, ABC):

    def get_optimizations(self, opt_filter_mask: int, logger) -> [AbstractOptimization]:
        opt_mask = self._get_opt_mask()
        return list(
            opt() for opt, mask in opt_mask
            if mask & opt_filter_mask != AbstractArchitecture.OPT_FILTER_MASK_NONE
        )

    @abstractmethod
    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        pass


class DefaultArchitecture(AbstractArchitecture):

    @staticmethod
    def _get_optimizations_in_module(mdl: ModuleType) -> [Type[TOpt]]:
        optimizations = []
        for name in dir(mdl):
            obj = getattr(mdl, name)
            if ismodule(obj) and 'msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations' in obj.__name__:
                optimizations.extend(DefaultArchitecture._get_optimizations_in_module(obj))
            elif isclass(obj) and not isabstract(obj) and issubclass(obj, AbstractOptimization):
                optimizations.append(obj)
        return list(set(optimizations))

    def get_name(self) -> str:
        return 'default'

    def get_optimizations(self, opt_filter_mask: int, logger) -> [AbstractOptimization]:
        mdl = sys.modules.get(__name__, None)
        opt_types = DefaultArchitecture._get_optimizations_in_module(mdl)
        return [t() for t in opt_types]


class DummyArchitecture(AbstractArchitecture):

    def get_name(self) -> str:
        return 'dummy'

    def get_optimizations(self, opt_filter_mask: int, logger) -> [AbstractOptimization]:
        return []


class _AbstractBertBasedArchitecture(AbstractOptMaskArchitecture, ABC):

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (gelu.GeluTanh2SigmoidOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (gelu.GeluErf2TanhOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (gemm_matmul.CombineMatmulOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (gemm_matmul.FuseDivMatmulOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (gemm_matmul.Matmul2GemmOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (misc.LayerNormOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (misc.ReplaceSoftmaxOptimizationV1, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (misc.ReplaceSoftmaxOptimizationV2, AbstractArchitecture.OPT_FILTER_MASK_FP),
        ]


class BERTArchitecture(_AbstractBertBasedArchitecture):

    def get_name(self) -> str:
        return 'bert'


class RoBERTaArchitecture(_AbstractBertBasedArchitecture):

    def get_name(self) -> str:
        return 'roberta'


class EfficientNetArchitecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'efficientnet'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (quant.ChangeGAPCONVOptimization, AbstractArchitecture.OPT_FILTER_MASK_QUANT),
        ]


class MobileNetV2Architecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'mobilenetv2'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (misc.ReplaceRelu6Optimization, AbstractArchitecture.OPT_FILTER_MASK_FP)
        ]


class MobileNetV3Architecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'mobilenetv3'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (conv.FuseBatchNormOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (gelu.FastClipOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
        ]


class ShuffleNetV2Architecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'shufflenetv2'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (conv.ReplaceReshapeTransposeOptimizationV1, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (conv.ReplaceReshapeTransposeOptimizationV2, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (conv.ReplaceReshapeTransposeOptimizationV3, AbstractArchitecture.OPT_FILTER_MASK_FP),
        ]


class SWINArchitecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'swin'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (conv.PatchMerging2ConvOptimizationV0, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (conv.PatchMerging2ConvOptimizationV1, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (conv.PatchMerging2ConvOptimizationV2, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (conv.PatchMerging2ConvOptimizationV3, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (reshape.DoubleReshapeOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (gelu.GeluErf2TanhOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (misc.ReplaceSoftmaxOptimizationV1, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (misc.ReplaceSoftmaxOptimizationV2, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (gemm_matmul.Matmul2GemmOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
        ]


class SeResnetArchitecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'seresnet'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (quant.ChangeGAPCONVOptimization, AbstractArchitecture.OPT_FILTER_MASK_QUANT),
        ]


class DenseNetArchitecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'densenet'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (misc.ReplaceReluOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
        ]


class U2NetArchitecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'u2net'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (quant.ReplaceAscendQuantOptimizationV2, AbstractArchitecture.OPT_FILTER_MASK_QUANT),
            (quant.ReplaceAscendQuantOptimizationV1, AbstractArchitecture.OPT_FILTER_MASK_QUANT),
            (quant.SimplifyShapeOptimizationV2, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (misc.ChangeResizeOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (conv.DeleteConcatOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (misc.ReplaceSigmoidOptimizationV1, AbstractArchitecture.OPT_FILTER_MASK_FP),
        ]


class YoloV5Architecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'yolov5'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (quant.ReplaceAscendQuantOptimizationV1, AbstractArchitecture.OPT_FILTER_MASK_QUANT),
            (quant.ReplaceConcatQuantOptimizationV1, AbstractArchitecture.OPT_FILTER_MASK_QUANT),
            (quant.ReplaceConcatQuantOptimizationV5, AbstractArchitecture.OPT_FILTER_MASK_QUANT),
            (quant.ReplaceConcatQuantOptimizationV8, AbstractArchitecture.OPT_FILTER_MASK_QUANT),
            (conv.DoubleFuseBatchNormOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (gelu.ReplaceLeakyReluOptimization, AbstractArchitecture.OPT_FILTER_MASK_FP),
            (misc.ReplaceMaxPoolBlockOptimizationV2, AbstractArchitecture.OPT_FILTER_MASK_FP),
        ]


class YoloV5s1Architecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'yolov5s1'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (quant.ReplaceConcatQuantOptimizationV5, AbstractArchitecture.OPT_FILTER_MASK_QUANT),
        ]


class YoloV7Architecture(AbstractOptMaskArchitecture):

    def get_name(self) -> str:
        return 'yolov7'

    def _get_opt_mask(self) -> List[Tuple[Type[TOpt], int]]:
        return [
            (quant.ReplaceAscendQuantOptimizationV2, AbstractArchitecture.OPT_FILTER_MASK_QUANT),
            (misc.ReplaceMaxPoolBlockOptimizationV2, AbstractArchitecture.OPT_FILTER_MASK_FP),
        ]
