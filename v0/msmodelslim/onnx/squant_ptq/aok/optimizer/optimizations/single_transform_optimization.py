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
from abc import ABC, abstractmethod
from inspect import isabstract
from typing import Generic, TypeVar, Type, Union
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations.abstract_optimization import AbstractOptimization
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations.abstract_transformation import AbstractTransformation

T = TypeVar('T', bound=AbstractTransformation)


class SingleTransformOptimization(AbstractOptimization, Generic[T], ABC):

    @abstractmethod
    def __init__(self, t: Union[T, Type[T]]):
        super(AbstractOptimization, self).__init__()
        if not isinstance(t, AbstractTransformation) and not issubclass(t, AbstractTransformation):
            raise TypeError(f't must be an instance or subclass of AbstractTransformation but got {type(t)}')
        self.__transform = t

    def get_simple_name(self):
        return self.__class__.__name__ if not isabstract(self.__class__) \
            else self.__transform.get_simple_name() if isinstance(self.__transform, AbstractTransformation) \
            else self.__transform.__name__ if issubclass(self.__transform, AbstractTransformation) \
            else None

    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        t = self.__transform if isinstance(self.__transform, AbstractTransformation) \
            else self.__transform(op_version) if issubclass(self.__transform, AbstractTransformation) \
            else None
        return [t]
