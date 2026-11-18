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
from onnx import GraphProto
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.onnx.squant_ptq.aok.utils import utilities
from msmodelslim.onnx.squant_ptq.aok.optimizer.transformations.abstract_transformation import AbstractTransformation


class AbstractOptimization(ABC):

    __ACCURACY_CHECK_THRESHOLD: float = 1e-4

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_simple_name(self) -> str:
        pass

    @abstractmethod
    def _setup_transformations(self, op_version: int) -> [AbstractTransformation]:
        pass

    def apply(self,
              model_graph: GraphProto,
              op_version: int,
              debug: bool = False) -> bool:
        if debug:
            msmodelslim_logger.debug(f'<<< {self.get_simple_name()}')
        transformations = self._setup_transformations(op_version=op_version)
        is_applied = False
        for t in transformations:
            found_count = t.apply(model_graph, debug=debug)
            if debug:
                msmodelslim_logger.debug(f'{t.get_simple_name()} => {found_count} occurrences')
            if found_count > 0:
                is_applied = True
                utilities.clean_constant_nodes(model_graph)
                utilities.clean_initializer(model_graph)
        if debug:
            msmodelslim_logger.debug(f"{self.get_simple_name()} is{'' if is_applied else ' not'} applicable >>>")
        return is_applied
