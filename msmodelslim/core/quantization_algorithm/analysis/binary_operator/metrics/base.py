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

from abc import abstractmethod
from typing import Any, Dict

from msmodelslim.core.quantization_algorithm.analysis.methods_base import LayerAnalysisMethod


class BinaryAnalysisMethod(LayerAnalysisMethod):
    """Base class for binary analysis methods."""

    @abstractmethod
    def compute_score(self, layer_data_before: Dict[str, Any], layer_data_after: Dict[str, Any]) -> float:
        """Compute analysis score for a layer given collected data before and after."""
        ...
