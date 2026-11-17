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

THIS SOFTWARE IS PROVIDED ON an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from abc import abstractmethod
from typing import Any, List

from msmodelslim.core.quantization_algorithm.analysis.methods_base import LayerAnalysisMethod


class LayerWiseAnalysisMethod(LayerAnalysisMethod):
    """Model-wise analysis method base.

    Note: The package name keeps ``binary_operator_layer_wise`` for backward
    compatibility, but the implementation is model-wise: score is computed from
    two forward-path outputs (float reference vs quantized candidate).
    """

    @abstractmethod
    def compute_score(
        self,
        ref_outputs: List[Any],
        cand_outputs: List[Any],
    ) -> float:
        """Compute a scalar score for one block given two output lists."""
        raise NotImplementedError

    def get_hook(self) -> Any:
        """Compatibility with LayerAnalysisMethod; model-wise does not use hooks."""
        return None
