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
from typing import Any, Dict, List, Optional

from msmodelslim.processor.analysis.methods_base import LayerAnalysisMethod


class UnaryAnalysisMethod(LayerAnalysisMethod):
    """Base class for unary analysis methods.

    Algorithm contract for new metrics:
    - required: ``compute_score`` (plus ``name`` / ``get_hook`` from the parent)
    - optional: ``pack_stats_for_distributed_merge`` + ``merge_distributed_stats``
      when ``score(merged_stats) != avg(score)`` (e.g. std/quantile/kurtosis)

    Session lifecycle (when to score / merge under DP) is owned by
    ``UnaryAnalysisProcessor``, not by this class.
    """

    @abstractmethod
    def compute_score(self, layer_data: Dict[str, Any]) -> float:
        """Compute analysis score for a layer given collected data."""
        raise NotImplementedError

    def pack_stats_for_distributed_merge(self, layer_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optionally pack picklable per-layer stats for DP merge.

        Return a dict so the processor merges stats across ranks before
        ``compute_score``. Return ``None`` (default) so the processor scores
        locally and averages scores under DP (e.g. ra_compress).
        """
        _ = layer_data

    def merge_distributed_stats(self, packed_stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reduce packed stats from ranks that saw this layer into ``compute_score`` input."""
        _ = packed_stats_list
        raise NotImplementedError(f"{self.name} returned packed stats but did not implement merge_distributed_stats.")
