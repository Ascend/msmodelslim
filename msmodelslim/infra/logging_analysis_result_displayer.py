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
from itertools import groupby
from typing import Any, Dict, List, Optional

from msmodelslim.app.analysis.result_displayer_infra import AnalysisResultDisplayerInfra
from msmodelslim.core.analysis_service import AnalysisResult, AnalysisScope
from msmodelslim.utils.logging import get_logger, clean_output


def _yaml_disable_entry_name(layer_name: str, scope: Optional[AnalysisScope]) -> str:
    """layer scope 下整块回退需匹配子模块，YAML 中为 ``block.*``；其它 scope 保持原名。"""
    if scope == AnalysisScope.LAYER:
        if layer_name.endswith(".*"):
            return layer_name
        return f"{layer_name}.*"
    return layer_name


class LoggingAnalysisResultDisplayer(AnalysisResultDisplayerInfra):
    def get_sorted_layers(
        self, result: AnalysisResult, reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """按分数排序返回层列表。"""
        return sorted(result.layer_scores, key=lambda x: x['score'], reverse=reverse)

    def display_result(
        self,
        result: AnalysisResult,
        topk: int,
        scope: Optional[AnalysisScope] = None,
    ) -> None:
        """打印/导出层分析结果（分数 + 量化用 YAML）。"""
        sorted_layers = self.get_sorted_layers(result, reverse=True)
        layer_groups = [list(g) for _, g in groupby(sorted_layers, key=lambda x: x['score'])]

        get_logger().info("=== Layer Analysis Results (%s method) ===", result.method)
        get_logger().info("Patterns analyzed: %s", result.patterns)
        get_logger().info("Total layers analyzed: %d", len(result.layer_scores))
        get_logger().info("Layer Sensitivity Scores (higher score = more sensitive to quantization):")
        get_logger().info("-" * 80)

        if 0 <= topk <= len(layer_groups):
            selected_groups = layer_groups[:topk]
        else:
            selected_groups = layer_groups

        display_layers = []
        for group in selected_groups:
            display_layers.extend(group)

        for i, layer_info in enumerate(display_layers, 1):
            # 统一使用科学计数法，4 位有效数字
            get_logger().info("%3d. %-50s | Score: %12.4e", i, layer_info['name'], layer_info['score'])

        get_logger().info("-" * 80)
        get_logger().info("Top %d most sensitive layers selected for disable_names", len(display_layers))
        get_logger().info("")
        get_logger().info("=== YAML Format for quantization ===")
        get_logger().info("")

        with clean_output():
            get_logger().info("top %d:", len(display_layers))
            for layer_info in display_layers:
                yaml_name = _yaml_disable_entry_name(layer_info["name"], scope)
                get_logger().info("  - '%s'", yaml_name)

        get_logger().info("")
        get_logger().info("=== End of YAML Format ===")