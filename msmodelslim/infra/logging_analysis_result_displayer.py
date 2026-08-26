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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import os
import re
from itertools import groupby
from typing import Any, Dict, List, Optional

import torch

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


def _save_yaml(yaml_content: str, save_path: str, model_type: Optional[str], method: Optional[str]) -> str:
    """保存 YAML 文件，返回实际输出路径。"""
    from ascend_utils.common.security import SafeWriteUmask, get_valid_write_path

    if not save_path.endswith(('.yaml', '.yml')):
        mt = model_type or 'model'
        m = method or 'analysis'
        safe_mt = mt.lower().replace('/', '-').replace('\\', '-').replace(' ', '-')
        save_path = os.path.join(save_path, f'{safe_mt}-{m}.yaml')
    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)
    output_path = get_valid_write_path(save_path, extensions=".yaml")
    with SafeWriteUmask():
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
    return output_path


def _save_head_pt(head_dict: Dict, save_path: str) -> str:
    """保存 head.pt 文件，返回实际输出路径。"""
    from ascend_utils.common.security import SafeWriteUmask, get_valid_write_path

    if not save_path.endswith('.pt'):
        save_path = os.path.join(save_path, 'head.pt')
    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)
    output_path = get_valid_write_path(save_path, extensions=".pt")
    with SafeWriteUmask():
        torch.save(head_dict, output_path)
    return output_path


class StandardAnalysisResultDisplayer(AnalysisResultDisplayerInfra):
    """标准分析结果展示（std / quantile / kurtosis / mse 等）。

    按分数排序打印层列表，输出量化 YAML。
    """

    def get_sorted_layers(self, result: AnalysisResult, reverse: bool = True) -> List[Dict[str, Any]]:
        """按分数排序返回层列表。"""
        return sorted(result.layer_scores, key=lambda x: x['score'], reverse=reverse)

    def display_result(
        self,
        result: AnalysisResult,
        topk: int,
        scope: Optional[AnalysisScope] = None,
        save_path: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> None:
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
            get_logger().info("%3d. %-50s | Score: %12.4e", i, layer_info['name'], layer_info['score'])

        get_logger().info("-" * 80)
        get_logger().info("Top %d most sensitive layers selected for disable_names", len(display_layers))

        yaml_lines: List[str] = []
        yaml_lines.append("top {}:".format(len(display_layers)))
        for layer_info in display_layers:
            yaml_name = _yaml_disable_entry_name(layer_info["name"], scope)
            yaml_lines.append("  - '{}'".format(yaml_name))
        yaml_content = "\n".join(yaml_lines)

        if save_path:
            output_path = _save_yaml(yaml_content, save_path, model_type, result.method)
            get_logger().info("YAML saved to: %s", output_path)
        else:
            get_logger().info("")
            get_logger().info("=== YAML Format for quantization ===")
            get_logger().info("")
            with clean_output():
                get_logger().info(yaml_content)
            get_logger().info("")
            get_logger().info("=== End of YAML Format ===")


class RaCompressAnalysisResultDisplayer(AnalysisResultDisplayerInfra):
    """ra_compress 分析结果展示：每层分数 + induction/echo head 筛选。"""

    def display_result(
        self,
        result: AnalysisResult,
        topk: int,
        scope: Optional[AnalysisScope] = None,
        save_path: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> None:
        def _extract_layer_idx(name: str) -> int:
            m = re.search(r'layers\.(\d+)\.', name)
            return int(m.group(1)) if m else -1

        prefix_map: Dict[int, List[int]] = {}
        copying_map: Dict[int, List[int]] = {}
        for entry in result.layer_scores:
            name = entry['name']
            layer_idx = _extract_layer_idx(name)
            ind_heads = entry.get('induction_heads', [])
            echo_heads = entry.get('echo_heads', [])
            if ind_heads:
                prefix_map[layer_idx] = ind_heads
            if echo_heads:
                copying_map[layer_idx] = echo_heads
        head_dict = {
            'prefix_matching': prefix_map,
            'copying': copying_map,
        }

        get_logger().info("=" * 80)
        get_logger().info("=== RA Compress Analysis Results ===")
        get_logger().info("Method: %s", result.method)
        get_logger().info("-" * 80)

        get_logger().info("=== Induction Heads (prefix matching) ===")
        get_logger().info("Selected %d layers with induction heads:", len(prefix_map))
        for layer_idx in sorted(prefix_map.keys()):
            heads = prefix_map[layer_idx]
            get_logger().info("  Layer %3d: KV heads %s", layer_idx, heads)

        get_logger().info("-" * 80)

        get_logger().info("=== Echo Heads (copying matching) ===")
        get_logger().info("Selected %d layers with echo heads:", len(copying_map))
        for layer_idx in sorted(copying_map.keys()):
            heads = copying_map[layer_idx]
            get_logger().info("  Layer %3d: KV heads %s", layer_idx, heads)

        get_logger().info("-" * 80)

        if save_path:
            output_path = _save_head_pt(head_dict, save_path)
            get_logger().info("RA compress heads saved to: %s", output_path)
        else:
            get_logger().info("No --save_path specified, results printed to console only.")

        get_logger().info("=" * 80)


class AnalysisResultDisplayerFactory:
    """按 metrics 创建对应的分析结果展示器。"""

    _STANDARD_METRICS = {'std', 'quantile', 'kurtosis', 'mse', 'mse_layer_wise', 'mse_model_wise'}
    _RA_COMPRESS_METRICS = {'ra_compress'}

    @classmethod
    def create(cls, metrics: str) -> AnalysisResultDisplayerInfra:
        if metrics in cls._RA_COMPRESS_METRICS:
            return RaCompressAnalysisResultDisplayer()
        if metrics in cls._STANDARD_METRICS:
            return StandardAnalysisResultDisplayer()
        return StandardAnalysisResultDisplayer()


# 向后兼容别名
LoggingAnalysisResultDisplayer = StandardAnalysisResultDisplayer
