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

Normalize handle_dataset outputs for Calibrator / AntiOutlier.

Calibrator.run supports dict via model(**data), but AntiOutlier.init_dag
indexes calib_data[0][0] and therefore needs positional-arg lists for VLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def vlm_dict_to_calibrator_args(processed: Dict[str, Any]) -> List[Any]:
    """Map processor / V1-style dicts to Calibrator positional-arg lists."""
    return [
        processed.get("input_ids"),
        processed.get("attention_mask"),
        None,
        None,
        None,
        None,
        processed.get("pixel_values"),
        None,
        processed.get("image_grid_thw"),
        None,
        None,
        processed.get("logits_to_keep", 0),
    ]


def normalize_calib_data_for_calibrator(data: Sequence[Any]) -> List[Any]:
    """Convert dict batches to positional lists; pass list batches through."""
    if not data:
        return list(data)
    first = data[0]
    if isinstance(first, dict):
        return [vlm_dict_to_calibrator_args(item) for item in data]
    return list(data)
