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

from typing import List, Optional

from msmodelslim.app.eval.result_displayer_infra import InferenceResultDisplayerInfra
from msmodelslim.core.infer_engine.interface import InferenceResult
from msmodelslim.utils.logging import get_logger


class LoggingInferenceResultDisplayer(InferenceResultDisplayerInfra):
    """Log inference results."""

    def display_result(
        self,
        result: InferenceResult,
        device_indices: Optional[List[int]] = None,
    ) -> None:
        world_size = len(device_indices) if device_indices and len(device_indices) > 1 else 1
        distributed = world_size > 1
        get_logger().info(
            "Fake-quant inference results (%d sample(s)%s):",
            len(result.generated_texts),
            f", {world_size} device(s)" if distributed else "",
        )
        for global_i, text in enumerate(result.generated_texts):
            token_ids = result.generated_token_ids[global_i] if global_i < len(result.generated_token_ids) else []
            if distributed and device_indices is not None:
                rank = global_i % world_size
                device = device_indices[rank]
                get_logger().info(
                    "Fake-quant sample[%d] (rank=%d, device=%d) token_ids=%s generated_text=%r",
                    global_i,
                    rank,
                    device,
                    token_ids,
                    text,
                )
            else:
                get_logger().info(
                    "Fake-quant sample[%d] token_ids=%s generated_text=%r",
                    global_i,
                    token_ids,
                    text,
                )
