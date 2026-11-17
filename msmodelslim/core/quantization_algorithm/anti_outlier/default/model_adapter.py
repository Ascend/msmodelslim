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
# pylint: disable=consider-using-from-import

from typing import List
import torch.nn as nn  # noqa: F401
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.utils.logging import get_logger
from .model_structure_retriever import collect_shared_input_modules, is_layernorm, _is_moe_module
from msmodelslim.core.base.protocol import BatchProcessRequest


def get_adapter_config_for_subgraph(request: BatchProcessRequest) -> List[AdapterConfig]:
    if request.module is None:
        get_logger().warning(
            "DefaultModelAdapter.get_adapter_config_for_subgraph: request.module is not set, returning empty list. "
            "Ensure the Processor calls adapter.set_model(request.module) before invoking this method."
        )
        return []
    try:
        input_to_linear, output_to_layernorm = collect_shared_input_modules(request)
        subgraphs_list: List[AdapterConfig] = []
        for input_tensor, consumer_modules in input_to_linear.items():
            producer = output_to_layernorm.get(input_tensor)
            if producer is None:
                continue
            moe_names = [getattr(m, "name", "") for m in consumer_modules if _is_moe_module(getattr(m, "name", ""))]
            if moe_names:
                get_logger().warning(
                    "MoE structure is not suitable for the default outlier suppression processing and will be skipped (e.g. %s).",
                    moe_names[0],
                )
                continue
            prefix = f"{request.name}." if request.name else ""
            source_name = f"{prefix}{producer.name}"
            target_names = [f"{prefix}{m.name}" for m in consumer_modules]

            if is_layernorm(producer):
                subgraphs_list.append(
                    AdapterConfig(
                        subgraph_type="norm-linear",
                        mapping=MappingConfig(source=source_name, targets=target_names),
                    )
                )
        return subgraphs_list
    except Exception as e:
        get_logger().warning(
            "DefaultModelAdapter auto-detect subgraph structure failed: %s, returning empty config.", e
        )
        return []
