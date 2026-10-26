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

from msmodelslim.core.graph.adapter_filter import (
    adapter_entry_module_name,
    filter_adapter_configs,
    filter_adapter_configs_for_processor,
)
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig


def _adapter(subgraph_type, source, targets):
    return AdapterConfig(
        subgraph_type=subgraph_type,
        mapping=MappingConfig(source=source, targets=targets),
    )


def test_adapter_entry_module_name_honors_entry_mode():
    source_adapter = _adapter("norm-linear", "model.layers.0.input_layernorm", ["model.layers.0.q_proj"])
    target_adapter = _adapter("norm-linear", None, ["model.layers.1.q_proj"])

    assert adapter_entry_module_name(source_adapter) == "model.layers.0.input_layernorm"
    assert adapter_entry_module_name(source_adapter, entry="source_only") == "model.layers.0.input_layernorm"
    assert adapter_entry_module_name(target_adapter) == "model.layers.1.q_proj"
    assert adapter_entry_module_name(target_adapter, entry="source_only") is None


def test_filter_adapter_configs_applies_scope_type_include_and_exclude():
    configs = [
        _adapter("norm-linear", "model.layers.0.input_layernorm", ["model.layers.0.q_proj"]),
        _adapter("linear-linear", "model.layers.0.q_proj", ["model.layers.0.k_proj"]),
        _adapter("norm-linear", "model.layers.1.input_layernorm", ["model.layers.1.q_proj"]),
    ]

    selected = filter_adapter_configs(
        configs,
        scope="model.layers.0",
        enabled_subgraph_types=["norm-linear", "linear-linear"],
        include=["*norm"],
        exclude=["*linear-linear*"],
    )

    assert selected == [configs[0]]


def test_filter_adapter_configs_for_processor_reads_config_defaults():
    configs = [
        _adapter("norm-linear", "model.layers.0.input_layernorm", ["model.layers.0.q_proj"]),
        _adapter("linear-linear", "model.layers.0.q_proj", ["model.layers.0.o_proj"]),
    ]

    class ProcessorConfig:
        enable_subgraph_type = ["norm-linear"]
        include = ["model.layers.0.*"]
        exclude = []

    selected = filter_adapter_configs_for_processor(configs, ProcessorConfig(), scope="model.layers.0")

    assert selected == [configs[0]]
