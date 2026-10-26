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

import pytest

import msmodelslim.core.graph.fast_ops_grapher.formatters.dot_formatter  # noqa: F401
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import ComputationGraph
from msmodelslim.core.graph.fast_ops_grapher.formatters import formatter_mng


def test_default_dot_formatter_is_listed_and_renders_empty_graph():
    assert "dot" in formatter_mng.list_formatters()
    assert formatter_mng.format_graph("dot", ComputationGraph()) == "digraph fx_graph {\n    rankdir=TB;\n}"


def test_unknown_formatter_reports_available_names():
    with pytest.raises(KeyError, match="Formatter 'unknown' not found"):
        formatter_mng.format_graph("unknown", ComputationGraph())


def test_register_formatter_enables_exact_format_lookup():
    formatter_mng.register_formatter("test_exact")(lambda graph: f"graph:{graph.node_count}")

    graph = ComputationGraph()
    assert "test_exact" in formatter_mng.list_formatters()
    assert formatter_mng.format_graph("test_exact", graph) == "graph:0"


def test_register_formatter_overwrites_existing_name():
    formatter_mng.register_formatter("test_exact_overwrite")(lambda graph: f"first:{graph.node_count}")
    formatter_mng.register_formatter("test_exact_overwrite")(lambda graph: f"second:{graph.node_count}")

    assert formatter_mng.format_graph("test_exact_overwrite", ComputationGraph()) == "second:0"


def test_computation_graph_format_uses_registered_preset():
    formatter_mng.register_formatter("test_graph_summary")(lambda graph: f"nodes={graph.node_count}")

    assert ComputationGraph().format("test_graph_summary") == "nodes=0"


def test_computation_graph_format_raises_for_missing_preset():
    with pytest.raises(KeyError, match="Formatter 'test_missing' not found"):
        ComputationGraph().format("test_missing")
