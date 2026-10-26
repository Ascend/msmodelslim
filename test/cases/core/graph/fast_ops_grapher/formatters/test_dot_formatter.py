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

# pylint: disable=protected-access,import-outside-toplevel,no-name-in-module

from test.cases.core.graph.fast_ops_grapher.conftest import _make_operator_record
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import ComputationGraph, GraphNode
from msmodelslim.core.graph.fast_ops_grapher.formatters.dot_formatter import dot_formatter


class TestDotFormatter:
    def test_output_is_dot_document_for_traced_graph(self, simple_model, sample_input):
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import trace_to_graph
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import trace_module

        graph = trace_to_graph(trace_module(simple_model, args=(sample_input,)))
        result = dot_formatter(graph)

        assert result.startswith("digraph fx_graph {")
        assert "rankdir=TB;" in result
        assert result.rstrip().endswith("}")

    def test_output_contains_node_labels_and_edge_arrows(self, simple_model, sample_input):
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import trace_to_graph
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import trace_module

        graph = trace_to_graph(trace_module(simple_model, args=(sample_input,)))
        result = dot_formatter(graph)

        assert 'label="' in result
        assert " -> " in result

    def test_empty_graph_is_rendered_as_empty_dot_document(self):
        result = dot_formatter(ComputationGraph())

        assert result == "digraph fx_graph {\n    rankdir=TB;\n}"

    def test_call_stack_quotes_are_escaped(self):
        graph = ComputationGraph()
        op = _make_operator_record(op_name="test.default")
        op.traceback = [
            {
                "filename": "test.py",
                "lineno": 1,
                "module": "mod",
                "function": "f",
                "code_context": 'x = "hello"\ny = 1',
            }
        ]
        graph.add_node(GraphNode(id_=0, operator=op))

        result = dot_formatter(graph)

        assert '\\"' in result
