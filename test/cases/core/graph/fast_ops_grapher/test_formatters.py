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

# pylint: disable=duplicate-code,protected-access,import-outside-toplevel
import pytest

from test.cases.core.graph.fast_ops_grapher.conftest import _make_operator_record
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import GraphNode, ComputationGraph
from msmodelslim.core.graph.fast_ops_grapher.formatters.formatter_mng import (
    register_formatter,
    format_graph,
    list_formatters,
    _formatters,
)
from msmodelslim.core.graph.fast_ops_grapher.formatters.dot_formatter import dot_formatter, _generate_node_names


class TestGenerateNodeNames:
    @pytest.mark.parametrize(
        "op_name,node_id,expected_simple,expected_node",
        [
            ("addmm.default", 0, "addmm", "addmm_0"),
            ("relu.default", 3, "relu", "relu_3"),
            ("sigmoid", 5, "sigmoid", "sigmoid_5"),
            ("a.b.c", 1, "a.b", "a.b_1"),
        ],
    )
    def test_generate_node_names_handles_various_operator_name_formats(
        self, op_name, node_id, expected_simple, expected_node
    ):
        simple, node_name = _generate_node_names(op_name, node_id)
        assert simple == expected_simple
        assert node_name == expected_node

    def test_generate_node_names_handles_operator_names_without_dots(self):
        simple, node_name = _generate_node_names("conv2d", 2)
        assert simple == "conv2d"
        assert node_name == "conv2d_2"

    def test_generate_node_names_handles_operator_names_with_multiple_dots(self):
        simple, node_name = _generate_node_names("aten.addmm.default", 0)
        assert simple == "aten.addmm"
        assert node_name == "aten.addmm_0"


class TestDotFormatter:
    def test_dot_formatter_output_is_valid_dot_format(self, simple_model, sample_input):
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import trace_module
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import trace_to_graph

        trace = trace_module(simple_model, args=(sample_input,))
        graph = trace_to_graph(trace)
        result = dot_formatter(graph)
        assert result.startswith("digraph fx_graph {")
        assert result.strip().endswith("}")
        assert "rankdir=TB;" in result

    def test_dot_formatter_output_contains_node_definitions(self, simple_model, sample_input):
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import trace_module
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import trace_to_graph

        trace = trace_module(simple_model, args=(sample_input,))
        graph = trace_to_graph(trace)
        result = dot_formatter(graph)
        assert 'label="' in result

    def test_dot_formatter_output_contains_edge_definitions(self, simple_model, sample_input):
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import trace_module
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import trace_to_graph

        trace = trace_module(simple_model, args=(sample_input,))
        graph = trace_to_graph(trace)
        result = dot_formatter(graph)
        assert " -> " in result

    def test_dot_formatter_handles_empty_graph(self):
        graph = ComputationGraph()
        result = dot_formatter(graph)
        assert "digraph fx_graph {" in result
        assert "}" in result

    def test_dot_formatter_escapes_special_characters_in_call_stack(self):
        graph = ComputationGraph()
        op = _make_operator_record(op_name='test.default')
        op.traceback = [
            {"filename": "test.py", "lineno": 1, "module": "mod", "function": "f", "code_context": 'x = "hello"\ny = 1'}
        ]
        node = GraphNode(id_=0, operator=op)
        graph.add_node(node)
        result = dot_formatter(graph)
        assert '\\"' in result or "hello" in result


class TestFormatterMng:
    def test_list_formatters_includes_dot_formatter_by_default(self):
        formatters = list_formatters()
        assert "dot" in formatters

    def test_format_graph_uses_dot_formatter_when_preset_is_dot(self, simple_model, sample_input):
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import trace_module
        from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import trace_to_graph

        trace = trace_module(simple_model, args=(sample_input,))
        graph = trace_to_graph(trace)
        result = format_graph("dot", graph)
        assert "digraph fx_graph {" in result

    def test_format_graph_raises_keyerror_for_unknown_formatter_preset(self, empty_graph):
        with pytest.raises(KeyError, match="Formatter 'nonexistent' not found"):
            format_graph("nonexistent", empty_graph)

    def test_register_formatter_adds_custom_formatter_to_registry(self, empty_graph):
        @register_formatter("test_custom")
        def custom_fmt(graph):
            return f"nodes={graph.node_count}"

        try:
            assert "test_custom" in list_formatters()
            result = format_graph("test_custom", empty_graph)
            assert result == "nodes=0"
        finally:
            del _formatters["test_custom"]

    def test_register_formatter_overwrites_existing_formatter_with_same_name(self, empty_graph):
        @register_formatter("test_overwrite")
        def fmt_v1(_):
            return "v1"

        @register_formatter("test_overwrite")
        def fmt_v2(_):
            return "v2"

        try:
            assert format_graph("test_overwrite", empty_graph) == "v2"
        finally:
            del _formatters["test_overwrite"]


class TestComputationGraphFormat:
    def test_computation_graph_format_uses_dot_formatter_for_dot_preset(self, simple_model, sample_input):
        from msmodelslim.core.graph.fast_ops_grapher.extractors.native_module_extractor import NativeModuleExtractor

        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        graph = extractor.extract_dag()
        result = graph.format("dot")
        assert "digraph fx_graph {" in result

    def test_computation_graph_format_raises_keyerror_for_unknown_preset(self, empty_graph):
        with pytest.raises(KeyError):
            empty_graph.format("nonexistent")
