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

# pylint: disable=protected-access
import pytest
import torch
import networkx as nx

from test.cases.core.graph.fast_ops_grapher.conftest import _make_operator_record
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import (
    TensorInfo,
    OperatorRecord,
    ExecutionTrace,
    trace_module,
)
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import (
    GraphNode,
    GraphEdge,
    ComputationGraph,
    tensor_identifier,
    tensor_matcher,
    trace_to_graph,
)


class TestGraphNode:
    def test_graph_node_creation_sets_attributes(self, sample_operator_record):
        node = GraphNode(id_=0, operator=sample_operator_record)
        assert node.id == 0
        assert node.operator is sample_operator_record
        assert node.graph is None

    def test_get_outgoing_edges_returns_empty_when_not_in_graph(self, sample_operator_record):
        node = GraphNode(id_=0, operator=sample_operator_record)
        assert not node.get_outgoing_edges()

    def test_get_incoming_edges_returns_empty_when_not_in_graph(self, sample_operator_record):
        node = GraphNode(id_=0, operator=sample_operator_record)
        assert not node.get_incoming_edges()

    def test_get_successors_returns_empty_when_not_in_graph(self, sample_operator_record):
        node = GraphNode(id_=0, operator=sample_operator_record)
        assert not node.get_successors()

    def test_get_predecessors_returns_empty_when_not_in_graph(self, sample_operator_record):
        node = GraphNode(id_=0, operator=sample_operator_record)
        assert not node.get_predecessors()

    def test_get_outgoing_edges_returns_empty_for_last_node_in_graph(self, graph_with_two_nodes):
        _, _, node1, _ = graph_with_two_nodes
        out_edges = node1.get_outgoing_edges()
        assert len(out_edges) == 0

    def test_get_incoming_edges_returns_empty_for_first_node_in_graph(self, graph_with_two_nodes):
        _, node0, _, _ = graph_with_two_nodes
        in_edges = node0.get_incoming_edges()
        assert len(in_edges) == 0

    def test_get_successors_returns_next_node_in_graph(self, graph_with_two_nodes):
        _, node0, _, _ = graph_with_two_nodes
        successors = node0.get_successors()
        assert len(successors) == 1
        assert successors[0].id == 1

    def test_get_predecessors_returns_previous_node_in_graph(self, graph_with_two_nodes):
        _, _, node1, _ = graph_with_two_nodes
        predecessors = node1.get_predecessors()
        assert len(predecessors) == 1
        assert predecessors[0].id == 0

    def test_get_successors_returns_empty_for_last_node(self, graph_with_two_nodes):
        _, _, node1, _ = graph_with_two_nodes
        assert not node1.get_successors()

    def test_get_predecessors_returns_empty_for_first_node(self, graph_with_two_nodes):
        _, node0, _, _ = graph_with_two_nodes
        assert not node0.get_predecessors()

    def test_format_info_returns_operator_details(self, sample_operator_record):
        node = GraphNode(id_=0, operator=sample_operator_record)
        info = node.format_info()
        assert "op_name" in info
        assert "call_stack" in info
        assert info["op_name"] == "addmm.default"


class TestGraphEdge:
    def test_graph_edge_creation_sets_attributes(self, sample_tensor_info):
        edge = GraphEdge(id_=1, source_node_id=0, target_node_id=1, tensor=sample_tensor_info)
        assert edge.id == 1
        assert edge.source_node_id == 0
        assert edge.target_node_id == 1
        assert edge.tensor is sample_tensor_info
        assert edge.graph is None

    def test_get_source_node_returns_none_when_not_in_graph(self, sample_tensor_info):
        edge = GraphEdge(id_=1, source_node_id=0, target_node_id=1, tensor=sample_tensor_info)
        assert edge.get_source_node() is None

    def test_get_target_node_returns_none_when_not_in_graph(self, sample_tensor_info):
        edge = GraphEdge(id_=1, source_node_id=0, target_node_id=1, tensor=sample_tensor_info)
        assert edge.get_target_node() is None

    def test_get_source_node_returns_correct_node_in_graph(self, graph_with_two_nodes):
        source = graph_with_two_nodes[3].get_source_node()
        assert source is not None
        assert source.id == 0

    def test_get_target_node_returns_correct_node_in_graph(self, graph_with_two_nodes):
        target = graph_with_two_nodes[3].get_target_node()
        assert target is not None
        assert target.id == 1

    def test_format_info_returns_tensor_details(self, sample_tensor_info):
        edge = GraphEdge(id_=1, source_node_id=0, target_node_id=1, tensor=sample_tensor_info)
        info = edge.format_info()
        assert "varname" in info
        assert "dtype" in info
        assert "shape" in info
        assert info["varname"] == "args[0]"


class TestComputationGraph:
    def test_computation_graph_creation_is_empty(self):
        graph = ComputationGraph()
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_computation_graph_inherits_from_nx_digraph(self):
        graph = ComputationGraph()
        assert isinstance(graph, nx.DiGraph)

    def test_add_node_adds_node_to_graph(self, sample_operator_record):
        graph = ComputationGraph()
        node = GraphNode(id_=0, operator=sample_operator_record)
        graph.add_node(node)
        assert graph.node_count == 1
        assert graph.get_node(0) is node
        assert node.graph is graph

    def test_add_edge_adds_edge_between_nodes(self, sample_tensor_info):
        graph = ComputationGraph()
        op0 = _make_operator_record(op_name="op0")
        op1 = _make_operator_record(op_name="op1")
        node0 = GraphNode(id_=0, operator=op0)
        node1 = GraphNode(id_=1, operator=op1)
        graph.add_node(node0)
        graph.add_node(node1)

        edge = GraphEdge(id_=1, source_node_id=0, target_node_id=1, tensor=sample_tensor_info)
        graph.add_edge(edge)
        assert graph.edge_count == 1
        assert graph.get_edge(1) is edge
        assert edge.graph is graph

    def test_get_node_returns_none_for_nonexistent_id(self, empty_graph):
        assert empty_graph.get_node(999) is None

    def test_get_edge_returns_none_for_nonexistent_id(self, empty_graph):
        assert empty_graph.get_edge(999) is None

    def test_remove_node_removes_node_and_connected_edges(self, graph_with_two_nodes):
        graph, _, node1, _ = graph_with_two_nodes
        graph.remove_node(node1.id)
        assert graph.node_count == 1
        assert graph.get_node(node1.id) is None
        assert graph.edge_count == 0

    def test_remove_node_raises_error_for_nonexistent_id(self, empty_graph):
        with pytest.raises(ValueError, match="does not exist"):
            empty_graph.remove_node(999)

    def test_remove_first_node_removes_node_and_edge(self, graph_with_two_nodes):
        graph, node0, _, _ = graph_with_two_nodes
        graph.remove_node(node0.id)
        assert graph.node_count == 1
        assert graph.get_node(node0.id) is None
        assert graph.edge_count == 0

    def test_iter_nodes_yields_all_nodes_in_order(self, graph_with_two_nodes):
        graph, node0, node1, _ = graph_with_two_nodes
        nodes = list(graph.iter_nodes())
        assert len(nodes) == 2
        assert nodes[0].id == node0.id
        assert nodes[1].id == node1.id

    def test_iter_edges_yields_all_edges(self, graph_with_two_nodes):
        graph, _, _, edge = graph_with_two_nodes
        edges = list(graph.iter_edges())
        assert len(edges) == 1
        assert edges[0].id == edge.id

    def test_iter_nodes_and_edges_returns_empty_for_empty_graph(self, empty_graph):
        assert not list(empty_graph.iter_nodes())
        assert not list(empty_graph.iter_edges())

    def test_next_edge_id_returns_1_for_empty_graph(self, empty_graph):
        assert empty_graph.next_edge_id == 1

    def test_next_edge_id_increments_after_adding_edge(self, sample_tensor_info):
        graph = ComputationGraph()
        op0 = _make_operator_record(op_name="op0")
        op1 = _make_operator_record(op_name="op1")
        node0 = GraphNode(id_=0, operator=op0)
        node1 = GraphNode(id_=1, operator=op1)
        graph.add_node(node0)
        graph.add_node(node1)

        edge = GraphEdge(id_=graph.next_edge_id, source_node_id=0, target_node_id=1, tensor=sample_tensor_info)
        graph.add_edge(edge)
        assert graph.next_edge_id == 2

    def test_reconcile_edges_removes_stale_edges_not_in_nx_graph(self, graph_with_two_nodes):
        graph, _, _, _ = graph_with_two_nodes
        stale_edge = GraphEdge(
            id_=99,
            source_node_id=0,
            target_node_id=1,
            tensor=TensorInfo(id=9, varname="x", dtype=torch.float32, shape=torch.Size([1])),
        )
        stale_edge.graph = graph
        graph._edges[99] = stale_edge
        assert graph.edge_count == 2

        graph.reconcile_edges()
        assert graph.edge_count == 1

    def test_nx_successors_returns_next_node_id(self, graph_with_two_nodes):
        graph, _, _, _ = graph_with_two_nodes
        assert list(graph.successors(0)) == [1]

    def test_nx_predecessors_returns_previous_node_id(self, graph_with_two_nodes):
        graph, _, _, _ = graph_with_two_nodes
        assert list(graph.predecessors(1)) == [0]


class TestTensorIdentifier:
    def test_for_output_assigns_incrementing_unique_ids(self):
        _, for_output = tensor_identifier()
        t1 = TensorInfo(id=100, varname="x", dtype=torch.float32, shape=torch.Size([1]))
        t2 = TensorInfo(id=200, varname="y", dtype=torch.float32, shape=torch.Size([1]))
        for_output(t1)
        for_output(t2)
        assert t1.id == 0
        assert t2.id == 1

    def test_for_input_reuses_matching_output_id(self):
        for_input, for_output = tensor_identifier()
        t_out = TensorInfo(id=100, varname="x", dtype=torch.float32, shape=torch.Size([1]))
        for_output(t_out)
        t_in = TensorInfo(id=100, varname="args[0]", dtype=torch.float32, shape=torch.Size([1]))
        for_input(t_in)
        assert t_in.id == 0

    def test_for_input_assigns_new_id_when_no_output_match(self):
        for_input, _ = tensor_identifier()
        t_in = TensorInfo(id=999, varname="args[0]", dtype=torch.float32, shape=torch.Size([1]))
        for_input(t_in)
        assert t_in.id == 0

    def test_for_output_assigns_sequential_incrementing_ids(self):
        _, for_output = tensor_identifier()
        infos = []
        for i in range(5):
            t = TensorInfo(id=100 + i, varname=f"t{i}", dtype=torch.float32, shape=torch.Size([1]))
            for_output(t)
            infos.append(t)
        for i, t in enumerate(infos):
            assert t.id == i


class TestTensorMatcher:
    def test_for_input_returns_none_for_unknown_tensor(self):
        for_input, _ = tensor_matcher()
        t = TensorInfo(id=1, varname="x", dtype=torch.float32, shape=torch.Size([1]))
        result = for_input(t)
        assert result is None

    def test_for_output_records_and_for_input_finds_producer_node(self):
        for_input, for_output = tensor_matcher()
        t = TensorInfo(id=42, varname="x", dtype=torch.float32, shape=torch.Size([1]))
        for_output(t, node_id=5)
        result = for_input(t)
        assert result == 5

    def test_for_output_overwrites_previous_producer_for_same_tensor(self):
        for_input, for_output = tensor_matcher()
        t = TensorInfo(id=42, varname="x", dtype=torch.float32, shape=torch.Size([1]))
        for_output(t, node_id=1)
        for_output(t, node_id=2)
        result = for_input(t)
        assert result == 2


class TestTraceToGraph:
    def test_trace_to_graph_returns_computation_graph_for_simple_model(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,))
        graph = trace_to_graph(trace)
        assert isinstance(graph, ComputationGraph)
        assert graph.node_count > 0
        assert graph.edge_count > 0

    def test_all_nodes_have_valid_operator_records(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,))
        graph = trace_to_graph(trace)
        for node in graph.iter_nodes():
            assert isinstance(node.operator, OperatorRecord)
            assert isinstance(node.operator.op_name, str)

    def test_all_edges_have_valid_tensor_info(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,))
        graph = trace_to_graph(trace)
        for edge in graph.iter_edges():
            assert isinstance(edge.tensor, TensorInfo)

    def test_edge_endpoints_exist_in_graph_nodes(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,))
        graph = trace_to_graph(trace)
        for edge in graph.iter_edges():
            assert graph.get_node(edge.source_node_id) is not None
            assert graph.get_node(edge.target_node_id) is not None

    def test_trace_to_graph_result_is_directed_acyclic_graph(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,))
        graph = trace_to_graph(trace)
        assert nx.is_directed_acyclic_graph(graph)

    def test_graph_node_count_matches_trace_operator_count(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,))
        graph = trace_to_graph(trace)
        assert graph.node_count == len(trace.operators)

    def test_trace_to_graph_returns_empty_graph_for_empty_trace(self):
        trace = ExecutionTrace(operators=[])
        graph = trace_to_graph(trace)
        assert graph.node_count == 0
        assert graph.edge_count == 0
