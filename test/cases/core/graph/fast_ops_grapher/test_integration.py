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
import pickle
import pytest
import torch
from torch import nn
import networkx as nx

from msmodelslim.core.graph.fast_ops_grapher import NativeModuleExtractor, ComputationGraph, TensorInfo, OperatorRecord
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import trace_module
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import trace_to_graph
from msmodelslim.core.graph.fast_ops_grapher.extractors.base_extractor import OPERATIONS_TO_REMOVE


class TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 8)
        self.linear2 = nn.Linear(8, 8)

    def forward(self, x):
        a = self.linear1(x)
        b = self.linear2(a)
        return a + b


class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x, y):
        return self.linear(x) + self.linear(y)


class TestFullPipeline:
    def test_full_pipeline_extracts_valid_computation_graph_for_two_layer_model(self):
        model = TwoLayerModel()
        inp = torch.randn(1, 10)
        extractor = NativeModuleExtractor.create(model, args=(inp,), kwargs={})
        graph = extractor.extract_dag()

        assert isinstance(graph, ComputationGraph)
        assert graph.node_count > 0
        assert graph.edge_count > 0
        assert nx.is_directed_acyclic_graph(graph)

    def test_full_pipeline_extracts_valid_computation_graph_for_residual_model(self):
        model = ResidualModel()
        inp = torch.randn(1, 8)
        extractor = NativeModuleExtractor.create(model, args=(inp,), kwargs={})
        graph = extractor.extract_dag()

        assert isinstance(graph, ComputationGraph)
        assert nx.is_directed_acyclic_graph(graph)

    def test_full_pipeline_extracts_valid_computation_graph_for_multi_input_model(self):
        model = MultiInputModel()
        x = torch.randn(1, 10)
        y = torch.randn(1, 10)
        extractor = NativeModuleExtractor.create(model, args=(x, y), kwargs={})
        graph = extractor.extract_dag()

        assert isinstance(graph, ComputationGraph)
        assert nx.is_directed_acyclic_graph(graph)

    def test_format_method_produces_valid_dot_format_output(self):
        model = TwoLayerModel()
        inp = torch.randn(1, 10)
        extractor = NativeModuleExtractor.create(model, args=(inp,), kwargs={})
        graph = extractor.extract_dag()
        dot_str = graph.format("dot")

        assert dot_str.startswith("digraph fx_graph {")
        assert dot_str.strip().endswith("}")
        assert "rankdir=TB;" in dot_str
        assert " -> " in dot_str

    def test_computation_graph_supports_pickle_serialization_roundtrip(self, tmp_path):
        model = TwoLayerModel()
        inp = torch.randn(1, 10)
        extractor = NativeModuleExtractor.create(model, args=(inp,), kwargs={})
        graph = extractor.extract_dag()

        pkl_path = tmp_path / "graph.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(graph, f)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        assert isinstance(loaded, ComputationGraph)
        assert loaded.node_count == graph.node_count
        assert loaded.edge_count == graph.edge_count

    def test_post_processing_removes_decorative_operators_from_graph(self):
        model = TwoLayerModel()
        inp = torch.randn(1, 10)
        extractor = NativeModuleExtractor.create(model, args=(inp,), kwargs={})
        graph = extractor.extract_dag()

        for node in graph.iter_nodes():
            base_op = node.operator.op_name.split('.')[0]
            assert base_op not in OPERATIONS_TO_REMOVE

    def test_raw_dag_has_equal_or_more_nodes_than_processed_graph(self):
        model = TwoLayerModel()
        inp = torch.randn(1, 10)
        extractor = NativeModuleExtractor.create(model, args=(inp,), kwargs={})
        raw_graph = extractor._extract_raw_dag()
        processed_graph = extractor.extract_dag()
        assert raw_graph.node_count >= processed_graph.node_count

    def test_graph_node_traversal_provides_valid_operators_and_edges(self):
        model = TwoLayerModel()
        inp = torch.randn(1, 10)
        extractor = NativeModuleExtractor.create(model, args=(inp,), kwargs={})
        graph = extractor.extract_dag()

        for node in graph.iter_nodes():
            op = node.operator
            assert isinstance(op, OperatorRecord)
            assert isinstance(op.op_name, str)
            assert len(op.op_name) > 0
            for edge in node.get_outgoing_edges():
                assert edge.source_node_id == node.id
                assert isinstance(edge.tensor, TensorInfo)
            for edge in node.get_incoming_edges():
                assert edge.target_node_id == node.id

    def test_trace_to_graph_pipeline_works_directly_without_extractor(self):
        model = TwoLayerModel()
        inp = torch.randn(1, 10)
        trace = trace_module(model, args=(inp,))
        graph = trace_to_graph(trace)

        assert isinstance(graph, ComputationGraph)
        assert graph.node_count == len(trace.operators)
        assert nx.is_directed_acyclic_graph(graph)

    def test_repeated_extraction_on_same_model_produces_consistent_results(self):
        model = TwoLayerModel()
        inp = torch.randn(1, 10)
        extractor = NativeModuleExtractor.create(model, args=(inp,), kwargs={})

        graph1 = extractor.extract_dag()
        graph2 = extractor.extract_dag()

        assert graph1.node_count == graph2.node_count


class TestTransformerIntegration:
    MODEL_PATH = "Qwen/Qwen3-0.6B"

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_auto_extractor_full_pipeline_works(self):
        from msmodelslim.core.graph.fast_ops_grapher import TransformerAutoExtractor

        extractor = TransformerAutoExtractor.create(model_path=self.MODEL_PATH)
        graph = extractor.extract_dag()

        assert isinstance(graph, ComputationGraph)
        assert graph.node_count > 0
        assert nx.is_directed_acyclic_graph(graph)

        dot_str = graph.format("dot")
        assert "digraph fx_graph {" in dot_str

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_extractor_full_pipeline_with_manual_model_loading_works(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from msmodelslim.core.graph.fast_ops_grapher import TransformerExtractor

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATH)
        extractor = TransformerExtractor.create(model=model, tokenizer=tokenizer)
        graph = extractor.extract_dag()

        assert isinstance(graph, ComputationGraph)
        assert nx.is_directed_acyclic_graph(graph)

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_computation_graph_supports_pickle_roundtrip(self, tmp_path):
        from msmodelslim.core.graph.fast_ops_grapher import TransformerAutoExtractor

        extractor = TransformerAutoExtractor.create(model_path=self.MODEL_PATH)
        graph = extractor.extract_dag()

        pkl_path = tmp_path / "transformer_graph.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(graph, f)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded.node_count == graph.node_count
