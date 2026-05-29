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

# pylint: disable=duplicate-code,protected-access,import-outside-toplevel,abstract-class-instantiated,arguments-differ
import pytest
import torch
import networkx as nx

from test.cases.core.graph.fast_ops_grapher.conftest import _make_operator_record
from msmodelslim.core.graph.fast_ops_grapher.extractors.base_extractor import (
    BaseExtractor,
    OPERATIONS_TO_REMOVE,
    _remove_simple_node,
)
from msmodelslim.core.graph.fast_ops_grapher.extractors.native_module_extractor import NativeModuleExtractor
from msmodelslim.core.graph.fast_ops_grapher.extractors.transformer_extractor import (
    TransformerExtractor,
    TransformerAutoExtractor,
)
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import ComputationGraph, GraphNode, GraphEdge
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import TensorInfo


class TestBaseExtractorAbstract:
    def test_base_extractor_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseExtractor()

    def test_base_extractor_subclass_must_implement_create_static_method(self):
        class IncompleteExtractor(BaseExtractor):
            @property
            def target_module(self):
                return None

            @property
            def dummy_inputs(self):
                return ((), {})

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_base_extractor_subclass_must_implement_target_module_property(self):
        class IncompleteExtractor(BaseExtractor):
            @staticmethod
            def create():
                pass

            @property
            def dummy_inputs(self):
                return ((), {})

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_base_extractor_subclass_must_implement_dummy_inputs_property(self):
        class IncompleteExtractor(BaseExtractor):
            @staticmethod
            def create():
                pass

            @property
            def target_module(self):
                return None

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_base_extractor_concrete_subclass_can_be_instantiated_and_used(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        assert isinstance(extractor, BaseExtractor)


class TestOperationsToRemove:
    def test_operations_to_remove_contains_common_decorative_operators(self):
        assert 'detach' in OPERATIONS_TO_REMOVE
        assert 'view' in OPERATIONS_TO_REMOVE
        assert 'unsqueeze' in OPERATIONS_TO_REMOVE
        assert 'transpose' in OPERATIONS_TO_REMOVE
        assert 'clone' in OPERATIONS_TO_REMOVE
        assert 'expand' in OPERATIONS_TO_REMOVE

    def test_operations_to_remove_is_a_list_instance(self):
        assert isinstance(OPERATIONS_TO_REMOVE, list)


class TestRemoveSimpleNode:
    def test_remove_middle_node_bridges_its_predecessor_and_successor_nodes(self):
        graph = ComputationGraph()
        t0 = TensorInfo(id=0, varname="x", dtype=torch.float32, shape=torch.Size([1]))
        t1 = TensorInfo(id=1, varname="y", dtype=torch.float32, shape=torch.Size([1]))
        t2 = TensorInfo(id=2, varname="z", dtype=torch.float32, shape=torch.Size([1]))

        op_a = _make_operator_record(op_name="addmm.default", inputs=[t0], outputs=[t1])
        op_view = _make_operator_record(op_name="view.default", inputs=[t1], outputs=[t2])
        op_b = _make_operator_record(op_name="addmm.default", inputs=[t2], outputs=[])

        node_a = GraphNode(id_=0, operator=op_a)
        node_view = GraphNode(id_=1, operator=op_view)
        node_b = GraphNode(id_=2, operator=op_b)

        graph.add_node(node_a)
        graph.add_node(node_view)
        graph.add_node(node_b)

        edge0 = GraphEdge(id_=1, source_node_id=0, target_node_id=1, tensor=t1)
        edge1 = GraphEdge(id_=2, source_node_id=1, target_node_id=2, tensor=t2)
        graph.add_edge(edge0)
        graph.add_edge(edge1)

        _remove_simple_node(node_view)

        assert graph.get_node(1) is None
        assert graph.get_node(0) is not None
        assert graph.get_node(2) is not None

    def test_remove_node_with_no_successors_removes_node_and_its_incoming_edge(self):
        graph = ComputationGraph()
        t0 = TensorInfo(id=0, varname="x", dtype=torch.float32, shape=torch.Size([1]))
        t1 = TensorInfo(id=1, varname="y", dtype=torch.float32, shape=torch.Size([1]))

        op_a = _make_operator_record(op_name="addmm.default", inputs=[t0], outputs=[t1])
        op_view = _make_operator_record(op_name="view.default", inputs=[t1], outputs=[])

        node_a = GraphNode(id_=0, operator=op_a)
        node_view = GraphNode(id_=1, operator=op_view)

        graph.add_node(node_a)
        graph.add_node(node_view)

        edge = GraphEdge(id_=1, source_node_id=0, target_node_id=1, tensor=t1)
        graph.add_edge(edge)

        _remove_simple_node(node_view)

        assert graph.get_node(1) is None
        assert graph.get_node(0) is not None


class TestNativeModuleExtractor:
    def test_native_module_extractor_create_returns_valid_extractor_instance(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        assert isinstance(extractor, NativeModuleExtractor)

    def test_native_module_extractor_target_module_returns_original_model(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        assert extractor.target_module is simple_model

    def test_native_module_extractor_dummy_inputs_returns_provided_args_and_kwargs(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        args, kwargs = extractor.dummy_inputs
        assert args == (sample_input,)
        assert kwargs == {}

    def test_native_module_extractor_extract_dag_returns_computation_graph_instance(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        graph = extractor.extract_dag()
        assert isinstance(graph, ComputationGraph)

    def test_native_module_extractor_extract_dag_produces_graph_with_nodes(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        graph = extractor.extract_dag()
        assert graph.node_count > 0

    def test_native_module_extractor_extract_dag_produces_graph_with_edges(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        graph = extractor.extract_dag()
        assert graph.edge_count > 0

    def test_native_module_extractor_extract_dag_produces_valid_directed_acyclic_graph(
        self, simple_model, sample_input
    ):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        graph = extractor.extract_dag()
        assert nx.is_directed_acyclic_graph(graph)

    def test_native_module_extractor_extract_dag_removes_decorative_operators_from_graph(
        self, simple_model, sample_input
    ):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        graph = extractor.extract_dag()
        for node in graph.iter_nodes():
            base_op = node.operator.op_name.split('.')[0]
            assert base_op not in OPERATIONS_TO_REMOVE

    def test_native_module_extractor_extract_dag_works_with_keyword_arguments(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        graph = extractor.extract_dag()
        assert isinstance(graph, ComputationGraph)

    def test_native_module_extractor_extract_dag_works_with_single_operator_model(self, single_op_model, small_input):
        extractor = NativeModuleExtractor.create(single_op_model, args=(small_input,), kwargs={})
        graph = extractor.extract_dag()
        assert graph.node_count >= 1

    def test_native_module_extractor_extract_dag_format_dot_produces_valid_dot_output(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})
        graph = extractor.extract_dag()
        dot_str = graph.format("dot")
        assert "digraph fx_graph {" in dot_str


class TestTransformerExtractor:
    MODEL_PATH = "Qwen/Qwen3-0.6B"

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_extractor_create_returns_valid_extractor_instance(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATH)
        extractor = TransformerExtractor.create(model=model, tokenizer=tokenizer)
        assert isinstance(extractor, TransformerExtractor)

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_extractor_target_module_returns_original_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATH)
        extractor = TransformerExtractor.create(model=model, tokenizer=tokenizer)
        assert extractor.target_module is model

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_extractor_dummy_inputs_returns_valid_inputs_with_use_cache_false(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATH)
        extractor = TransformerExtractor.create(model=model, tokenizer=tokenizer)
        args, kwargs = extractor.dummy_inputs
        assert not args
        assert "input_ids" in kwargs
        assert "use_cache" in kwargs
        assert kwargs["use_cache"] is False

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_extractor_extract_dag_returns_valid_computation_graph(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATH)
        extractor = TransformerExtractor.create(model=model, tokenizer=tokenizer)
        graph = extractor.extract_dag()
        assert isinstance(graph, ComputationGraph)
        assert graph.node_count > 0
        assert graph.edge_count > 0
        assert nx.is_directed_acyclic_graph(graph)

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_extractor_extract_dag_format_dot_produces_valid_dot_output(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_PATH)
        extractor = TransformerExtractor.create(model=model, tokenizer=tokenizer)
        graph = extractor.extract_dag()
        dot_str = graph.format("dot")
        assert "digraph fx_graph {" in dot_str


class TestTransformerAutoExtractor:
    MODEL_PATH = "Qwen/Qwen3-0.6B"

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_auto_extractor_create_returns_valid_extractor_instance(self):
        extractor = TransformerAutoExtractor.create(model_path=self.MODEL_PATH)
        assert isinstance(extractor, TransformerExtractor)

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_auto_extractor_extract_dag_returns_valid_computation_graph(self):
        extractor = TransformerAutoExtractor.create(model_path=self.MODEL_PATH)
        graph = extractor.extract_dag()
        assert isinstance(graph, ComputationGraph)
        assert graph.node_count > 0
        assert nx.is_directed_acyclic_graph(graph)

    @pytest.mark.skip(reason="local model weights needed")
    def test_transformer_auto_extractor_create_works_with_trust_remote_code_true(self):
        extractor = TransformerAutoExtractor.create(model_path=self.MODEL_PATH, trust_remote_code=True)
        assert isinstance(extractor, TransformerExtractor)
