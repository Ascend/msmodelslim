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

# pylint: disable=protected-access,abstract-class-instantiated,arguments-differ,no-name-in-module

import pytest
import torch

from test.cases.core.graph.fast_ops_grapher.conftest import _make_operator_record
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import (
    ComputationGraph,
    GraphEdge,
    GraphNode,
)
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import TensorInfo
from msmodelslim.core.graph.fast_ops_grapher.extractors.base_extractor import (
    OPERATIONS_TO_REMOVE,
    BaseExtractor,
    _remove_simple_node,
)


class TestBaseExtractorContract:
    def test_base_extractor_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseExtractor()

    def test_subclass_missing_create_cannot_be_instantiated(self):
        class IncompleteExtractor(BaseExtractor):
            @property
            def target_module(self):
                return None

            @property
            def dummy_inputs(self):
                return ((), {})

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_subclass_missing_target_module_cannot_be_instantiated(self):
        class IncompleteExtractor(BaseExtractor):
            @staticmethod
            def create():
                return None

            @property
            def dummy_inputs(self):
                return ((), {})

        with pytest.raises(TypeError):
            IncompleteExtractor()

    def test_subclass_missing_dummy_inputs_cannot_be_instantiated(self):
        class IncompleteExtractor(BaseExtractor):
            @staticmethod
            def create():
                return None

            @property
            def target_module(self):
                return None

        with pytest.raises(TypeError):
            IncompleteExtractor()


class TestOperationsToRemove:
    def test_contains_decorative_operator_bases(self):
        assert OPERATIONS_TO_REMOVE == [
            "detach",
            "view",
            "_unsafe_view",
            "unsqueeze",
            "transpose",
            "t",
            "slice",
            "select",
            "expand",
            "clone",
            "alias",
            "lift_fresh",
            "_to_copy",
            "to",
        ]

    def test_is_list_instance(self):
        assert isinstance(OPERATIONS_TO_REMOVE, list)


class TestRemoveSimpleNode:
    def test_remove_middle_node_bridges_predecessor_and_successor(self):
        graph = ComputationGraph()
        t0 = TensorInfo(id=0, varname="x", dtype=torch.float32, shape=torch.Size([1]))
        t1 = TensorInfo(id=1, varname="y", dtype=torch.float32, shape=torch.Size([1]))
        t2 = TensorInfo(id=2, varname="z", dtype=torch.float32, shape=torch.Size([1]))
        node_a = GraphNode(id_=0, operator=_make_operator_record(op_name="addmm.default", inputs=[t0], outputs=[t1]))
        node_view = GraphNode(id_=1, operator=_make_operator_record(op_name="view.default", inputs=[t1], outputs=[t2]))
        node_b = GraphNode(id_=2, operator=_make_operator_record(op_name="addmm.default", inputs=[t2], outputs=[]))
        graph.add_node(node_a)
        graph.add_node(node_view)
        graph.add_node(node_b)
        graph.add_edge(GraphEdge(id_=1, source_node_id=0, target_node_id=1, tensor=t1))
        graph.add_edge(GraphEdge(id_=2, source_node_id=1, target_node_id=2, tensor=t2))

        _remove_simple_node(node_view)

        assert graph.get_node(0) is node_a
        assert graph.get_node(1) is None
        assert graph.get_node(2) is node_b
        assert graph.edge_count == 1

    def test_remove_terminal_node_removes_node_and_incoming_edge(self):
        graph = ComputationGraph()
        t0 = TensorInfo(id=0, varname="x", dtype=torch.float32, shape=torch.Size([1]))
        t1 = TensorInfo(id=1, varname="y", dtype=torch.float32, shape=torch.Size([1]))
        node_a = GraphNode(id_=0, operator=_make_operator_record(op_name="addmm.default", inputs=[t0], outputs=[t1]))
        node_view = GraphNode(id_=1, operator=_make_operator_record(op_name="view.default", inputs=[t1], outputs=[]))
        graph.add_node(node_a)
        graph.add_node(node_view)
        graph.add_edge(GraphEdge(id_=1, source_node_id=0, target_node_id=1, tensor=t1))

        _remove_simple_node(node_view)

        assert graph.get_node(0) is node_a
        assert graph.get_node(1) is None
        assert graph.edge_count == 0
