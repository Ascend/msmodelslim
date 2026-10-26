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

# pylint: disable=protected-access,arguments-differ

import networkx as nx

from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import ComputationGraph
from msmodelslim.core.graph.fast_ops_grapher.extractors.base_extractor import OPERATIONS_TO_REMOVE
from msmodelslim.core.graph.fast_ops_grapher.extractors.native_module_extractor import NativeModuleExtractor


class TestNativeModuleExtractor:
    def test_create_returns_extractor_and_preserves_public_inputs(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})

        assert isinstance(extractor, NativeModuleExtractor)
        assert extractor.target_module is simple_model
        args, kwargs = extractor.dummy_inputs
        assert len(args) == 1
        assert args[0] is sample_input
        assert kwargs == {}

    def test_extract_dag_produces_non_empty_directed_acyclic_graph(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})

        graph = extractor.extract_dag()

        assert isinstance(graph, ComputationGraph)
        assert graph.node_count > 0
        assert graph.edge_count > 0
        assert nx.is_directed_acyclic_graph(graph)

    def test_extract_dag_removes_decorative_operators(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})

        graph = extractor.extract_dag()
        remaining_base_ops = {node.operator.op_name.split('.')[0] for node in graph.iter_nodes()}

        assert remaining_base_ops.isdisjoint(set(OPERATIONS_TO_REMOVE))

    def test_extract_dag_formats_non_empty_dot_output(self, simple_model, sample_input):
        extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs={})

        dot_str = extractor.extract_dag().format("dot")

        assert dot_str.startswith("digraph fx_graph {")
        assert dot_str.rstrip().endswith("}")


def test_create_preserves_keyword_arguments(simple_model, sample_input):
    kwargs = {"foo": "bar"}
    extractor = NativeModuleExtractor.create(simple_model, args=(sample_input,), kwargs=kwargs)

    args, actual_kwargs = extractor.dummy_inputs

    assert args == (sample_input,)
    assert actual_kwargs == kwargs


def test_extract_dag_supports_single_operator_model(single_op_model, small_input):
    extractor = NativeModuleExtractor.create(single_op_model, args=(small_input,), kwargs={})

    graph = extractor.extract_dag()

    assert graph.node_count >= 1
