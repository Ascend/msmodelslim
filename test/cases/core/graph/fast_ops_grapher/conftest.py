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

# pylint: disable=redefined-outer-name
from unittest.mock import patch

import pytest
import torch
from torch import nn

from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import TensorInfo, OperatorRecord
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import GraphNode, GraphEdge, ComputationGraph


def _make_operator_record(op_name="", inputs=None, outputs=None):
    with patch.object(OperatorRecord, '_OperatorRecord__capture_traceback', return_value=[]):
        record = OperatorRecord(op_name=op_name, inputs=inputs, outputs=outputs)
    return record


class SimpleModel(nn.Module):
    """A simple two-layer linear model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class BranchModel(nn.Module):
    """A model with branching (fan-out) structure for testing DAG topology."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
        self.linear3 = nn.Linear(5, 3)

    def forward(self, x):
        a = self.linear1(x)
        b = torch.relu(a)
        c = self.linear2(b)
        d = self.linear3(c)
        return d


class SingleOpModel(nn.Module):
    """A minimal model with a single operation."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def branch_model():
    return BranchModel()


@pytest.fixture
def single_op_model():
    return SingleOpModel()


@pytest.fixture
def sample_input():
    return torch.randn(1, 10)


@pytest.fixture
def small_input():
    return torch.randn(1, 4)


@pytest.fixture
def sample_tensor_info():
    return TensorInfo(id=1, varname="args[0]", dtype=torch.float32, shape=torch.Size([1, 10]))


@pytest.fixture
def sample_operator_record(sample_tensor_info):
    output_info = TensorInfo(id=2, varname="output", dtype=torch.float32, shape=torch.Size([1, 20]))
    record = _make_operator_record(op_name="addmm.default", inputs=[sample_tensor_info], outputs=[output_info])
    return record


@pytest.fixture
def empty_graph():
    return ComputationGraph()


@pytest.fixture
def graph_with_one_node(sample_operator_record):
    graph = ComputationGraph()
    node = GraphNode(id_=0, operator=sample_operator_record)
    graph.add_node(node)
    return graph, node


@pytest.fixture
def graph_with_two_nodes():
    graph = ComputationGraph()
    t0 = TensorInfo(id=0, varname="args[0]", dtype=torch.float32, shape=torch.Size([1, 10]))
    t1 = TensorInfo(id=1, varname="output", dtype=torch.float32, shape=torch.Size([1, 20]))
    t2 = TensorInfo(id=2, varname="output", dtype=torch.float32, shape=torch.Size([1, 5]))

    op0 = _make_operator_record(op_name="addmm.default", inputs=[t0], outputs=[t1])
    op1 = _make_operator_record(op_name="addmm.default", inputs=[t1], outputs=[t2])

    node0 = GraphNode(id_=0, operator=op0)
    node1 = GraphNode(id_=1, operator=op1)
    graph.add_node(node0)
    graph.add_node(node1)

    edge = GraphEdge(id_=1, source_node_id=0, target_node_id=1, tensor=t1)
    graph.add_edge(edge)

    return graph, node0, node1, edge
