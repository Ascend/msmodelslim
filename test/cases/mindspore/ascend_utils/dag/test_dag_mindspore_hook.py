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
# pylint: disable=consider-using-from-import,redefined-outer-name,super-with-arguments

import mindspore
import mindspore.nn as nn
import pytest
from mindspore.common.initializer import Normal

from ascend_utils.core.dag.dag_node import DagNode
from ascend_utils.mindspore.dag.dag_mindspore_hook import DagMindSporeHook


class FakeModuleAA(mindspore.nn.Cell):
    def construct(self, tensor_x):
        return tensor_x + 1


class FakeModuleQuant(mindspore.nn.Cell):
    def construct(self, tensor_x):
        return tensor_x * 2


class FakeModuleBB(mindspore.nn.Cell):
    def construct(self, tensor_x):
        return tensor_x * 2


class FakeModuleCC(mindspore.nn.Cell):
    def construct(self, tensor_x):
        return tensor_x


class FakeModuleDD(mindspore.nn.Cell):
    def construct(self, tensor_x, tensor_x2):
        return tensor_x + tensor_x2


class Network2Search(mindspore.nn.Cell):
    def __init__(self):
        super(Network2Search, self).__init__()
        self.g1_a1 = FakeModuleAA()
        self.g1_b1 = FakeModuleBB()
        self.g1_c1 = FakeModuleCC()
        self.g1_a2 = FakeModuleAA()

        self.g2_a1 = FakeModuleAA()
        self.g2_a2 = FakeModuleAA()
        self.g2_b1 = FakeModuleBB()
        self.g2_b2 = FakeModuleBB()

        self.g3_a1 = FakeModuleAA()
        self.g3_a2 = FakeModuleAA()
        self.g3_b1 = FakeModuleBB()
        self.g3_b2 = FakeModuleBB()

        self.g4_a1 = FakeModuleAA()
        self.g4_a2 = FakeModuleAA()
        self.g4_b1 = FakeModuleBB()
        self.g4_d1 = FakeModuleDD()

        self.g5_a1 = FakeModuleAA()
        self.g5_b1 = FakeModuleBB()
        self.g5_b2 = FakeModuleBB()
        self.g5_d1 = FakeModuleDD()

        self.g6_a1 = FakeModuleAA()
        self.g6_b1 = FakeModuleBB()
        self.g6_c2 = FakeModuleCC()
        self.g6_d1 = FakeModuleDD()

        self.g7_a1 = FakeModuleAA()
        self.g7_a2 = FakeModuleAA()
        self.g7_b1 = FakeModuleBB()
        self.g7_c1 = FakeModuleCC()
        self.g7_d1 = FakeModuleDD()
        self.g7_d2 = FakeModuleDD()
        self.g7_d3 = FakeModuleDD()

        self.conv1 = mindspore.nn.Conv2d(3, 64, 3)
        self.conv2 = mindspore.nn.Conv2d(64, 3, 3)
        self.flatten = mindspore.ops.Flatten()
        self.dense = mindspore.nn.Dense(768, 16)

    def forward_group1(self, tensor_x):
        # group 1: a >> b >> c >> a
        tensor_x = self.g1_a1(tensor_x)
        tensor_x = self.g1_b1(tensor_x)
        tensor_x = self.g1_c1(tensor_x)
        tensor_x = self.g1_a2(tensor_x)
        return tensor_x

    def forward_group2(self, tensor_x):
        # group 2: a >> a >> b >> b
        tensor_x = self.g2_a1(tensor_x)
        tensor_x = self.g2_a2(tensor_x)
        tensor_x = self.g2_b1(tensor_x)
        tensor_x = self.g2_b2(tensor_x)
        return tensor_x

    def forward_group3(self, tensor_x):
        # group 3: a >> b >> a >> b
        tensor_x = self.g3_a1(tensor_x)
        tensor_x = self.g3_b1(tensor_x)
        tensor_x = self.g3_a2(tensor_x)
        tensor_x = self.g3_b2(tensor_x)
        return tensor_x

    def forward_group4(self, tensor_x):
        # group 4: a >> < b, a> >> d
        tensor_x = self.g4_a1(tensor_x)
        tensor_x1 = self.g4_b1(tensor_x)
        tensor_x2 = self.g4_a2(tensor_x)
        tensor_x = self.g4_d1(tensor_x1, tensor_x2)
        return tensor_x

    def forward_group5(self, tensor_x):
        # group 5: a >> < b, b> >> d
        tensor_x = self.g5_a1(tensor_x)
        tensor_x1 = self.g5_b1(tensor_x)
        tensor_x2 = self.g5_b2(tensor_x)
        tensor_x = self.g5_d1(tensor_x1, tensor_x2)
        return tensor_x

    def forward_group6(self, tensor_x):
        # group 6: a >> < b, c> >> d
        tensor_x = self.g6_a1(tensor_x)
        tensor_x1 = self.g6_b1(tensor_x)
        tensor_x2 = self.g6_c2(tensor_x)
        tensor_x = self.g6_d1(tensor_x1, tensor_x2)
        return tensor_x

    def forward_group7(self, tensor_x):
        # group 7:  a1
        #         /    \
        #        b1 a2 c1
        #        \ / \ /
        #         d1  d2
        #          \  /
        #           d3
        tensor_xa1 = self.g7_a1(tensor_x)
        tensor_xa2 = self.g7_a2(tensor_x)
        tensor_xb1 = self.g7_b1(tensor_xa1)
        tensor_xc1 = self.g7_c1(tensor_xa1)
        tensor_xd1 = self.g7_d1(tensor_xb1, tensor_xa2)
        tensor_xd2 = self.g7_d2(tensor_xc1, tensor_xa2)
        return self.g7_d3(tensor_xd2, tensor_xd1)

    def construct(self, x_in):
        x_out = self.forward_group1(x_in)
        x_out = self.forward_group2(x_out)
        x_out = self.forward_group3(x_out)
        x_out = self.forward_group4(x_out)
        x_out = self.forward_group5(x_out)
        x_out = self.forward_group6(x_out)
        x_out = self.forward_group7(x_out)
        return self.dense(self.flatten(self.conv2(self.conv1(x_out))))


@pytest.fixture(scope="module")
def inputs_of_model():
    return mindspore.Tensor(shape=(1, 3, 16, 16), dtype=mindspore.float32, init=Normal())


@pytest.fixture
def dag(inputs_of_model):
    model = Network2Search()
    input_data = inputs_of_model
    return DagMindSporeHook(
        model, input_data, hook_ops=[FakeModuleAA, FakeModuleBB, FakeModuleCC, FakeModuleDD, FakeModuleQuant]
    )


@pytest.mark.skip()
class TestNetworkParse:
    @staticmethod
    def test_parse_network_given_sample_network_when_any_pass(dag):
        # first node : FakeModuleAA
        first_node: DagNode = next(dag.search_nodes_by_op_type("FakeModuleAA"))
        assert isinstance(first_node.node, FakeModuleAA)
        assert len(list(first_node.output_nodes)) == 1

        # 2th node : FakeModuleBB
        node_2th = next(first_node.output_nodes)
        assert isinstance(node_2th.node, FakeModuleBB)
        assert list(node_2th.input_nodes) == [first_node]
        assert len(list(node_2th.output_nodes)) == 1

        # 3th node : FakeModuleCC
        node_3th = next(node_2th.output_nodes)
        assert isinstance(node_3th.node, FakeModuleCC)
        assert list(node_3th.input_nodes) == [node_2th]
        assert len(list(node_3th.output_nodes)) == 1

        # 4th node : FakeModuleAA
        node_4th = next(node_3th.output_nodes)
        assert isinstance(node_4th.node, FakeModuleAA)
        assert list(node_4th.input_nodes) == [node_3th]
        assert len(list(node_4th.output_nodes)) == 1

        # more...

    @staticmethod
    def test_search_by_class_given_aa_when_any_pass(dag):
        assert len(list(dag.search_nodes_by_class(FakeModuleAA))) == 12

    @staticmethod
    def test_search_by_class_given_bb_when_any_pass(dag):
        assert len(list(dag.search_nodes_by_class(FakeModuleBB))) == 10

    @staticmethod
    def test_search_nodes_by_class_given_cc_when_any_pass(dag):
        assert len(list(dag.search_nodes_by_class(FakeModuleCC))) == 3

    @staticmethod
    def test_search_nodes_by_class_given_dd_when_any_pass(dag):
        assert len(list(dag.search_nodes_by_class(FakeModuleDD))) == 6

    @staticmethod
    def test_search_by_op_type_given_aa_when_any_pass(dag):
        assert len(list(dag.search_nodes_by_op_type("FakeModuleAA"))) == 12

    @staticmethod
    def test_search_by_op_type_given_bb_when_any_pass(dag):
        assert len(list(dag.search_nodes_by_op_type("FakeModuleBB"))) == 10

    @staticmethod
    def test_search_nodes_by_op_type_given_cc_when_any_pass(dag):
        assert len(list(dag.search_nodes_by_op_type("FakeModuleCC"))) == 3

    @staticmethod
    def test_search_nodes_by_op_type_given_dd_when_any_pass(dag):
        assert len(list(dag.search_nodes_by_op_type("FakeModuleDD"))) == 6

    @staticmethod
    def test_get_node_by_name_given_g6_a1_when_any_pass(dag):
        assert isinstance(dag.get_node_by_name("g6_a1").node, FakeModuleAA)

    @staticmethod
    def test_get_node_by_name_given_g7_d3_when_any_pass(dag):
        assert isinstance(dag.get_node_by_name("g7_d3").node, FakeModuleDD)

    @staticmethod
    def test_search_by_name_prefix_given_embedding_when_any_pass(dag):
        gen_embedding = dag.get_nodes_by_name_prefix("g1")
        nodes = [next(gen_embedding).node for _ in range(4)]
        assert isinstance(nodes[0], FakeModuleAA)
        assert isinstance(nodes[1], FakeModuleBB)
        assert isinstance(nodes[2], FakeModuleCC)
        assert isinstance(nodes[3], FakeModuleAA)


@pytest.mark.skip()
class TestNetworkModify:
    @staticmethod
    def test_replace_node_given_leakyrelu_when_g1_b1_pass(dag, inputs_of_model):
        model: mindspore.nn.Cell = dag.network
        relu_node = dag.get_node_by_name("g1_b1")
        new_leaky_relu_node = mindspore.nn.LeakyReLU(0.1)
        dag.replace_node(relu_node, new_leaky_relu_node)
        mindspore.set_context(mode=mindspore.GRAPH_MODE)
        output = model(inputs_of_model)  # noqa: F841

        assert relu_node.node == new_leaky_relu_node
        assert mindspore.nn.LeakyReLU in [type(module) for _, module in model.cells_and_names()]

    @staticmethod
    def test_add_node_before_given_fake_node_when_g1_b1_pass(dag, inputs_of_model):
        model = dag.network
        embedding_node = dag.get_node_by_name("g1_b1")
        ori_embedding_module = embedding_node.node
        new_fake_node = FakeModuleAA()
        ori_node_count = len(dag.dag_node_list)
        with dag:
            dag.add_node_before(embedding_node, new_fake_node)
        mindspore.set_context(mode=mindspore.GRAPH_MODE)
        output = model(inputs_of_model)  # noqa: F841

        assert new_fake_node in [module for _, module in model.cells_and_names()]
        assert ori_node_count + 1 == len(dag.dag_node_list)
        assert dag.get_node_by_name("g1_b1.0").node == new_fake_node
        assert dag.get_node_by_name("g1_b1.1").node == ori_embedding_module

    @staticmethod
    def test_add_node_after_given_fake_node_when_g3_b2_pass(dag, inputs_of_model):
        model = dag.network
        embedding_node = dag.get_node_by_name("g3_b2")
        ori_embedding_module = embedding_node.node
        new_fake_node = FakeModuleAA()
        ori_node_count = len(dag.dag_node_list)
        with dag:
            dag.add_node_after(embedding_node, new_fake_node)
        mindspore.set_context(mode=mindspore.GRAPH_MODE)
        output = model(inputs_of_model)  # noqa: F841

        assert new_fake_node in [module for _, module in model.cells_and_names()]
        assert dag.get_node_by_name("g3_b2.0").node == ori_embedding_module
        assert dag.get_node_by_name("g3_b2.1").node == new_fake_node
        assert ori_node_count + 1 == len(dag.dag_node_list)

    @staticmethod
    def test_remove_node_given_none_when_bb_pass(dag, inputs_of_model):
        model = dag.network
        ori_node_count = len(dag.dag_node_list)
        remove_cnt = 0
        with dag:
            for relu_node in dag.search_nodes_by_op_type("FakeModuleBB"):
                remove_cnt += 1
                dag.remove_node(relu_node)
        mindspore.set_context(mode=mindspore.GRAPH_MODE)
        output = model(inputs_of_model)  # noqa: F841

        assert nn.ReLU not in [type(module) for _, module in model.cells_and_names()]
        assert ori_node_count == len(dag.dag_node_list) + remove_cnt

    @staticmethod
    def test_like_quant_given_any_when_any_pass(dag, inputs_of_model):
        model = dag.network
        linear_node = dag.get_node_by_name("g1_b1")
        with dag:
            dag.replace_node(linear_node, nn.SequentialCell(FakeModuleQuant(), linear_node.node, FakeModuleQuant()))
        mindspore.set_context(mode=mindspore.GRAPH_MODE)
        output = model(inputs_of_model)  # noqa: F841

        assert dag.get_node_by_name("g1_b1.0").op_type == "FakeModuleQuant"
        assert dag.get_node_by_name("g1_b1.1").op_type == "FakeModuleBB"
        assert dag.get_node_by_name("g1_b1.2").op_type == "FakeModuleQuant"

    @staticmethod
    def test_like_prune_conv_given_any_when_any_pass(dag, inputs_of_model):
        model = dag.network
        conv1_node = dag.get_node_by_name("conv1")
        conv2_node = dag.get_node_by_name("conv2")
        with dag:
            dag.replace_node(conv1_node, mindspore.nn.Conv2d(3, 32, 3))
            dag.replace_node(conv2_node, mindspore.nn.Conv2d(32, 3, 3))
        mindspore.set_context(mode=mindspore.GRAPH_MODE)
        output = model(inputs_of_model)  # noqa: F841

        assert dag.get_node_by_name("conv1").op_type == "Conv2d"
        assert dag.get_node_by_name("conv1").node.out_channels == 32
        assert dag.get_node_by_name("conv2").op_type == "Conv2d"
        assert dag.get_node_by_name("conv2").node.in_channels == 32
        assert [module.out_channels for name, module in model.cells_and_names() if name == "conv1"] == [32]
        assert [module.in_channels for name, module in model.cells_and_names() if name == "conv2"] == [32]

    @staticmethod
    def test_like_low_rank_decomposition_given_any_when_any_pass(dag, inputs_of_model):
        model = dag.network
        linear_node = dag.get_node_by_name("dense")
        with dag:
            dag.replace_node(
                linear_node,
                mindspore.nn.SequentialCell(
                    nn.Dense(768, 8),
                    nn.Dense(8, 16),
                ),
            )
        mindspore.set_context(mode=mindspore.GRAPH_MODE)
        output = model(inputs_of_model)  # noqa: F841

        assert dag.get_node_by_name("dense.0").op_type == "Dense"
        assert dag.get_node_by_name("dense.1").op_type == "Dense"
        assert [module.out_channels for name, module in model.cells_and_names() if name == "dense.0"] == [8]
        assert [module.in_channels for name, module in model.cells_and_names() if name == "dense.1"] == [8]
