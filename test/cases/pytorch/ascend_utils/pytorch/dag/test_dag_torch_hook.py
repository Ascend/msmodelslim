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
# pylint: disable=consider-using-from-import,pointless-statement,redefined-outer-name,super-with-arguments

import pytest

# 第三方库
import numpy as np
import torch
import torch.nn as nn

# 应用程序自定义模块
from ascend_utils.core.dag.dag import DirectedAcyclicGraph
from ascend_utils.core.dag.dag_node import DagNode
from ascend_utils.pytorch.dag.dag_torch_hook import DagTorchHook

from resources.sample_net_torch import LrdSampleNetwork


class FakeModuleAA(torch.nn.Module):
    def forward(self, tensor_x):
        return tensor_x + 1


class FakeModuleBB(torch.nn.Module):
    def forward(self, tensor_x):
        return tensor_x * 2


class FakeModuleCC(torch.nn.Module):
    def forward(self, tensor_x):
        return tensor_x


class FakeModuleDD(torch.nn.Module):
    def forward(self, tensor_x, tensor_x2):
        return tensor_x + tensor_x2


class Network2Search(torch.nn.Module):
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
        # group 4: a >> < b, a > >> d
        tensor_x = self.g4_a1(tensor_x)
        tensor_x1 = self.g4_b1(tensor_x)
        tensor_x2 = self.g4_a2(tensor_x)
        tensor_x = self.g4_d1(tensor_x1, tensor_x2)
        return tensor_x

    def forward_group5(self, tensor_x):
        # group 5: a >> < b, b > >> d
        tensor_x = self.g5_a1(tensor_x)
        tensor_x1 = self.g5_b1(tensor_x)
        tensor_x2 = self.g5_b2(tensor_x)
        tensor_x = self.g5_d1(tensor_x1, tensor_x2)
        return tensor_x

    def forward_group6(self, tensor_x):
        # group 6: a >> < b, c > >> d
        tensor_x = self.g6_a1(tensor_x)
        tensor_x1 = self.g6_b1(tensor_x)
        tensor_x2 = self.g6_c2(tensor_x)
        tensor_x = self.g6_d1(tensor_x1, tensor_x2)
        return tensor_x

    def forward_group7(self, tensor_x):
        # group 7:   a1
        #          /    \
        #         b1 a2 c1
        #         \ / \ /
        #          d1  d2
        #           \  /
        #             d3
        tensor_xa1 = self.g7_a1(tensor_x)
        tensor_xa2 = self.g7_a2(tensor_x)
        tensor_xb1 = self.g7_b1(tensor_xa1)
        tensor_xc1 = self.g7_c1(tensor_xa1)
        tensor_xd1 = self.g7_d1(tensor_xb1, tensor_xa2)
        tensor_xd2 = self.g7_d2(tensor_xc1, tensor_xa2)
        return self.g7_d3(tensor_xd2, tensor_xd1)

    def forward(self, x_in):
        x_out = self.forward_group1(x_in)
        x_out = self.forward_group2(x_out)
        x_out = self.forward_group3(x_out)
        x_out = self.forward_group4(x_out)
        x_out = self.forward_group5(x_out)
        x_out = self.forward_group6(x_out)
        x_out = self.forward_group7(x_out)
        return x_out


@pytest.fixture(scope="module")
def inputs_of_model():
    return torch.from_numpy(np.random.uniform(size=[2, 16, 16]).astype('long'))


@pytest.fixture(scope="module")
def dag(inputs_of_model):
    model = LrdSampleNetwork()
    return DagTorchHook(model, inputs_of_model, hook_ops=[FakeModuleAA])


@pytest.fixture(scope="module")
def dag_to_search(inputs_of_model):
    model = Network2Search()
    return DagTorchHook(model, inputs_of_model, hook_ops=[FakeModuleAA, FakeModuleBB, FakeModuleCC, FakeModuleDD])


class TestNetworkParse:
    def test_parse_network_given_sample_network_when_any_pass(self, dag):
        # first node:Embedding
        first_node: DagNode = dag.dag_node_list[0]
        assert isinstance(first_node.node, nn.Embedding)
        assert len(list(first_node.output_nodes)) == 1

        # 2th node : Linear
        node_2th = next(first_node.output_nodes)
        assert isinstance(node_2th.node, nn.Linear)
        assert list(node_2th.input_nodes) == [first_node]
        assert len(list(node_2th.output_nodes)) == 1

        # 3th node : ReLU
        node_3th = next(node_2th.output_nodes)
        assert isinstance(node_3th.node, nn.ReLU)
        assert list(node_3th.input_nodes) == [node_2th]
        assert len(list(node_3th.output_nodes)) == 1

        # 4th node : permute in torch.Tensor, 2 outputs
        node_4th = next(node_3th.output_nodes)
        assert node_4th.node == torch.Tensor.permute
        assert list(node_4th.input_nodes) == [node_3th]
        assert len(list(node_4th.output_nodes)) == 2

        # 5th node : conv2d
        node_5th = next(node_4th.output_nodes)
        assert isinstance(node_5th.node, nn.Conv2d)
        assert list(node_5th.input_nodes) == [node_4th]
        assert len(list(node_5th.output_nodes)) == 1

        # 6.7.8th node
        node_8th = next(next(next(node_5th.output_nodes).output_nodes).output_nodes)

        # 9th node : add, 2 inputs
        node_9th = next(node_8th.output_nodes)
        node_9th_from4th = list(node_4th.output_nodes)[1]
        assert node_9th_from4th == node_9th
        assert node_9th.node == torch.Tensor.__add__
        assert list(node_9th.input_nodes) == [node_8th, node_4th]
        assert len(list(node_9th.output_nodes)) == 1

        # 10th node : AdaptiveAvgPool2d
        node_10th = next(node_9th.output_nodes)
        assert isinstance(node_10th.node, nn.AdaptiveAvgPool2d)
        assert list(node_10th.input_nodes) == [node_9th]
        assert len(list(node_10th.output_nodes)) == 1

        # 11th node : flatten of torch
        node_11th = next(node_10th.output_nodes)
        assert node_11th.node == torch.flatten
        assert list(node_11th.input_nodes) == [node_10th]
        assert len(list(node_11th.output_nodes)) == 1

        # more...

    def test_search_by_class_given_relu_when_any_pass(self, dag):
        assert len(list(dag.search_nodes_by_class(nn.ReLU))) == 3

    def test_search_by_class_given_conv2d_when_any_pass(self, dag):
        assert len(list(dag.search_nodes_by_class(nn.Conv2d))) == 2

    def test_search_by_class_given_linear_when_any_pass(self, dag):
        assert len(list(dag.search_nodes_by_class(nn.Linear))) == 4

    def test_search_by_class_given_pool2d_when_any_pass(self, dag):
        assert len(list(dag.search_nodes_by_class(nn.AdaptiveAvgPool2d))) == 1

    def test_search_by_op_type_given_relu_when_any_pass(self, dag):
        assert len(list(dag.search_nodes_by_op_type("ReLU"))) == 3

    def test_search_by_op_type_given_conv2d_when_any_pass(self, dag):
        assert len(list(dag.search_nodes_by_op_type("Conv2d"))) == 2

    def test_search_by_op_type_given_linear_when_any_pass(self, dag):
        assert len(list(dag.search_nodes_by_op_type("Linear"))) == 4

    def test_search_by_op_type_given_pool2d_when_any_pass(self, dag):
        assert len(list(dag.search_nodes_by_op_type("AdaptiveAvgPool2d"))) == 1

    def test_get_node_by_name_type_given_embedding_0_when_any_pass(self, dag):
        assert isinstance(dag.get_node_by_name("embedding.0").node, nn.Embedding)

    def test_get_node_by_name_type_given_embedding_2_when_any_pass(self, dag):
        assert isinstance(dag.get_node_by_name("embedding.1").node, nn.Linear)

    def test_get_node_by_name_type_given_embedding_3_when_any_pass(self, dag):
        assert isinstance(dag.get_node_by_name("embedding.2").node, nn.ReLU)

    def test_get_node_by_name_type_given_classifier_0_when_any_pass(self, dag):
        assert isinstance(dag.get_node_by_name("classifier.0").node, nn.Linear)

    def test_get_node_by_name_type_given_classifier_1_when_any_pass(self, dag):
        assert isinstance(dag.get_node_by_name("classifier.1").node, nn.Linear)

    def test_get_node_by_name_type_given_feature_0_when_any_pass(self, dag):
        assert isinstance(dag.get_node_by_name("feature.0").node, nn.Conv2d)

    def test_get_node_by_name_type_given_feature_1_when_any_pass(self, dag):
        assert isinstance(dag.get_node_by_name("feature.1").node, nn.ReLU)

    def test_get_node_by_name_type_given_pool_when_any_pass(self, dag):
        assert isinstance(dag.get_node_by_name("pool").node, nn.AdaptiveAvgPool2d)

    def test_search_by_name_prefix_given_embedding_when_any_pass(self, dag):
        gen_embedding = dag.get_nodes_by_name_prefix("embedding.")
        assert isinstance(next(gen_embedding).node, nn.Embedding)
        assert isinstance(next(gen_embedding).node, nn.Linear)
        assert isinstance(next(gen_embedding).node, nn.ReLU)

    def test_search_by_name_prefix_given_feature_when_any_pass(self, dag):
        gen_feature = dag.get_nodes_by_name_prefix("feature.")
        assert isinstance(next(gen_feature).node, nn.Conv2d)
        assert isinstance(next(gen_feature).node, nn.ReLU)


class TestNetworkModify:
    def test_replace_node_given_leakyrelu_when_embedding_2_pass(self, dag, inputs_of_model):
        model = dag.network
        relu_node = dag.get_node_by_name("embedding.2")
        new_leaky_relu_node = torch.nn.LeakyReLU(0.1)
        dag.replace_node(relu_node, new_leaky_relu_node)
        output = model(inputs_of_model)  # noqa: F841

        assert relu_node.node == new_leaky_relu_node
        assert torch.nn.LeakyReLU in [type(module) for _, module in model.named_modules()]

    def test_add_node_before_given_fake_node_when_embedding_0_pass(self, dag, inputs_of_model):
        model = dag.network
        embedding_node = dag.get_node_by_name("embedding.0")
        ori_embedding_module = embedding_node.node
        new_fake_node = FakeModuleAA()
        ori_node_count = len(dag.dag_node_list)
        with dag:
            dag.add_node_before(embedding_node, new_fake_node)
        output = model(inputs_of_model)  # noqa: F841

        assert new_fake_node in [module for _, module in model.named_modules()]
        assert ori_node_count + 1 == len(dag.dag_node_list)
        assert dag.get_node_by_name("embedding.0.0").node == new_fake_node
        assert dag.get_node_by_name("embedding.0.1").node == ori_embedding_module

    def test_add_node_after_given_fake_node_when_embedding_0_pass(self, dag, inputs_of_model):
        model = dag.network
        embedding_node = dag.get_node_by_name("embedding.2")
        ori_embedding_module = embedding_node.node
        new_fake_node = FakeModuleAA()
        ori_node_count = len(dag.dag_node_list)
        with dag:
            dag.add_node_after(embedding_node, new_fake_node)
        output = model(inputs_of_model)  # noqa: F841

        assert new_fake_node in [module for _, module in model.named_modules()]
        assert dag.get_node_by_name("embedding.2.0").node == ori_embedding_module
        assert dag.get_node_by_name("embedding.2.1").node == new_fake_node
        assert ori_node_count + 1 == len(dag.dag_node_list)

    def test_remove_node_given_fake_node_when_embedding_0_pass(self, dag, inputs_of_model):
        model = dag.network
        ori_node_count = len(dag.dag_node_list)
        remove_cnt = 0
        with dag:
            for relu_node in dag.search_nodes_by_op_type("ReLU"):
                remove_cnt += 1
                dag.remove_node(relu_node)
        out_put = model(inputs_of_model)  # noqa: F841

        assert nn.ReLU not in [type(module) for _, module in model.named_modules()]
        assert ori_node_count == len(dag.dag_node_list) + remove_cnt

    def test_like_quant_given_any_when_any_pass(self, dag, inputs_of_model):
        model = dag.network
        linear_node = dag.get_node_by_name("embedding.1")
        with dag:
            dag.replace_node(linear_node, nn.Sequential(FakeModuleAA(), linear_node.node, FakeModuleAA()))
        output = model(inputs_of_model)  # noqa: F841

        assert dag.get_node_by_name("embedding.1.0").op_type == "FakeModuleAA"
        assert dag.get_node_by_name("embedding.1.1").op_type == "Linear"
        assert dag.get_node_by_name("embedding.1.2").op_type == "FakeModuleAA"

    def test_like_prune_conv_given_any_when_any_pass(self, dag, inputs_of_model):
        model = dag.network
        conv1_node = dag.get_node_by_name("feature.0")
        conv2_node = dag.get_node_by_name("feature.2")
        with dag:
            dag.replace_node(conv1_node, torch.nn.Conv2d(64, 32, 3, 1, 1))
            dag.replace_node(conv2_node, torch.nn.Conv2d(32, 64, 3, 1, 1))
        output = model(inputs_of_model)  # noqa: F841

        assert dag.get_node_by_name("feature.0").op_type == "Conv2d"
        assert dag.get_node_by_name("feature.0").node.out_channels == 32
        assert dag.get_node_by_name("feature.2").op_type == "Conv2d"
        assert dag.get_node_by_name("feature.2").node.in_channels == 32
        assert [module.out_channels for name, module in model.named_modules() if name == "feature.0"] == [32]
        assert [module.in_channels for name, module in model.named_modules() if name == "feature.2"] == [32]

    def test_like_low_rank_decomposition_given_any_when_any_pass(self, dag, inputs_of_model):
        model = dag.network
        linear_node = dag.get_node_by_name("classifier.0")
        with dag:
            dag.replace_node(
                linear_node,
                torch.nn.Sequential(
                    nn.Linear(512, 16),
                    nn.Linear(16, 256),
                ),
            )
        output = model(inputs_of_model)  # noqa: F841

        assert dag.get_node_by_name("classifier.0.0").op_type == "Linear"
        assert dag.get_node_by_name("classifier.0.1").op_type == "Linear"
        assert [module.out_features for name, module in model.named_modules() if name == "classifier.0.0"] == [16]
        assert [module.in_features for name, module in model.named_modules() if name == "classifier.0.1"] == [16]


class TestNetworkSearchGraph:
    def test_search_given_abca_when_any_pass(self, dag_to_search):
        node_a = DagNode(op_type="FakeModuleAA", name="node_a")
        node_b = DagNode(op_type="FakeModuleBB", name="node_b")
        node_c = DagNode(op_type="FakeModuleCC", name="node_c")
        node_a2 = DagNode(op_type="FakeModuleAA", name="node_a2")

        node_a >> node_b >> node_c >> node_a2

        search_list = list(dag_to_search.search_sub_graph([node_a2, node_b, node_c, node_a]))
        assert len(search_list) == 1
        assert search_list[0]["node_a"].name == "g1_a1"
        assert search_list[0]["node_b"].name == "g1_b1"
        assert search_list[0]["node_c"].name == "g1_c1"
        assert search_list[0]["node_a2"].name == "g1_a2"
        assert search_list[0]["node_a"].op_type == "FakeModuleAA"
        assert search_list[0]["node_b"].op_type == "FakeModuleBB"
        assert search_list[0]["node_c"].op_type == "FakeModuleCC"
        assert search_list[0]["node_a2"].op_type == "FakeModuleAA"

    def test_search_given_ab_when_any_pass(self, dag_to_search):
        node_a = DagNode(op_type="FakeModuleAA", name="node_a")
        node_b = DagNode(op_type="FakeModuleBB", name="node_b")

        node_a >> node_b

        cnt = 0
        for sub_graph_in_network in dag_to_search.search_sub_graph([node_b, node_a]):
            assert sub_graph_in_network["node_a"].op_type == "FakeModuleAA"
            assert sub_graph_in_network["node_b"].op_type == "FakeModuleBB"
            cnt += 1
        assert cnt == 4

    def test_search_given_aabb_when_any_pass(self, dag_to_search):
        node_a = DagNode(op_type="FakeModuleAA", name="node_a")
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_b = DagNode(op_type="FakeModuleBB", name="node_b")
        node_b1 = DagNode(op_type="FakeModuleBB", name="node_b1")

        node_a >> node_a1 >> node_b >> node_b1

        search_list = list(dag_to_search.search_sub_graph([node_b, node_b1, node_a, node_a1]))
        assert len(search_list) == 1
        assert search_list[0]["node_a"].name == "g2_a1"
        for sub_graph_in_network in search_list:
            assert sub_graph_in_network["node_a"].op_type == "FakeModuleAA"
            assert sub_graph_in_network["node_a1"].op_type == "FakeModuleAA"
            assert sub_graph_in_network["node_b"].op_type == "FakeModuleBB"
            assert sub_graph_in_network["node_b1"].op_type == "FakeModuleBB"

    def test_search_given_abab_when_any_pass(self, dag_to_search):
        node_a = DagNode(op_type="FakeModuleAA", name="node_a")
        node_b = DagNode(op_type="FakeModuleBB", name="node_b")
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_b1 = DagNode(op_type="FakeModuleBB", name="node_b1")

        node_a >> node_b >> node_a1 >> node_b1

        search_list = list(dag_to_search.search_sub_graph([node_b, node_b1, node_a, node_a1]))
        assert len(search_list) == 1
        assert search_list[0]["node_a"].name == "g3_a1"
        for sub_graph_in_network in search_list:
            assert sub_graph_in_network["node_a"].op_type == "FakeModuleAA"
            assert sub_graph_in_network["node_a1"].op_type == "FakeModuleAA"
            assert sub_graph_in_network["node_b"].op_type == "FakeModuleBB"
            assert sub_graph_in_network["node_b1"].op_type == "FakeModuleBB"

    def test_search_given_a_ba_d_when_any_pass(self, dag_to_search):
        node_a = DagNode(op_type="FakeModuleAA", name="node_a")
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_b = DagNode(op_type="FakeModuleBB", name="node_b")
        node_d = DagNode(op_type="FakeModuleDD", name="node_d")

        node_a >> node_b >> node_d
        node_a >> node_a1 >> node_d

        search_list = list(dag_to_search.search_sub_graph([node_b, node_d, node_a, node_a1]))
        assert len(search_list) == 1
        assert search_list[0]["node_a"].name == "g4_a1"
        for sub_graph_in_network in search_list:
            assert sub_graph_in_network["node_a"].op_type == "FakeModuleAA"
            assert sub_graph_in_network["node_a1"].op_type == "FakeModuleAA"
            assert sub_graph_in_network["node_b"].op_type == "FakeModuleBB"
            assert sub_graph_in_network["node_d"].op_type == "FakeModuleDD"

    def test_search_given_a_bb_d_when_any_pass(self, dag_to_search):
        node_a = DagNode(op_type="FakeModuleAA", name="node_a")
        node_b2 = DagNode(op_type="FakeModuleBB", name="node_b1")
        node_b1 = DagNode(op_type="FakeModuleBB", name="node_b2")
        node_d = DagNode(op_type="FakeModuleDD", name="node_d")

        node_a >> node_b1 >> node_d
        node_a >> node_b2 >> node_d

        search_list = list(dag_to_search.search_sub_graph([node_b2, node_d, node_a, node_b1]))
        assert len(search_list) == 2
        name_list = []
        for sub_graph in search_list:
            assert sub_graph["node_a"].op_type == "FakeModuleAA"
            assert sub_graph["node_b1"].op_type == "FakeModuleBB"
            assert sub_graph["node_b2"].op_type == "FakeModuleBB"
            assert sub_graph["node_d"].op_type == "FakeModuleDD"
            name_list.append(
                [
                    sub_graph["node_a"].name,
                    sub_graph["node_b1"].name,
                    sub_graph["node_b2"].name,
                    sub_graph["node_d"].name,
                ]
            )

        name_list.sort()
        assert name_list == [["g5_a1", "g5_b1", "g5_b2", "g5_d1"], ["g5_a1", "g5_b2", "g5_b1", "g5_d1"]]

    def test_search_given_a_bc_d_when_any_pass(self, dag_to_search):
        node_a = DagNode(op_type="FakeModuleAA", name="node_a")
        node_c = DagNode(op_type="FakeModuleCC", name="node_c")
        node_b = DagNode(op_type="FakeModuleBB", name="node_b")
        node_d = DagNode(op_type="FakeModuleDD", name="node_d")

        node_a >> node_b >> node_d
        node_a >> node_c >> node_d

        search_list = list(dag_to_search.search_sub_graph([node_b, node_d, node_a, node_c]))
        assert len(search_list) == 1
        assert search_list[0]["node_a"].name == "g6_a1"
        for sub_graph_in_network in search_list:
            assert sub_graph_in_network["node_a"].op_type == "FakeModuleAA"
            assert sub_graph_in_network["node_c"].op_type == "FakeModuleCC"
            assert sub_graph_in_network["node_b"].op_type == "FakeModuleBB"
            assert sub_graph_in_network["node_d"].op_type == "FakeModuleDD"

    def test_search_error_no_input_given_aaa_when_any_pass(self, dag_to_search):
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_a2 = DagNode(op_type="FakeModuleAA", name="node_a2")
        node_a3 = DagNode(op_type="FakeModuleAA", name="node_a3")

        node_a1 >> node_a2 >> node_a3
        node_a2 >> node_a1

        with pytest.raises(ValueError, match="You must have an input and an output."):
            list(dag_to_search.search_sub_graph([node_a1, node_a2, node_a3]))

    def test_search_error_no_output_given_aaa_when_any_pass(self, dag_to_search):
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_a2 = DagNode(op_type="FakeModuleAA", name="node_a2")
        node_a3 = DagNode(op_type="FakeModuleAA", name="node_a3")

        node_a1 >> node_a2 >> node_a3
        node_a3 >> node_a2

        with pytest.raises(ValueError, match="You must have an input and an output."):
            list(dag_to_search.search_sub_graph([node_a1, node_a2, node_a3]))

    def test_search_error_multi_output_given_aaa_when_any_pass(self, dag_to_search):
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_a2 = DagNode(op_type="FakeModuleAA", name="node_a2")
        node_a3 = DagNode(op_type="FakeModuleAA", name="node_a3")

        node_a1 >> node_a2
        node_a1 >> node_a3

        with pytest.raises(ValueError, match="There can only be one output node."):
            list(dag_to_search.search_sub_graph([node_a1, node_a2, node_a3]))

    def test_search_error_multi_input_given_aaa_when_any_pass(self, dag_to_search):
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_a2 = DagNode(op_type="FakeModuleAA", name="node_a2")
        node_a3 = DagNode(op_type="FakeModuleAA", name="node_a3")

        node_a1 >> node_a3
        node_a2 >> node_a3

        with pytest.raises(ValueError, match="There can only be one input node."):
            list(dag_to_search.search_sub_graph([node_a1, node_a2, node_a3]))

    def test_search_error_name_same_given_aaa_when_any_pass(self, dag_to_search):
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_a2 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_a3 = DagNode(op_type="FakeModuleAA", name="node_a3")

        node_a1 >> node_a2
        node_a2 >> node_a3

        with pytest.raises(ValueError, match="The node name must be different."):
            list(dag_to_search.search_sub_graph([node_a1, node_a2, node_a3]))

    def test_get_possible_sub_graph_given_none_when_any_pass(self):
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_a2 = DagNode(op_type=set(["FakeModuleAA", None]), name="node_a2")
        node_a3 = DagNode(op_type=set(["FakeModuleAA"]), name="node_a3")
        node_a4 = DagNode(op_type="FakeModuleAA", name="node_a4")

        node_a1 >> node_a2 >> node_a3 >> node_a4

        dag_fake = DirectedAcyclicGraph(None)

        possibles = list(dag_fake._get_possible_sub_graph([node_a1, node_a2, node_a3, node_a4]))

        possibles_name = [{node.name for node in possible} for possible in possibles]
        assert len(possibles) == 2
        assert {"node_a1", "node_a2", "node_a3", "node_a4"} in possibles_name
        assert {"node_a1", "node_a3", "node_a4"} in possibles_name

    def test_get_possible_sub_graph_given_none_none_when_any_pass(self):
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_a2 = DagNode(op_type=set(["FakeModuleAA", None]), name="node_a2")
        node_a3 = DagNode(op_type=set(["FakeModuleAA", None]), name="node_a3")
        node_a4 = DagNode(op_type="FakeModuleAA", name="node_a4")

        node_a1 >> node_a2 >> node_a4
        node_a1 >> node_a3 >> node_a4
        dag = DirectedAcyclicGraph(None)

        possibles = list(dag._get_possible_sub_graph([node_a1, node_a2, node_a3, node_a4]))

        possibles_name = [{node.name for node in possible} for possible in possibles]
        assert len(possibles) == 4
        assert {"node_a1", "node_a4"} in possibles_name
        assert {"node_a1", "node_a2", "node_a3", "node_a4"} in possibles_name
        assert {"node_a1", "node_a2", "node_a4"} in possibles_name
        assert {"node_a1", "node_a3", "node_a4"} in possibles_name

    def test_get_possible_sub_graph_given_not_none_when_any_pass(self):
        node_a1 = DagNode(op_type="FakeModuleAA", name="node_a1")
        node_a2 = DagNode(op_type=set(["FakeModuleAA"]), name="node_a2")
        node_a3 = DagNode(op_type=set(["FakeModuleAA"]), name="node_a3")
        node_a4 = DagNode(op_type="FakeModuleAA", name="node_a4")

        node_a1 >> node_a2 >> node_a4
        node_a1 >> node_a3 >> node_a4
        dag = DirectedAcyclicGraph(None)

        possibles = list(dag._get_possible_sub_graph([node_a1, node_a2, node_a3, node_a4]))

        possibles_name = [{node.name for node in possible} for possible in possibles]
        assert len(possibles) == 1
        assert {"node_a1", "node_a2", "node_a3", "node_a4"} in possibles_name
