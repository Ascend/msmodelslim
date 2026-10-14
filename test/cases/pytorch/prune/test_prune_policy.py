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
# pylint: disable=abstract-class-instantiated

from unittest.mock import MagicMock, patch
from typing import List
import pytest
import torch
from torch.nn import Conv2d, Linear
from torch.nn.modules import activation, pooling

from msmodelslim.pytorch.prune.prune_policy import (
    PrunePolicyGraph,
    ImportanceInfo,
    PrunePolicy,
    PrunePolicyGraphConv2D,
    PrunePolicyGraphLinear,
    DagNode,
)


def create_dag_node(name: str, node=None) -> DagNode:
    """创建DagNode Mock实例"""
    dag_node = MagicMock(spec=DagNode)
    dag_node.name = name
    if node is not None:
        dag_node.node = node
    return dag_node


def create_chn_eq_node(name: str) -> DagNode:
    """创建无weight属性的chn_eq节点"""
    node = MagicMock()
    # 确保weight属性不存在（避免MagicMock自动创建）
    if hasattr(node, "weight"):
        delattr(node, "weight")
    return create_dag_node(name, node)


# ------------------------------ PrunePolicyGraph 测试 ------------------------------
class TestPrunePolicyGraph:
    @staticmethod
    def test_get_module_attr_name_returns_module_classes():
        attr_names = list(PrunePolicyGraph.get_module_attr_name(activation))
        assert "ReLU" in attr_names
        assert "Sigmoid" in attr_names
        assert "Tanh" in attr_names

        attr_names = list(PrunePolicyGraph.get_module_attr_name(pooling))
        assert "MaxPool2d" in attr_names
        assert "AvgPool2d" in attr_names


# ------------------------------ ImportanceInfo 测试 ------------------------------
class TestImportanceInfo:
    @staticmethod
    def test_init_success_with_valid_params(valid_params):
        info = ImportanceInfo(**valid_params)
        assert info.importance == 0.5
        assert info.params == 100
        assert info.out_weight_idxes == [0, 1]
        assert info.policy == valid_params["policy"]
        assert info["in_weight_idxes"] == [2, 3]
        assert info["out_chn_idxes"] == [4, 5]

    @staticmethod
    def test_dict_inheritance(valid_params):
        info = ImportanceInfo(**valid_params)
        assert info["importance"] == 0.5
        info["extra_key"] = "extra_val"
        assert info["extra_key"] == "extra_val"

    @pytest.fixture
    def valid_params(self):
        return {
            "importance": 0.5,
            "params": 100,
            "out_weight_idxes": [0, 1],
            "policy": MagicMock(),
            "in_weight_idxes": [2, 3],
            "out_chn_idxes": [4, 5],
        }

    @pytest.mark.parametrize("none_param", ["importance", "params", "out_weight_idxes", "policy"])
    def test_init_raises_value_error_when_param_none(self, valid_params, none_param):
        valid_params[none_param] = None
        with pytest.raises(ValueError, match="all input param of ImportanceInfo must not None"):
            ImportanceInfo(**valid_params)


# ------------------------------ PrunePolicy 测试 ------------------------------
class TestPrunePolicy:
    @staticmethod
    def test_create_item_in_desc():
        desc = {}
        PrunePolicy.create_item_in_desc(desc, "node1", "output", [1, 3], 5)
        assert desc == {"node1": {"output": [3, ['-', 'x', '-', 'x', '-']]}}

        PrunePolicy.create_item_in_desc(desc, "node1", "input", [0], 5)
        # 使用 get() 方法安全访问，指定默认值避免 KeyError
        node1_desc = desc.get("node1", {})
        assert node1_desc.get("input") == [4, ['x', '-', '-', '-', '-']]

        PrunePolicy.create_item_in_desc(desc, "node1", "output", [2], 5)
        # 再次使用 get() 方法安全访问
        node1_desc = desc.get("node1", {})
        assert node1_desc.get("output") == [2, ['-', 'x', 'x', 'x', '-']]

    @staticmethod
    def test_write_desc_basic(mock_policy):
        with patch.object(PrunePolicy, "create_item_in_desc") as mock_create:
            importance_info = {"out_weight_idxes": [0], "in_weight_idxes": [1], "out_chn_idxes": [2]}
            desc = {}
            mock_policy.write_desc(desc, importance_info)

            assert mock_create.call_count == 2
            mock_create.assert_any_call(desc, "node_out", "output", [0], 10)
            mock_create.assert_any_call(desc, "node_in", "input", [1], 10)

    # 只读属性测试
    @staticmethod
    def test_name_property_returns_node_out_name(mock_policy):
        assert mock_policy.name == "node_out"
        with pytest.raises(AttributeError):
            mock_policy.name = "new_name"

    @staticmethod
    def test_node_out_property_returns_correct_instance(mock_policy):
        assert mock_policy.node_out.name == "node_out"
        with pytest.raises(AttributeError):
            mock_policy.node_out = MagicMock()

    @staticmethod
    def test_node_in_property_returns_correct_instance(mock_policy):
        assert mock_policy.node_in.name == "node_in"
        with pytest.raises(AttributeError):
            mock_policy.node_in = MagicMock()

    # importance_infos 懒加载测试
    @staticmethod
    def test_importance_infos_lazy_load_first_access(mock_policy):
        assert mock_policy._importance_infos is None
        infos = mock_policy.importance_infos
        assert len(infos) == 1
        assert infos[0].importance == 0.8

    @staticmethod
    def test_importance_infos_lazy_load_subsequent_access(mock_policy):
        first_infos = mock_policy.importance_infos
        with patch.object(mock_policy, "_calc_importance_infos") as mock_calc:
            second_infos = mock_policy.importance_infos
            mock_calc.assert_not_called()
            assert second_infos == first_infos

    @staticmethod
    def test_write_desc_with_none_idxes(mock_policy, importance_info_fixture):
        none_idxes_info = {key: None for key in importance_info_fixture.keys()}
        mock_policy._chn_eq1.node.weight = MagicMock()
        mock_policy._chn_eq2.node.weight = MagicMock()

        desc = {}
        with patch.object(PrunePolicy, "create_item_in_desc") as mock_create:
            mock_policy.write_desc(desc, none_idxes_info)
            assert mock_create.call_count == 4

    @staticmethod
    def test_write_desc_with_custom_out_chn(mock_policy, importance_info_fixture):
        mock_policy._out_chn = 20
        mock_policy._in_weight_dims = 15
        mock_policy._chn_eq1.node.weight = MagicMock()

        desc = {}
        with patch.object(PrunePolicy, "create_item_in_desc") as mock_create:
            mock_policy.write_desc(desc, importance_info_fixture)
            mock_create.assert_any_call(desc, "node_out", "output", [0, 2, 4], 20)
            mock_create.assert_any_call(desc, "node_in", "input", [1, 3], 15)
            mock_create.assert_any_call(desc, "chn_eq1", "input", [0, 1], 20)

    @pytest.fixture
    def mock_policy(self):
        class ConcretePrunePolicy(PrunePolicy):
            def __init__(self, node_out: DagNode, node_in: DagNode, chn_eq1: DagNode = None, chn_eq2: DagNode = None):
                super().__init__(node_out, node_in, chn_eq1, chn_eq2)
                self._out_chn = 10
                self._in_weight_dims = 10

            @property
            def out_chn(self) -> int:
                return self._out_chn

            @property
            def out_weight_dims(self) -> int:
                return super().out_weight_dims

            @property
            def in_weight_dims(self) -> int:
                return self._in_weight_dims

            def _calc_importance_infos(self) -> List:
                self._importance_infos = [MagicMock(spec="ImportanceInfo", importance=0.8)]

        node_out = create_dag_node("node_out")
        node_in = create_dag_node("node_in")
        chn_eq1 = create_chn_eq_node("chn_eq1")
        chn_eq2 = create_chn_eq_node("chn_eq2")

        return ConcretePrunePolicy(node_out, node_in, chn_eq1, chn_eq2)

    @pytest.fixture
    def importance_info_fixture(self):
        return {"out_weight_idxes": [0, 2, 4], "in_weight_idxes": [1, 3], "out_chn_idxes": [0, 1]}

    @pytest.mark.parametrize(
        "chn_eq1_has_weight, chn_eq2_has_weight, expected_call_count",
        [(False, False, 2), (True, False, 3), (False, True, 3), (True, True, 4)],
    )
    def test_write_desc_all_branches(
        self, mock_policy, importance_info_fixture, chn_eq1_has_weight, chn_eq2_has_weight, expected_call_count
    ):
        # 配置weight属性
        if chn_eq1_has_weight:
            mock_policy._chn_eq1.node.weight = MagicMock()
        else:
            if hasattr(mock_policy._chn_eq1.node, "weight"):
                delattr(mock_policy._chn_eq1.node, "weight")

        if chn_eq2_has_weight:
            mock_policy._chn_eq2.node.weight = MagicMock()
        else:
            if hasattr(mock_policy._chn_eq2.node, "weight"):
                delattr(mock_policy._chn_eq2.node, "weight")

        desc = {}
        with patch.object(PrunePolicy, "create_item_in_desc") as mock_create:
            mock_policy.write_desc(desc, importance_info_fixture)
            assert mock_create.call_count == expected_call_count

            mock_create.assert_any_call(desc, "node_out", "output", [0, 2, 4], mock_policy.out_weight_dims)
            mock_create.assert_any_call(desc, "node_in", "input", [1, 3], mock_policy.in_weight_dims)

            if chn_eq1_has_weight:
                mock_create.assert_any_call(desc, "chn_eq1", "input", [0, 1], mock_policy.out_chn)
            if chn_eq2_has_weight:
                mock_create.assert_any_call(desc, "chn_eq2", "input", [0, 1], mock_policy.out_chn)


# ------------------------------ PrunePolicyGraphConv2D 测试 ------------------------------
class TestPrunePolicyGraphConv2D:
    @staticmethod
    def test_init_success(mock_conv_nodes, mock_importance_eval):
        policy = PrunePolicyGraphConv2D(mock_conv_nodes["graph"], mock_importance_eval)
        assert policy._conv_out_node == mock_conv_nodes["conv_out"]
        assert policy._conv_in_node == mock_conv_nodes["conv_in"]
        assert policy.importance_eval == mock_importance_eval
        assert policy._importance_infos is None

    @staticmethod
    def test_out_chn(mock_conv_nodes):
        policy = PrunePolicyGraphConv2D(mock_conv_nodes["graph"], MagicMock())
        assert policy.out_chn == 8

    @staticmethod
    def test_in_weight_dims_valid_groups(mock_conv_nodes):
        policy = PrunePolicyGraphConv2D(mock_conv_nodes["graph"], MagicMock())
        assert policy.in_weight_dims == 4

    @staticmethod
    def test_in_weight_dims_zero_groups(mock_conv_nodes):
        mock_conv_nodes["conv_in"].groups = 0
        policy = PrunePolicyGraphConv2D(mock_conv_nodes["graph"], MagicMock())
        with pytest.raises(ValueError, match="Check whether the node is a normal Conv2d operator."):
            _ = policy.in_weight_dims

    @staticmethod
    def test_get_search_graph():
        graph = PrunePolicyGraphConv2D.get_search_graph()
        assert len(graph) == 4
        assert graph[0].op_type == "Conv2d" and graph[0].name == "conv_out"
        assert graph[-1].op_type == "Conv2d" and graph[-1].name == "conv_in"

    @staticmethod
    def test_calc_importance_infos_zero_groups(mock_conv_nodes):
        mock_conv_nodes["conv_out"].groups = 0
        policy = PrunePolicyGraphConv2D(mock_conv_nodes["graph"], MagicMock())
        with pytest.raises(ValueError, match="Check whether the node is a normal Conv2d operator."):
            policy._calc_importance_infos()

    @staticmethod
    def test_calc_importance_infos_raise_error_when_groups_zero(mock_conv_nodes, mock_importance_eval):
        policy = PrunePolicyGraphConv2D(mock_conv_nodes["graph"], mock_importance_eval)
        policy._out_chn = 8

        mock_conv_nodes["conv_in"].groups = 0
        with pytest.raises(ValueError, match="Check whether the node is a normal Conv2d operator."):
            policy._calc_importance_infos()

        mock_conv_nodes["conv_in"].groups = 2
        mock_conv_nodes["conv_out"].groups = 0
        with pytest.raises(ValueError, match="Check whether the node is a normal Conv2d operator."):
            policy._calc_importance_infos()

    @staticmethod
    def test_calc_importance_infos_return_empty_when_groups_not_divisible(mock_conv_nodes, mock_importance_eval):
        policy = PrunePolicyGraphConv2D(mock_conv_nodes["graph"], mock_importance_eval)
        policy._out_chn = 8
        mock_conv_nodes["conv_in"].groups = 3
        mock_conv_nodes["conv_out"].groups = 2

        policy._calc_importance_infos()
        assert len(policy._importance_infos) == 0
        mock_importance_eval.assert_not_called()

    @staticmethod
    def test_calc_importance_infos_out_groups_bigger(mock_conv_nodes, mock_importance_eval):
        conv_out = mock_conv_nodes["conv_out"]
        conv_in = mock_conv_nodes["conv_in"]
        conv_out.groups = 4
        conv_in.groups = 2
        out_chn = 8
        policy = PrunePolicyGraphConv2D(mock_conv_nodes["graph"], mock_importance_eval)
        policy._out_chn = out_chn

        expected_out_weight_idxes_list = [[0, 2, 4, 6], [1, 3, 5, 7]]
        expected_in_weight_idxes_list = [[0, 2], [1, 3]]
        out_weight_nelement = 4 * 3 * 3 * 3
        in_weight_nelement = 8 * 2 * 3 * 3
        expected_params = out_weight_nelement + in_weight_nelement

        policy._calc_importance_infos()

        assert len(policy._importance_infos) == 2
        assert mock_importance_eval.call_count == 2

        for idx in range(2):
            info = policy._importance_infos[idx]
            assert info.importance == 0.75
            assert info.params == expected_params
            assert info.policy == policy
            assert info.out_weight_idxes == expected_out_weight_idxes_list[idx]
            assert info["in_weight_idxes"] == expected_in_weight_idxes_list[idx]
            assert info["out_chn_idxes"] == expected_out_weight_idxes_list[idx]

            out_weight = conv_out.weight.data[expected_out_weight_idxes_list[idx]]
            assert out_weight.shape == (4, 3, 3, 3)
            in_weight = conv_in.weight.data[:, expected_in_weight_idxes_list[idx]]
            assert in_weight.shape == (8, 2, 3, 3)

    @staticmethod
    def test_calc_importance_infos_in_groups_bigger(mock_conv_nodes, mock_importance_eval):
        conv_out = mock_conv_nodes["conv_out"]
        conv_in = mock_conv_nodes["conv_in"]
        conv_out.groups = 2
        conv_in.groups = 4
        out_chn = 8
        policy = PrunePolicyGraphConv2D(mock_conv_nodes["graph"], mock_importance_eval)
        policy._out_chn = out_chn

        expected_in_weight_idxes_list = [[0], [1]]
        expected_out_weight_idxes_list = [[0, 2], [1, 3]]
        expected_out_chn_list = [[0, 2, 4, 6], [1, 3, 5, 7]]
        out_weight_nelement = 2 * 3 * 3 * 3
        in_weight_nelement = 8 * 1 * 3 * 3
        expected_params = out_weight_nelement + in_weight_nelement

        policy._calc_importance_infos()

        assert len(policy._importance_infos) == 2
        assert mock_importance_eval.call_count == 2

        for idx in range(2):
            info = policy._importance_infos[idx]
            assert info.importance == 0.75
            assert info.params == expected_params
            assert info.policy == policy
            assert info["in_weight_idxes"] == expected_in_weight_idxes_list[idx]
            assert info.out_weight_idxes == expected_out_weight_idxes_list[idx]
            assert info["out_chn_idxes"] == expected_out_chn_list[idx]

            out_weight = conv_out.weight.data[expected_out_weight_idxes_list[idx]]
            assert out_weight.shape == (2, 3, 3, 3)
            in_weight = conv_in.weight.data[:, expected_in_weight_idxes_list[idx]]
            assert in_weight.shape == (8, 1, 3, 3)

    @staticmethod
    def test_calc_importance_infos_lazy_load(mock_conv_nodes, mock_importance_eval):
        policy = PrunePolicyGraphConv2D(mock_conv_nodes["graph"], mock_importance_eval)
        policy._out_chn = 8
        mock_conv_nodes["conv_out"].groups = 2
        mock_conv_nodes["conv_in"].groups = 2

        assert policy._importance_infos is None
        mock_infos = [MagicMock(spec=ImportanceInfo)]

        with patch.object(policy, "_calc_importance_infos") as mock_calc:

            def mock_calc_logic():
                policy._importance_infos = mock_infos

            mock_calc.side_effect = mock_calc_logic

            infos = policy.importance_infos
            mock_calc.assert_called_once()
            assert infos == mock_infos

        with patch.object(policy, "_calc_importance_infos") as mock_calc:
            infos2 = policy.importance_infos
            mock_calc.assert_not_called()
            assert infos2 == mock_infos

    @pytest.fixture
    def mock_conv_nodes(self):
        # Conv2d配置
        conv_out = MagicMock(spec=Conv2d, out_channels=8, groups=2)
        conv_out.weight = MagicMock(data=torch.randn(8, 3, 3, 3))

        conv_in = MagicMock(spec=Conv2d, in_channels=8, groups=2)
        conv_in.weight = MagicMock(data=torch.randn(8, 4, 3, 3))

        # DagNode配置
        dag_out = create_dag_node("conv_out", conv_out)
        dag_in = create_dag_node("conv_in", conv_in)
        chn_eq1 = create_dag_node("chn_eq1")
        chn_eq2 = create_dag_node("chn_eq2")

        return {
            "graph": {"conv_out": dag_out, "conv_in": dag_in, "chn_eq1": chn_eq1, "chn_eq2": chn_eq2},
            "conv_out": conv_out,
            "conv_in": conv_in,
            "dag_out": dag_out,
            "dag_in": dag_in,
        }

    @pytest.fixture
    def mock_importance_eval(self):
        return MagicMock(return_value=0.75)

    @pytest.mark.parametrize("missing_key", ["conv_out", "conv_in"])
    def test_init_raises_value_error_when_missing_node(self, mock_conv_nodes, missing_key):
        mock_conv_nodes["graph"].pop(missing_key)
        with pytest.raises(ValueError, match="inner Error, search nothing"):
            PrunePolicyGraphConv2D(mock_conv_nodes["graph"], MagicMock())


# ------------------------------ PrunePolicyGraphLinear 测试 ------------------------------
class TestPrunePolicyGraphLinear:
    @staticmethod
    def test_init_success(mock_linear_nodes):
        importance_eval = MagicMock(return_value=0.7)
        policy = PrunePolicyGraphLinear(mock_linear_nodes["graph"], importance_eval)
        assert policy.linear_out_node == mock_linear_nodes["linear_out"]
        assert policy.linear_in_node == mock_linear_nodes["linear_in"]

    @staticmethod
    def test_out_chn(mock_linear_nodes):
        policy = PrunePolicyGraphLinear(mock_linear_nodes["graph"], MagicMock())
        assert policy.out_chn == 5

    @staticmethod
    def test_get_search_graph():
        graph = PrunePolicyGraphLinear.get_search_graph()
        assert len(graph) == 4
        assert graph[0].op_type == "Linear" and graph[0].name == "linear_out"
        assert graph[-1].op_type == "Linear" and graph[-1].name == "linear_in"

    @pytest.fixture
    def mock_linear_nodes(self):
        linear_out = MagicMock(spec=Linear, out_features=5, weight=MagicMock(data=torch.randn(5, 10)))
        linear_in = MagicMock(spec=Linear, in_features=5, weight=MagicMock(data=torch.randn(10, 5)))

        dag_out = create_dag_node("linear_out", linear_out)
        dag_in = create_dag_node("linear_in", linear_in)
        chn_eq1 = create_dag_node("chn_eq1")
        chn_eq2 = create_dag_node("chn_eq2")

        return {
            "graph": {"linear_out": dag_out, "linear_in": dag_in, "chn_eq1": chn_eq1, "chn_eq2": chn_eq2},
            "linear_out": linear_out,
            "linear_in": linear_in,
        }

    @pytest.mark.parametrize("missing_key", ["linear_out", "linear_in"])
    def test_init_raises_value_error_when_missing_node(self, mock_linear_nodes, missing_key):
        mock_linear_nodes["graph"].pop(missing_key)
        with pytest.raises(ValueError, match="inner Error, search nothing"):
            PrunePolicyGraphLinear(mock_linear_nodes["graph"], MagicMock())
