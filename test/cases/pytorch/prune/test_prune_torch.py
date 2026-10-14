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
# pylint: disable=consider-using-from-import

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import torch
import torch.nn as nn
from ascend_utils.pytorch.dag.dag_torch_hook import DagTorchHook
from msmodelslim.pytorch.prune.prune_policy import ImportanceInfo
from msmodelslim.pytorch.prune.prune_torch import PruneTorch, chn_weight


class TestPruneTorch(unittest.TestCase):
    def setUp(self):
        self.simple_model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

        self.test_input = torch.randn(1, 3, 32, 32)

        self.conv1_node = Mock(name="conv1")
        self.conv2_node = Mock(name="conv2")
        self.linear_node = Mock(name="linear")

        self.mock_dag = MagicMock(spec=DagTorchHook)
        self.mock_dag.network = self.simple_model
        self.mock_dag.get_params.return_value = 1000
        self.mock_dag.search_nodes_by_op_type.return_value = [self.conv1_node, self.conv2_node, self.linear_node]
        self.mock_dag.dag_node_list = [self.conv1_node, self.conv2_node, self.linear_node]
        self.mock_dag.get_node_by_name.return_value = Mock()

    def test_chn_weight_function_returns_correct_value(self):
        test_tensor = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        result = chn_weight(test_tensor)
        expected = (1.0 + 2.0 + 3.0 + 4.0) / 4
        self.assertEqual(result, expected)

    def test_init_with_torch_module_creates_dag_network(self):
        prune_torch = PruneTorch(self.simple_model, self.test_input)
        self.assertEqual(prune_torch.network, self.simple_model)
        self.assertIsInstance(prune_torch.dag, DagTorchHook)

    def test_init_with_dag_torch_hook_sets_correct_attributes(self):
        prune_torch = PruneTorch(self.mock_dag)
        self.assertEqual(prune_torch.network, self.simple_model)
        self.assertEqual(prune_torch.dag, self.mock_dag)

    def test_init_with_invalid_network_raises_value_error(self):
        with self.assertRaises(ValueError):
            PruneTorch("invalid_network")

    def test_set_importance_evaluation_function_with_valid_function(self):
        prune_torch = PruneTorch(self.mock_dag)

        def test_eval_func(x):
            return torch.sum(x).item()

        result = prune_torch.set_importance_evaluation_function(test_eval_func)
        self.assertEqual(result, prune_torch)
        self.assertEqual(prune_torch._importance_evaluation_function, test_eval_func)

    def test_set_importance_evaluation_function_with_invalid_type_raises_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        with self.assertRaises(ValueError):
            prune_torch.set_importance_evaluation_function("not_callable")

    def test_set_node_reserved_ratio_with_valid_value(self):
        prune_torch = PruneTorch(self.mock_dag)
        result = prune_torch.set_node_reserved_ratio(0.7)
        self.assertEqual(result, prune_torch)
        self.assertEqual(prune_torch._node_reserved_ratio, 0.7)

    def test_set_node_reserved_ratio_with_invalid_type_raises_type_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        with self.assertRaises(TypeError):
            prune_torch.set_node_reserved_ratio("0.5")

    def test_set_node_reserved_ratio_with_out_of_range_value_raises_value_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        with self.assertRaises(ValueError):
            prune_torch.set_node_reserved_ratio(1.0)
        with self.assertRaises(ValueError):
            prune_torch.set_node_reserved_ratio(0.0)

    def test_analysis_with_invalid_reserved_ratio_type_raises_type_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        with self.assertRaises(TypeError):
            prune_torch.analysis(reserved_ratio="0.5")

    def test_analysis_with_abnormal_reserved_ratio_logs_warning(self):
        prune_torch = PruneTorch(self.mock_dag)
        prune_torch.analysis(reserved_ratio=0.2)

    @patch.object(PruneTorch, '_preprocess_un_prune_list')
    @patch.object(PruneTorch, '_assessment_importance_conv')
    @patch.object(PruneTorch, '_assessment_importance_linear')
    def test_analysis_method_calls_required_functions(self, mock_assess_linear, mock_assess_conv, mock_preprocess):
        prune_torch = PruneTorch(self.mock_dag)
        mock_preprocess.return_value = set()
        left_params, desc = prune_torch.analysis(reserved_ratio=0.8)
        mock_preprocess.assert_called_once()
        mock_assess_conv.assert_called_once()
        mock_assess_linear.assert_called_once()

    def test_analysis_method_sorts_importance_info(self):
        prune_torch = PruneTorch(self.mock_dag)

        with (
            patch.object(prune_torch, '_preprocess_un_prune_list') as mock_preprocess,
            patch.object(prune_torch, '_assessment_importance_conv') as mock_conv,
            patch.object(prune_torch, '_assessment_importance_linear') as mock_linear,
        ):
            mock_preprocess.return_value = set()

            mock_policy1 = Mock()
            mock_policy1.name = "conv1"
            mock_policy1.out_weight_dims = 16
            mock_policy1.write_desc = Mock()

            mock_policy2 = Mock()
            mock_policy2.name = "conv2"
            mock_policy2.out_weight_dims = 32
            mock_policy2.write_desc = Mock()

            importance_info1 = Mock(spec=ImportanceInfo)
            importance_info1.importance = 0.5
            importance_info1.params = 100
            importance_info1.policy = mock_policy1
            importance_info1.out_weight_idxes = [0, 1, 2]

            importance_info2 = Mock(spec=ImportanceInfo)
            importance_info2.importance = 0.2
            importance_info2.params = 200
            importance_info2.policy = mock_policy2
            importance_info2.out_weight_idxes = [0, 1]

            def fill_importance_list(lst, un_prune_set):
                lst.extend([importance_info1, importance_info2])

            def empty_importance_linear(lst, un_prune_set):
                pass

            mock_conv.side_effect = fill_importance_list
            mock_linear.side_effect = empty_importance_linear

            left_params, desc = prune_torch.analysis(reserved_ratio=0.8)
            self.assertIsInstance(left_params, (int, float))
            self.assertIsInstance(desc, dict)

    @patch.object(PruneTorch, '_preprocess_un_prune_list')
    def test_preprocess_un_prune_list_with_none_returns_default_set(self, mock_method):
        prune_torch = PruneTorch(self.mock_dag)
        expected_result = {"conv1", "linear"}
        mock_method.return_value = expected_result
        result = prune_torch._preprocess_un_prune_list(None)
        self.assertEqual(result, expected_result)

    def test_preprocess_un_prune_list_with_invalid_type_raises_type_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        with self.assertRaises(TypeError):
            prune_torch._preprocess_un_prune_list("not_a_list")

    @patch.object(PruneTorch, '_preprocess_un_prune_list')
    def test_preprocess_un_prune_list_with_mixed_int_and_str(self, mock_method):
        prune_torch = PruneTorch(self.mock_dag)
        expected_result = {"conv1", "custom_layer"}
        mock_method.return_value = expected_result
        result = prune_torch._preprocess_un_prune_list([0, "custom_layer"])
        self.assertEqual(result, expected_result)

    def test_preprocess_un_prune_list_with_invalid_index_issues_warning(self):
        prune_torch = PruneTorch(self.mock_dag)
        prune_torch._preprocess_un_prune_list([10])

    def test_preprocess_un_prune_list_with_invalid_element_type_raises_value_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        with self.assertRaises(ValueError):
            prune_torch._preprocess_un_prune_list([1.5])

    def test_prune_by_desc_with_invalid_desc_type_raises_type_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        with self.assertRaises(TypeError):
            prune_torch.prune_by_desc("invalid_desc")

    @patch('ascend_utils.common.security.type.check_dict_character')
    def test_prune_by_desc_with_empty_network_raises_value_error(self, mock_check):
        prune_torch = PruneTorch(self.mock_dag)
        self.mock_dag.get_params.return_value = 0
        with self.assertRaises(ValueError):
            prune_torch.prune_by_desc({"layer1": {"input": (10, "----------")}})

    @patch('ascend_utils.common.security.type.check_dict_character')
    def test_prune_by_desc_calls_prune_one_node_for_each_desc(self, mock_check):
        prune_torch = PruneTorch(self.mock_dag)

        mock_node1 = Mock()
        mock_node2 = Mock()

        def get_node_side_effect(name):
            return {"node1": mock_node1, "node2": mock_node2}.get(name)

        self.mock_dag.get_node_by_name.side_effect = get_node_side_effect

        desc = {"node1": {"input": (5, "-----")}, "node2": {"output": (8, "--------")}}

        with patch.object(prune_torch, '_prune_one_node') as mock_prune:
            prune_torch.prune_by_desc(desc)
            self.assertEqual(mock_prune.call_count, 2)
            mock_prune.assert_has_calls([call(mock_node1, desc["node1"]), call(mock_node2, desc["node2"])])

    def test_check_desc_input_output_with_valid_input(self):
        prune_torch = PruneTorch(self.mock_dag)
        prune_torch._check_desc_input_output((10, "----------"))

    def test_check_desc_input_output_with_invalid_input_raises_value_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        with self.assertRaises(ValueError):
            prune_torch._check_desc_input_output(('string', 0, 1))

    def test_prune_conv2d_updates_parameters_correctly(self):
        prune_torch = PruneTorch(self.mock_dag)
        conv = nn.Conv2d(10, 20, 3)
        node_input = (5, "-----+++++")
        node_output = (8, "--------++++++++++++")

        prune_torch._prune_conv2d(conv, node_input, node_output)

        self.assertEqual(conv.in_channels, 5)
        self.assertEqual(conv.out_channels, 8)
        self.assertEqual(conv.weight.shape[0], 8)
        self.assertEqual(conv.weight.shape[1], 5)

    def test_prune_conv2d_with_depthwise_convolution(self):
        prune_torch = PruneTorch(self.mock_dag)
        conv = nn.Conv2d(10, 20, 3, groups=2)
        node_input = (4, "----++++++")
        node_output = (8, "--------++++++++++++")

        with self.assertRaises(ValueError):
            prune_torch._prune_conv2d(conv, node_input, node_output)

    def test_prune_conv2d_with_invalid_channels_raises_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        conv = nn.Conv2d(10, 20, 3)
        with self.assertRaises(ValueError):
            prune_torch._prune_conv2d(conv, (15, "---------------"), (8, "--------"))

    def test_prune_linear_updates_parameters_correctly(self):
        prune_torch = PruneTorch(self.mock_dag)
        linear = nn.Linear(20, 10)
        node_input = (15, "---------------+++++")
        node_output = (8, "--------++")

        prune_torch._prune_linear(linear, node_input, node_output)

        self.assertEqual(linear.in_features, 15)
        self.assertEqual(linear.out_features, 8)
        self.assertEqual(linear.weight.shape[0], 8)
        self.assertEqual(linear.weight.shape[1], 15)

    def test_prune_linear_with_invalid_features_raises_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        linear = nn.Linear(20, 10)
        with self.assertRaises(ValueError):
            prune_torch._prune_linear(linear, (25, "-------------------------"), (8, "--------"))

    def test_prune_batchnorm_updates_parameters_correctly(self):
        prune_torch = PruneTorch(self.mock_dag)
        batchnorm = nn.BatchNorm2d(10)
        node_feature_num = (6, "------++++")

        prune_torch._prune_batchnorm(batchnorm, node_feature_num)

        self.assertEqual(batchnorm.num_features, 6)
        self.assertEqual(batchnorm.running_mean.shape[0], 6)
        self.assertEqual(batchnorm.running_var.shape[0], 6)
        self.assertEqual(batchnorm.weight.shape[0], 6)
        self.assertEqual(batchnorm.bias.shape[0], 6)

    def test_prune_batchnorm_with_layernorm_does_nothing(self):
        prune_torch = PruneTorch(self.mock_dag)
        layer_norm = nn.LayerNorm(10)
        prune_torch._prune_batchnorm(layer_norm, (5, "-----"))

    def test_prune_one_node_with_conv2d_calls_correct_method(self):
        prune_torch = PruneTorch(self.mock_dag)
        mock_dag_node = Mock()
        mock_dag_node.node = nn.Conv2d(10, 20, 3)
        node_desc = {"input": (5, "-----+++++"), "output": (8, "--------++++++++++++")}

        with patch.object(prune_torch, '_prune_conv2d') as mock_prune_conv:
            prune_torch._prune_one_node(mock_dag_node, node_desc)
            mock_prune_conv.assert_called_once()

    def test_prune_one_node_with_linear_calls_correct_method(self):
        prune_torch = PruneTorch(self.mock_dag)
        mock_dag_node = Mock()
        mock_dag_node.node = nn.Linear(20, 10)
        node_desc = {"input": (15, "---------------+++++"), "output": (8, "--------++")}

        with patch.object(prune_torch, '_prune_linear') as mock_prune_linear:
            prune_torch._prune_one_node(mock_dag_node, node_desc)
            mock_prune_linear.assert_called_once_with(mock_dag_node.node, node_desc["input"], node_desc["output"])

    def test_prune_one_node_with_batchnorm_calls_correct_method(self):
        prune_torch = PruneTorch(self.mock_dag)
        mock_dag_node = Mock()
        mock_dag_node.node = nn.BatchNorm2d(10)
        node_desc = {"input": (6, "------++++")}

        with patch.object(prune_torch, '_prune_batchnorm') as mock_prune_bn:
            prune_torch._prune_one_node(mock_dag_node, node_desc)
            mock_prune_bn.assert_called_once_with(mock_dag_node.node, node_desc["input"])

    def test_prune_one_node_with_unknown_layer_type_does_nothing(self):
        prune_torch = PruneTorch(self.mock_dag)

        class CustomLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_features = 10

        mock_dag_node = Mock()
        mock_dag_node.node = CustomLayer()
        node_desc = {"input": (5, "-----")}

        with (
            patch.object(prune_torch, '_prune_conv2d') as mock_conv,
            patch.object(prune_torch, '_prune_linear') as mock_linear,
            patch.object(prune_torch, '_prune_batchnorm') as mock_bn,
        ):
            prune_torch._prune_one_node(mock_dag_node, node_desc)
            mock_conv.assert_not_called()
            mock_linear.assert_not_called()
            mock_bn.assert_called_once()

    def test_prune_one_node_with_invalid_node_raises_value_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        with self.assertRaises(ValueError):
            prune_torch._prune_one_node(None, {})

    def test_prune_one_node_with_invalid_desc_raises_type_error(self):
        prune_torch = PruneTorch(self.mock_dag)
        mock_dag_node = Mock()
        with self.assertRaises(TypeError):
            prune_torch._prune_one_node(mock_dag_node, "invalid_desc")

    def test_prune_one_node_with_missing_input_output_uses_defaults(self):
        prune_torch = PruneTorch(self.mock_dag)
        mock_dag_node = Mock()
        mock_dag_node.node = nn.Conv2d(10, 20, 3)
        node_desc = {"output": (8, "--------++++++++++++")}

        with patch.object(prune_torch, '_prune_conv2d') as mock_prune_conv:
            prune_torch._prune_one_node(mock_dag_node, node_desc)
            args, kwargs = mock_prune_conv.call_args
            self.assertEqual(args[1], (-1, []))
            self.assertEqual(args[2], node_desc["output"])

    @patch.object(PruneTorch, 'analysis')
    def test_prune_method_calls_analysis_and_prune_by_desc(self, mock_analysis):
        prune_torch = PruneTorch(self.mock_dag)
        expected_desc = {"layer1": {"input": (10, "----------")}}
        mock_analysis.return_value = (500, expected_desc)

        with patch.object(prune_torch, 'prune_by_desc') as mock_prune_by_desc:
            result_desc = prune_torch.prune(reserved_ratio=0.5)
            mock_analysis.assert_called_once_with(0.5, None)
            mock_prune_by_desc.assert_called_once_with(expected_desc)
            self.assertEqual(result_desc, expected_desc)

    def test_network_and_dag_properties_return_correct_values(self):
        prune_torch = PruneTorch(self.mock_dag)
        self.assertEqual(prune_torch.network, self.simple_model)
        self.assertEqual(prune_torch.dag, self.mock_dag)

    @patch.object(PruneTorch, '_assessment_importance_conv')
    def test_assessment_importance_conv_skips_unprune_nodes(self, mock_assess):
        prune_torch = PruneTorch(self.mock_dag)
        mock_sub_graph = Mock()
        self.mock_dag.search_sub_graph.return_value = [mock_sub_graph]

        mock_policy = Mock()
        mock_policy.node_out = self.conv1_node
        mock_policy.node_in = self.conv2_node
        mock_policy.importance_infos = [Mock()]

        with patch('msmodelslim.pytorch.prune.prune_policy.PrunePolicyGraphConv2D') as mock_policy_class:
            mock_policy_class.return_value = mock_policy
            unprune_set = {"conv1"}
            all_importance = []
            prune_torch._assessment_importance_conv(all_importance, unprune_set)
            self.assertEqual(len(all_importance), 0)

    @patch.object(PruneTorch, '_assessment_importance_linear')
    def test_assessment_importance_linear_skips_unprune_nodes(self, mock_assess):
        prune_torch = PruneTorch(self.mock_dag)
        mock_sub_graph = Mock()
        self.mock_dag.search_sub_graph.return_value = [mock_sub_graph]

        mock_policy = Mock()
        mock_policy.node_out = self.linear_node
        mock_policy.node_in = self.conv2_node
        mock_policy.importance_infos = [Mock()]

        with patch('msmodelslim.pytorch.prune.prune_policy.PrunePolicyGraphLinear') as mock_policy_class:
            mock_policy_class.return_value = mock_policy
            unprune_set = {"linear"}
            all_importance = []
            prune_torch._assessment_importance_linear(all_importance, unprune_set)
            self.assertEqual(len(all_importance), 0)


if __name__ == '__main__':
    unittest.main()
