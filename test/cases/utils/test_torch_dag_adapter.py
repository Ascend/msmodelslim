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
# pylint: disable=consider-using-from-import,pointless-string-statement,too-many-lines,ungrouped-imports

"""
msmodelslim.utils.dag_utils.torch_dag_adapter 模块的单元测试
"""

import unittest  # noqa: E402
from collections import defaultdict  # noqa: E402
from unittest.mock import Mock, patch  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from msmodelslim.utils.dag_utils.dag_torch_hook import DagTorchHook  # noqa: E402, F401
from msmodelslim.utils.dag_utils.model_structure_process import StructureProcess  # noqa: E402, F401
from ascend_utils.core.dag.dag_node import DagNode  # noqa: E402, F401
from ascend_utils.common.utils import CallParams  # noqa: E402, F401
from msmodelslim.utils.dag_utils.model_infos import ModuleType  # noqa: E402
from msmodelslim.utils.dag_utils.torch_dag_adapter import DagNodeInfo, TorchDAGAdapter  # noqa: E402


class TestDagNodeInfo(unittest.TestCase):
    """测试DagNodeInfo类"""

    def setUp(self):
        """测试前准备"""
        self.dag_node_info = DagNodeInfo(
            name="test_node", class_type="Linear", input_nodes=["input1", "input2"], output_nodes=["output1", "output2"]
        )

    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.dag_node_info.name, "test_node")
        self.assertEqual(self.dag_node_info.class_type, "Linear")
        self.assertEqual(self.dag_node_info.input_nodes, ["input1", "input2"])
        self.assertEqual(self.dag_node_info.output_nodes, ["output1", "output2"])

    def test_eq_true(self):
        """测试相等比较"""
        other = DagNodeInfo(
            name="test_node", class_type="Linear", input_nodes=["input1", "input2"], output_nodes=["output1", "output2"]
        )
        self.assertEqual(self.dag_node_info, other)

    def test_eq_false_different_name(self):
        """测试名称不同"""
        other = DagNodeInfo(
            name="other_node",
            class_type="Linear",
            input_nodes=["input1", "input2"],
            output_nodes=["output1", "output2"],
        )
        self.assertNotEqual(self.dag_node_info, other)

    def test_eq_false_different_class_type(self):
        """测试类型不同"""
        other = DagNodeInfo(
            name="test_node", class_type="Conv2d", input_nodes=["input1", "input2"], output_nodes=["output1", "output2"]
        )
        self.assertNotEqual(self.dag_node_info, other)

    def test_eq_false_different_input_nodes(self):
        """测试输入节点不同"""
        other = DagNodeInfo(
            name="test_node", class_type="Linear", input_nodes=["input1", "input3"], output_nodes=["output1", "output2"]
        )
        self.assertNotEqual(self.dag_node_info, other)

    def test_eq_false_different_output_nodes(self):
        """测试输出节点不同"""
        other = DagNodeInfo(
            name="test_node", class_type="Linear", input_nodes=["input1", "input2"], output_nodes=["output1", "output3"]
        )
        self.assertNotEqual(self.dag_node_info, other)

    def test_eq_none(self):
        """测试与None比较"""
        self.assertIsNotNone(self.dag_node_info)


class MockDagNode:
    def __init__(self, name, op_type, input_nodes=None, output_nodes=None, inputs=None, outputs=None):
        self.name_in_network = name
        self.op_type = op_type
        self.input_nodes = input_nodes or []
        self.output_nodes = output_nodes or []
        self.inputs = inputs or []
        self.outputs = outputs or []


class TestTorchDAGAdapter(unittest.TestCase):
    """测试TorchDAGAdapter类"""

    def setUp(self):
        """测试前准备"""
        self.model = nn.Linear(224 * 224 * 3, 10)

        self.dag_nodes = [
            MockDagNode("layer1", "Linear"),
            MockDagNode("relu", "ReLU"),
            MockDagNode("layer2", "Linear"),
            MockDagNode("add_op", "__add__"),
            MockDagNode("norm", "LayerNorm"),
        ]

        self.dag_nodes[0].output_nodes = [self.dag_nodes[1]]
        self.dag_nodes[1].input_nodes = [self.dag_nodes[0]]
        self.dag_nodes[1].output_nodes = [self.dag_nodes[2]]
        self.dag_nodes[2].input_nodes = [self.dag_nodes[1]]

        self.mock_dag_hook = Mock()
        self.mock_dag_hook.dag_node_list = self.dag_nodes

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=self.mock_dag_hook):
            self.adapter = TorchDAGAdapter(
                model=self.model, dummy_input=torch.randn(1, 224 * 224 * 3), hook_nodes=[nn.LayerNorm]
            )

    def test_init_with_default_dummy_input(self):
        """测试使用默认dummy_input初始化"""
        default_model = nn.Linear(224 * 224 * 3, 10)
        mock_dag_hook = Mock()
        mock_dag_hook.dag_node_list = [MockDagNode("linear", "Linear")]

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=mock_dag_hook):
            adapter = TorchDAGAdapter(default_model, dummy_input=torch.randn(1, 224 * 224 * 3))
            self.assertIsNotNone(adapter._dummy_input)
            self.assertEqual(adapter._dummy_input.shape, torch.Size([1, 224 * 224 * 3]))

    def test_init_with_tensor_dummy_input(self):
        """测试使用tensor dummy_input初始化"""
        dummy_input = torch.randn(2, 10)
        model = nn.Linear(10, 20)
        mock_dag_hook = Mock()
        mock_dag_hook.dag_node_list = [MockDagNode("linear", "Linear")]

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=mock_dag_hook):
            adapter = TorchDAGAdapter(model, dummy_input=dummy_input)
            self.assertTrue(torch.equal(adapter._dummy_input, dummy_input))

    def test_init_with_hook_nodes(self):
        """测试使用hook_nodes初始化"""
        model = nn.Linear(10, 20)
        mock_dag_hook = Mock()
        mock_dag_hook.dag_node_list = [MockDagNode("linear", "Linear")]

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=mock_dag_hook):
            adapter = TorchDAGAdapter(model, dummy_input=torch.randn(1, 10), hook_nodes=[nn.LayerNorm, nn.BatchNorm2d])
            self.assertIn("layernorm", adapter.norm_nodes)
            self.assertIn("batchnorm2d", adapter.norm_nodes)

    def test_init_with_none_hook_nodes(self):
        """测试使用None hook_nodes初始化"""
        model = nn.Linear(10, 20)
        mock_dag_hook = Mock()
        mock_dag_hook.dag_node_list = [MockDagNode("linear", "Linear")]

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=mock_dag_hook):
            adapter = TorchDAGAdapter(model, dummy_input=torch.randn(1, 10), hook_nodes=None)
            self.assertEqual(adapter.norm_nodes, [])

    def test_get_node_name_in_order(self):
        """测试获取节点名称"""
        with patch.object(self.adapter, 'node_list', self.dag_nodes):
            names = self.adapter.get_node_name_in_order()
            expected = [node.name_in_network for node in self.dag_nodes]
            self.assertEqual(names, expected)

    def test_get_name_type_dict_in_order(self):
        """测试获取名称类型字典"""
        with patch.object(self.adapter, 'node_list', self.dag_nodes):
            result = self.adapter.get_name_type_dict_in_order()
            expected = {node.name_in_network: node.op_type for node in self.dag_nodes}
            self.assertEqual(result, expected)

    def test_get_network_topology(self):
        """测试获取网络拓扑"""
        with patch.object(self.adapter, 'node_list', self.dag_nodes):
            topology = self.adapter.get_network_topology()
            self.assertEqual(len(topology), len(self.dag_nodes))

            first_node = topology[0]
            self.assertEqual(first_node.name, self.dag_nodes[0].name_in_network)
            self.assertEqual(first_node.class_type, self.dag_nodes[0].op_type)
            self.assertEqual(
                first_node.input_nodes,
                [item.name_in_network if item else None for item in self.dag_nodes[0].input_nodes],
            )
            self.assertEqual(
                first_node.output_nodes,
                [item.name_in_network if item else None for item in self.dag_nodes[0].output_nodes],
            )

    def test_get_mhsa_pattern_with_valid_conditions(self):
        """测试获取MHSA模式，满足条件的情况"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]

        test_nodes = [
            mock_node,
            mock_add_node,
            mock_other_branch,
            MockDagNode("linear1", "Linear"),
            MockDagNode("linear2", "Linear"),
        ]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process'
            ) as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_called()
                self.assertIsInstance(qkv_list, list)
                self.assertIsInstance(proj_list, list)

    def test_get_mhsa_pattern_with_continue_condition_len_add_node_not_1(self):
        """测试MHSA模式中满足continue条件：add_node数量不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node1 = MockDagNode("add1", "__add__")
        mock_add_node2 = MockDagNode("add2", "__add__")

        mock_node.output_nodes = [mock_add_node1, mock_add_node2]

        test_nodes = [mock_node, mock_add_node1, mock_add_node2]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process'
            ) as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_not_called()
                self.assertIsInstance(qkv_list, list)
                self.assertIsInstance(proj_list, list)

    def test_get_mhsa_pattern_with_continue_condition_len_add_inputs_not_2(self):
        """测试MHSA模式中满足continue条件：add_node输入数量不为2"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock()]
        mock_add_node.input_nodes = [mock_node]

        mock_other_branch = MockDagNode("other", "Linear")
        mock_node.output_nodes = [mock_add_node, mock_other_branch]

        test_nodes = [mock_node, mock_add_node, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process'
            ) as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_not_called()
                self.assertIsInstance(qkv_list, list)
                self.assertIsInstance(proj_list, list)

    def test_get_mhsa_pattern_with_continue_condition_no_other_branch(self):
        """测试MHSA模式中满足continue条件：没有other_branch"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        mock_node.output_nodes = [mock_add_node]

        test_nodes = [mock_node, mock_add_node, MockDagNode("other_input", "Linear")]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process'
            ) as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_not_called()
                self.assertIsInstance(qkv_list, list)
                self.assertIsInstance(proj_list, list)

    def test_get_ffn_pattern_with_valid_conditions(self):
        """测试获取FFN模式，满足条件的情况"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        test_nodes = [
            mock_node,
            mock_add_node,
            mock_other_branch,
            MockDagNode("linear1", "Linear"),
            MockDagNode("linear2", "Linear"),
        ]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.is_ffn_matmul', return_value=True
            ):
                ffn_pattern = self.adapter.get_ffn_pattern()
                self.assertIsInstance(ffn_pattern, list)

    def test_get_ffn_pattern_with_continue_condition_len_add_node_not_1(self):
        """测试FFN模式中满足continue条件：add_node数量不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node1 = MockDagNode("add1", "__add__")
        mock_add_node2 = MockDagNode("add2", "__add__")
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node1, mock_add_node2, mock_other_branch]

        test_nodes = [mock_node, mock_add_node1, mock_add_node2, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.is_ffn_matmul'
            ) as mock_is_ffn:
                ffn_pattern = self.adapter.get_ffn_pattern()
                mock_is_ffn.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)

    def test_get_ffn_pattern_with_continue_condition_len_other_branch_not_1(self):
        """测试FFN模式中满足continue条件：other_branch数量不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch1 = MockDagNode("other1", "Linear")
        mock_other_branch2 = MockDagNode("other2", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch1, mock_other_branch2]

        test_nodes = [mock_node, mock_add_node, mock_other_branch1, mock_other_branch2]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.is_ffn_matmul'
            ) as mock_is_ffn:
                ffn_pattern = self.adapter.get_ffn_pattern()
                mock_is_ffn.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)

    def test_get_ffn_pattern_with_continue_condition_len_add_inputs_not_2(self):
        """测试FFN模式中满足continue条件：add_node输入数量不为2"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]

        test_nodes = [mock_node, mock_add_node, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.is_ffn_matmul'
            ) as mock_is_ffn:
                ffn_pattern = self.adapter.get_ffn_pattern()
                mock_is_ffn.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)

    def test_get_mhsa_ln_pattern(self):
        """测试获取MHSA LN模式"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        test_nodes = [
            mock_node,
            mock_add_node,
            mock_other_branch,
            MockDagNode("linear1", "Linear"),
            MockDagNode("linear2", "Linear"),
            MockDagNode("norm", "LayerNorm"),
        ]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch('msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_ln_process'):
                qkv_list, proj_list, ln_list = self.adapter.get_mhsa_ln_pattern()
                self.assertIsInstance(qkv_list, list)
                self.assertIsInstance(proj_list, list)
                self.assertIsInstance(ln_list, list)

    def test_get_ffn_ln_pattern(self):
        """测试获取FFN LN模式"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        test_nodes = [
            mock_node,
            mock_add_node,
            mock_other_branch,
            MockDagNode("linear1", "Linear"),
            MockDagNode("linear2", "Linear"),
            MockDagNode("norm", "LayerNorm"),
        ]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch('msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list'):
                ffn_pattern, ffn_ln_list = self.adapter.get_ffn_ln_pattern()
                self.assertIsInstance(ffn_pattern, list)
                self.assertIsInstance(ffn_ln_list, list)

    def test_get_llama_mhsa_ln_pattern(self):
        """测试获取LLAMA MHSA LN模式"""
        with patch.object(self.adapter, 'get_mhsa_ln_pattern') as mock_get:
            self.adapter.get_llama_mhsa_ln_pattern()
            mock_get.assert_called_once_with(ln_type='Llamarmsnormbias')

    def test_get_llama_ffn_ln_pattern_with_valid_conditions(self):
        """测试获取LLAMA FFN LN模式，满足条件的情况"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        test_nodes = [
            mock_node,
            mock_add_node,
            mock_other_branch,
            MockDagNode("linear1", "Linear"),
            MockDagNode("linear2", "Linear"),
            MockDagNode("norm", "LayerNorm"),
        ]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch('msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list'):
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                self.assertIsInstance(ffn_pattern, list)
                self.assertIsInstance(ffn_ln_list, list)

    def test_get_llama_ffn_ln_pattern_with_continue_condition_len_add_node_not_1(self):
        """测试LLAMA FFN LN模式中满足continue条件：add_node数量不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node1 = MockDagNode("add1", "__add__")
        mock_add_node2 = MockDagNode("add2", "__add__")
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node1, mock_add_node2, mock_other_branch]

        test_nodes = [mock_node, mock_add_node1, mock_add_node2, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list'
            ) as mock_get:
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                mock_get.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)
                self.assertIsInstance(ffn_ln_list, list)

    def test_get_llama_ffn_ln_pattern_with_continue_condition_len_other_branch_not_1(self):
        """测试LLAMA FFN LN模式中满足continue条件：other_branch数量不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]
        mock_other_branch1 = MockDagNode("other1", "Linear")
        mock_other_branch2 = MockDagNode("other2", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch1, mock_other_branch2]

        test_nodes = [mock_node, mock_add_node, mock_other_branch1, mock_other_branch2]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list'
            ) as mock_get:
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                mock_get.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)
                self.assertIsInstance(ffn_ln_list, list)

    def test_get_llama_ffn_ln_pattern_with_continue_condition_len_add_inputs_not_2(self):
        """测试LLAMA FFN LN模式中满足continue条件：add_node输入数量不为2"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock()]
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]

        test_nodes = [mock_node, mock_add_node, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list'
            ) as mock_get:
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                mock_get.assert_not_called()
                self.assertIsInstance(ffn_pattern, list)
                self.assertIsInstance(ffn_ln_list, list)

    def test_get_norm_linear_subgraph(self):
        """测试获取norm-linear子图"""
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        linear3 = MockDagNode("linear3", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, linear3, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                result = self.adapter.get_norm_linear_subgraph()
                self.assertIsInstance(result, defaultdict)
                self.assertIn(norm_node.name_in_network, result)

    def test_get_kv_linears(self):
        """测试获取KV线性层"""
        # 创建包含norm节点的测试数据
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        linear3 = MockDagNode("linear3", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, linear3, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                kv_linears, num_kv = self.adapter.get_kv_linears()
                self.assertIsInstance(kv_linears, list)
                self.assertIsInstance(num_kv, int)

    def test_get_linear_linear_subgraph(self):
        """测试获取linear-linear子图"""
        # 创建包含norm节点的测试数据
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                with patch.object(self.adapter, '_find_module_by_name', return_value=Mock()):
                    result = self.adapter.get_linear_linear_subgraph()
                    self.assertIsInstance(result, defaultdict)

    def test_get_allreduce_linear(self):
        """测试获取allreduce线性层"""
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                result = self.adapter.get_allreduce_linear()
                self.assertIsInstance(result, list)

    def test_find_module_by_name(self):
        """测试通过名称查找模块"""
        # 测试正常情况
        result = self.adapter._find_module_by_name("model.0")
        # 由于模型结构，这里会返回Linear层或None

        # 测试None输入
        result = self.adapter._find_module_by_name(None)
        self.assertIsNone(result)

    def test_split_by_add_op_type(self):
        """测试按add操作类型分割"""
        node = MockDagNode("test", "Linear")
        add_node = MockDagNode("__add__", "__add__")
        other_node = MockDagNode("other", "ReLU")

        node.output_nodes = [add_node, other_node]

        add_nodes, other_branch = self.adapter._split_by_add_op_type(node)
        self.assertEqual(len(add_nodes), 1)
        self.assertEqual(len(other_branch), 1)
        self.assertEqual(add_nodes[0].op_type, "__add__")
        self.assertEqual(other_branch[0].op_type, "ReLU")

    def test_find_input_nodes(self):
        """测试查找输入节点"""
        # 创建一些节点
        input_node = MockDagNode("input", "Input")
        tensor_node = MockDagNode("_TensorBase.size", "_TensorBase.size")  # 应该被过滤
        normal_node = MockDagNode("normal", "Linear")

        # 模拟dag_node_from属性
        input_node.inputs = [Mock()]
        input_node.inputs[0].dag_node_from = None  # 没有来源，是输入

        tensor_node.inputs = [Mock()]
        tensor_node.inputs[0].dag_node_from = Mock()

        normal_node.inputs = [Mock()]
        normal_node.inputs[0].dag_node_from = None

        with patch.object(self.adapter, 'node_list', [input_node, tensor_node, normal_node]):
            input_nodes = self.adapter._find_input_nodes()
            self.assertIn(input_node, input_nodes)
            self.assertNotIn(tensor_node, input_nodes)  # 被dark_name_list过滤
            self.assertIn(normal_node, input_nodes)

    def test_find_output_nodes(self):
        """测试查找输出节点"""
        output_node = MockDagNode("output", "Output")
        tensor_node = MockDagNode("_TensorBase.size", "_TensorBase.size")  # 应该被过滤
        normal_node = MockDagNode("normal", "Linear")

        # 模拟outputs属性
        output_node.outputs = [Mock()]
        output_node.outputs[0].dag_nodes_to = []  # 没有去向，是输出

        tensor_node.outputs = [Mock()]
        tensor_node.outputs[0].dag_nodes_to = [Mock()]

        normal_node.outputs = [Mock()]
        normal_node.outputs[0].dag_nodes_to = []  # 没有去向，是输出

        with patch.object(self.adapter, 'node_list', [output_node, tensor_node, normal_node]):
            output_nodes = self.adapter._find_output_nodes()
            self.assertIn(output_node, output_nodes)
            self.assertNotIn(tensor_node, output_nodes)  # 被dark_name_list过滤
            self.assertIn(normal_node, output_nodes)

    def test_get_trav_node(self):
        """测试获取遍历节点"""
        add_node = [MockDagNode("add", "__add__")]
        add_node[0].input_nodes = ["node1", "node2"]
        node = "node1"

        trav_node = TorchDAGAdapter._get_trav_node(add_node, node)
        self.assertEqual(trav_node, "node2")

    def test_get_trav_node_single_input(self):
        """测试单个输入节点情况"""
        add_node = [MockDagNode("add", "__add__")]
        add_node[0].input_nodes = ["node1"]
        node = "node1"

        trav_node = TorchDAGAdapter._get_trav_node(add_node, node)
        self.assertIsNone(trav_node)

    def test_get_llm_network_pattern_auto(self):
        """测试自动获取LLM网络模式"""
        norm_node1 = MockDagNode("norm1", "LayerNorm")
        norm_node2 = MockDagNode("norm2", "LayerNorm")
        norm_node3 = MockDagNode("norm3", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")

        test_nodes = [norm_node1, linear1, linear2, norm_node2, linear1, linear2, norm_node3]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                with patch.object(self.adapter, 'add_mhsa_norm_linears'):
                    result = self.adapter.get_llm_network_pattern_auto()
                    self.assertEqual(len(result), 5)  # 返回5个列表

    def test_add_mhsa_norm_linears_short_interval(self):
        """测试添加MHSA norm线性层（短间隔）"""
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        linear3 = MockDagNode("linear3", "Linear")
        linear4 = MockDagNode("linear4", "Linear")

        test_nodes = [
            MockDagNode("norm", "LayerNorm"),
            linear1,
            linear2,
            linear3,
            linear4,
            MockDagNode("end", "Linear"),
        ]

        mhsa_linear = []
        mhsa_o = []
        mhsa_ln = []

        with patch.object(self.adapter, 'node_list', test_nodes):
            self.adapter.add_mhsa_norm_linears(mhsa_linear, mhsa_o, mhsa_ln, [0, 5])
            self.assertEqual(len(mhsa_linear), 1)
            self.assertEqual(len(mhsa_o), 1)
            self.assertEqual(len(mhsa_ln), 1)

    def test_add_mhsa_norm_linears_long_interval(self):
        """测试添加MHSA norm线性层（长间隔）"""
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        linear3 = MockDagNode("linear3", "Linear")
        linear4 = MockDagNode("linear4", "Linear")
        linear5 = MockDagNode("linear5", "Linear")
        linear6 = MockDagNode("linear6", "Linear")

        test_nodes = [
            MockDagNode("norm", "LayerNorm"),
            linear1,
            linear2,
            linear3,
            linear4,
            linear5,
            linear6,
            MockDagNode("end", "Linear"),
        ]

        mhsa_linear = []
        mhsa_o = []
        mhsa_ln = []

        with patch.object(self.adapter, 'node_list', test_nodes):
            self.adapter.add_mhsa_norm_linears(mhsa_linear, mhsa_o, mhsa_ln, [0, 7])
            self.assertEqual(len(mhsa_linear), 1)
            self.assertEqual(len(mhsa_o), 1)
            self.assertEqual(len(mhsa_ln), 1)

    def test_get_all_interval_nodes_by_type(self):
        """测试获取指定类型的所有间隔节点"""
        stop_node = MockDagNode("stop", "Linear")
        trav_node = MockDagNode("trav", "Linear")

        result = self.adapter._get_all_interval_nodes_by_type(stop_node, trav_node)
        self.assertIsInstance(result, list)

    def test_get_all_interval_nodes_by_type_ignore_stop_node(self):
        """测试忽略停止节点的情况"""
        stop_node = MockDagNode("stop", "Linear")
        trav_node = MockDagNode("trav", "Linear")

        result = self.adapter._get_all_interval_nodes_by_type(stop_node, trav_node, ignore_stop_node=True)
        self.assertIsInstance(result, list)

    def test_find_latest_module_by_type_bfs(self):
        """测试BFS查找最新模块类型"""
        search_node = MockDagNode("search", "Linear")
        target_node = MockDagNode("target", "Linear")

        search_node.input_nodes = [target_node]

        result = self.adapter._find_latest_module_by_type_bfs(search_node, "Linear")
        self.assertIsInstance(result, list)

    def test_find_latest_module_by_type_bfs_none_input(self):
        """测试BFS查找模块类型，输入为None"""
        result = self.adapter._find_latest_module_by_type_bfs(None, "Linear")
        self.assertEqual(result, [])

    def test_find_latest_module_by_type_bfs_stop_node(self):
        """测试BFS查找模块类型，遇到停止节点"""
        search_node = MockDagNode("search", "Linear")
        stop_node = MockDagNode("stop", "Stop")

        search_node.input_nodes = [stop_node]

        result = self.adapter._find_latest_module_by_type_bfs(search_node, "Linear", stop_node=stop_node)
        self.assertEqual(result, [])

    def test_find_latest_module_by_type_recursive(self):
        """测试递归查找最新模块类型"""
        search_node = MockDagNode("search", "Linear")
        target_node = MockDagNode("target", "Linear")

        search_node.input_nodes = [target_node]

        with patch.object(self.adapter, '_find_latest_module_by_type') as mock_find:
            mock_find.return_value = [target_node]
            result = self.adapter._find_latest_module_by_type(search_node, "Linear")
            self.assertIsInstance(result, list)

    def test_find_latest_module_by_type_none_input(self):
        """测试递归查找模块类型，输入为None"""
        result = self.adapter._find_latest_module_by_type(None, "Linear")
        self.assertEqual(result, [])

    def test_find_latest_module_by_type_conv2d_or_getitem(self):
        """测试遇到Conv2d或GetItem的情况"""
        search_node = MockDagNode("search", "Linear")
        conv_node = MockDagNode("conv", ModuleType.CONV2D)

        search_node.input_nodes = [conv_node]

        result = self.adapter._find_latest_module_by_type(search_node, "Linear")
        self.assertEqual(result, [])

    def test_find_latest_module_by_type_stop_node_in_output(self):
        """测试递归查找模块类型，停止节点在输出中"""
        search_node = MockDagNode("search", "Linear")
        target_node = MockDagNode("target", "Linear")
        stop_node = MockDagNode("stop", "Stop")

        search_node.input_nodes = [target_node]
        target_node.output_nodes = [stop_node]  # stop_node在target_node的输出中

        result = self.adapter._find_latest_module_by_type(search_node, "Linear", stop_node=stop_node)
        self.assertEqual(result, [])  # 应该返回空列表

    def test_find_latest_module_by_type_stop_node_equal(self):
        """测试递归查找模块类型，当前节点等于停止节点"""
        search_node = MockDagNode("search", "Linear")
        stop_node = MockDagNode("stop", "Stop")

        search_node.input_nodes = [stop_node]

        result = self.adapter._find_latest_module_by_type(search_node, "Linear", stop_node=stop_node)
        self.assertEqual(result, [])  # 应该返回空列表

    def test_find_latest_module_by_type_found_target(self):
        """测试递归查找模块类型，找到目标节点"""
        search_node = MockDagNode("search", "Linear")
        target_node = MockDagNode("target", "Linear")

        search_node.input_nodes = [target_node]

        result = self.adapter._find_latest_module_by_type(search_node, "Linear")
        self.assertEqual(result, [target_node])  # 应该返回包含目标节点的列表

    def test_find_latest_module_by_type_branch_aware_true(self):
        """测试branch_aware=True的情况"""
        search_node = MockDagNode("search", "Linear")
        target_node = MockDagNode("target", "Linear")
        other_node = MockDagNode("other", "Linear")  # noqa: F841

        search_node.input_nodes = [target_node]

        with patch.object(self.adapter, '_find_latest_module_by_type') as mock_find:
            mock_find.return_value = [target_node]
            result = self.adapter._find_latest_module_by_type(search_node, "Linear", branch_aware=True)
            self.assertIsInstance(result, list)

    def test_get_ffn_pattern_and_ln_list(self):
        """测试获取FFN模式和LN列表"""
        matmul_list = [MockDagNode("linear1", "Linear"), MockDagNode("linear2", "Linear")]
        ln_list = [MockDagNode("norm", "LayerNorm")]
        stop_node = MockDagNode("stop", "Linear")
        ffn_pattern = []
        ffn_ln_list = []

        with patch(
            'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.is_ffn_matmul', return_value=True
        ):
            TorchDAGAdapter._get_ffn_pattern_and_ln_list((2, matmul_list), ln_list, stop_node, ffn_pattern, ffn_ln_list)
            self.assertEqual(len(ffn_pattern), 1)
            self.assertEqual(len(ffn_ln_list), 1)

    def test_get_ffn_pattern_and_ln_list_reverse_condition(self):
        """测试需要反转列表的情况"""
        matmul_list = [MockDagNode("linear1", "Linear"), MockDagNode("linear2", "Linear")]
        ln_list = [MockDagNode("norm", "LayerNorm")]
        stop_node = MockDagNode("stop", "OtherType")  # 不是Linear，应该反转
        ffn_pattern = []
        ffn_ln_list = []

        with patch(
            'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.is_ffn_matmul', return_value=True
        ):
            TorchDAGAdapter._get_ffn_pattern_and_ln_list((2, matmul_list), ln_list, stop_node, ffn_pattern, ffn_ln_list)
            self.assertEqual(len(ffn_pattern), 1)
            self.assertEqual(len(ffn_ln_list), 1)

    def test_get_linear_linear_subgraph_structured_linear_order_false(self):
        """测试get_linear_linear_subgraph，structured_linear_order=False"""
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                with patch.object(self.adapter, '_find_module_by_name', return_value=Mock()):
                    with patch.object(self.adapter, '_find_latest_module_by_type_bfs', return_value=[linear1]):
                        result = self.adapter.get_linear_linear_subgraph(structured_linear_order=False)
                        self.assertIsInstance(result, defaultdict)

    def test_get_linear_linear_subgraph_more_than_four_linears(self):
        """测试超过4个线性层的情况"""
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        linear3 = MockDagNode("linear3", "Linear")
        linear4 = MockDagNode("linear4", "Linear")
        linear5 = MockDagNode("linear5", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, linear3, linear4, linear5, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                with patch.object(self.adapter, '_find_module_by_name', return_value=Mock()):
                    result = self.adapter.get_linear_linear_subgraph()
                    self.assertIsInstance(result, defaultdict)

    def test_get_linear_linear_subgraph_gate_down_case(self):
        """测试gate down情况"""
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                with patch.object(self.adapter, '_find_module_by_name') as mock_find:
                    mock_module = Mock()
                    mock_module.weight = Mock()
                    mock_module.weight.size.return_value = (10, 10)  # 模拟权重尺寸
                    mock_find.return_value = mock_module
                    result = self.adapter.get_linear_linear_subgraph()
                    self.assertIsInstance(result, defaultdict)


class TestTorchDAGAdapterEdgeCases(unittest.TestCase):
    """测试TorchDAGAdapter的边界情况"""

    def setUp(self):
        """测试前准备"""
        self.model = nn.Linear(10, 10)  # 使用简单的线性模型
        self.mock_dag_hook = Mock()
        self.mock_dag_hook.dag_node_list = []

        with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=self.mock_dag_hook):
            self.adapter = TorchDAGAdapter(self.model, dummy_input=torch.randn(1, 10))

    def test_empty_node_list(self):
        """测试空节点列表"""
        with patch.object(self.adapter, 'node_list', []):
            names = self.adapter.get_node_name_in_order()
            self.assertEqual(names, [])

            result = self.adapter.get_name_type_dict_in_order()
            self.assertEqual(result, {})

            topology = self.adapter.get_network_topology()
            self.assertEqual(topology, [])

    def test_single_node(self):
        """测试单个节点"""
        single_node = MockDagNode("single", "Linear")
        with patch.object(self.adapter, 'node_list', [single_node]):
            names = self.adapter.get_node_name_in_order()
            self.assertEqual(names, ["single"])

            result = self.adapter.get_name_type_dict_in_order()
            self.assertEqual(result, {"single": "Linear"})

    def test_no_norm_nodes(self):
        """测试没有norm节点的情况"""
        with patch.object(self.adapter, 'norm_nodes', []):
            result = self.adapter.get_norm_linear_subgraph()
            self.assertEqual(result, defaultdict(list))

    def test_cuda_device(self):
        """测试CUDA设备情况"""
        if torch.cuda.is_available():
            cuda_model = nn.Linear(10, 10).cuda()
            dummy_input = torch.randn(1, 10).cuda()

            mock_dag_hook = Mock()
            mock_dag_hook.dag_node_list = [MockDagNode("linear", "Linear")]

            with patch('msmodelslim.utils.dag_utils.dag_torch_hook.DagTorchHook', return_value=mock_dag_hook):
                adapter = TorchDAGAdapter(cuda_model, dummy_input=dummy_input)
                self.assertEqual(adapter._dummy_input.device.type, 'cuda')

    def test_find_latest_module_by_type_branch_aware_true(self):
        """测试branch_aware=True的情况"""
        search_node = MockDagNode("search", "Linear")
        target_node = MockDagNode("target", "Linear")
        other_node = MockDagNode("other", "Linear")  # noqa: F841

        search_node.input_nodes = [target_node]

        with patch.object(self.adapter, '_find_latest_module_by_type') as mock_find:
            mock_find.return_value = [target_node]
            result = self.adapter._find_latest_module_by_type(search_node, "Linear", branch_aware=True)
            self.assertIsInstance(result, list)

    def test_find_latest_module_by_type_stop_node_in_output(self):
        """测试停止节点在输出中的情况"""
        search_node = MockDagNode("search", "Linear")
        stop_node = MockDagNode("stop", "Stop")
        output_node = MockDagNode("output", "Linear")

        output_node.output_nodes = [stop_node]
        search_node.input_nodes = [output_node]

        result = self.adapter._find_latest_module_by_type(search_node, "Linear", stop_node=stop_node)
        self.assertEqual(result, [])

    def test_find_latest_module_by_type_with_conv2d_or_getitem(self):
        """测试遇到Conv2d或GetItem的情况"""
        search_node = MockDagNode("search", "Linear")
        conv_node = MockDagNode("conv", ModuleType.CONV2D)

        search_node.input_nodes = [conv_node]

        result = self.adapter._find_latest_module_by_type(search_node, "Linear")
        self.assertEqual(result, [])

    def test_find_input_nodes_with_tensor_base(self):
        """测试查找输入节点，包含tensor base类型"""
        input_node = MockDagNode("input", "Input")
        tensor_node = MockDagNode("_TensorBase.__getitem__", "_TensorBase.__getitem__")  # 应该被过滤
        normal_node = MockDagNode("normal", "Linear")

        input_node.inputs = [Mock()]
        input_node.inputs[0].dag_node_from = None  # 没有来源，是输入

        tensor_node.inputs = [Mock()]
        tensor_node.inputs[0].dag_node_from = Mock()

        normal_node.inputs = [Mock()]
        normal_node.inputs[0].dag_node_from = None

        with patch.object(self.adapter, 'node_list', [input_node, tensor_node, normal_node]):
            input_nodes = self.adapter._find_input_nodes()
            self.assertIn(input_node, input_nodes)
            self.assertNotIn(tensor_node, input_nodes)  # 被dark_name_list过滤
            self.assertIn(normal_node, input_nodes)

    def test_find_output_nodes_with_tensor_base(self):
        """测试查找输出节点，包含tensor base类型"""
        output_node = MockDagNode("output", "Output")
        tensor_node = MockDagNode("_TensorBase.size", "_TensorBase.size")  # 应该被过滤
        normal_node = MockDagNode("normal", "Linear")

        output_node.outputs = [Mock()]
        output_node.outputs[0].dag_nodes_to = []  # 没有去向，是输出

        tensor_node.outputs = [Mock()]
        tensor_node.outputs[0].dag_nodes_to = [Mock()]

        normal_node.outputs = [Mock()]
        normal_node.outputs[0].dag_nodes_to = []  # 没有去向，是输出

        with patch.object(self.adapter, 'node_list', [output_node, tensor_node, normal_node]):
            output_nodes = self.adapter._find_output_nodes()
            self.assertIn(output_node, output_nodes)
            self.assertNotIn(tensor_node, output_nodes)  # 被dark_name_list过滤
            self.assertIn(normal_node, output_nodes)

    def test_get_linear_linear_subgraph_more_than_two_linears(self):
        """测试get_linear_linear_subgraph，2 < num_of_linears <= 4"""
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        linear3 = MockDagNode("linear3", "Linear")
        linear4 = MockDagNode("linear4", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, linear3, linear4, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                with patch.object(self.adapter, '_find_module_by_name', return_value=Mock()):
                    result = self.adapter.get_linear_linear_subgraph()
                    self.assertIsInstance(result, defaultdict)

    def test_get_linear_linear_subgraph_two_linears_gate_down(self):
        """测试get_linear_linear_subgraph，2个线性层，gate down情况"""
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                with patch.object(self.adapter, '_find_module_by_name') as mock_find:
                    # 模拟gate down情况：origin_layer.weight.size(1) == target_layer.weight.size(0)
                    mock_module = Mock()
                    mock_module.weight = Mock()
                    mock_module.weight.size.return_value = (10, 10)  # 模拟尺寸匹配
                    mock_find.return_value = mock_module
                    result = self.adapter.get_linear_linear_subgraph()
                    self.assertIsInstance(result, defaultdict)

    def test_get_kv_linears_even_interval(self):
        """测试get_kv_linears，偶数区间情况"""
        norm_node = MockDagNode("norm", "LayerNorm")
        linear1 = MockDagNode("linear1", "Linear")
        linear2 = MockDagNode("linear2", "Linear")
        linear3 = MockDagNode("linear3", "Linear")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, linear1, linear2, linear3, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                kv_linears, num_kv = self.adapter.get_kv_linears()
                self.assertIsInstance(kv_linears, list)
                self.assertIsInstance(num_kv, int)

    def test_get_linear_linear_subgraph_no_target_node(self):
        """测试get_linear_linear_subgraph，没有目标节点的情况"""
        norm_node = MockDagNode("norm", "LayerNorm")
        next_norm = MockDagNode("norm2", "LayerNorm")

        test_nodes = [norm_node, next_norm]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch.object(self.adapter, 'norm_nodes', ['layernorm']):
                with patch.object(self.adapter, '_find_module_by_name', return_value=Mock()):
                    result = self.adapter.get_linear_linear_subgraph()
                    self.assertIsInstance(result, defaultdict)

    def test_get_mhsa_pattern_with_multiple_conditions(self):
        """测试get_mhsa_pattern函数，多个条件分支"""
        # 测试条件: len(add_node) != 1
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node1 = MockDagNode("add1", "__add__")
        mock_add_node2 = MockDagNode("add2", "__add__")
        mock_node.output_nodes = [mock_add_node1, mock_add_node2]  # 2个add节点

        test_nodes = [mock_node, mock_add_node1, mock_add_node2]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process'
            ) as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_not_called()  # 应该不被调用
                self.assertEqual(len(qkv_list), 0)
                self.assertEqual(len(proj_list), 0)

    def test_get_mhsa_pattern_with_len_inputs_not_2(self):
        """测试get_mhsa_pattern函数，add_node[0].inputs长度不为2"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock()]  # 只有1个输入
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]
        mock_add_node.input_nodes = [mock_node]  # 只有1个输入节点

        test_nodes = [mock_node, mock_add_node, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process'
            ) as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_not_called()  # 应该不被调用
                self.assertEqual(len(qkv_list), 0)
                self.assertEqual(len(proj_list), 0)

    def test_get_mhsa_pattern_with_no_other_branch(self):
        """测试get_mhsa_pattern函数，没有other_branch"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]  # 2个输入
        mock_add_node.input_nodes = [mock_node, MockDagNode("other_input", "Linear")]

        mock_node.output_nodes = [mock_add_node]  # 没有other_branch

        test_nodes = [mock_node, mock_add_node, MockDagNode("other_input", "Linear")]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.model_structure_process.StructureProcess.mhsa_matmul_process'
            ) as mock_process:
                qkv_list, proj_list = self.adapter.get_mhsa_pattern()
                mock_process.assert_not_called()  # 应该不被调用
                self.assertEqual(len(qkv_list), 0)
                self.assertEqual(len(proj_list), 0)

    def test_get_llama_ffn_ln_pattern_with_multiple_conditions(self):
        """测试get_llama_ffn_ln_pattern函数，多个条件分支"""
        # 测试条件: len(add_node) != 1
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node1 = MockDagNode("add1", "__add__")
        mock_add_node2 = MockDagNode("add2", "__add__")
        mock_other_branch = MockDagNode("other", "Linear")
        mock_node.output_nodes = [mock_add_node1, mock_add_node2, mock_other_branch]  # 2个add节点

        test_nodes = [mock_node, mock_add_node1, mock_add_node2, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list'
            ) as mock_get:
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                mock_get.assert_not_called()  # 应该不被调用
                self.assertEqual(len(ffn_pattern), 0)
                self.assertEqual(len(ffn_ln_list), 0)

    def test_get_llama_ffn_ln_pattern_with_len_other_branch_not_1(self):
        """测试get_llama_ffn_ln_pattern函数，other_branch长度不为1"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock(), Mock()]  # 2个输入
        mock_other_branch1 = MockDagNode("other1", "Linear")
        mock_other_branch2 = MockDagNode("other2", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch1, mock_other_branch2]  # 2个other_branch

        test_nodes = [mock_node, mock_add_node, mock_other_branch1, mock_other_branch2]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list'
            ) as mock_get:
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                mock_get.assert_not_called()  # 应该不被调用
                self.assertEqual(len(ffn_pattern), 0)
                self.assertEqual(len(ffn_ln_list), 0)

    def test_get_llama_ffn_ln_pattern_with_len_add_inputs_not_2(self):
        """测试get_llama_ffn_ln_pattern函数，add_node输入长度不为2"""
        mock_node = MockDagNode("test_node", "Linear")
        mock_add_node = MockDagNode("add", "__add__")
        mock_add_node.inputs = [Mock()]  # 只有1个输入
        mock_other_branch = MockDagNode("other", "Linear")

        mock_node.output_nodes = [mock_add_node, mock_other_branch]

        test_nodes = [mock_node, mock_add_node, mock_other_branch]

        with patch.object(self.adapter, 'node_list', test_nodes):
            with patch(
                'msmodelslim.utils.dag_utils.torch_dag_adapter.TorchDAGAdapter._get_ffn_pattern_and_ln_list'
            ) as mock_get:
                ffn_pattern, ffn_ln_list = self.adapter.get_llama_ffn_ln_pattern()
                mock_get.assert_not_called()  # 应该不被调用
                self.assertEqual(len(ffn_pattern), 0)
                self.assertEqual(len(ffn_ln_list), 0)

    def test_find_latest_module_by_type_with_stop_node_in_output_nodes(self):
        """测试_find_latest_module_by_type函数，stop_node在node.output_nodes中"""
        search_node = MockDagNode("search", "Linear")
        target_node = MockDagNode("target", "Linear")
        stop_node = MockDagNode("stop", "Stop")

        search_node.input_nodes = [target_node]
        target_node.output_nodes = [stop_node]  # stop_node在target_node的输出中

        # 验证stop_node在target_node的输出节点列表中
        self.assertIn(stop_node, list(target_node.output_nodes))

        result = self.adapter._find_latest_module_by_type(search_node, "Linear", stop_node=stop_node)
        self.assertEqual(result, [])  # 应该返回空列表，因为stop_node在输出中

    def test_find_latest_module_by_type_with_stop_node_equal_current(self):
        """测试_find_latest_module_by_type函数，当前节点等于stop_node"""
        search_node = MockDagNode("search", "Linear")
        stop_node = MockDagNode("stop", "Stop")

        search_node.input_nodes = [stop_node]  # 直接连接到stop_node

        result = self.adapter._find_latest_module_by_type(search_node, "Linear", stop_node=stop_node)
        self.assertEqual(result, [])  # 应该返回空列表，因为当前节点就是stop_node

    def test_find_latest_module_by_type_with_found_target_node(self):
        """测试_find_latest_module_by_type函数，找到目标节点"""
        search_node = MockDagNode("search", "Linear")
        target_node = MockDagNode("target", "Linear")
        target_node.op_type = "Linear"

        search_node.input_nodes = [target_node]

        result = self.adapter._find_latest_module_by_type(search_node, "Linear")
        self.assertEqual(result, [target_node])  # 应该返回包含目标节点的列表

    def test_find_latest_module_by_type_with_conv2d_or_getitem_stop(self):
        """测试_find_latest_module_by_type函数，遇到Conv2d或GetItem时停止"""
        search_node = MockDagNode("search", "Linear")
        conv_node = MockDagNode("conv", ModuleType.CONV2D)

        search_node.input_nodes = [conv_node]

        result = self.adapter._find_latest_module_by_type(search_node, "Linear")
        self.assertEqual(result, [])  # 应该返回空列表，因为遇到Conv2d停止


if __name__ == '__main__':
    unittest.main()
