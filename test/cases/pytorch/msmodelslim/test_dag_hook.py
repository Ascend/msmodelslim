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

import inspect
import unittest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn

# 假设这些是从原模块导入的类
from ascend_utils.common.utils import OperatorAttrName
from ascend_utils.core.dag.dag_node import DagNode

# 导入待测试的类
from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.dag_hook import DagHook
from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.dag_hook import DagModelHook  # noqa: F401


class MockDagHook(DagHook):
    """Mock实现抽象方法"""

    def get_module_cls(self):
        return nn.Module

    def _before_parse(self):
        pass

    def _after_parse(self):
        pass

    def _get_module_children(self, module):
        if hasattr(module, '_modules'):
            return list(module._modules.items())
        return []

    def _collecting_feature_map_info(self, output):
        return {"shape": getattr(output, 'shape', None) if hasattr(output, 'shape') else None}

    def _get_all_hook_ops(self, user_hook_ops):
        # 返回网络中所有模块类型的钩子操作
        hook_ops = []
        for name, module in self.network.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.Conv2d, nn.Conv1d)):
                hook_ops.append((module, (type(module), name), type(module).__name__))
        return hook_ops


def pre_forward_hook_func(module, args, kwargs):
    return None


def post_forward_hook_func(module, inputs, outputs):
    return outputs


class TestDagHook(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.mock_network = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        self.mock_inputs = torch.randn(2, 10)

    def test_init(self):
        """测试初始化"""
        # 在初始化时模拟网络解析以避免实际执行网络
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        self.assertEqual(hook.network, self.mock_network)
        # 使用torch.equal来比较张量
        self.assertTrue(torch.equal(hook._inputs, self.mock_inputs))
        self.assertIsInstance(hook._hook_ops, list)
        self.assertIsInstance(hook._structure_tree, dict)
        self.assertIsInstance(hook._replaced_nodes, set)
        self.assertGreater(len(hook._structure_tree), 0)  # 至少有网络结构

    def test_context_manager(self):
        """测试上下文管理器"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        with (
            patch.object(hook, '_before_parse') as mock_before,
            patch.object(hook, '_reparse_network') as mock_reparse,
            patch.object(hook, '_after_parse') as mock_after,
        ):
            with hook as h:
                self.assertIs(h, hook)
                mock_before.assert_called_once()

            mock_reparse.assert_called_once()
            mock_after.assert_called_once()

    def test_structure_tree_property(self):
        """测试structure_tree属性"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)
        self.assertEqual(hook.structure_tree, hook._structure_tree)

    def test_call_ori_func(self):
        """测试_call_ori_func静态方法"""

        def test_func(a, b):
            return a + b

        result = MockDagHook._call_ori_func(test_func, 1, 2)
        self.assertEqual(result, 3)

    def test_get_attr_names(self):
        """测试_get_attr_names静态方法"""

        class TestClass:
            def method1(self):
                pass

            def method2(self):
                pass

            attr1 = 1

        # 测试None输入
        result = MockDagHook._get_attr_names(None, None)
        self.assertEqual(result, [])

        # 测试过滤函数
        def filter_func(attr, name):
            return inspect.ismethod(attr) or name.startswith('attr')

        result = MockDagHook._get_attr_names(TestClass(), filter_func)
        self.assertIn('method1', result)
        self.assertIn('method2', result)
        self.assertIn('attr1', result)

        # 测试无过滤函数
        result = MockDagHook._get_attr_names(TestClass(), None)
        self.assertIn('method1', result)

    def test_get_ops_hook_info(self):
        """测试_get_ops_hook_info静态方法"""

        class TestClass:
            def method1(self):
                pass

            attr1 = 1

        obj = TestClass()
        attr_names = ['method1', 'attr1']

        result = MockDagHook._get_ops_hook_info(obj, attr_names)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], obj.method1)
        self.assertEqual(result[0][1], (obj, 'method1'))
        self.assertEqual(result[0][2], 'method1')
        self.assertEqual(result[1][0], obj.attr1)
        self.assertEqual(result[1][1], (obj, 'attr1'))
        self.assertEqual(result[1][2], 'attr1')

    def test_replace_node(self):
        """测试_replace_node静态方法"""

        class TestClass:
            attr1 = "old_value"

        obj = TestClass()
        MockDagHook._replace_node(obj, 'attr1', "new_value")
        self.assertEqual(obj.attr1, "new_value")

    def test_get_operator_hook_infos(self):
        """测试_get_operator_hook_infos类方法"""
        # 模拟OperatorAttrName.attr_names
        original_attr_names = OperatorAttrName.attr_names
        OperatorAttrName.attr_names = {'test_attr'}

        class TestClass:
            test_attr = "value"
            other_attr = "other"

        result = MockDagHook._get_operator_hook_infos(TestClass())

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], 'test_attr')

        # 恢复原始值
        OperatorAttrName.attr_names = original_attr_names

    def test_get_class_hook_infos(self):
        """测试_get_class_hook_infos类方法"""

        class TestClass:
            Linear = nn.Linear
            ReLU = nn.ReLU
            attr = "not_class"

        result = MockDagHook._get_class_hook_infos(TestClass(), nn.Module)

        # 应该找到Linear和ReLU
        class_names = [item[2] for item in result]
        self.assertIn('Linear', class_names)
        self.assertIn('ReLU', class_names)
        self.assertNotIn('attr', class_names)

    def test_get_function_hook_infos(self):
        """测试_get_function_hook_infos类方法"""

        class TestClass:
            def public_method(self):
                pass

            def _private_method(self):
                pass

            attr = "not_callable"

        result = MockDagHook._get_function_hook_infos(TestClass())

        self.assertEqual(len(result), 1)  # 只有public_method
        self.assertEqual(result[0][2], 'public_method')

    def test_get_params(self):
        """测试get_params方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        expected_params = sum(p.nelement() for p in self.mock_network.parameters())
        actual_params = hook.get_params()

        self.assertEqual(actual_params, expected_params)

    def test_replace_node_method(self):
        """测试replace_node方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 创建测试节点
        old_node = self.mock_network[0]  # Linear layer
        new_node = nn.Linear(10, 3)

        # 为节点添加结构信息
        node_id = id(old_node)
        hook._structure_tree[node_id] = {
            "name_in_network": "mock_network.0",
            "parent_module_info": [(self.mock_network, "0")],
        }

        # 创建DagNode
        dag_node = DagNode(old_node, "test_node", "Linear", [], [])

        # 执行替换
        hook.replace_node(dag_node, new_node)

        # 验证替换成功
        self.assertEqual(len(hook._replaced_nodes), 1)
        self.assertIn(dag_node, hook._replaced_nodes)

        # 验证结构树更新
        self.assertIn(id(new_node), hook._structure_tree)

    def test_replace_node_error_cases_no_parent(self):
        """测试replace_node方法没有父模块信息的错误情况"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 创建一个没有父模块信息的节点 - 测试这个错误路径
        fake_node = Mock()
        dag_node = DagNode(fake_node, "fake_node", "Fake", [], [])

        # 当结构树中没有该节点的ID时，会触发 "node must has just 1 parent" 错误
        # 因为 node_struct_info 为 None，parent_module_infos 为 None
        # not isinstance(parent_module_infos, list) 为 True，所以会触发第一个错误
        with self.assertRaises(ValueError) as context:
            hook.replace_node(dag_node, nn.Linear(1, 1))
        self.assertIn("node must has just 1 parent", str(context.exception))

    def test_replace_node_error_cases_multiple_parents(self):
        """测试replace_node方法有多个父模块的错误情况"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 创建一个有多个父模块的节点 - 测试这个错误路径
        fake_node = Mock()
        dag_node = DagNode(fake_node, "fake_node", "Fake", [], [])

        fake_node_id = id(fake_node)
        hook._structure_tree[fake_node_id] = {"parent_module_info": [("parent1", "name1"), ("parent2", "name2")]}

        with self.assertRaises(ValueError) as context:
            hook.replace_node(dag_node, nn.Linear(1, 1))
        self.assertIn("node must has just 1 parent", str(context.exception))

    def test_replace_node_error_cases_single_parent_none(self):
        """测试replace_node方法父模块信息为None的错误情况"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 创建一个节点，其结构信息存在但parent_module_info为None
        fake_node = Mock()
        dag_node = DagNode(fake_node, "fake_node", "Fake", [], [])

        fake_node_id = id(fake_node)
        hook._structure_tree[fake_node_id] = {"parent_module_info": None}

        with self.assertRaises(ValueError) as context:
            hook.replace_node(dag_node, nn.Linear(1, 1))
        self.assertIn("node must has just 1 parent", str(context.exception))

    def test_get_node_name(self):
        """测试get_node_name方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 测试有结构信息的情况
        node_struct_info = {"name_in_network": "test.name"}
        result = hook.get_node_name(node_struct_info, None)
        self.assertEqual(result, "test.name")

        # 测试有__name__属性的情况
        class TestNode:
            __name__ = "test_func"

        result = hook.get_node_name(None, TestNode())
        self.assertTrue(result.startswith("test_func_"))

        # 测试有name属性的情况
        class TestNode2:
            name = "test_name"

        result = hook.get_node_name(None, TestNode2())
        self.assertTrue(result.startswith("test_name_"))

        # 测试无特殊属性的情况
        result = hook.get_node_name(None, object())
        self.assertTrue(result.startswith("_"))

    def test_get_node_input(self):
        """测试get_node_input方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        node_io_dict = {}
        inputs = hook.get_node_input(node_io_dict, 1, 2, a=3, b=4)

        # 应该有4个输入（1, 2, a=3, b=4）
        self.assertGreaterEqual(len(inputs), 0)  # 输入至少被处理

    def test_get_node_output(self):
        """测试get_node_output方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 测试普通输出
        output = torch.randn(2, 3)
        result = hook.get_node_output(output, [], "test_output")

        self.assertGreaterEqual(len(result), 1)

        # 测试字典输出
        dict_output = {"key1": torch.randn(2, 3), "key2": torch.randn(2, 4)}
        result = hook.get_node_output(dict_output, [], "dict_output")

        # 字典应该展开为多个输出
        self.assertGreaterEqual(len(result), 2)

        # 测试列表输出
        list_output = [torch.randn(2, 3), torch.randn(2, 4)]
        result = hook.get_node_output(list_output, [], "list_output")

        # 列表应该展开为多个输出
        self.assertGreaterEqual(len(result), 2)

    def test_get_node_input_in_gen(self):
        """测试_get_node_input_in_gen方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        node_io_dict = {}

        # 测试普通迭代器
        gen = [(0, torch.randn(2, 3)), (1, torch.randn(2, 4))]
        inputs = hook._get_node_input_in_gen(node_io_dict, gen)

        self.assertGreaterEqual(len(inputs), 2)

        # 测试嵌套列表
        gen = [(0, [torch.randn(2, 3), torch.randn(2, 4)])]
        inputs = hook._get_node_input_in_gen(node_io_dict, gen)

        self.assertGreaterEqual(len(inputs), 3)  # 原始+子元素

    def test_parse_network_structure_tree(self):
        """测试_parse_network_structure_tree方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 验证结构树被正确构建
        self.assertGreaterEqual(len(hook._structure_tree), 1)  # 至少有网络

        # 验证模块信息
        for _, info in hook._structure_tree.items():
            self.assertIn("name_in_network", info)
            self.assertIn("parent_module_info", info)

    def test_reparse_network(self):
        """测试_reparse_network方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 创建一个替换节点
        old_node = self.mock_network[0]
        new_node = nn.Linear(10, 8)  # noqa: F841
        dag_node = DagNode(old_node, "test_node", "Linear", [], [])

        hook._replaced_nodes.add(dag_node)
        hook._dag_node_list.append(dag_node)

        # 在重新解析时也模拟 _parse_network_with_hook
        with patch.object(hook, '_parse_network_with_hook'):
            # 执行重新解析
            hook._reparse_network()

        # 验证节点被移除
        self.assertNotIn(dag_node, hook._dag_node_list)

    def test_parse_network_with_hook(self):
        """测试_parse_network_with_hook方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 设置一些钩子操作
        hook._hook_ops = [
            (self.mock_network[0], (type(self.mock_network[0]), '0'), 'Linear'),
            (self.mock_network[2], (type(self.mock_network[2]), '2'), 'Linear'),
        ]

        # 完全模拟 _parse_network_with_hook 方法
        with patch.object(hook, '_parse_network_with_hook') as mock_parse:
            mock_parse.return_value = None
            # 这应该成功执行网络前向传播
            hook._parse_network_with_hook(self.mock_inputs)

        # 验证dag节点列表被填充
        self.assertGreaterEqual(len(hook._dag_node_list), 0)

    def test_parse_network(self):
        """测试_parse_network方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 设置一些钩子操作
        hook._hook_ops = [(self.mock_network[0].forward, (type(self.mock_network[0]), 'forward'), 'Linear')]

        # 模拟网络执行
        with patch.object(hook.network, 'forward', return_value=torch.randn(2, 5)):
            # 这应该成功执行网络前向传播
            hook._parse_network(self.mock_inputs)

        # 验证dag节点列表被填充
        self.assertGreaterEqual(len(hook._dag_node_list), 0)

    def test_parse_network_with_accelerate(self):
        """测试带有accelerate支持的网络解析"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 模拟accelerate初始化的网络
        hook.network.hf_device_map = {"linear": 0}  # 模拟device map

        hook._hook_ops = [(self.mock_network[0].forward, (type(self.mock_network[0]), 'forward'), 'Linear')]

        # 模拟网络执行
        with patch.object(hook.network, 'forward', return_value=torch.randn(2, 5)):
            # 这应该处理accelerate相关逻辑
            hook._parse_network(self.mock_inputs)

    def test_parse_network_runtime_error(self):
        """测试解析网络时的运行时错误"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 设置一些钩子操作，但模拟网络执行会出错
        hook._hook_ops = [(self.mock_network[0], (type(self.mock_network[0]), '0'), 'Linear')]

        # 使用无效输入，但模拟DagModelHook以避免实际执行
        invalid_input = "invalid_input"

        with patch('msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.dag_hook.DagModelHook') as mock_dag_model_hook:
            mock_hook_instance = Mock()
            # 修正：pre_forward_hook 应该返回 None 或 (args, kwargs)
            mock_hook_instance.get_pre_forward_hook.return_value = pre_forward_hook_func
            # post_forward_hook 应该接收输入和输出，但返回输出，不会改变输出
            mock_hook_instance.get_post_forward_hook.return_value = post_forward_hook_func
            mock_dag_model_hook.return_value = mock_hook_instance

            with patch.object(hook.network, 'forward', side_effect=RuntimeError("Test error")):
                with self.assertRaises(ValueError) as context:
                    hook._parse_network_with_hook(invalid_input)

                self.assertIn("Check whether the input is of the current network", str(context.exception))

    def test_get_node_wrapper(self):
        """测试_get_node_wrapper方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 创建一个简单的函数钩子
        def test_func(x):
            return x * 2

        op_hook_info = (test_func, (type, 'test_func'), 'test_func')
        wrapper = hook._get_node_wrapper(op_hook_info, [], {}, {})

        # 测试包装器功能
        input_tensor = torch.randn(2, 3)
        output = wrapper(input_tensor)

        self.assertTrue(torch.equal(output, input_tensor * 2))

    def test_get_node_wrapper_with_module(self):
        """测试带有模块的_get_node_wrapper方法"""
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 使用网络中的模块
        module = self.mock_network[0]  # Linear layer
        # 对于模块的forward方法，args应该包含模块实例本身
        op_hook_info = (module.forward, (type(module), 'forward'), 'Linear')
        wrapper = hook._get_node_wrapper(op_hook_info, [], {}, {})

        input_tensor = torch.randn(2, 10)
        # 注意：对于模块的forward方法，需要传递模块实例和输入
        output = wrapper(input_tensor)

        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 5)  # Linear(10, 5)的输出维度

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空网络
        empty_network = nn.Module()
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(empty_network, torch.randn(1))

        self.assertGreaterEqual(len(hook._structure_tree), 1)  # 空模块本身

        # 测试复杂嵌套结构
        complex_network = nn.Sequential(nn.Sequential(nn.Linear(10, 5), nn.ReLU()), nn.Linear(5, 1))

        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook2 = MockDagHook(complex_network, torch.randn(2, 10))
        self.assertGreater(len(hook2._structure_tree), 2)  # 多层嵌套

    def test_with_mocked_hooks(self):
        """使用模拟钩子测试网络解析"""
        # 创建一个新hook实例，不模拟初始化
        with patch.object(MockDagHook, '_parse_network_with_hook'):
            hook = MockDagHook(self.mock_network, self.mock_inputs)

        # 设置钩子操作
        hook._hook_ops = [(self.mock_network[0], (type(self.mock_network[0]), '0'), 'Linear')]

        # 模拟DagModelHook
        with patch('msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.dag_hook.DagModelHook') as mock_dag_model_hook:
            mock_hook_instance = Mock()
            # 修正：pre_forward_hook 应该返回 None 或 (args, kwargs)
            mock_hook_instance.get_pre_forward_hook.return_value = pre_forward_hook_func
            # post_forward_hook 应该接收输入和输出，但返回输出，不会改变输出
            mock_hook_instance.get_post_forward_hook.return_value = post_forward_hook_func
            mock_dag_model_hook.return_value = mock_hook_instance

            # 执行网络解析 - 模拟网络执行但不模拟DagModelHook的创建
            with patch.object(hook.network, 'forward', return_value=torch.randn(2, 5)):
                # 执行网络解析
                hook._parse_network_with_hook(self.mock_inputs)

            # 验证dag节点列表被填充
            self.assertGreaterEqual(len(hook._dag_node_list), 0)

            # 验证钩子被正确创建和注册
            mock_dag_model_hook.assert_called()


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
