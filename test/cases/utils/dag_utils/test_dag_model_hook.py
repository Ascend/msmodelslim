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
# pylint: disable=consider-using-from-import,pointless-string-statement

"""
msmodelslim.utils.dag_utils.dag_model_hook 模块的单元测试
测试DagModelHook类的功能，包括模型hook的前向和后向处理
"""

import unittest  # noqa: E402
from unittest.mock import Mock, patch  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ascend_utils.core.dag.dag_node import DagNode  # noqa: E402
from ascend_utils.core.dag.dag_node_io import DagNodeIO  # noqa: E402
from msmodelslim.utils.dag_utils.dag_model_hook import DagModelHook  # noqa: E402


class TestDagModelHook(unittest.TestCase):
    """测试DagModelHook类"""

    def setUp(self):
        """测试前的准备工作"""
        self.ops_name = "Linear"
        self.replace_stack = []
        self.node_io_dict = {}
        self.parsed_node_list = {}

        # 创建模拟的dag_hook
        self.mock_dag_hook = Mock()
        self.mock_dag_hook.get_node_input = Mock(return_value=[Mock(spec=DagNodeIO)])
        self.mock_dag_hook.structure_tree = {}
        self.mock_dag_hook.get_node_name = Mock(return_value="test_node")
        self.mock_dag_hook.get_node_output = Mock(return_value={123: Mock(spec=DagNodeIO, name="output")})
        self.mock_dag_hook.get_module_cls = Mock(return_value=torch.nn.Module)
        self.mock_dag_hook.dag_node_list = []

        # 创建测试用的模型
        self.test_model = nn.Linear(10, 5)

    def test_init_when_create_dag_model_hook_then_properties_initialized(self):
        """测试初始化：创建DagModelHook时，属性应该被正确初始化"""
        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        # 验证属性初始化
        self.assertEqual(hook.ops_type, self.ops_name)
        self.assertEqual(hook.replace_stack, self.replace_stack)
        self.assertEqual(hook.node_io_dict, self.node_io_dict)
        self.assertEqual(hook.parsed_node_list, self.parsed_node_list)
        self.assertEqual(hook.dag_hook, self.mock_dag_hook)
        self.assertIsInstance(hook.infos, dict)

    def test_get_pre_forward_hook_when_called_then_return_callable(self):
        """测试get_pre_forward_hook：调用时应该返回可调用的lambda函数"""
        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        pre_hook_func = hook.get_pre_forward_hook()

        # 验证返回的是可调用的函数
        self.assertTrue(callable(pre_hook_func))

    def test_get_post_forward_hook_when_called_then_return_callable(self):
        """测试get_post_forward_hook：调用时应该返回可调用的lambda函数"""
        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        post_hook_func = hook.get_post_forward_hook()

        # 验证返回的是可调用的函数
        self.assertTrue(callable(post_hook_func))

    def test_pre_forward_when_replace_stack_empty_then_process_normally(self):
        """测试pre_forward：当replace_stack为空时，应该正常处理并记录输入信息"""
        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        # 测试参数
        args = (torch.randn(1, 10), torch.randn(1, 3))
        kwargs = {"bias": True}

        # 模拟dag_hook的方法返回值
        mock_inputs = [Mock(spec=DagNodeIO)]
        self.mock_dag_hook.get_node_input.return_value = mock_inputs
        self.mock_dag_hook.structure_tree = {id(self.test_model): {"name": "test"}}
        self.mock_dag_hook.get_node_name.return_value = "linear_layer"

        result_args, result_kwargs = hook.pre_forward(self.test_model, *args, **kwargs)

        # 验证参数原样返回
        self.assertEqual(result_args, args)
        self.assertEqual(result_kwargs, kwargs)

        # 验证replace_stack被更新
        self.assertIn(self.test_model, hook.replace_stack)

        # 验证dag_hook方法被调用
        self.mock_dag_hook.get_node_input.assert_called_once()
        self.mock_dag_hook.get_node_name.assert_called_once()

        # 验证infos被正确更新
        self.assertIn(id(self.test_model), hook.infos)
        inputs, name = hook.infos[id(self.test_model)]
        self.assertEqual(inputs, mock_inputs)
        self.assertEqual(name, "linear_layer")

    def test_pre_forward_when_replace_stack_not_empty_then_return_args_directly(self):
        """测试pre_forward：当replace_stack不为空时，应该直接返回参数而不做处理"""
        # 在replace_stack中添加一个模型
        self.replace_stack.append(Mock())

        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        args = (torch.randn(1, 10),)
        kwargs = {"bias": True}

        result_args, result_kwargs = hook.pre_forward(self.test_model, *args, **kwargs)

        # 验证直接返回参数，不做处理
        self.assertEqual(result_args, args)
        self.assertEqual(result_kwargs, kwargs)

        # 验证dag_hook方法未被调用
        self.mock_dag_hook.get_node_input.assert_not_called()

        # 验证infos没有被更新
        self.assertNotIn(id(self.test_model), hook.infos)

    def test_post_forward_when_replace_stack_empty_then_return_output_directly(self):
        """测试post_forward：当replace_stack为空时，应该直接返回输出"""
        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        output = torch.randn(1, 5)
        result = hook.post_forward(self.test_model, output)

        # 验证直接返回输出
        self.assertTrue(torch.equal(result, output))

    def test_post_forward_when_different_model_on_stack_then_return_output_directly(self):
        """测试post_forward：当栈顶模型不匹配时，应该直接返回输出"""
        # 在replace_stack中添加不同的模型
        other_model = nn.Linear(5, 3)
        self.replace_stack.append(other_model)

        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        output = torch.randn(1, 5)
        result = hook.post_forward(self.test_model, output)

        # 验证直接返回输出
        self.assertTrue(torch.equal(result, output))

    def test_post_forward_when_model_matches_stack_top_then_process_normally(self):
        """测试post_forward：当模型匹配栈顶时，应该正常处理并创建DAG节点"""
        # 在replace_stack中添加测试模型
        self.replace_stack.append(self.test_model)

        # 创建mock的dag_node_list，使其具有append方法
        mock_dag_node_list = Mock()
        self.mock_dag_hook.dag_node_list = mock_dag_node_list

        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        # 预先设置infos
        mock_inputs = [Mock(spec=DagNodeIO)]
        hook.infos[id(self.test_model)] = (mock_inputs, "linear_layer")

        # 模拟dag_hook的返回值
        output_dict = {123: Mock(spec=DagNodeIO)}
        self.mock_dag_hook.get_node_output.return_value = output_dict

        output = torch.randn(1, 5)

        with patch('msmodelslim.utils.dag_utils.dag_model_hook.DagNode') as mock_dag_node_cls:
            mock_dag_node = Mock(spec=DagNode)
            mock_dag_node_cls.return_value = mock_dag_node

            result = hook.post_forward(self.test_model, output)

            # 验证返回原始输出
            self.assertTrue(torch.equal(result, output))

            # 验证dag_hook方法被调用
            self.mock_dag_hook.get_node_output.assert_called_once()

            # 验证node_io_dict被更新
            self.assertEqual(hook.node_io_dict, output_dict)

            # 验证DAG节点被创建
            mock_dag_node_cls.assert_called_once()

            # 验证DAG节点被添加到dag_hook
            mock_dag_node_list.append.assert_called_once_with(mock_dag_node)

            # 验证replace_stack被弹出
            self.assertNotIn(self.test_model, hook.replace_stack)

            # 验证infos被清理
            self.assertNotIn(id(self.test_model), hook.infos)

    def test_post_forward_when_model_in_parsed_list_then_update_existing_node(self):
        """测试post_forward：当模型在已解析列表中时，应该更新现有节点而不是创建新节点"""
        # 在replace_stack中添加测试模型
        self.replace_stack.append(self.test_model)

        # 创建已存在的DAG节点
        existing_dag_node = Mock(spec=DagNode)
        self.parsed_node_list[self.test_model] = existing_dag_node

        # 创建mock的dag_node_list
        mock_dag_node_list = Mock()
        self.mock_dag_hook.dag_node_list = mock_dag_node_list

        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        # 预先设置infos
        mock_inputs = [Mock(spec=DagNodeIO)]
        hook.infos[id(self.test_model)] = (mock_inputs, "linear_layer")

        # 模拟dag_hook的返回值
        output_dict = {123: Mock(spec=DagNodeIO)}
        self.mock_dag_hook.get_node_output.return_value = output_dict

        output = torch.randn(1, 5)
        hook.post_forward(self.test_model, output)

        # 验证现有节点的set_node_io被调用
        existing_dag_node.set_node_io.assert_called_once()

        # 验证现有节点被添加到dag_hook
        mock_dag_node_list.append.assert_called_once_with(existing_dag_node)

    def test_post_forward_integration_when_full_workflow_then_all_steps_executed(self):
        """测试post_forward集成：完整的pre_forward和post_forward工作流"""
        # 创建mock的dag_node_list
        mock_dag_node_list = Mock()
        self.mock_dag_hook.dag_node_list = mock_dag_node_list

        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        # 先调用pre_forward来设置状态
        args = (torch.randn(1, 10),)
        kwargs = {}
        hook.pre_forward(self.test_model, *args, **kwargs)

        # 验证状态已正确设置
        self.assertIn(self.test_model, hook.replace_stack)
        self.assertIn(id(self.test_model), hook.infos)

        # 然后调用post_forward
        output = torch.randn(1, 5)

        with patch('msmodelslim.utils.dag_utils.dag_model_hook.DagNode'):
            result = hook.post_forward(self.test_model, output)

            # 验证返回原始输出
            self.assertTrue(torch.equal(result, output))

            # 验证状态已清理
            self.assertNotIn(self.test_model, hook.replace_stack)
            self.assertNotIn(id(self.test_model), hook.infos)

    def test_memory_cleanup_when_post_forward_completes_then_infos_cleaned(self):
        """测试内存清理：post_forward完成时，infos应该被正确清理以避免内存泄漏"""
        # 创建mock的dag_node_list
        mock_dag_node_list = Mock()
        self.mock_dag_hook.dag_node_list = mock_dag_node_list

        hook = DagModelHook(
            ops_name=self.ops_name,
            replace_stack=self.replace_stack,
            node_io_dict=self.node_io_dict,
            parsed_node_list=self.parsed_node_list,
            dag_hook=self.mock_dag_hook,
        )

        # 设置初始状态
        hook.replace_stack.append(self.test_model)
        hook.infos[id(self.test_model)] = ([Mock()], "test_name")

        # 验证infos有数据
        self.assertGreater(len(hook.infos), 0)

        output = torch.randn(1, 5)

        with patch('msmodelslim.utils.dag_utils.dag_model_hook.DagNode'):
            hook.post_forward(self.test_model, output)

        # 验证infos被清理
        self.assertNotIn(id(self.test_model), hook.infos)


if __name__ == '__main__':
    unittest.main()
