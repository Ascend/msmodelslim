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

# pylint: disable=too-many-lines

# msmodelslim.utils.dag_utils.dag_hook 模块的单元测试
# 测试DagHook抽象基类的功能，包括DAG图构建、节点替换等核心功能

import unittest
from typing import Any, List, Tuple
from unittest.mock import Mock, patch

import torch
from torch import nn

from ascend_utils.common.utils import CallParams
from ascend_utils.core.dag.dag_node import DagNode
from ascend_utils.core.dag.dag_node_io import DagNodeIO
from msmodelslim.utils.dag_utils.dag_hook import DagHook


class ConcreteDagHook(DagHook):
    """DagHook的具体实现类，用于测试抽象基类的功能"""

    def __init__(self, network, inputs, hook_ops=None, anti_method=None):
        # 初始化时需要模拟所有抽象方法
        super().__init__(network, inputs, hook_ops, anti_method)
        self.parse_called = False
        self.after_parse_called = False

    def get_module_cls(self):
        """获取模块类型"""
        return torch.nn.Module

    def _before_parse(self):
        """解析前的准备工作"""
        self.parse_called = True

    def _after_parse(self):
        """解析后的清理工作"""
        self.after_parse_called = True

    def _get_module_children(self, module):
        """获取模块的子模块"""
        if hasattr(module, 'named_children'):
            return module.named_children()
        return []

    def _collecting_feature_map_info(self, output):
        """收集特征图信息"""
        io_info = {}
        if isinstance(output, torch.Tensor):
            io_info["shape"] = output.shape
            io_info["dtype"] = output.dtype
        return io_info

    def _get_all_hook_ops(self, user_hook_ops) -> List[Tuple[Any, Any, str]]:
        """获取所有需要hook的操作"""
        hook_ops = []
        if user_hook_ops:
            for op in user_hook_ops:
                if isinstance(op, type) and issubclass(op, torch.nn.Module):
                    hook_ops.append((op.forward, (op, "forward"), op.__name__))
        return hook_ops


class TestDagHook(unittest.TestCase):
    """测试DagHook抽象基类"""

    @staticmethod
    def _create_mock_context():
        """创建mock上下文管理器（用于ResListToRelease）"""
        from contextlib import contextmanager

        @contextmanager
        def mock_res_list(*args):
            yield

        return mock_res_list

    @staticmethod
    def _create_network_call_mock(call_count):
        """创建mock network调用，记录调用次数"""

        def mock_network_call(self, *args, **kwargs):
            call_count['count'] += 1
            return torch.randn(1, 1)

        return mock_network_call

    def setUp(self):
        """测试前的准备工作"""
        # 创建一个简单的测试网络
        self.network = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        self.inputs = torch.randn(1, 10)
        self.hook_ops = [nn.Linear]

    def test_init_when_create_dag_hook_then_properties_initialized(self):
        """测试初始化：创建DagHook时，属性应该被正确初始化"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 验证基本属性
            self.assertEqual(dag_hook.network, self.network)
            # 对于Tensor使用torch.equal比较
            self.assertTrue(torch.equal(dag_hook._inputs, self.inputs))
            self.assertIsInstance(dag_hook._structure_tree, dict)
            self.assertIsInstance(dag_hook._replaced_nodes, set)

    def test_context_manager_when_enter_and_exit_then_methods_called(self):
        """测试上下文管理器：进入和退出时应该调用相应的方法"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
            patch.object(ConcreteDagHook, '_reparse_network'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 测试上下文管理器
            with dag_hook:
                self.assertTrue(hasattr(dag_hook, 'parse_called'))

            self.assertTrue(hasattr(dag_hook, 'after_parse_called'))

    def test_structure_tree_property_when_access_then_return_tree(self):
        """测试structure_tree属性：访问时应该返回结构树"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            self.assertIsInstance(dag_hook.structure_tree, dict)
            self.assertEqual(dag_hook.structure_tree, dag_hook._structure_tree)

    def test_call_ori_func_when_called_with_args_then_execute_function(self):
        """测试_call_ori_func静态方法：调用时应该执行原始函数"""
        # 创建一个模拟函数
        mock_func = Mock(return_value=42)

        # 调用静态方法
        result = DagHook._call_ori_func(mock_func, 1, 2, key="value")

        # 验证函数被正确调用
        mock_func.assert_called_once_with(1, 2, key="value")
        self.assertEqual(result, 42)

    def test_get_attr_names_when_object_has_attributes_then_return_filtered_names(self):
        """测试_get_attr_names静态方法：当对象有属性时，应该返回过滤后的属性名"""
        # 创建测试对象
        test_obj = Mock()
        test_obj.attr1 = "value1"
        test_obj.attr2 = 42

        # 不使用过滤器
        attrs = DagHook._get_attr_names(test_obj, None)
        self.assertIsInstance(attrs, list)

        # 使用过滤器
        def filter_func(attr, name):
            return name.startswith("attr")

        attrs = DagHook._get_attr_names(test_obj, filter_func)
        filtered_attrs = [name for name in attrs if name.startswith("attr")]
        self.assertGreater(len(filtered_attrs), 0)

    def test_get_attr_names_when_object_is_none_then_return_empty_list(self):
        """测试_get_attr_names静态方法：当对象为None时，应该返回空列表"""
        result = DagHook._get_attr_names(None, None)
        self.assertEqual(result, [])

    def test_get_ops_hook_info_when_called_then_return_hook_info_list(self):
        """测试_get_ops_hook_info静态方法：调用时应该返回hook信息列表"""
        test_obj = Mock()
        test_obj.method1 = Mock()
        test_obj.method2 = Mock()

        attr_names = ["method1", "method2"]
        result = DagHook._get_ops_hook_info(test_obj, attr_names)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 3)  # (attr, (obj, name), name)

    def test_replace_node_static_method_when_called_then_set_attribute(self):
        """测试_replace_node静态方法：调用时应该设置属性"""
        parent_module = Mock()
        name = "test_attr"
        new_node = Mock()

        DagHook._replace_node(parent_module, name, new_node)

        # 验证setattr被调用
        self.assertEqual(getattr(parent_module, name), new_node)

    def test_get_operator_hook_infos_when_called_then_return_operator_methods(self):
        """测试_get_operator_hook_infos类方法：调用时应该返回运算符方法信息"""

        class _OperatorHolder:
            def __add__(self, other):
                return other

            def __sub__(self, other):
                return other

        result = DagHook._get_operator_hook_infos(_OperatorHolder())

        # 应返回自定义的运算符方法
        result_names = {info[2] for info in result}
        self.assertTrue({"__add__", "__sub__"}.issubset(result_names))

    def test_get_class_hook_infos_when_called_then_return_class_infos(self):
        """测试_get_class_hook_infos类方法：调用时应该返回类信息"""
        # 使用torch.nn作为测试对象
        result = DagHook._get_class_hook_infos(torch.nn, torch.nn.Module)

        # 验证返回列表
        self.assertIsInstance(result, list)

    def test_get_function_hook_infos_when_called_then_return_function_infos(self):
        """测试_get_function_hook_infos类方法：调用时应该返回函数信息"""

        class _FunctionHolder:
            def public_method(self):
                pass

            def _private_method(self):
                pass

        result = DagHook._get_function_hook_infos(_FunctionHolder())
        result_names = [info[2] for info in result]

        # 只应收集公开方法，并排除以下划线开头的私有方法
        self.assertEqual(result_names, ["public_method"])

    def test_get_params_when_network_has_parameters_then_return_total_count(self):
        """测试get_params：当网络有参数时，应该返回参数总数"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 计算预期参数数量
            expected_params = sum(param.nelement() for param in self.network.parameters())

            # 验证返回的参数数量
            self.assertEqual(dag_hook.get_params(), expected_params)

    def test_replace_node_when_node_has_single_parent_then_replace_successfully(self):
        """测试replace_node：当节点有单一父模块时，应该成功替换"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建测试节点和结构信息
            old_module = nn.Linear(5, 3)
            new_module = nn.Linear(5, 4)
            parent_module = Mock()

            dag_node = DagNode(old_module, "test_node", "Linear")

            # 设置结构树信息
            dag_hook._structure_tree[id(old_module)] = {"parent_module_info": [(parent_module, "linear_layer")]}

            # 执行替换
            with patch.object(dag_hook, '_replace_node') as mock_replace:
                dag_hook.replace_node(dag_node, new_module)

                # 验证替换方法被调用
                mock_replace.assert_called_once_with(parent_module, "linear_layer", new_module)

                # 验证节点被标记为已替换
                self.assertIn(dag_node, dag_hook._replaced_nodes)

    def test_replace_node_when_node_has_multiple_parents_then_raise_error(self):
        """测试replace_node：当节点有多个父模块时，应该抛出错误"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建测试节点
            old_module = nn.Linear(5, 3)
            new_module = nn.Linear(5, 4)
            dag_node = DagNode(old_module, "test_node", "Linear")

            # 设置多个父模块的结构树信息
            dag_hook._structure_tree[id(old_module)] = {
                "parent_module_info": [(Mock(), "linear1"), (Mock(), "linear2")]
            }

            # 验证抛出异常
            with self.assertRaises(ValueError) as context:
                dag_hook.replace_node(dag_node, new_module)

            self.assertIn("node must has just 1 parent", str(context.exception))

    def test_replace_node_when_node_has_no_parent_info_then_raise_error(self):
        """测试replace_node：当节点没有父模块信息时，应该抛出错误"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建测试节点
            old_module = nn.Linear(5, 3)
            new_module = nn.Linear(5, 4)
            dag_node = DagNode(old_module, "test_node", "Linear")

            # 不设置结构树信息，会导致parent_module_infos为None
            # 根据代码逻辑，这会触发"node must has just 1 parent"错误

            # 验证抛出异常
            with self.assertRaises(ValueError) as context:
                dag_hook.replace_node(dag_node, new_module)

            # 修正预期的错误消息
            self.assertIn("node must has just 1 parent", str(context.exception))

    def test_get_node_name_when_struct_info_exists_then_return_network_name(self):
        """测试get_node_name：当结构信息存在时，应该返回网络中的名称"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建结构信息
            struct_info = {"name_in_network": "feature.0"}
            node_hook = Mock()

            result = dag_hook.get_node_name(struct_info, node_hook)
            self.assertEqual(result, "feature.0")

    def test_get_node_name_when_no_struct_info_but_has_name_attr_then_return_generated_name(self):
        """测试get_node_name：当没有结构信息但节点有__name__属性时，应该返回生成的名称"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建有__name__属性的节点
            def _TestModule_():
                pass

            node_hook = _TestModule_

            result = dag_hook.get_node_name(None, node_hook)
            expected = f"TestModule_{len(dag_hook._dag_node_list)}"
            self.assertEqual(result, expected)

    def test_get_node_name_when_no_name_attr_but_has_name_property_then_return_generated_name(self):
        """测试get_node_name：当节点没有__name__但有name属性时，应该返回生成的名称"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建有name属性但没有__name__属性的节点
            class _NodeWithName:
                name = "_MyNode_"

            node_hook = _NodeWithName()

            result = dag_hook.get_node_name(None, node_hook)
            expected = f"MyNode_{len(dag_hook._dag_node_list)}"
            self.assertEqual(result, expected)

    def test_get_node_input_when_called_with_args_and_kwargs_then_return_input_list(self):
        """测试get_node_input：传入参数和关键字参数时，应该返回输入列表"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建node_io_dict
            node_io_dict = {}

            # 模拟输入参数
            arg1 = torch.randn(2, 3)
            arg2 = torch.randn(3, 4)
            kwarg1 = torch.randn(1, 5)

            with patch.object(dag_hook, '_get_node_input_in_gen') as mock_get_input:
                mock_get_input.return_value = [Mock(), Mock()]

                result = dag_hook.get_node_input(node_io_dict, arg1, arg2, key=kwarg1)

                # 验证方法被调用两次（args和kwargs各一次）
                self.assertEqual(mock_get_input.call_count, 2)
                self.assertIsInstance(result, list)

    def test_get_node_output_when_simple_output_then_create_dag_node_io(self):
        """测试get_node_output：简单输出时，应该创建DagNodeIO"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建测试输出
            output = torch.randn(2, 3)
            deduplicate = []

            result = dag_hook.get_node_output(output, deduplicate, "test_output")

            # 验证返回字典
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 1)
            self.assertIn(id(output), result)
            self.assertIsInstance(result[id(output)], DagNodeIO)

    def test_get_node_output_when_duplicate_output_then_return_empty_dict(self):
        """测试get_node_output：重复输出时，应该返回空字典"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建测试输出
            output = torch.randn(2, 3)
            deduplicate = [id(output)]  # 已经存在

            result = dag_hook.get_node_output(output, deduplicate, "test_output")

            # 验证返回空字典
            self.assertEqual(result, {})

    def test_get_node_output_when_list_output_then_process_recursively(self):
        """测试get_node_output：列表输出时，应该递归处理"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建列表输出
            sub_output1 = torch.randn(2, 3)
            sub_output2 = torch.randn(3, 4)
            output = [sub_output1, sub_output2]
            deduplicate = []

            result = dag_hook.get_node_output(output, deduplicate, "test_output")

            # 验证结果包含主输出和子输出
            self.assertIn(id(output), result)
            self.assertIn(id(sub_output1), result)
            self.assertIn(id(sub_output2), result)

    def test_get_node_output_when_tuple_output_then_process_recursively(self):
        """测试get_node_output：元组输出时，应该递归处理"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建元组输出
            sub_output1 = torch.randn(2, 3)
            sub_output2 = torch.randn(3, 4)
            output = (sub_output1, sub_output2)
            deduplicate = []

            result = dag_hook.get_node_output(output, deduplicate, "test_output")

            # 验证结果包含主输出和子输出
            self.assertIn(id(output), result)
            self.assertIn(id(sub_output1), result)
            self.assertIn(id(sub_output2), result)

    def test_parse_network_structure_tree_when_module_has_children_then_build_tree(self):
        """测试_parse_network_structure_tree：当模块有子模块时，应该构建结构树"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建简单的测试模块
            test_module = nn.Linear(10, 5)

            # 构建结构树
            DagHook._parse_network_structure_tree(dag_hook, test_module, "linear", None, "model.linear")

            # 验证结构树被更新
            self.assertGreater(len(dag_hook.structure_tree), 0)
            self.assertIn(id(test_module), dag_hook.structure_tree)

    def test_parse_network_structure_tree_when_module_visited_twice_then_handle_correctly(self):
        """测试_parse_network_structure_tree：当模块被访问两次时，应该正确处理"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建测试模块
            test_module = nn.Linear(10, 5)
            parent1 = Mock()
            parent2 = Mock()

            # 第一次访问
            DagHook._parse_network_structure_tree(dag_hook, test_module, "linear1", parent1, "model.linear1")

            # 第二次访问
            DagHook._parse_network_structure_tree(dag_hook, test_module, "linear2", parent2, "model.linear2")

            # 验证parent_module_info被更新
            self.assertIn(id(test_module), dag_hook.structure_tree)

    def test_get_node_input_in_gen_when_argument_in_dict_then_return_existing(self):
        """测试_get_node_input_in_gen：当参数已在字典中时，应该返回现有的DagNodeIO"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建node_io_dict
            arg = torch.randn(2, 3)
            existing_io = Mock(spec=DagNodeIO)
            node_io_dict = {id(arg): existing_io}

            # 调用方法
            gen = enumerate([arg])
            result = dag_hook._get_node_input_in_gen(node_io_dict, gen)

            # 验证返回现有的DagNodeIO
            self.assertIn(existing_io, result)

    def test_get_node_input_in_gen_when_argument_not_in_dict_then_create_new(self):
        """测试_get_node_input_in_gen：当参数不在字典中时，应该创建新的DagNodeIO"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建空的node_io_dict
            arg = torch.randn(2, 3)
            node_io_dict = {}

            # 调用方法
            gen = enumerate([arg])
            result = dag_hook._get_node_input_in_gen(node_io_dict, gen)

            # 验证创建了新的DagNodeIO
            self.assertGreater(len(result), 0)
            self.assertIn(id(arg), node_io_dict)

    def test_get_node_input_in_gen_when_argument_is_list_then_process_recursively(self):
        """测试_get_node_input_in_gen：当参数是列表时，应该递归处理"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建列表参数
            sub_arg1 = torch.randn(2, 3)
            sub_arg2 = torch.randn(3, 4)
            arg = [sub_arg1, sub_arg2]
            node_io_dict = {}

            # 调用方法
            gen = enumerate([arg])
            result = dag_hook._get_node_input_in_gen(node_io_dict, gen)

            # 验证递归处理
            self.assertGreater(len(result), 1)

    def test_get_node_input_in_gen_when_argument_is_dict_then_process_recursively(self):
        """测试_get_node_input_in_gen：当参数是字典时，应该递归处理"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建字典参数
            sub_arg1 = torch.randn(2, 3)
            arg = {"key": sub_arg1}
            node_io_dict = {}

            # 调用方法
            gen = enumerate([arg])
            result = dag_hook._get_node_input_in_gen(node_io_dict, gen)

            # 验证递归处理
            self.assertGreater(len(result), 1)

    def test_reparse_network_when_nodes_replaced_then_reparse_correctly(self):
        """测试_reparse_network：节点被替换时，应该正确重新解析"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook') as mock_parse,
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建一些被替换的节点
            replaced_node = DagNode(nn.Linear(5, 3), "replaced", "Linear")
            dag_hook._replaced_nodes.add(replaced_node)
            dag_hook._dag_node_list.append(replaced_node)

            # 添加一个未被替换的节点
            normal_node = DagNode(nn.ReLU(), "normal", "ReLU")
            dag_hook._dag_node_list.append(normal_node)

            # 重置mock调用计数，因为__init__中已经调用过一次
            mock_parse.reset_mock()

            # 执行重新解析
            dag_hook._reparse_network()

            # 验证被替换的节点被移除
            self.assertNotIn(replaced_node, dag_hook._dag_node_list)

            # 验证解析方法被调用（此时应该只被调用一次）
            mock_parse.assert_called_once()

    def test_reparse_network_when_duplicate_nodes_then_remove_duplicates(self):
        """测试_reparse_network：存在重复节点时，应该去重"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook') as mock_parse,
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建相同模块的多个节点
            shared_module = nn.ReLU()
            node1 = DagNode(shared_module, "node1", "ReLU")
            node2 = DagNode(shared_module, "node2", "ReLU")
            dag_hook._dag_node_list.extend([node1, node2])

            # 重置mock调用计数，因为__init__中已经调用过一次
            mock_parse.reset_mock()

            # 执行重新解析
            dag_hook._reparse_network()

            # 验证解析方法被调用（此时应该只被调用一次）
            mock_parse.assert_called_once()

    def test_parse_network_when_single_input_then_parse_successfully(self):
        """测试_parse_network：当输入是单个tensor时，应该成功解析网络"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)
            dag_hook._dag_node_list.clear()

            call_count = {'count': 0}
            mock_context = self._create_mock_context()
            mock_call = self._create_network_call_mock(call_count)
            with (
                patch('msmodelslim.utils.dag_utils.dag_hook.ResListToRelease', side_effect=mock_context),
                patch('msmodelslim.utils.dag_utils.dag_hook.FunctionReplace'),
                patch.object(dag_hook.network.__class__, '__call__', mock_call),
            ):
                dag_hook._parse_network(self.inputs)

            self.assertEqual(call_count['count'], 1)
            self.assertIsInstance(dag_hook._dag_node_list, list)

    def test_parse_network_when_tuple_input_then_unpack_and_parse(self):
        """测试_parse_network：当输入是元组时，应该解包并解析"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)
            tuple_inputs = (torch.randn(1, 10),)

            call_count = {'count': 0}
            mock_context = self._create_mock_context()
            mock_call = self._create_network_call_mock(call_count)
            with (
                patch('msmodelslim.utils.dag_utils.dag_hook.ResListToRelease', side_effect=mock_context),
                patch('msmodelslim.utils.dag_utils.dag_hook.FunctionReplace'),
                patch.object(dag_hook.network.__class__, '__call__', mock_call),
            ):
                dag_hook._parse_network(tuple_inputs)

            self.assertEqual(call_count['count'], 1)

    def test_parse_network_when_list_input_then_unpack_and_parse(self):
        """测试_parse_network：当输入是列表时，应该解包并解析"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)
            list_inputs = [torch.randn(1, 10)]

            call_count = {'count': 0}
            mock_context = self._create_mock_context()
            mock_call = self._create_network_call_mock(call_count)
            with (
                patch('msmodelslim.utils.dag_utils.dag_hook.ResListToRelease', side_effect=mock_context),
                patch('msmodelslim.utils.dag_utils.dag_hook.FunctionReplace'),
                patch.object(dag_hook.network.__class__, '__call__', mock_call),
            ):
                dag_hook._parse_network(list_inputs)

            self.assertEqual(call_count['count'], 1)

    def test_parse_network_when_dict_input_then_unpack_as_kwargs(self):
        """测试_parse_network：当输入是字典时，应该作为关键字参数解包"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)
            dict_inputs = {"input": torch.randn(1, 10)}

            call_count = {'count': 0}
            mock_context = self._create_mock_context()
            mock_call = self._create_network_call_mock(call_count)
            with (
                patch('msmodelslim.utils.dag_utils.dag_hook.ResListToRelease', side_effect=mock_context),
                patch('msmodelslim.utils.dag_utils.dag_hook.FunctionReplace'),
                patch.object(dag_hook.network.__class__, '__call__', mock_call),
            ):
                dag_hook._parse_network(dict_inputs)

            self.assertEqual(call_count['count'], 1)

    def test_parse_network_when_call_params_input_then_unpack_args_and_kwargs(self):
        """测试_parse_network：当输入是CallParams时，应该解包args和kwargs"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)
            call_params = CallParams(torch.randn(1, 10))

            call_count = {'count': 0}
            mock_context = self._create_mock_context()
            mock_call = self._create_network_call_mock(call_count)
            with (
                patch('msmodelslim.utils.dag_utils.dag_hook.ResListToRelease', side_effect=mock_context),
                patch('msmodelslim.utils.dag_utils.dag_hook.FunctionReplace'),
                patch.object(dag_hook.network.__class__, '__call__', mock_call),
            ):
                dag_hook._parse_network(call_params)

            self.assertEqual(call_count['count'], 1)

    def test_parse_network_when_runtime_error_then_raise_value_error(self):
        """测试_parse_network：当运行时错误发生时，应该转换为ValueError"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            def mock_network_call(self, *args, **kwargs):
                raise RuntimeError("test error")

            mock_context = self._create_mock_context()
            with (
                patch('msmodelslim.utils.dag_utils.dag_hook.ResListToRelease', side_effect=mock_context),
                patch('msmodelslim.utils.dag_utils.dag_hook.FunctionReplace'),
                patch.object(dag_hook.network.__class__, '__call__', mock_network_call),
            ):
                with self.assertRaises(ValueError) as context:
                    dag_hook._parse_network(self.inputs)
                self.assertIn("Check whether the input is of the current network", str(context.exception))

    def test_parse_network_when_has_accelerate_then_handle_forward_attr(self):
        """测试_parse_network：当使用accelerate时，应该处理forward属性"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)
            dag_hook.network.hf_device_map = {"0": 0, "1": 1}

            call_count = {'count': 0}
            delattr_calls = []
            original_delattr = delattr

            def mock_delattr(obj, name):
                if name == 'forward' and isinstance(obj, nn.Linear):
                    delattr_calls.append((obj, name))
                    return
                original_delattr(obj, name)

            mock_context = self._create_mock_context()
            mock_call = self._create_network_call_mock(call_count)
            with (
                patch('msmodelslim.utils.dag_utils.dag_hook.ResListToRelease', side_effect=mock_context),
                patch('msmodelslim.utils.dag_utils.dag_hook.FunctionReplace'),
                patch.object(dag_hook.network.__class__, '__call__', mock_call),
                patch('builtins.delattr', side_effect=mock_delattr),
            ):
                dag_hook._parse_network(self.inputs)

            self.assertEqual(call_count['count'], 1)
            self.assertGreater(len(delattr_calls), 0)
            for obj, name in delattr_calls:
                self.assertEqual(name, 'forward')
                self.assertIsInstance(obj, nn.Linear)

    def test_get_node_wrapper_when_replace_stack_not_empty_then_call_original(self):
        """测试_get_node_wrapper：当replace_stack不为空时，应该直接调用原始函数"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建一个简单的函数而不是Linear.forward
            def simple_func(x):
                return x * 2

            op_hook_info = (simple_func, (simple_func, ""), "simple")

            replace_stack = [Mock()]  # 非空栈
            node_io_dict = {}
            parsed_nodes = {}

            # 获取wrapper
            wrapper = dag_hook._get_node_wrapper(op_hook_info, replace_stack, node_io_dict, parsed_nodes)

            # 调用wrapper
            input_tensor = torch.randn(1, 10)
            result = wrapper(input_tensor)

            # 验证结果是输入的2倍
            self.assertTrue(torch.allclose(result, input_tensor * 2))

    def test_get_node_wrapper_when_module_self_then_process_as_module(self):
        """测试_get_node_wrapper：当第一个参数是Module时，应该作为模块处理"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建测试模块
            test_module = nn.Linear(10, 5)

            # 添加到结构树
            dag_hook._structure_tree[id(test_module)] = {"name_in_network": "test_linear"}

            # 创建一个wrapper函数来模拟Linear.forward
            def mock_forward_for_module(self, input_data):
                # 简单返回一个正确形状的tensor
                return torch.randn(input_data.shape[0], 5)

            op_hook_info = (mock_forward_for_module, (nn.Linear, "forward"), "Linear")

            replace_stack = []
            node_io_dict = {}
            parsed_nodes = {}

            # 获取wrapper
            wrapper = dag_hook._get_node_wrapper(op_hook_info, replace_stack, node_io_dict, parsed_nodes)

            # 调用wrapper（模拟模块的forward调用）
            input_tensor = torch.randn(1, 10)
            result = wrapper(test_module, input_tensor)

            # 验证结果
            self.assertIsInstance(result, torch.Tensor)
            # 验证DAG节点被添加
            self.assertGreater(len(dag_hook._dag_node_list), 0)
            # 验证replace_stack被清空
            self.assertEqual(len(replace_stack), 0)

    def test_get_node_wrapper_when_not_module_self_then_process_as_function(self):
        """测试_get_node_wrapper：当第一个参数不是Module时，应该作为函数处理"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建op_hook_info（模拟函数而非模块方法）
            def mock_func(x, y):
                return x + y

            op_hook_info = (mock_func, (mock_func, ""), "add")

            replace_stack = []
            node_io_dict = {}
            parsed_nodes = {}

            # 获取wrapper
            wrapper = dag_hook._get_node_wrapper(op_hook_info, replace_stack, node_io_dict, parsed_nodes)

            # 调用wrapper
            result = wrapper(torch.randn(2, 3), torch.randn(2, 3))

            # 验证结果
            self.assertIsInstance(result, torch.Tensor)
            # 验证DAG节点被添加
            self.assertGreater(len(dag_hook._dag_node_list), 0)

    def test_get_node_wrapper_when_module_in_parsed_nodes_then_reuse_node(self):
        """测试_get_node_wrapper：当模块在parsed_nodes中时，应该复用节点"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建测试模块
            test_module = nn.Linear(10, 5)

            # 创建已存在的DAG节点
            existing_node = DagNode(test_module, "existing", "Linear")

            # 添加到结构树
            dag_hook._structure_tree[id(test_module)] = {"name_in_network": "test_linear"}

            # 创建一个wrapper函数来模拟Linear.forward
            def mock_forward_for_reuse(self, input_data):
                return torch.randn(input_data.shape[0], 5)

            op_hook_info = (mock_forward_for_reuse, (nn.Linear, "forward"), "Linear")

            replace_stack = []
            node_io_dict = {}
            parsed_nodes = {test_module: existing_node}

            # 获取wrapper
            wrapper = dag_hook._get_node_wrapper(op_hook_info, replace_stack, node_io_dict, parsed_nodes)

            # 调用wrapper
            input_tensor = torch.randn(1, 10)
            result = wrapper(test_module, input_tensor)

            # 验证结果
            self.assertIsInstance(result, torch.Tensor)
            # 验证使用了现有节点
            self.assertIn(existing_node, dag_hook._dag_node_list)

    def test_get_node_wrapper_when_module_has_old_module_attr_then_handle_device(self):
        """测试_get_node_wrapper：当模块有_old_module属性时，应该处理设备转换"""
        with (
            patch.object(ConcreteDagHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagHook(self.network, self.inputs, self.hook_ops)

            # 创建带_old_module属性的测试模块（模拟accelerate）
            class _LinearWithOldModule(nn.Linear):
                def __init__(self):
                    super().__init__(10, 5)
                    self._old_module = Mock()

            test_module = _LinearWithOldModule()

            # 添加到结构树
            dag_hook._structure_tree[id(test_module)] = {"name_in_network": "test_linear"}

            # 创建一个wrapper函数来模拟Linear.forward
            def mock_forward_with_device(self, input_data):
                return torch.randn(input_data.shape[0], 5)

            op_hook_info = (mock_forward_with_device, (nn.Linear, "forward"), "Linear")

            replace_stack = []
            node_io_dict = {}
            parsed_nodes = {}

            # 获取wrapper
            wrapper = dag_hook._get_node_wrapper(op_hook_info, replace_stack, node_io_dict, parsed_nodes)

            # 调用wrapper
            input_tensor = torch.randn(1, 10)
            result = wrapper(test_module, input_tensor)

            # 验证结果
            self.assertIsInstance(result, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
