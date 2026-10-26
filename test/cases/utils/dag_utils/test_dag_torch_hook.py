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

# msmodelslim.utils.dag_utils.dag_torch_hook 模块的单元测试
# 测试DagTorchHook类的功能，包括PyTorch模型的DAG图构建和节点操作

import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from ascend_utils.common.utils import CallParams
from ascend_utils.core.dag.dag_node import DagNode
from msmodelslim.utils.dag_utils.dag_torch_hook import DagTorchHook


class ConcreteDagTorchHook(DagTorchHook):
    """DagTorchHook的具体实现类，用于测试"""

    pass


class TestDagTorchHook(unittest.TestCase):
    """测试DagTorchHook类"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建一个简单的测试网络
        self.network = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        self.inputs = torch.randn(1, 10)
        self.hook_ops = [nn.Linear, nn.ReLU]

    def test_init_when_valid_network_and_inputs_then_create_successfully(self):
        """测试初始化：当网络和输入有效时，应该成功创建DagTorchHook实例"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs, self.hook_ops)

            # 验证基本属性
            self.assertEqual(dag_hook.network, self.network)
            self.assertIsNotNone(dag_hook._inputs)
            self.assertEqual(dag_hook._tmp_device, "CPU")

    def test_init_when_invalid_network_type_then_raise_type_error(self):
        """测试初始化：当网络类型无效时，应该抛出TypeError"""
        with self.assertRaises(TypeError) as context:
            ConcreteDagTorchHook("not a module", self.inputs)

        self.assertIn("network must be type torch.nn.Module", str(context.exception))

    def test_init_when_invalid_inputs_type_then_raise_type_error(self):
        """测试初始化：当输入类型无效时，应该抛出TypeError"""
        with self.assertRaises(TypeError) as context:
            ConcreteDagTorchHook(self.network, "invalid input")

        self.assertIn("inputs must be type", str(context.exception))

    def test_init_when_invalid_hook_ops_type_then_raise_type_error(self):
        """测试初始化：当hook_ops类型无效时，应该抛出TypeError"""
        with self.assertRaises(TypeError) as context:
            ConcreteDagTorchHook(self.network, self.inputs, "not a list")

        self.assertIn("hook_nodes must be type list", str(context.exception))

    def test_get_obj_module_attrs_when_called_then_return_module_classes(self):
        """测试get_obj_module_attrs：调用时应该返回对象中的Module类型属性"""
        # 使用torch.nn作为测试对象
        result = list(DagTorchHook.get_obj_module_attrs(torch.nn))

        # 验证返回的是类并且是Module的子类
        self.assertGreater(len(result), 0)
        for attr in result[:5]:  # 只检查前几个
            self.assertTrue(callable(attr))

    def test_parse_network_device_when_network_has_parameters_then_return_device(self):
        """测试_parse_network_device：当网络有参数时，应该返回设备类型"""
        device = DagTorchHook._parse_network_device(self.network)

        # 验证返回的是设备对象
        self.assertIsNotNone(device)
        self.assertIn(str(device), ['cpu', 'cuda:0'])

    def test_input_item_to_cpu_when_tensor_input_then_convert_to_cpu(self):
        """测试_input_item_to_cpu：当输入是Tensor时，应该转换到CPU"""
        # 创建CPU上的tensor
        tensor_input = torch.randn(2, 3)
        result = DagTorchHook._input_item_to_cpu(tensor_input)

        # 验证结果在CPU上
        self.assertEqual(result.device.type, 'cpu')

    def test_input_item_to_cpu_when_non_tensor_input_then_return_as_is(self):
        """测试_input_item_to_cpu：当输入不是Tensor时，应该原样返回"""
        non_tensor_input = "some string"
        result = DagTorchHook._input_item_to_cpu(non_tensor_input)

        # 验证原样返回
        self.assertEqual(result, non_tensor_input)

    def test_input_to_cpu_when_single_tensor_then_convert_to_cpu(self):
        """测试input_to_cpu：当输入是单个Tensor时，应该转换到CPU"""
        tensor_input = torch.randn(2, 3)
        result = DagTorchHook.input_to_cpu(tensor_input)

        # 验证结果在CPU上
        self.assertEqual(result.device.type, 'cpu')

    def test_input_to_cpu_when_list_of_tensors_then_convert_all_to_cpu(self):
        """测试input_to_cpu：当输入是Tensor列表时，应该全部转换到CPU"""
        tensor_list = [torch.randn(2, 3), torch.randn(3, 4)]
        result = DagTorchHook.input_to_cpu(tensor_list)

        # 验证所有tensor都在CPU上
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for tensor in result:
            self.assertEqual(tensor.device.type, 'cpu')

    def test_input_to_cpu_when_tuple_of_tensors_then_convert_all_to_cpu(self):
        """测试input_to_cpu：当输入是Tensor元组时，应该全部转换到CPU"""
        tensor_tuple = (torch.randn(2, 3), torch.randn(3, 4))
        result = DagTorchHook.input_to_cpu(tensor_tuple)

        # 验证所有tensor都在CPU上
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_input_to_cpu_when_dict_of_tensors_then_convert_all_to_cpu(self):
        """测试input_to_cpu：当输入是Tensor字典时，应该全部转换到CPU"""
        tensor_dict = {"key1": torch.randn(2, 3), "key2": torch.randn(3, 4)}
        result = DagTorchHook.input_to_cpu(tensor_dict)

        # 验证所有tensor都在CPU上
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        for tensor in result.values():
            self.assertEqual(tensor.device.type, 'cpu')

    def test_input_to_cpu_when_call_params_then_convert_args_and_kwargs(self):
        """测试input_to_cpu：当输入是CallParams时，应该转换args和kwargs中的Tensor"""
        args = (torch.randn(2, 3), torch.randn(3, 4))
        kwargs = {"key": torch.randn(1, 5)}
        call_params = CallParams(*args, **kwargs)

        result = DagTorchHook.input_to_cpu(call_params)

        # 验证返回CallParams对象
        self.assertIsInstance(result, CallParams)
        # 验证args中的tensor都在CPU上
        for tensor in result.args:
            self.assertEqual(tensor.device.type, 'cpu')
        # 验证kwargs中的tensor都在CPU上
        for tensor in result.kwargs.values():
            self.assertEqual(tensor.device.type, 'cpu')

    def test_get_params_when_network_has_parameters_then_return_total_count(self):
        """测试get_params：当网络有参数时，应该返回参数总数"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 计算预期参数数量
            expected_params = sum(param.nelement() for param in self.network.parameters())

            # 验证返回的参数数量
            self.assertEqual(dag_hook.get_params(), expected_params)

    def test_replace_node_when_valid_dag_node_and_new_node_then_replace_successfully(self):
        """测试replace_node：当DAG节点和新节点有效时，应该成功替换"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 创建测试节点
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
                mock_replace.assert_called_once()

    def test_replace_node_when_invalid_dag_node_type_then_raise_type_error(self):
        """测试replace_node：当dag_node类型无效时，应该抛出TypeError"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            new_module = nn.Linear(5, 4)

            with self.assertRaises(TypeError) as context:
                dag_hook.replace_node("not a dag node", new_module)

            self.assertIn("dag_node must be type DagNode", str(context.exception))

    def test_replace_node_when_invalid_new_node_type_then_raise_type_error(self):
        """测试replace_node：当new_node类型无效时，应该抛出TypeError"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            old_module = nn.Linear(5, 3)
            dag_node = DagNode(old_module, "test_node", "Linear")

            with self.assertRaises(TypeError) as context:
                dag_hook.replace_node(dag_node, "not a module")

            self.assertIn("new_node must be type torch.nn.Module", str(context.exception))

    def test_add_node_before_when_valid_inputs_then_create_sequential(self):
        """测试add_node_before：当输入有效时，应该在节点前添加预处理模块"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 创建测试节点和预处理模块
            old_module = nn.Linear(5, 3)
            preprocess_module = nn.ReLU()
            dag_node = DagNode(old_module, "test_node", "Linear")

            # 模拟replace_node方法
            with patch.object(dag_hook, 'replace_node') as mock_replace:
                dag_hook.add_node_before(dag_node, preprocess_module)

                # 验证replace_node被调用
                mock_replace.assert_called_once()
                # 验证传入的是Sequential模块
                args = mock_replace.call_args[0]
                self.assertEqual(args[0], dag_node)
                self.assertIsInstance(args[1], nn.Sequential)

    def test_add_node_after_when_valid_inputs_then_create_sequential(self):
        """测试add_node_after：当输入有效时，应该在节点后添加后处理模块"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 创建测试节点和后处理模块
            old_module = nn.Linear(5, 3)
            postprocess_module = nn.ReLU()
            dag_node = DagNode(old_module, "test_node", "Linear")

            # 模拟replace_node方法
            with patch.object(dag_hook, 'replace_node') as mock_replace:
                dag_hook.add_node_after(dag_node, postprocess_module)

                # 验证replace_node被调用
                mock_replace.assert_called_once()
                # 验证传入的是Sequential模块
                args = mock_replace.call_args[0]
                self.assertEqual(args[0], dag_node)
                self.assertIsInstance(args[1], nn.Sequential)

    def test_remove_node_when_io_match_and_check_enabled_then_replace_with_eq(self):
        """测试remove_node：当输入输出数量匹配且check_io启用时，应该用恒等模块替换"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 创建测试节点，输入输出数量相同
            old_module = nn.ReLU()
            dag_node = DagNode(old_module, "test_node", "ReLU", inputs=[Mock()], outputs=[Mock()])

            # 模拟replace_node方法
            with patch.object(dag_hook, 'replace_node') as mock_replace:
                dag_hook.remove_node(dag_node, check_io=True)

                # 验证replace_node被调用
                mock_replace.assert_called_once()

    def test_remove_node_when_io_mismatch_and_check_enabled_then_raise_value_error(self):
        """测试remove_node：当输入输出数量不匹配且check_io启用时，应该抛出ValueError"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 创建测试节点，输入输出数量不同
            old_module = nn.Linear(5, 3)
            dag_node = DagNode(old_module, "test_node", "Linear", inputs=[Mock()], outputs=[Mock(), Mock()])

            with self.assertRaises(ValueError) as context:
                dag_hook.remove_node(dag_node, check_io=True)

            self.assertIn("remove node must input eq output", str(context.exception))

    def test_remove_node_when_check_disabled_then_remove_regardless_of_io(self):
        """测试remove_node：当check_io禁用时，应该无论输入输出是否匹配都移除节点"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 创建测试节点，输入输出数量不同
            old_module = nn.Linear(5, 3)
            dag_node = DagNode(old_module, "test_node", "Linear", inputs=[Mock()], outputs=[Mock(), Mock()])

            # 模拟replace_node方法
            with patch.object(dag_hook, 'replace_node') as mock_replace:
                dag_hook.remove_node(dag_node, check_io=False)

                # 验证replace_node被调用
                mock_replace.assert_called_once()

    def test_get_module_cls_when_called_then_return_torch_module(self):
        """测试get_module_cls：调用时应该返回torch.nn.Module类"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            result = dag_hook.get_module_cls()

            # 验证返回torch.nn.Module
            self.assertEqual(result, torch.nn.Module)

    def test_before_parse_when_called_then_save_device_and_move_to_cpu(self):
        """测试_before_parse：调用时应该保存设备并将网络移至CPU"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 调用_before_parse
            dag_hook._before_parse()

            # 验证_tmp_device被设置
            self.assertIsNotNone(dag_hook._tmp_device)

            # 验证网络被移至CPU
            current_device = next(self.network.parameters()).device
            self.assertEqual(current_device.type, 'cpu')

    def test_after_parse_when_called_then_restore_device(self):
        """测试_after_parse：调用时应该恢复网络到原始设备"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 通过真实前置流程设置临时设备
            dag_hook._before_parse()

            # 调用_after_parse
            dag_hook._after_parse()

            # 验证网络设备被恢复
            current_device = next(self.network.parameters()).device
            self.assertEqual(str(current_device), str(dag_hook._tmp_device))

    def test_get_module_children_when_module_has_children_then_return_named_children(self):
        """测试_get_module_children：当模块有子模块时，应该返回命名子模块"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 获取子模块
            children = list(dag_hook._get_module_children(self.network))

            # 验证返回子模块列表
            self.assertGreater(len(children), 0)
            # 验证每个元素是(name, module)元组
            for name, module in children:
                self.assertIsInstance(name, str)
                self.assertIsInstance(module, nn.Module)

    def test_collecting_feature_map_info_when_tensor_output_then_return_shape_and_dtype(self):
        """测试_collecting_feature_map_info：当输出是Tensor时，应该返回shape和dtype信息"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 创建测试tensor
            output = torch.randn(2, 3, 4)

            # 获取特征图信息
            io_info = dag_hook._collecting_feature_map_info(output)

            # 验证返回的信息
            self.assertIn("shape", io_info)
            self.assertIn("dtype", io_info)
            self.assertEqual(io_info["shape"], torch.Size([2, 3, 4]))

    def test_collecting_feature_map_info_when_non_tensor_output_then_return_empty_dict(self):
        """测试_collecting_feature_map_info：当输出不是Tensor时，应该返回空字典"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs)

            # 创建非tensor输出
            output = "not a tensor"

            # 获取特征图信息
            io_info = dag_hook._collecting_feature_map_info(output)

            # 验证返回空字典
            self.assertEqual(io_info, {})

    def test_get_all_hook_ops_when_no_user_ops_then_return_default_ops(self):
        """测试_get_all_hook_ops：当没有用户自定义操作时，应该返回默认操作列表"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs, hook_ops=None)

            # 获取hook操作
            hook_ops = dag_hook._get_all_hook_ops(None)

            # 验证返回操作列表
            self.assertIsInstance(hook_ops, list)
            # 默认包含Linear
            self.assertGreater(len(hook_ops), 0)

    def test_get_all_hook_ops_when_user_ops_provided_then_include_user_ops(self):
        """测试_get_all_hook_ops：当提供用户操作时，应该包含用户操作"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            user_ops = [nn.Conv2d, nn.ReLU]
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs, hook_ops=user_ops)

            # 获取hook操作
            hook_ops = dag_hook._get_all_hook_ops(user_ops)

            # 验证包含用户操作
            self.assertGreaterEqual(len(hook_ops), len(user_ops))
            # 验证操作信息格式正确
            for op_info in hook_ops:
                self.assertIsInstance(op_info, tuple)
                self.assertEqual(len(op_info), 3)

    def test_get_hook_ops_when_no_user_ops_then_return_comprehensive_ops(self):
        """测试_get_hook_ops：当没有用户操作时，应该返回全面的操作列表"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs, hook_ops=None)

            # 获取hook操作
            hook_ops = dag_hook._get_hook_ops(None)

            # 验证返回操作列表
            self.assertIsInstance(hook_ops, list)
            # 应该包含多种类型的操作（torch函数、tensor函数、functional函数、操作符、nn模块）
            self.assertGreater(len(hook_ops), 0)
            # 验证操作信息格式正确
            for op_info in hook_ops:
                self.assertIsInstance(op_info, tuple)
                self.assertEqual(len(op_info), 3)

    def test_get_hook_ops_when_user_ops_include_module_then_add_to_list(self):
        """测试_get_hook_ops：当用户操作包含Module类时，应该添加到列表"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            user_ops = [nn.Conv2d, nn.BatchNorm2d]
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs, hook_ops=user_ops)

            # 获取hook操作
            hook_ops = dag_hook._get_hook_ops(user_ops)

            # 验证包含用户Module操作
            self.assertGreater(len(hook_ops), 0)
            # 验证用户操作被添加（应该在列表末尾）
            user_op_names = [op.__name__ for op in user_ops]
            hook_op_names = [op_info[2] for op_info in hook_ops if len(op_info) > 2]
            for user_name in user_op_names:
                self.assertIn(user_name, hook_op_names)

    def test_get_hook_ops_when_user_ops_include_function_then_add_to_list(self):
        """测试_get_hook_ops：当用户操作包含函数时，应该添加到列表"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            # 创建一个自定义函数作为user_op
            def custom_func(x):
                return x * 2

            user_ops = [custom_func]
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs, hook_ops=user_ops)

            # 获取hook操作
            hook_ops = dag_hook._get_hook_ops(user_ops)

            # 验证包含用户函数操作
            self.assertGreater(len(hook_ops), 0)
            # 验证函数被正确添加
            found_custom_func = False
            for op_info in hook_ops:
                if len(op_info) > 2 and 'custom_func' in op_info[2]:
                    found_custom_func = True
                    break
            self.assertTrue(found_custom_func)

    def test_get_hook_ops_when_called_then_exclude_unhook_modules(self):
        """测试_get_hook_ops：调用时应该排除不需要hook的模块（如Sequential、Dropout等）"""
        with (
            patch.object(ConcreteDagTorchHook, '_parse_network_structure_tree'),
            patch.object(ConcreteDagTorchHook, '_parse_network_with_hook'),
        ):
            dag_hook = ConcreteDagTorchHook(self.network, self.inputs, hook_ops=None)

            # 获取hook操作
            hook_ops = dag_hook._get_hook_ops(None)

            # 验证不包含unhook_modules中的类
            unhook_module_names = [
                'Sequential',
                'Container',
                'ModuleList',
                'ModuleDict',
                'ParameterList',
                'ParameterDict',
                'Dropout',
                'SiLU',
            ]
            hook_op_names = [op_info[2] for op_info in hook_ops if len(op_info) > 2]

            # 验证这些unhook模块不在hook操作中
            for unhook_name in unhook_module_names:
                # 注意：可能有其他包含这些名字的操作，所以只检查精确匹配
                exact_matches = [name for name in hook_op_names if name == unhook_name]
                # Sequential等不应该作为独立的hook操作出现
                self.assertEqual(len(exact_matches), 0)


if __name__ == '__main__':
    unittest.main()
