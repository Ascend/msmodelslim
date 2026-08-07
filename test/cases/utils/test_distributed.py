#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

msmodelslim.utils.distributed 模块的单元测试
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.distributed as dist
from torch import nn
from msmodelslim.utils.distributed import DistHelper
from msmodelslim.utils.distributed.dist_setup import find_free_port, setup_distributed
from msmodelslim.utils.distributed.dist_ops import ReduceOperation, sync_base_operation, sync_gather_tensors
from msmodelslim.utils.exception import SchemaValidateError, EnvError, UnsupportedError


class TestDistHelper(unittest.TestCase):
    def setUp(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)

        self.test_model = TestModel()
        self.test_model.child_module = nn.Linear(5, 2)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_init_with_mocked_distributed(self, mock_all_gather_object, mock_get_world_size):
        """测试初始化方法"""
        mock_get_world_size.return_value = 2
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            gathered_modules[1] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertEqual(helper._model, self.test_model)
        expected_local_modules = {'', 'linear1', 'relu', 'dropout', 'child_module'}
        self.assertEqual(helper._local_modules, expected_local_modules)
        self.assertEqual(helper._shared_modules, expected_local_modules)
        self.assertEqual(helper._all_modules, expected_local_modules)
        self.assertEqual(helper._local_only_modules, set())

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_local_modules_generator(self, mock_all_gather_object, mock_get_world_size):
        """测试本地模块生成器"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        local_modules = list(helper.local_modules())

        self.assertEqual(len(local_modules), 5)
        self.assertIn(self.test_model, local_modules)
        self.assertIn(self.test_model.linear1, local_modules)
        self.assertIn(self.test_model.relu, local_modules)
        self.assertIn(self.test_model.dropout, local_modules)
        self.assertIn(self.test_model.child_module, local_modules)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_shared_modules_generator(self, mock_all_gather_object, mock_get_world_size):
        """测试共享模块生成器"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        shared_modules = list(helper.shared_modules())

        self.assertEqual(len(shared_modules), 5)
        self.assertIn(self.test_model, shared_modules)
        self.assertIn(self.test_model.linear1, shared_modules)
        self.assertIn(self.test_model.relu, shared_modules)
        self.assertIn(self.test_model.dropout, shared_modules)
        self.assertIn(self.test_model.child_module, shared_modules)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_all_modules_generator(self, mock_all_gather_object, mock_get_world_size):
        """测试所有模块生成器"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        all_modules = list(helper.all_modules())

        self.assertEqual(len(all_modules), 5)
        self.assertIn(self.test_model, all_modules)
        self.assertIn(self.test_model.linear1, all_modules)
        self.assertIn(self.test_model.relu, all_modules)
        self.assertIn(self.test_model.dropout, all_modules)
        self.assertIn(self.test_model.child_module, all_modules)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_local_only_modules_generator(self, mock_all_gather_object, mock_get_world_size):
        """测试仅本地模块生成器"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        local_only_modules = list(helper.local_only_modules())

        self.assertEqual(len(local_only_modules), 0)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    @patch('torch.distributed.get_rank')
    def test_local_only_modules_generator_with_different_modules(
        self, mock_get_rank, mock_all_gather_object, mock_get_world_size
    ):
        """测试仅本地模块生成器在不同模块配置下的行为"""
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = {'', 'module_a', 'module_b'}
            gathered_modules[1] = {'', 'module_a', 'module_c'}

        mock_all_gather_object.side_effect = side_effect

        class LocalOnlyTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.module_a = nn.Linear(10, 5)
                self.module_b = nn.Linear(5, 2)

        local_only_model = LocalOnlyTestModel()
        helper = DistHelper(local_only_model)

        local_only_modules = list(helper.local_only_modules())

        self.assertEqual(len(local_only_modules), 1)
        self.assertIn(local_only_model.module_b, local_only_modules)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_is_local_method(self, mock_all_gather_object, mock_get_world_size):
        """测试is_local方法"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertTrue(helper.is_local('linear1'))
        self.assertTrue(helper.is_local(''))

        self.assertFalse(helper.is_local('non_existent_module'))

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_is_local_only_method(self, mock_all_gather_object, mock_get_world_size):
        """测试is_local_only方法"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertFalse(helper.is_local_only('linear1'))
        self.assertFalse(helper.is_local_only(''))

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_is_shared_method(self, mock_all_gather_object, mock_get_world_size):
        """测试is_shared方法"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertTrue(helper.is_shared('linear1'))
        self.assertTrue(helper.is_shared(''))

        self.assertFalse(helper.is_shared('non_existent_module'))

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_is_all_method(self, mock_all_gather_object, mock_get_world_size):
        """测试is_all方法"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertTrue(helper.is_all('linear1'))
        self.assertTrue(helper.is_all(''))

        self.assertFalse(helper.is_all('non_existent_module'))

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    @patch('torch.distributed.get_rank')
    def test_get_shared_modules_slice(self, mock_get_rank, mock_all_gather_object, mock_get_world_size):
        """测试get_shared_modules_slice方法"""
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            gathered_modules[1] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        result = helper.get_shared_modules_slice()
        expected = sorted(['', 'child_module', 'dropout', 'linear1', 'relu'])[0::2]
        self.assertEqual(result, expected)

        result_with_prefix = helper.get_shared_modules_slice(prefix="model")
        expected_with_prefix = sorted([f"model.{name}" for name in ['', 'child_module', 'dropout', 'linear1', 'relu']])[
            0::2
        ]
        self.assertEqual(result_with_prefix, expected_with_prefix)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    @patch('torch.distributed.get_rank')
    def test_get_shared_modules_slice_different_rank(self, mock_get_rank, mock_all_gather_object, mock_get_world_size):
        """测试不同rank下的get_shared_modules_slice方法"""
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules
            gathered_modules[1] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        result = helper.get_shared_modules_slice()
        expected = sorted(['', 'child_module', 'dropout', 'linear1', 'relu'])[1::2]
        self.assertEqual(result, expected)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    @patch('torch.distributed.get_rank')
    def test_get_rank_method(self, mock_get_rank, mock_all_gather_object, mock_get_world_size):
        """测试get_rank方法"""
        mock_get_world_size.return_value = 1
        mock_get_rank.return_value = 42
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        helper = DistHelper(self.test_model)

        self.assertEqual(helper.get_rank(), 42)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather')
    def test_gather_variable_shapes_static_method(self, mock_all_gather, mock_get_world_size):
        """测试gather_variable_shapes静态方法"""
        mock_get_world_size.return_value = 2
        mock_all_gather.return_value = None

        local_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

        def all_gather_side_effect(tensor_list, tensor):
            tensor_list[:] = [x.clone() for x in [tensor] * len(tensor_list)]

        mock_all_gather.side_effect = all_gather_side_effect

        result = DistHelper.gather_variable_shapes(local_tensor)

        self.assertEqual(len(result), 2)
        for tensor in result:
            self.assertTrue(torch.equal(tensor, local_tensor))
            self.assertEqual(tensor.dtype, local_tensor.dtype)
            self.assertEqual(tensor.shape, local_tensor.shape)
            self.assertEqual(tensor.device, local_tensor.device)

        # all_gather 两次：先 shape，再 data；输入与缓冲应同 device
        self.assertEqual(mock_all_gather.call_count, 2)
        shape_list, local_shape = mock_all_gather.call_args_list[0][0]
        self.assertEqual(local_shape.device, local_tensor.device)
        for buf in shape_list:
            self.assertEqual(buf.device, local_tensor.device)
        data_list, gathered_local = mock_all_gather.call_args_list[1][0]
        self.assertEqual(gathered_local.device, local_tensor.device)
        for buf in data_list:
            self.assertEqual(buf.device, local_tensor.device)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather')
    def test_gather_variable_shapes_with_different_shapes(self, mock_all_gather, mock_get_world_size):
        """测试gather_variable_shapes静态方法处理不同形状的张量"""
        mock_get_world_size.return_value = 2
        mock_all_gather.return_value = None

        local_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)

        def all_gather_side_effect(tensor_list, tensor):
            tensor_list[:] = [x.clone() for x in [tensor] * len(tensor_list)]

        mock_all_gather.side_effect = all_gather_side_effect

        result = DistHelper.gather_variable_shapes(local_tensor)

        self.assertEqual(len(result), 2)
        for tensor in result:
            self.assertTrue(torch.equal(tensor, local_tensor))
            self.assertEqual(tensor.dtype, local_tensor.dtype)
            self.assertEqual(tensor.shape, local_tensor.shape)
            self.assertEqual(tensor.device, local_tensor.device)

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_init_with_empty_model(self, mock_all_gather_object, mock_get_world_size):
        """测试使用空模型初始化"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        empty_model = nn.Module()
        helper = DistHelper(empty_model)

        self.assertEqual(helper._local_modules, {''})
        self.assertEqual(helper._shared_modules, {''})
        self.assertEqual(helper._all_modules, {''})
        self.assertEqual(helper._local_only_modules, set())

    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_init_with_nested_modules(self, mock_all_gather_object, mock_get_world_size):
        """测试带有嵌套模块的模型"""
        mock_get_world_size.return_value = 1
        mock_all_gather_object.return_value = None

        def side_effect(gathered_modules, local_modules):
            gathered_modules[0] = local_modules

        mock_all_gather_object.side_effect = side_effect

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
                self.layer2 = nn.Linear(5, 1)

        nested_model = NestedModel()
        helper = DistHelper(nested_model)

        expected_modules = {'', 'layer1', 'layer1.0', 'layer1.1', 'layer2'}
        self.assertEqual(helper._local_modules, expected_modules)


class TestFindFreePort(unittest.TestCase):
    """测试 find_free_port 函数"""

    @patch('socket.socket')
    def test_find_free_port_retries_on_oserror(self, mock_socket_class):
        """测试端口被占用时会尝试下一个端口"""
        mock_socket = MagicMock()
        mock_socket.__enter__ = MagicMock(return_value=mock_socket)
        mock_socket.__exit__ = MagicMock(return_value=False)
        # 前两次失败，第三次成功
        mock_socket.bind.side_effect = [
            OSError("Port 29500 in use"),
            OSError("Port 29501 in use"),
            None,  # 成功
        ]
        mock_socket_class.return_value = mock_socket

        port = find_free_port(start_port=29500, max_attempts=10)
        self.assertEqual(port, 29502)

    def test_find_free_port_success(self):
        """测试成功找到可用端口"""
        port = find_free_port(start_port=29500, max_attempts=100)
        self.assertIsInstance(port, int)
        self.assertGreaterEqual(port, 29500)
        self.assertLessEqual(port, 29600)

    def test_find_free_port_with_custom_start_port(self):
        """测试使用自定义起始端口"""
        port = find_free_port(start_port=30000, max_attempts=50)
        self.assertIsInstance(port, int)
        self.assertGreaterEqual(port, 30000)
        self.assertLessEqual(port, 30050)

    def test_find_free_port_start_port_too_low(self):
        """测试起始端口小于1024时抛出异常"""
        with self.assertRaises(SchemaValidateError) as context:
            find_free_port(start_port=1023)
        self.assertIn("start_port must be >= 1024", str(context.exception))

    def test_find_free_port_start_port_too_high(self):
        """测试起始端口大于65535时抛出异常"""
        with self.assertRaises(SchemaValidateError) as context:
            find_free_port(start_port=65536)
        self.assertIn("start_port must be <= 65535", str(context.exception))

    @patch('socket.socket')
    def test_find_free_port_all_ports_in_use(self, mock_socket_class):
        """测试所有端口都被占用时抛出异常"""
        mock_socket = MagicMock()
        mock_socket.__enter__ = MagicMock(return_value=mock_socket)
        mock_socket.__exit__ = MagicMock(return_value=False)
        mock_socket.bind.side_effect = OSError("Port in use")
        mock_socket_class.return_value = mock_socket

        with self.assertRaises(EnvError) as context:
            find_free_port(start_port=29500, max_attempts=5)
        self.assertIn("Cannot find a free port", str(context.exception))


class TestSetupDistributed(unittest.TestCase):
    """测试 setup_distributed 函数（CPU环境，通过mock模拟设备操作）"""

    def test_setup_distributed_hccl_backend(self):
        """测试使用 hccl 后端的分布式设置"""
        # 创建 mock 对象
        mock_npu = MagicMock()
        mock_init_process_group = MagicMock()

        # 使用 patch 来 mock torch.npu 和 dist.init_process_group
        with (
            patch.object(torch, 'npu', mock_npu, create=True),
            patch('msmodelslim.utils.distributed.dist_setup.dist.init_process_group', mock_init_process_group),
        ):
            setup_distributed(rank=0, world_size=4, backend='hccl', master_port=29500, device_index=0)

            self.assertEqual(os.environ['MASTER_ADDR'], '127.0.0.1')
            self.assertEqual(os.environ['MASTER_PORT'], '29500')
            self.assertEqual(os.environ['RANK'], '0')
            self.assertEqual(os.environ['WORLD_SIZE'], '4')

            mock_npu.set_device.assert_called_once_with("npu:0")
            mock_init_process_group.assert_called_once_with(backend='hccl', world_size=4, rank=0)

    def test_setup_distributed_device_index_none(self):
        """测试 device_index 为 None 时使用 rank 作为设备索引"""
        mock_npu = MagicMock()
        mock_init_process_group = MagicMock()

        with (
            patch.object(torch, 'npu', mock_npu, create=True),
            patch('msmodelslim.utils.distributed.dist_setup.dist.init_process_group', mock_init_process_group),
        ):
            setup_distributed(rank=2, world_size=4, backend='hccl', master_port=29502, device_index=None)

            mock_npu.set_device.assert_called_once_with("npu:2")
            mock_init_process_group.assert_called_once()

    def test_setup_distributed_device_index_different_from_rank(self):
        """测试 device_index 与 rank 不同的情况"""
        mock_npu = MagicMock()
        mock_init_process_group = MagicMock()

        with (
            patch.object(torch, 'npu', mock_npu, create=True),
            patch('msmodelslim.utils.distributed.dist_setup.dist.init_process_group', mock_init_process_group),
        ):
            setup_distributed(rank=0, world_size=4, backend='hccl', master_port=29503, device_index=3)

            mock_npu.set_device.assert_called_once_with("npu:3")
            mock_init_process_group.assert_called_once_with(backend='hccl', world_size=4, rank=0)


class TestReduceOperation(unittest.TestCase):
    """测试 ReduceOperation 枚举"""

    def test_reduce_operation_from_string(self):
        """测试从字符串创建 ReduceOperation"""
        self.assertEqual(ReduceOperation("min"), ReduceOperation.MIN)
        self.assertEqual(ReduceOperation("max"), ReduceOperation.MAX)
        self.assertEqual(ReduceOperation("sum"), ReduceOperation.SUM)
        self.assertEqual(ReduceOperation("mean"), ReduceOperation.MEAN)
        self.assertEqual(ReduceOperation("prod"), ReduceOperation.PROD)

    def test_reduce_operation_invalid_value(self):
        """测试无效值抛出异常"""
        with self.assertRaises(ValueError):
            ReduceOperation("invalid")


class TestSyncBaseOperation(unittest.TestCase):
    """测试 sync_base_operation 函数（CPU环境，通过mock模拟分布式操作）"""

    @patch('msmodelslim.utils.distributed.dist_ops.dist.all_reduce')
    def test_sync_base_operation_min_with_string(self, mock_all_reduce):
        """测试使用字符串的 min 操作"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = sync_base_operation(tensor, "min")

        mock_all_reduce.assert_called_once_with(tensor, op=dist.ReduceOp.MIN, group=None)
        self.assertIs(result, tensor)

    @patch('msmodelslim.utils.distributed.dist_ops.dist.all_reduce')
    def test_sync_base_operation_max_with_string(self, mock_all_reduce):
        """测试使用字符串的 max 操作"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = sync_base_operation(tensor, "max")

        mock_all_reduce.assert_called_once_with(tensor, op=dist.ReduceOp.MAX, group=None)
        self.assertIs(result, tensor)

    @patch('msmodelslim.utils.distributed.dist_ops.dist.all_reduce')
    def test_sync_base_operation_sum_with_string(self, mock_all_reduce):
        """测试使用字符串的 sum 操作"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = sync_base_operation(tensor, "sum")

        mock_all_reduce.assert_called_once_with(tensor, op=dist.ReduceOp.SUM, group=None)
        self.assertIs(result, tensor)

    @patch('msmodelslim.utils.distributed.dist_ops.dist.all_reduce')
    def test_sync_base_operation_prod_with_string(self, mock_all_reduce):
        """测试使用字符串的 prod 操作"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = sync_base_operation(tensor, "prod")

        mock_all_reduce.assert_called_once_with(tensor, op=dist.ReduceOp.PRODUCT, group=None)
        self.assertIs(result, tensor)

    @patch('msmodelslim.utils.distributed.dist_ops.dist.get_world_size')
    @patch('msmodelslim.utils.distributed.dist_ops.dist.all_reduce')
    def test_sync_base_operation_mean_with_string(self, mock_all_reduce, mock_get_world_size):
        """测试使用字符串的 mean 操作"""
        mock_get_world_size.return_value = 2
        tensor = torch.tensor([2.0, 4.0, 6.0])
        result = sync_base_operation(tensor, "mean")

        mock_all_reduce.assert_called_once_with(tensor, op=dist.ReduceOp.SUM, group=None)
        mock_get_world_size.assert_called_once_with(None)
        self.assertIs(result, tensor)

    def test_sync_base_operation_invalid_string(self):
        """测试无效操作字符串抛出异常"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        with self.assertRaises(UnsupportedError) as context:
            sync_base_operation(tensor, "invalid_op")
        self.assertIn("Unsupported operation", str(context.exception))

    @patch('msmodelslim.utils.distributed.dist_ops.dist.all_reduce')
    def test_sync_base_operation_case_insensitive(self, mock_all_reduce):
        """测试字符串操作不区分大小写"""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        # 测试各种大小写组合
        sync_base_operation(tensor, "MIN")
        sync_base_operation(tensor, "Min")
        sync_base_operation(tensor, "mIn")

        self.assertEqual(mock_all_reduce.call_count, 3)


class TestSyncGatherTensors(unittest.TestCase):
    """测试 sync_gather_tensors 函数（CPU环境，通过mock模拟分布式操作）"""

    @patch('msmodelslim.utils.distributed.dist_ops.dist.get_rank')
    @patch('msmodelslim.utils.distributed.dist_ops.dist.get_world_size')
    @patch('msmodelslim.utils.distributed.dist_ops.dist.all_gather')
    def test_sync_gather_tensors_same_shape_on_device(self, mock_all_gather, mock_get_world_size, mock_get_rank):
        """测试在设备上聚合相同形状的张量"""
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0

        tensor = torch.tensor([1.0, 2.0, 3.0])

        def all_gather_side_effect(tensor_list, tensor, group=None):
            for _, t in enumerate(tensor_list):
                t.copy_(tensor)

        mock_all_gather.side_effect = all_gather_side_effect

        result = sync_gather_tensors(tensor, variable_shapes=False, on_cpu=False)

        self.assertEqual(len(result), 2)
        mock_all_gather.assert_called_once()

    @patch('msmodelslim.utils.distributed.dist_ops.dist.get_rank')
    @patch('msmodelslim.utils.distributed.dist_ops.dist.get_world_size')
    @patch('msmodelslim.utils.distributed.dist_ops.dist.all_gather_object')
    def test_sync_gather_tensors_on_cpu(self, mock_all_gather_object, mock_get_world_size, mock_get_rank):
        """测试在 CPU 上聚合张量"""
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0

        tensor = torch.tensor([1.0, 2.0, 3.0])

        def all_gather_object_side_effect(tensor_list, tensor_cpu, group=None):
            tensor_list[0] = tensor_cpu
            tensor_list[1] = tensor_cpu

        mock_all_gather_object.side_effect = all_gather_object_side_effect

        result = sync_gather_tensors(tensor, on_cpu=True)

        self.assertEqual(len(result), 2)
        mock_all_gather_object.assert_called_once()

    @patch('msmodelslim.utils.distributed.dist_ops.torch.device')
    @patch('msmodelslim.utils.distributed.dist_ops.dist.get_rank')
    @patch('msmodelslim.utils.distributed.dist_ops.dist.get_world_size')
    @patch('msmodelslim.utils.distributed.dist_ops.dist.all_gather')
    def test_sync_gather_tensors_variable_shapes(
        self, mock_all_gather, mock_get_world_size, mock_get_rank, mock_device
    ):
        """测试聚合不同形状的张量（CPU环境）"""
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0
        # Mock torch.device 上下文管理器，使其在 CPU 上也能正常工作
        mock_device.return_value.__enter__ = MagicMock(return_value=None)
        mock_device.return_value.__exit__ = MagicMock(return_value=False)

        tensor = torch.tensor([1.0, 2.0, 3.0])

        call_count = [0]

        def all_gather_side_effect(tensor_list, tensor, group=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # 第一次调用：收集形状信息
                for t in tensor_list:
                    t.copy_(torch.tensor([3], dtype=torch.long))
            else:
                # 第二次调用：收集实际数据
                for t in tensor_list:
                    t.copy_(tensor)

        mock_all_gather.side_effect = all_gather_side_effect

        result = sync_gather_tensors(tensor, variable_shapes=True, on_cpu=False)

        self.assertEqual(len(result), 2)
        self.assertEqual(mock_all_gather.call_count, 2)

    @patch('msmodelslim.utils.distributed.dist_ops.dist.get_rank')
    @patch('msmodelslim.utils.distributed.dist_ops.dist.get_world_size')
    @patch('msmodelslim.utils.distributed.dist_ops.dist.all_gather')
    def test_sync_gather_tensors_2d_tensor(self, mock_all_gather, mock_get_world_size, mock_get_rank):
        """测试聚合二维张量"""
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0

        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        def all_gather_side_effect(tensor_list, tensor, group=None):
            for t in tensor_list:
                t.copy_(tensor)

        mock_all_gather.side_effect = all_gather_side_effect

        result = sync_gather_tensors(tensor, variable_shapes=False, on_cpu=False)

        self.assertEqual(len(result), 2)
        for t in result:
            self.assertEqual(t.shape, torch.Size([2, 2]))


if __name__ == '__main__':
    unittest.main()
