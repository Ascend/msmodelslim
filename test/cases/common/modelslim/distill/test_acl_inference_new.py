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

import unittest
from unittest.mock import Mock, patch
import numpy as np
import acl  # noqa: F401
from ascend_utils.common import acl_inference
from ascend_utils.common.acl_inference import (
    AclInference,
    init_acl,
    release_acl,
    _check_ret,
    ACL_ERROR_NONE,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEM_MALLOC_HUGE_FIRST,
    NodeType,
)


class TestAclInference(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        self.device_id = 0
        self.model_path = "test_model.om"

        # Mock ACL相关函数
        self.acl_mock_patcher = patch('ascend_utils.common.acl_inference.acl')
        self.mock_acl = self.acl_mock_patcher.start()

        # 设置ACL函数返回值
        self.mock_acl.ERROR_NONE = ACL_ERROR_NONE
        self.mock_acl.MEM_MALLOC_HUGE_FIRST = ACL_MEM_MALLOC_HUGE_FIRST
        self.mock_acl.memcpy.HOST_TO_DEVICE = ACL_MEMCPY_HOST_TO_DEVICE
        self.mock_acl.memcpy.DEVICE_TO_HOST = ACL_MEMCPY_DEVICE_TO_HOST

        # 模拟成功返回值
        self.mock_acl.init.return_value = ACL_ERROR_NONE
        self.mock_acl.rt.set_device.return_value = ACL_ERROR_NONE
        self.mock_acl.rt.create_context.return_value = (Mock(), ACL_ERROR_NONE)
        self.mock_acl.mdl.load_from_file.return_value = (Mock(), ACL_ERROR_NONE)
        self.mock_acl.mdl.create_desc.return_value = Mock()
        self.mock_acl.mdl.get_desc.return_value = ACL_ERROR_NONE
        self.mock_acl.rt.malloc.return_value = (Mock(), ACL_ERROR_NONE)
        self.mock_acl.mdl.execute.return_value = ACL_ERROR_NONE
        self.mock_acl.create_data_buffer.return_value = Mock()
        self.mock_acl.mdl.add_dataset_buffer.return_value = (Mock(), ACL_ERROR_NONE)
        self.mock_acl.destroy_data_buffer.return_value = ACL_ERROR_NONE
        self.mock_acl.mdl.destroy_dataset.return_value = ACL_ERROR_NONE
        self.mock_acl.mdl.unload.return_value = ACL_ERROR_NONE
        self.mock_acl.rt.destroy_context.return_value = ACL_ERROR_NONE
        self.mock_acl.finalize.return_value = ACL_ERROR_NONE
        self.mock_acl.rt.reset_device.return_value = ACL_ERROR_NONE
        self.mock_acl.rt.memcpy.return_value = ACL_ERROR_NONE
        self.mock_acl.mdl.set_dataset_tensor_desc.return_value = (Mock(), ACL_ERROR_NONE)

        # 模拟模型描述信息
        self.mock_model_desc = Mock()
        self.mock_acl.mdl.create_desc.return_value = self.mock_model_desc

        # 模拟输入输出信息
        self.mock_input_dims = [{"name": "input1", "dims": [1, 3, 224, 224]}]
        self.mock_output_dims = [{"name": "output1", "dims": [1, 1000]}]

        # 设置模型描述函数返回值
        self.mock_acl.mdl.get_num_inputs.return_value = 1
        self.mock_acl.mdl.get_num_outputs.return_value = 1
        self.mock_acl.mdl.get_input_dims.side_effect = [(self.mock_input_dims[0], ACL_ERROR_NONE)]
        self.mock_acl.mdl.get_output_dims.side_effect = [(self.mock_output_dims[0], ACL_ERROR_NONE)]
        self.mock_acl.mdl.get_input_format.return_value = 0
        self.mock_acl.mdl.get_output_format.return_value = 0
        self.mock_acl.mdl.get_input_size_by_index.return_value = 1024
        self.mock_acl.mdl.get_output_size_by_index.return_value = 4096
        self.mock_acl.mdl.get_input_data_type.return_value = 0  # float32
        self.mock_acl.mdl.get_output_data_type.return_value = 0  # float32
        self.mock_acl.mdl.get_output_name_by_index.return_value = "output1"

    def tearDown(self):
        """测试后清理"""
        self.acl_mock_patcher.stop()

    def test_check_ret_success(self):
        """测试_check_ret成功情况"""
        # 不应该抛出异常
        _check_ret("test operation", ACL_ERROR_NONE)

    def test_check_ret_failure(self):
        """测试_check_ret失败情况"""
        with self.assertRaises(Exception) as context:
            _check_ret("test operation", 1)  # 非零返回码

        self.assertIn("test operation failed", str(context.exception))

    @patch('ascend_utils.common.acl_inference.logger')
    def test_init_acl_success(self, mock_logger):
        """测试ACL初始化成功"""
        # 模拟设备已初始化
        self.mock_acl.rt.get_device.return_value = (Mock(), ACL_ERROR_NONE)

        init_acl(device_id=0)

        # 验证日志记录
        mock_logger.info.assert_called_with("acl set_device 0")

    @patch('ascend_utils.common.acl_inference.logger')
    def test_init_acl_first_time(self, mock_logger):
        """测试首次ACL初始化"""
        # 模拟设备未初始化
        self.mock_acl.rt.get_device.return_value = (Mock(), 1)

        init_acl(device_id=0)

        # 验证ACL初始化被调用
        self.mock_acl.init.assert_called_once()
        self.mock_acl.rt.set_device.assert_called_with(0)
        mock_logger.info.assert_called_with("acl set_device 0")

    @patch('ascend_utils.common.acl_inference.logger')
    def test_release_acl(self, mock_logger):
        """测试ACL释放"""
        # 设置全局变量
        acl_inference.IS_ACL_INITIALIZED_BY_THIS_MODULE = True

        release_acl(device_id=0)

        # 验证资源释放调用
        self.mock_acl.rt.reset_device.assert_called_with(0)
        self.mock_acl.finalize.assert_called_once()
        mock_logger.info.assert_any_call("end to reset device 0")
        mock_logger.info.assert_any_call("end to finalize acl")

    def test_acl_inference_init_success(self):
        """测试AclInference初始化成功"""
        with (
            patch.object(AclInference, '_init_input_device_buffer') as mock_input_buf,
            patch.object(AclInference, '_init_output_device_buffer') as mock_output_buf,
            patch.object(AclInference, '_init_output_host_buffer') as mock_host_buf,
        ):
            mock_input_buf.return_value = [{"buffer": Mock(), "size": 1024}]
            mock_output_buf.return_value = [{"buffer": Mock(), "size": 4096}]
            mock_host_buf.return_value = ([np.array([0])], [{"buffer": 1234, "size": 4096}])

        with patch('ascend_utils.common.acl_inference.get_valid_read_path', side_effect=lambda x, **kwargs: x):
            # 创建实例
            acl_infer = AclInference(self.model_path, self.device_id)

            # 验证初始化成功
            self.assertTrue(acl_infer._init_success)
            self.assertEqual(acl_infer.num_inputs, 1)
            self.assertEqual(acl_infer.num_outputs, 1)
            self.assertIsNotNone(acl_infer.context)
            self.assertIsNotNone(acl_infer.model_id)
            self.assertIsNotNone(acl_infer.model_desc)

    def test_acl_inference_init_failure(self):
        """测试AclInference初始化失败"""
        # 模拟模型加载失败
        self.mock_acl.mdl.load_from_file.return_value = (None, 1)

        with self.assertRaises(Exception):
            AclInference(self.model_path, self.device_id)

    def test_acl_inference_init_zero_inputs(self):
        """测试零输入模型初始化失败"""
        self.mock_acl.mdl.get_num_inputs.return_value = 0

        with self.assertRaises(ValueError) as context:
            AclInference(self.model_path, self.device_id)

        self.assertIn("doesn't exist or not a file", str(context.exception))

    def test_get_inputs(self):
        """测试获取输入信息"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer.model_desc = self.mock_model_desc

        inputs = acl_infer.get_inputs()

        # 验证ACL函数调用
        self.mock_acl.mdl.get_num_inputs.assert_called_with(self.mock_model_desc)
        self.mock_acl.mdl.get_input_dims.assert_called()

        # 验证返回的NodeType
        self.assertEqual(len(inputs), 1)
        self.assertIsInstance(inputs[0], NodeType)
        self.assertEqual(inputs[0].name, "input1")

    def test_get_outputs(self):
        """测试获取输出信息"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer.model_desc = self.mock_model_desc

        outputs = acl_infer.get_outputs()

        # 验证ACL函数调用
        self.mock_acl.mdl.get_num_outputs.assert_called_with(self.mock_model_desc)
        self.mock_acl.mdl.get_output_dims.assert_called()

        # 验证返回的NodeType
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], NodeType)
        self.assertEqual(outputs[0].name, "output1")

    @patch('ascend_utils.common.acl_inference.time')
    def test_call_success(self, mock_time):
        """测试模型推理调用成功"""
        mock_time.time.side_effect = [0, 0.1]  # 模拟执行时间

        # 创建AclInference实例
        acl_infer = AclInference.__new__(AclInference)
        acl_infer._init_success = True
        acl_infer.context = Mock()
        acl_infer.model_id = Mock()
        acl_infer.inputs = [
            NodeType("input1", [1, 3, 224, 224], 0, 1024, 0)  # dtype 0 = float32
        ]
        acl_infer.outputs = [NodeType("output1", [1, 1000], 0, 4096, 0)]
        acl_infer.num_inputs = 1
        acl_infer.num_outputs = 1
        acl_infer.model_desc = Mock()
        acl_infer.input_data_buffer = [{"buffer": Mock(), "size": 1024}]
        acl_infer.output_data_buffer = [{"buffer": Mock(), "size": 4096}]
        acl_infer.output_host_buffer = [{"buffer": 1234, "size": 4096}]
        acl_infer.output_host_bytes_data = [np.zeros(4096, dtype=np.bool_)]
        acl_infer.execute_time_ms = 0

        # Mock内部方法
        with (
            patch.object(acl_infer, '_input_data_from_host_to_device') as mock_input,
            patch.object(acl_infer, '_create_output_data_device_buffer') as mock_output,
            patch.object(acl_infer, '_output_data_from_device_to_host') as mock_output_host,
            patch.object(acl_infer, '_destroy_data_buffer') as mock_destroy,
        ):
            mock_input.return_value = Mock()
            mock_output.return_value = Mock()
            mock_output_host.return_value = [np.array([0.1, 0.2])]

            # 准备输入数据
            input_data = np.ones([1, 3, 224, 224], dtype=np.float32)

            # 执行推理
            acl_infer(input_data)

            # 验证调用
            mock_input.assert_called_once_with([input_data])
            mock_output.assert_called_once()
            self.mock_acl.mdl.execute.assert_called_once()
            mock_output_host.assert_called_once_with(output_shape=[[1, 1000]])
            mock_destroy.assert_any_call(mock_input.return_value)
            mock_destroy.assert_any_call(mock_output.return_value)

            # 验证执行时间记录
            self.assertEqual(acl_infer.execute_time_ms, 100)  # 100ms

    def test_call_input_shape_mismatch(self):
        """测试输入形状不匹配"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer._init_success = True
        acl_infer.context = None
        acl_infer.inputs = [NodeType("input1", [1, 3, 224, 224], 0, 1024, 0)]
        acl_infer.num_inputs = 1

        # 形状不匹配的输入数据
        input_data = np.ones([1, 3, 112, 112], dtype=np.float32)

        with self.assertRaises(ValueError) as context:
            acl_infer(input_data)

        self.assertIn("input data shape", str(context.exception))

    def test_get_execute_time(self):
        """测试获取执行时间"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer.execute_time_ms = 150.5

        result = acl_infer.get_execute_time()

        self.assertEqual(result, 150.5)

    def test_release_resource(self):
        """测试资源释放"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer.context = Mock()
        acl_infer.model_id = Mock()
        acl_infer.model_desc = Mock()
        acl_infer.input_data_buffer = [{"buffer": Mock(), "size": 1024}]
        acl_infer.output_data_buffer = [{"buffer": Mock(), "size": 4096}]
        acl_infer.output_host_bytes_data = [Mock()]
        acl_infer.output_host_buffer = [Mock()]

        with patch.object(acl_infer, '_destroy_data_buffer') as mock_destroy:  # noqa: F841
            acl_infer.release_resource()

            # 验证内存释放调用
            self.mock_acl.rt.free.assert_called()
            self.mock_acl.mdl.destroy_desc.assert_called_with(acl_infer.model_desc)
            self.mock_acl.mdl.unload.assert_called_with(acl_infer.model_id)
            self.mock_acl.rt.destroy_context.assert_called_with(acl_infer.context)

    def test_init_input_device_buffer(self):
        """测试初始化输入设备缓冲区"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer.model_desc = self.mock_model_desc
        acl_infer.num_inputs = 1

        result = acl_infer._init_input_device_buffer()

        # 验证内存分配和初始化
        self.mock_acl.rt.malloc.assert_called_with(1024, ACL_MEM_MALLOC_HUGE_FIRST)
        self.mock_acl.rt.memset.assert_called()
        self.assertEqual(len(result), 1)
        self.assertIn("buffer", result[0])
        self.assertIn("size", result[0])

    def test_init_output_device_buffer(self):
        """测试初始化输出设备缓冲区"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer.model_desc = self.mock_model_desc
        acl_infer.num_outputs = 1

        result = acl_infer._init_output_device_buffer()

        # 验证内存分配和初始化
        self.mock_acl.rt.malloc.assert_called_with(4096, ACL_MEM_MALLOC_HUGE_FIRST)
        self.mock_acl.rt.memset.assert_called()
        self.assertEqual(len(result), 1)
        self.assertIn("buffer", result[0])
        self.assertIn("size", result[0])

    def test_init_output_host_buffer(self):
        """测试初始化输出主机缓冲区"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer.model_desc = self.mock_model_desc
        acl_infer.num_outputs = 1

        bytes_data, buffer_info = acl_infer._init_output_host_buffer()

        # 验证缓冲区创建
        self.assertEqual(len(bytes_data), 1)
        self.assertEqual(len(buffer_info), 1)
        self.assertIsInstance(bytes_data[0], np.ndarray)
        self.assertEqual(bytes_data[0].dtype, np.bool_)
        self.assertIn("buffer", buffer_info[0])
        self.assertIn("size", buffer_info[0])

    def test_input_data_from_host_to_device(self):
        """测试主机到设备数据传输"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer.model_desc = self.mock_model_desc
        acl_infer.num_inputs = 1
        acl_infer.inputs = [NodeType("input1", [1, 3, 224, 224], 0, 1024, 0)]
        acl_infer.input_data_buffer = [{"buffer": Mock(), "size": 1024}]

        input_data = [np.ones([1, 3, 224, 224], dtype=np.float32)]

        with patch.object(acl_infer, '_init_acl_data_buffer') as mock_init_buf:
            acl_infer._input_data_from_host_to_device(input_data)

            mock_init_buf.assert_called()

    def test_create_output_data_device_buffer(self):
        """测试创建设备输出数据缓冲区"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer.num_outputs = 1
        acl_infer.output_data_buffer = [{"buffer": Mock(), "size": 4096}]

        with patch.object(acl_infer, '_init_acl_data_buffer') as mock_init_buf:
            result = acl_infer._create_output_data_device_buffer()  # noqa: F841

            # 验证数据缓冲区初始化
            mock_init_buf.assert_called_once()

    def test_output_data_from_device_to_host(self):
        """测试设备到主机数据传输"""
        acl_infer = AclInference.__new__(AclInference)
        acl_infer.num_outputs = 1
        acl_infer.outputs = [
            NodeType("output1", [1, 1000], 0, 4096, 0)  # float32
        ]
        acl_infer.output_data_buffer = [{"buffer": Mock(), "size": 4096}]
        acl_infer.output_host_buffer = [{"buffer": 1234, "size": 4096}]
        acl_infer.output_host_bytes_data = [np.zeros(4096, dtype=np.bool_)]

        output_shape = [[1, 1000]]

        result = acl_infer._output_data_from_device_to_host(output_shape)

        # 验证内存拷贝和结果转换
        self.mock_acl.rt.memcpy.assert_called_with(
            1234,  # 主机缓冲区指针
            1000 * 4,  # 数据大小 (1000个float32元素)
            acl_infer.output_data_buffer[0]["buffer"],
            1000 * 4,
            ACL_MEMCPY_DEVICE_TO_HOST,
        )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)

    def test_destroy_data_buffer(self):
        """测试数据缓冲区销毁"""
        mock_dataset = Mock()
        self.mock_acl.mdl.get_dataset_num_buffers.return_value = 2
        self.mock_acl.mdl.get_dataset_buffer.side_effect = [Mock(), Mock()]

        AclInference._destroy_data_buffer(mock_dataset)

        # 验证缓冲区销毁调用
        self.assertEqual(self.mock_acl.destroy_data_buffer.call_count, 2)
        self.mock_acl.mdl.destroy_dataset.assert_called_with(mock_dataset)

    def test_destroy_data_buffer_none(self):
        """测试销毁空数据缓冲区"""
        # 不应该抛出异常
        AclInference._destroy_data_buffer(None)


if __name__ == '__main__':
    unittest.main()
