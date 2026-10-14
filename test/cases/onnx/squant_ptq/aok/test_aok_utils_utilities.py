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
import os
import tempfile
import numpy as np
import onnx
from onnx import helper, TensorProto, ModelProto
from unittest.mock import patch, MagicMock
import logging

# 被测试模块
import msmodelslim.onnx.squant_ptq.aok.utils.utilities as aok_utilities
from msmodelslim.onnx.squant_ptq.aok.utils.utilities import __np_type_to_tf_type as np_type_to_tf_type
from msmodelslim.onnx.squant_ptq.aok.utils.utilities import _get_bytearray


class TestONNXUtils(unittest.TestCase):
    def setUp(self):
        # 创建临时目录和测试模型
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.onnx")

        # 创建简单ONNX模型
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])
        node = helper.make_node('Relu', ['X'], ['Y'], name='relu_node')
        graph = helper.make_graph([node], "test_graph", [X], [Y])
        model = helper.make_model(graph)
        onnx.save(model, self.model_path)

        # 设置文件权限
        os.chmod(self.model_path, 0o640)

        # 测试logger
        self.logger = logging.getLogger('test')
        self.logger.addHandler(logging.NullHandler())

    def tearDown(self):
        # 清理临时目录
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_create_empty_folder(self):
        """测试创建空文件夹"""
        test_dir = os.path.join(self.temp_dir, "test_dir")
        aok_utilities.create_empty_folder(test_dir)
        self.assertTrue(os.path.exists(test_dir))
        self.assertEqual(os.stat(test_dir).st_mode & 0o777, 0o750)  # 检查权限

    def test_encode_decode_vector(self):
        """测试向量编码解码"""
        test_vec = [1, 1, 1, 2, 2, 3, 3, 3, 3]
        encoded = aok_utilities.encode_vector(test_vec)
        decoded = aok_utilities.decode_vector(encoded)
        self.assertEqual(decoded, test_vec)

    def test_get_output_shape(self):
        """测试获取输出形状"""
        model = onnx.load(self.model_path)
        shape = aok_utilities.get_output_shape(model)
        self.assertEqual(shape, 3)  # [1,3] -> 1*3=3

    def test_get_model_info(self):
        """测试获取模型信息"""
        nodes_str, input_shape = aok_utilities.get_model_info(self.model_path, 2)
        self.assertIn("relu_node:0", nodes_str)
        self.assertIn("X:2,3", input_shape)

    @patch('subprocess.run')
    def test_onnx2om(self, mock_run):
        """测试ONNX转OM"""
        mock_run.return_value = MagicMock(returncode=0)
        aok_utilities.onnx2om(self.model_path, 1, "test_version", 0, "atc")
        mock_run.assert_called_once()

        with self.assertRaises(ValueError):
            aok_utilities.onnx2om(self.model_path, 1, "test_version", 0, "invalid")

    def test_generate_model_inputs_for_onnxruntime(self):
        # 创建一个简单的 ONNX 模型
        model = ModelProto()
        model.graph.CopyFrom(helper.make_graph([], "test_graph", [], []))

        # 添加输入
        input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [2, 3])
        input2 = helper.make_tensor_value_info("input2", TensorProto.INT64, [1, 4])
        input3 = helper.make_tensor_value_info("input3", TensorProto.DOUBLE, [3, 3])
        model.graph.input.extend([input1, input2, input3])

        # 生成输入数据
        ort_inputs = aok_utilities.generate_model_inputs_for_onnxruntime(model)

        # 验证生成的输入数据
        self.assertIn("input1", ort_inputs)
        self.assertIn("input2", ort_inputs)
        self.assertIn("input3", ort_inputs)

        self.assertEqual(ort_inputs["input1"].shape, (2, 3))
        self.assertEqual(ort_inputs["input1"].dtype, np.float32)

        self.assertEqual(ort_inputs["input2"].shape, (1, 4))
        self.assertEqual(ort_inputs["input2"].dtype, np.int64)

        self.assertEqual(ort_inputs["input3"].shape, (3, 3))
        self.assertEqual(ort_inputs["input3"].dtype, np.float64)

    def test_generate_model_inputs_for_onnxruntime_with_unsupported_type(self):
        # 创建一个包含不支持的数据类型的 ONNX 模型
        model = ModelProto()
        model.graph.CopyFrom(helper.make_graph([], "test_graph", [], []))

        # 添加输入
        input1 = helper.make_tensor_value_info("input1", TensorProto.STRING, [2, 3])
        model.graph.input.extend([input1])

        # 验证是否抛出 ValueError
        with self.assertRaises(ValueError) as context:
            aok_utilities.generate_model_inputs_for_onnxruntime(model)

        self.assertIn(f"Unsupported type: {TensorProto.STRING}", str(context.exception))

    def test_type_np_to_tf_supported_types(self):
        """测试支持的数据类型"""
        self.assertEqual(np_type_to_tf_type(np.float16), TensorProto.FLOAT)
        self.assertEqual(np_type_to_tf_type(np.float32), TensorProto.FLOAT)
        self.assertEqual(np_type_to_tf_type(np.int32), TensorProto.INT32)
        self.assertEqual(np_type_to_tf_type(np.float64), TensorProto.DOUBLE)
        self.assertEqual(np_type_to_tf_type(np.int64), TensorProto.INT64)

    def test_type_np_to_tf_unsupported_type(self):
        """测试不支持的数据类型"""
        with self.assertRaises(ValueError) as context:
            np_type_to_tf_type(np.uint8)
        self.assertIn(f"Unsupported type: {np.uint8}", str(context.exception))

    def test_get_bytearray_raw_data(self):
        """测试 raw_data 不为空的情况"""
        init_input = TensorProto()
        init_input.raw_data = b'\x01\x02\x03\x04'
        np_dtype = np.float32
        result = _get_bytearray(init_input, np_dtype)
        self.assertEqual(result, bytearray(b'\x01\x02\x03\x04'))

    def test_get_bytearray_int64_data(self):
        """测试 int64 数据类型"""
        init_input = TensorProto()
        init_input.int64_data.extend([1, 2, 3, 4])
        np_dtype = np.int64
        result = _get_bytearray(init_input, np_dtype)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_get_bytearray_int32_data(self):
        """测试 int32 数据类型"""
        init_input = TensorProto()
        init_input.int32_data.extend([1, 2, 3, 4])
        np_dtype = np.int32
        result = _get_bytearray(init_input, np_dtype)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_get_bytearray_float_data(self):
        """测试 float 数据类型"""
        init_input = TensorProto()
        init_input.float_data.extend([1.0, 2.0, 3.0, 4.0])
        np_dtype = np.float32
        result = _get_bytearray(init_input, np_dtype)
        self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])

    def test_get_bytearray_uint64_data(self):
        """测试 uint64 数据类型"""
        init_input = TensorProto()
        init_input.uint64_data.extend([1, 2, 3, 4])
        np_dtype = np.uint64
        result = _get_bytearray(init_input, np_dtype)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_get_bytearray_unsupported_data_type(self):
        """测试不支持的数据类型"""
        init_input = TensorProto()
        init_input.float_data.extend([1.0, 2.0, 3.0, 4.0])
        np_dtype = np.complex128  # 不支持的数据类型
        with self.assertRaises(TypeError) as context:
            _get_bytearray(init_input, np_dtype)
        self.assertIn("Unsupported data type of input!", str(context.exception))

    def test_node_io_operations(self):
        """测试节点输入输出操作"""
        node = helper.make_node('Add', ['A', 'B'], ['C'])

        # 测试替换输入
        aok_utilities.replace_node_inputs(node, ['X', 'Y'])
        self.assertEqual(list(node.input), ['X', 'Y'])

        # 测试替换输出
        aok_utilities.replace_node_outputs(node, ['Z'])
        self.assertEqual(list(node.output), ['Z'])

    def test_clean_constant_nodes(self):
        """测试清理常量节点"""
        # 创建带常量节点的模型
        tensor = helper.make_tensor('const', TensorProto.FLOAT, [1], [1.0])
        node = helper.make_node('Constant', [], ['const_out'], value=tensor)
        graph = helper.make_graph([node], "test", [], [])
        model = helper.make_model(graph)

        # 测试清理
        aok_utilities.clean_constant_nodes(model.graph)
        self.assertEqual(len(model.graph.node), 0)

    @patch('onnxruntime.InferenceSession')
    def test_fix_model_outputs(self, mock_session):
        """测试修复模型输出"""
        mock_output = [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]
        mock_session.return_value.run.return_value = mock_output

        model = onnx.load(self.model_path)
        fixed_model = aok_utilities.fix_model_outputs(model, 11, 7, self.logger)
        self.assertEqual(fixed_model.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT)

    def test_check_topology_sorting_valid(self):
        """测试拓扑排序正确的图"""
        # 创建简单 ONNX 模型
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])
        node = helper.make_node('Relu', ['X'], ['Y'], name='relu_node')
        graph = helper.make_graph([node], "test_graph", [X], [Y])
        model = helper.make_model(graph)  # noqa: F841

        logger = MagicMock()
        result = aok_utilities.check_topology_sorting(graph, logger)
        self.assertTrue(result)
        logger.info.assert_called_with('Topology sorting is OK')

    def test_check_topology_sorting_invalid(self):
        """测试拓扑排序错误的图"""
        # 创建简单 ONNX 模型
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])
        node1 = helper.make_node('Relu', ['X'], ['Y'], name='relu_node')
        node2 = helper.make_node('Add', ['Y'], ['Z'], name='add_node')
        graph = helper.make_graph([node1, node2], "test_graph", [X], [Y])
        model = helper.make_model(graph)  # noqa: F841

        # 修改图，使拓扑排序错误
        graph.node[1].input[0] = 'non_existent_input'

        logger = MagicMock()
        result = aok_utilities.check_topology_sorting(graph, logger)
        self.assertFalse(result)
        logger.error.assert_called_with(
            "ERROR: topology is broken: the input 'non_existent_input' of node 'add_node' "
            "(op_type = 'Add') is neither a model input nor an output of any of the previous nodes"
        )

    def test_convert_const_to_init(self):
        """测试常量转初始化器"""
        # 创建带常量节点的模型
        tensor = helper.make_tensor('value', TensorProto.FLOAT, [1], [1.0])
        node = helper.make_node('Constant', [], ['const_out'], value=tensor)
        graph = helper.make_graph([node], "test", [], [])
        model = helper.make_model(graph)

        # 测试转换
        converted = aok_utilities.convert_const_to_init(model, 11, 7, self.logger)
        self.assertEqual(len(converted.graph.node), 0)
        self.assertEqual(len(converted.graph.initializer), 1)

    def test_rebatch_model(self):
        """测试模型批处理修改"""
        model = onnx.load(self.model_path)
        rebatched = aok_utilities.rebatch_model(model, 2, 11, 7, self.logger)
        self.assertEqual(rebatched.graph.input[0].type.tensor_type.shape.dim[0].dim_value, 2)

    def test_define_batch_size_with_explicit_batch_size(self):
        """测试模型输入和输出中明确指定了批量大小"""
        model = ModelProto()
        graph = helper.make_graph([], "test_graph", [], [])
        model.graph.CopyFrom(graph)

        # 添加输入和输出，明确指定批量大小
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 4])
        model.graph.input.extend([input_tensor])
        model.graph.output.extend([output_tensor])

        batch_size = aok_utilities.define_batch_size(model)
        self.assertEqual(batch_size, 2)

    def test_define_batch_size_with_inconsistent_batch_size(self):
        """测试模型输入和输出中批量大小不一致"""
        model = ModelProto()
        graph = helper.make_graph([], "test_graph", [], [])
        model.graph.CopyFrom(graph)

        # 添加输入和输出，批量大小不一致
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 3, 4])
        model.graph.input.extend([input_tensor])
        model.graph.output.extend([output_tensor])

        with self.assertRaises(ValueError) as context:
            aok_utilities.define_batch_size(model)
        self.assertIn("The model has an ambiguously defined batch size", str(context.exception))

    def test_define_batch_size_with_undefined_batch_size(self):
        """测试模型输入和输出中批量大小未明确指定（为 0）"""
        model = ModelProto()
        graph = helper.make_graph([], "test_graph", [], [])
        model.graph.CopyFrom(graph)

        # 添加输入和输出，批量大小未明确指定
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [0, 3, 4])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [0, 3, 4])
        model.graph.input.extend([input_tensor])
        model.graph.output.extend([output_tensor])

        batch_size = aok_utilities.define_batch_size(model, default_batch_size=32)
        self.assertEqual(batch_size, 32)

    def test_load_model(self):
        """测试加载模型"""
        model = aok_utilities.load_model(self.temp_dir, "test_model", self.logger)
        self.assertIsInstance(model, onnx.ModelProto)

        with self.assertRaises(ValueError) as context:
            aok_utilities.load_model(self.temp_dir, "nonexist", self.logger)
        self.assertIn("doesn't exist or not a file", str(context.exception))

    def test_is_model_quantized(self):
        """测试检查模型是否量化"""
        model = onnx.load(self.model_path)
        self.assertFalse(aok_utilities.is_model_quantized(model))

        # 添加量化节点
        node = helper.make_node('AscendQuant', ['X'], ['Y'])
        model.graph.node.append(node)
        self.assertTrue(aok_utilities.is_model_quantized(model))


if __name__ == '__main__':
    unittest.main()
