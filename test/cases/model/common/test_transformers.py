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
# pylint: disable=attribute-defined-outside-init

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from msmodelslim.core.const import DeviceType
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.utils.exception import SchemaValidateError


class DummyConfig:
    """模拟配置对象"""

    def __init__(self):
        self.model_type = 'DummyModel'
        self.num_hidden_layers = 3


class TestTransformersModelLoadConfig(unittest.TestCase):
    """测试TransformersModel的_load_config方法"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_path = Path('.')

    @patch('msmodelslim.model.common.transformers.SafeGenerator.get_config_from_pretrained')
    def test_load_config_when_called_then_delegate_to_safe_generator(self, mock_get_config):
        """测试_load_config方法：应委托给SafeGenerator"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.model_path = self.model_path

            result = adapter._load_config(trust_remote_code=False)

            self.assertEqual(result, mock_config)
            mock_get_config.assert_called_once_with(model_path=str(self.model_path), trust_remote_code=False)

    @patch('msmodelslim.model.common.transformers.SafeGenerator.get_config_from_pretrained')
    def test_load_config_with_trust_remote_code_when_called_then_pass_trust_flag(self, mock_get_config):
        """测试_load_config方法：trust_remote_code=True时应正确传递"""
        mock_config = DummyConfig()
        mock_get_config.return_value = mock_config

        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.model_path = self.model_path

            result = adapter._load_config(trust_remote_code=True)  # noqa: F841

            mock_get_config.assert_called_once_with(model_path=str(self.model_path), trust_remote_code=True)


class TestTransformersModelLoadTokenizer(unittest.TestCase):
    """测试TransformersModel的_load_tokenizer方法"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_path = Path('.')

    @patch('msmodelslim.model.common.transformers.SafeGenerator.get_tokenizer_from_pretrained')
    def test_load_tokenizer_when_called_then_delegate_to_safe_generator(self, mock_get_tokenizer):
        """测试_load_tokenizer方法：应委托给SafeGenerator"""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.model_path = self.model_path

            result = adapter._load_tokenizer(trust_remote_code=False)

            self.assertEqual(result, mock_tokenizer)
            mock_get_tokenizer.assert_called_once_with(
                model_path=str(self.model_path), use_fast=False, legacy=False, trust_remote_code=False
            )

    @patch('msmodelslim.model.common.transformers.SafeGenerator.get_tokenizer_from_pretrained')
    def test_load_tokenizer_with_trust_remote_code_when_called_then_pass_trust_flag(self, mock_get_tokenizer):
        """测试_load_tokenizer方法：trust_remote_code=True时应正确传递"""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.model_path = self.model_path

            result = adapter._load_tokenizer(trust_remote_code=True)  # noqa: F841

            mock_get_tokenizer.assert_called_once_with(
                model_path=str(self.model_path), use_fast=False, legacy=False, trust_remote_code=True
            )


class TestTransformersModelLoadModel(unittest.TestCase):
    """测试TransformersModel的_load_model方法"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_path = Path('.')

    @patch('msmodelslim.model.common.transformers.SafeGenerator.get_model_from_pretrained')
    def test_load_model_with_npu_device_when_called_then_use_auto_device_map(self, mock_get_model):
        """测试_load_model方法：NPU设备时应使用auto device_map"""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.model_path = self.model_path
            adapter.config = DummyConfig()
            adapter.trust_remote_code = False

            result = adapter._load_model(device=DeviceType.NPU)

            self.assertEqual(result, mock_model)
            # 验证device_map为"auto"
            call_kwargs = mock_get_model.call_args[1]
            self.assertEqual(call_kwargs['device_map'], 'auto')

    @patch('msmodelslim.model.common.transformers.SafeGenerator.get_model_from_pretrained')
    def test_load_model_with_cpu_device_when_called_then_use_cpu_device_map(self, mock_get_model):
        """测试_load_model方法：CPU设备时应使用cpu device_map"""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.model_path = self.model_path
            adapter.config = DummyConfig()
            adapter.trust_remote_code = False

            result = adapter._load_model(device=DeviceType.CPU)  # noqa: F841

            # 验证device_map为"cpu"
            call_kwargs = mock_get_model.call_args[1]
            self.assertEqual(call_kwargs['device_map'], 'cpu')


class TestTransformersModelGetModelType(unittest.TestCase):
    """测试TransformersModel的_get_model_type方法"""

    def test_get_model_type_with_none_when_called_then_return_config_model_type(self):
        """测试_get_model_type方法：model_type为None时应返回config中的model_type"""
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.config = DummyConfig()
            adapter.config.model_type = 'ConfigModelType'

            result = adapter._get_model_type(None)

            self.assertEqual(result, 'ConfigModelType')

    def test_get_model_type_with_value_when_called_then_return_input_value(self):
        """测试_get_model_type方法：model_type有值时应返回输入值"""
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.config = DummyConfig()

            result = adapter._get_model_type('CustomModelType')

            self.assertEqual(result, 'CustomModelType')


class TestTransformersModelGetModelPedigree(unittest.TestCase):
    """测试TransformersModel的_get_model_pedigree方法"""

    def test_get_model_pedigree_with_none_when_called_then_return_config_model_type(self):
        """测试_get_model_pedigree方法：model_type为None时应返回config中的model_type"""
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.config = DummyConfig()
            adapter.config.model_type = 'Qwen2'

            result = adapter._get_model_pedigree(None)

            self.assertEqual(result, 'Qwen2')

    def test_get_model_pedigree_with_valid_name_when_called_then_extract_prefix(self):
        """测试_get_model_pedigree方法：有效名称时应提取前缀并转小写"""
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.config = DummyConfig()

            # 测试场景1: Llama3-8B
            result = adapter._get_model_pedigree('Llama3-8B')
            self.assertEqual(result, 'llama')

            # 测试场景2: DeepSeek-V3
            result = adapter._get_model_pedigree('DeepSeek-V3')
            self.assertEqual(result, 'deepseek')

    def test_get_model_pedigree_with_invalid_name_when_called_then_raise_schema_validate_error(self):
        """测试_get_model_pedigree方法：无效名称时应抛出SchemaValidateError"""
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.config = DummyConfig()

            # 测试没有字母开头的名称
            with self.assertRaises(SchemaValidateError) as context:
                adapter._get_model_pedigree('123-invalid')

            self.assertIn("Invalid model_name", str(context.exception))


class TestTransformersModelGetTokenizedData(unittest.TestCase):
    """测试TransformersModel的_get_tokenized_data方法"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_path = Path('.')

    def test_get_tokenized_data_with_non_list_when_called_then_raise_schema_validate_error(self):
        """测试_get_tokenized_data方法：非列表输入时应抛出SchemaValidateError"""
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)

            # 测试字符串输入
            with self.assertRaises(SchemaValidateError) as context:
                adapter._get_tokenized_data('not_a_list', DeviceType.CPU)

            self.assertIn("calib_list must be a list", str(context.exception))


class TestTransformersModelGetBatchTokenizedData(unittest.TestCase):
    """测试TransformersModel的_get_batch_tokenized_data方法"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_path = Path('.')

    def test_get_batch_tokenized_data_with_valid_list_when_called_then_return_batched_data(self):
        """测试_get_batch_tokenized_data方法：有效列表时应返回批量数据"""
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)

            # Mock _get_padding_data方法
            mock_batch1 = [torch.tensor([[1, 2, 3]])]
            mock_batch2 = [torch.tensor([[4, 5, 6]])]
            adapter._get_padding_data = MagicMock(side_effect=[mock_batch1, mock_batch2])

            calib_list = ['text1', 'text2', 'text3', 'text4']
            result = adapter._get_batch_tokenized_data(calib_list, batch_size=2, device=DeviceType.CPU)

            # 验证返回列表
            self.assertIsInstance(result, list)
            # 4个数据，batch_size=2，应该有2个batch
            self.assertEqual(len(result), 2)

            # 验证_get_padding_data被调用两次
            self.assertEqual(adapter._get_padding_data.call_count, 2)

    def test_get_batch_tokenized_data_with_non_list_when_called_then_raise_schema_validate_error(self):
        """测试_get_batch_tokenized_data方法：非列表输入时应抛出SchemaValidateError"""
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)

            # 测试字符串输入
            with self.assertRaises(SchemaValidateError) as context:
                adapter._get_batch_tokenized_data('not_a_list', batch_size=2, device=DeviceType.CPU)

            self.assertIn("calib_list must be a list", str(context.exception))


class TestTransformersModelGetPaddingData(unittest.TestCase):
    """测试TransformersModel的_get_padding_data方法"""

    def setUp(self):
        """测试前的准备工作"""
        self.model_path = Path('.')

    def test_get_padding_data_with_same_length_when_called_then_no_padding(self):
        """测试_get_padding_data方法：相同长度时不需要padding"""
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)

            # 创建模拟tokenizer，返回相同长度的输入
            mock_tokenizer = MagicMock()
            mock_inputs1 = MagicMock()
            mock_inputs1.data = {'input_ids': torch.tensor([[1, 2, 3]])}
            mock_inputs2 = MagicMock()
            mock_inputs2.data = {'input_ids': torch.tensor([[4, 5, 6]])}
            mock_tokenizer.side_effect = [mock_inputs1, mock_inputs2]
            adapter.tokenizer = mock_tokenizer

            import torch.nn.functional as F_torch

            with patch('msmodelslim.model.common.transformers.F', F_torch):
                calib_list = ['text1', 'text2']
                result = adapter._get_padding_data(calib_list, DeviceType.CPU)

                # 验证返回列表
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 1)  # 返回一个concatenated tensor的列表

    def test_get_padding_data_with_different_lengths_when_called_then_apply_padding(self):
        """测试_get_padding_data方法：不同长度时应应用padding"""
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)

            # 创建模拟tokenizer，返回不同长度的输入
            mock_tokenizer = MagicMock()
            mock_inputs1 = MagicMock()
            mock_inputs1.data = {'input_ids': torch.tensor([[1, 2]])}
            mock_inputs2 = MagicMock()
            mock_inputs2.data = {'input_ids': torch.tensor([[4, 5, 6, 7]])}
            mock_tokenizer.side_effect = [mock_inputs1, mock_inputs2]
            adapter.tokenizer = mock_tokenizer

            import torch.nn.functional as F_torch

            with patch('msmodelslim.model.common.transformers.F', F_torch):
                calib_list = ['short', 'longer text']
                result = adapter._get_padding_data(calib_list, DeviceType.CPU)

                # 验证返回列表
                self.assertIsInstance(result, list)
                # 验证第一个输入被padding到max_len=4
                self.assertEqual(len(result), 1)  # 返回一个concatenated tensor的列表
                self.assertEqual(result[0].shape[1], 4)  # 两个tensor都应该被padding到长度4


class TestTransformersModelGetGlobalModelTorchDtype(unittest.TestCase):
    """测试 TransformersModel.get_global_model_torch_dtype（AscendV1GlobalModelDtypeInterface）。"""

    def setUp(self):
        self.model_path = Path('.')

    def test_config_torch_dtype_none_returns_float32(self):
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.config = DummyConfig()
            self.assertFalse(hasattr(adapter.config, 'torch_dtype'))
        self.assertEqual(adapter.get_global_model_torch_dtype(), torch.float32)

    def test_config_torch_dtype_bfloat16_returns_bfloat16(self):
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.config = DummyConfig()
            adapter.config.torch_dtype = torch.bfloat16
        self.assertEqual(adapter.get_global_model_torch_dtype(), torch.bfloat16)

    def test_config_torch_dtype_str_bfloat16_returns_bfloat16(self):
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.config = DummyConfig()
            adapter.config.torch_dtype = "bfloat16"
        self.assertEqual(adapter.get_global_model_torch_dtype(), torch.bfloat16)

    def test_config_torch_dtype_str_float32_returns_float32(self):
        with patch('msmodelslim.model.common.transformers.TransformersModel.__init__', return_value=None):
            adapter = TransformersModel.__new__(TransformersModel)
            adapter.config = DummyConfig()
            adapter.config.torch_dtype = "float32"
        self.assertEqual(adapter.get_global_model_torch_dtype(), torch.float32)
