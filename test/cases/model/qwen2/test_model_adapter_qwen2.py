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
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.model.qwen2.model_adapter import Qwen2ModelAdapter
from msmodelslim.model.interface_hub import RaCompressAnalysisInterface
from msmodelslim.processor.kv_smooth import KVSmoothFusedType, KVSmoothFusedUnit
from msmodelslim.utils.exception import InvalidModelError


class DummyConfig:
    """模拟配置对象"""

    def __init__(self, num_hidden_layers=3):
        self.hidden_size = 128
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.num_hidden_layers = num_hidden_layers


class TestQwen2ModelAdapterGetModelType(unittest.TestCase):
    """测试Qwen2ModelAdapter的get_model_type方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_get_model_type_with_valid_type_when_called_then_return_model_type(self):
        """正常：应返回初始化时的model_type"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.model_type = self.model_type

            result = adapter.get_model_type()

            self.assertEqual(result, self.model_type)


class TestQwen2ModelAdapterGetModelPedigree(unittest.TestCase):
    """测试Qwen2ModelAdapter的get_model_pedigree方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_get_model_pedigree_when_called_then_return_qwen2(self):
        """正常：应返回qwen2谱系标识"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            result = adapter.get_model_pedigree()

            self.assertEqual(result, 'qwen2')


class TestQwen2ModelAdapterLoadModel(unittest.TestCase):
    """测试Qwen2ModelAdapter的load_model方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_load_model_with_npu_device_when_called_then_delegate_to_load_model(self):
        """正常：使用NPU设备时应委托给_load_model方法"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.load_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)


class TestQwen2ModelAdapterHandleDataset(unittest.TestCase):
    """测试Qwen2ModelAdapter的handle_dataset方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_handle_dataset_with_valid_dataset_when_called_then_return_tokenized_data(self):
        """正常：应委托_get_tokenized_data并返回结果"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_dataset = ['data1', 'data2']
            adapter._get_tokenized_data = MagicMock(return_value=mock_dataset)

            result = adapter.handle_dataset(dataset='test_data', device=DeviceType.CPU)

            self.assertEqual(result, mock_dataset)
            adapter._get_tokenized_data.assert_called_once_with('test_data', DeviceType.CPU)

    def test_handle_dataset_with_empty_dataset_when_called_then_return_empty_list(self):
        """边界：空数据集应返回空列表"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            adapter._get_tokenized_data = MagicMock(return_value=[])

            result = adapter.handle_dataset(dataset=[], device=DeviceType.NPU)

            self.assertEqual(result, [])
            adapter._get_tokenized_data.assert_called_once_with([], DeviceType.NPU)


class TestQwen2ModelAdapterHandleDatasetByBatch(unittest.TestCase):
    """测试Qwen2ModelAdapter的handle_dataset_by_batch方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_handle_dataset_by_batch_with_valid_params_when_called_then_return_batch_data(self):
        """正常：应按批次委托_get_batch_tokenized_data"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_batch_dataset = [['batch1'], ['batch2']]
            adapter._get_batch_tokenized_data = MagicMock(return_value=mock_batch_dataset)

            result = adapter.handle_dataset_by_batch(dataset='test_data', batch_size=2, device=DeviceType.CPU)

            self.assertEqual(result, mock_batch_dataset)
            adapter._get_batch_tokenized_data.assert_called_once_with(
                calib_list='test_data', batch_size=2, device=DeviceType.CPU
            )

    def test_handle_dataset_by_batch_with_batch_size_one_when_called_then_return_single_batches(self):
        """边界：batch_size为1时应正确传递参数"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_batch_dataset = [['item1'], ['item2'], ['item3']]
            adapter._get_batch_tokenized_data = MagicMock(return_value=mock_batch_dataset)

            result = adapter.handle_dataset_by_batch(dataset=['a', 'b', 'c'], batch_size=1, device=DeviceType.NPU)

            self.assertEqual(result, mock_batch_dataset)
            adapter._get_batch_tokenized_data.assert_called_once_with(
                calib_list=['a', 'b', 'c'], batch_size=1, device=DeviceType.NPU
            )


class TestQwen2ModelAdapterInitModel(unittest.TestCase):
    """测试Qwen2ModelAdapter的init_model方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_init_model_with_npu_device_when_called_then_delegate_to_load_model(self):
        """正常：init_model应委托_load_model加载模型"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._load_model = MagicMock(return_value=mock_model)

            result = adapter.init_model(device=DeviceType.NPU)

            self.assertIs(result, mock_model)
            adapter._load_model.assert_called_once_with(DeviceType.NPU)


class TestQwen2ModelAdapterGenerateModelVisit(unittest.TestCase):
    """测试Qwen2ModelAdapter的generate_model_visit方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_generate_model_visit_with_valid_model_when_called_then_yield_from_visit_func(self):
        """正常：应委托generated_decoder_layer_visit_func并yield结果"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            mock_request = MagicMock(spec=ProcessRequest)

            with patch('msmodelslim.model.qwen2.model_adapter.generated_decoder_layer_visit_func') as mock_visit_func:
                mock_visit_func.return_value = iter([mock_request])

                result = list(adapter.generate_model_visit(model=mock_model))

                mock_visit_func.assert_called_once_with(mock_model)
                self.assertEqual(len(result), 1)
                self.assertIs(result[0], mock_request)


class TestQwen2ModelAdapterGenerateModelForward(unittest.TestCase):
    """测试Qwen2ModelAdapter的generate_model_forward方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_generate_model_forward_with_valid_inputs_when_called_then_yield_from_forward_func(self):
        """正常：应委托transformers_generated_forward_func并yield结果"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            mock_inputs = {'input_ids': MagicMock()}
            mock_request = MagicMock(spec=ProcessRequest)

            with patch(
                'msmodelslim.model.qwen2.model_adapter.transformers_generated_forward_func'
            ) as mock_forward_func:
                mock_forward_func.return_value = iter([mock_request])

                result = list(adapter.generate_model_forward(model=mock_model, inputs=mock_inputs))

                mock_forward_func.assert_called_once_with(mock_model, mock_inputs)
                self.assertEqual(len(result), 1)
                self.assertIs(result[0], mock_request)


class TestQwen2ModelAdapterEnableKvCache(unittest.TestCase):
    """测试Qwen2ModelAdapter的enable_kv_cache方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_enable_kv_cache_with_need_cache_true_when_called_then_delegate_to_enable(self):
        """正常：need_kv_cache为True时应委托_enable_kv_cache"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)

            adapter._enable_kv_cache.assert_called_once_with(mock_model, True)

    def test_enable_kv_cache_with_need_cache_false_when_called_then_delegate_with_false(self):
        """边界：need_kv_cache为False时应传递False"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            mock_model = nn.Linear(10, 10)
            adapter._enable_kv_cache = MagicMock(return_value=None)

            adapter.enable_kv_cache(model=mock_model, need_kv_cache=False)

            adapter._enable_kv_cache.assert_called_once_with(mock_model, False)


class TestQwen2ModelAdapterGetKvcacheSmoothFusedSubgraph(unittest.TestCase):
    """测试Qwen2ModelAdapter的get_kvcache_smooth_fused_subgraph方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_get_kvcache_smooth_fused_subgraph_with_valid_config_when_called_then_return_units(self):
        """正常：每层应返回一个KVSmoothFusedUnit"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig(num_hidden_layers=3)

            result = adapter.get_kvcache_smooth_fused_subgraph()

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)

            first_unit = result[0]
            self.assertIsInstance(first_unit, KVSmoothFusedUnit)
            self.assertEqual(first_unit.attention_name, "model.layers.0.self_attn")
            self.assertEqual(first_unit.layer_idx, 0)
            self.assertEqual(first_unit.fused_from_query_states_name, "q_proj")
            self.assertEqual(first_unit.fused_from_key_states_name, "k_proj")
            self.assertEqual(first_unit.fused_type, KVSmoothFusedType.StateViaRopeToLinear)

            last_unit = result[2]
            self.assertEqual(last_unit.attention_name, "model.layers.2.self_attn")
            self.assertEqual(last_unit.layer_idx, 2)

    def test_get_kvcache_smooth_fused_subgraph_with_zero_layers_when_called_then_return_empty_list(self):
        """边界：num_hidden_layers为0时应返回空列表"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig(num_hidden_layers=0)

            result = adapter.get_kvcache_smooth_fused_subgraph()

            self.assertEqual(result, [])


class TestQwen2ModelAdapterGetHeadDim(unittest.TestCase):
    """测试Qwen2ModelAdapter的get_head_dim方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_get_head_dim_with_valid_config_when_called_then_return_calculated_dim(self):
        """正常：应通过hidden_size // num_attention_heads计算"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()

            result = adapter.get_head_dim()

            self.assertEqual(result, 16)

    def test_get_head_dim_with_different_sizes_when_called_then_return_correct_dim(self):
        """边界：不同hidden_size与head数比例应返回正确维度"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'hidden_size': 256, 'num_attention_heads': 16})()

            result = adapter.get_head_dim()

            self.assertEqual(result, 16)

    def test_get_head_dim_missing_hidden_size_when_called_then_raise_invalid_model_error(self):
        """异常：缺少hidden_size时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_head_dim()

            self.assertIn("hidden_size is not found", str(context.exception))

    def test_get_head_dim_missing_num_attention_heads_when_called_then_raise_invalid_model_error(self):
        """异常：缺少num_attention_heads时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'hidden_size': 128})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_head_dim()

            self.assertIn("num_attention_heads is not found", str(context.exception))

    def test_get_head_dim_with_zero_num_attention_heads_when_called_then_raise_invalid_model_error(self):
        """异常：num_attention_heads为0时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'hidden_size': 128, 'num_attention_heads': 0})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_head_dim()

            self.assertIn("num_attention_heads is 0", str(context.exception))


class TestQwen2ModelAdapterGetNumKeyValueGroups(unittest.TestCase):
    """测试Qwen2ModelAdapter的get_num_key_value_groups方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_get_num_key_value_groups_with_valid_config_when_called_then_return_groups(self):
        """正常：有效配置时应返回正确的组数"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()

            result = adapter.get_num_key_value_groups()

            self.assertEqual(result, 2)

    def test_get_num_key_value_groups_with_different_ratios_when_called_then_return_correct_groups(self):
        """边界：不同头数比例应返回正确的组数"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            adapter.config = type('Config', (), {'num_attention_heads': 16, 'num_key_value_heads': 2})()
            self.assertEqual(adapter.get_num_key_value_groups(), 8)

            adapter.config = type('Config', (), {'num_attention_heads': 12, 'num_key_value_heads': 12})()
            self.assertEqual(adapter.get_num_key_value_groups(), 1)

    def test_get_num_key_value_groups_missing_num_attention_heads_when_called_then_raise_error(self):
        """异常：缺少num_attention_heads时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_groups()

            self.assertIn("num_attention_heads is not found", str(context.exception))

    def test_get_num_key_value_groups_missing_num_key_value_heads_when_called_then_raise_error(self):
        """异常：缺少num_key_value_heads时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'num_attention_heads': 8})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_groups()

            self.assertIn("num_key_value_heads is not found", str(context.exception))

    def test_get_num_key_value_groups_with_zero_num_key_value_heads_when_called_then_raise_error(self):
        """异常：num_key_value_heads为0时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'num_attention_heads': 8, 'num_key_value_heads': 0})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_groups()

            self.assertIn("num_key_value_heads is 0", str(context.exception))


class TestQwen2ModelAdapterGetNumKeyValueHeads(unittest.TestCase):
    """测试Qwen2ModelAdapter的get_num_key_value_heads方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_get_num_key_value_heads_with_valid_config_when_called_then_return_heads(self):
        """正常：有效配置时应返回num_key_value_heads"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig()

            result = adapter.get_num_key_value_heads()

            self.assertEqual(result, 4)

    def test_get_num_key_value_heads_with_single_head_when_called_then_return_one(self):
        """边界：MHA场景num_key_value_heads为1时应返回1"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {'num_key_value_heads': 1})()

            result = adapter.get_num_key_value_heads()

            self.assertEqual(result, 1)

    def test_get_num_key_value_heads_missing_config_when_called_then_raise_error(self):
        """异常：缺少num_key_value_heads时应抛出InvalidModelError"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = type('Config', (), {})()

            with self.assertRaises(InvalidModelError) as context:
                adapter.get_num_key_value_heads()

            self.assertIn("num_key_value_heads is not found", str(context.exception))


class TestQwen2ModelAdapterLoadTokenizer(unittest.TestCase):
    """测试Qwen2ModelAdapter的_load_tokenizer方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_load_tokenizer_with_trust_remote_code_true_when_called_then_use_safe_generator(self):
        """正常：trust_remote_code为True时应调用SafeGenerator"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.model_path = self.model_path

            with patch(
                'msmodelslim.model.qwen2.model_adapter.SafeGenerator.get_tokenizer_from_pretrained'
            ) as mock_get_tokenizer:
                mock_tokenizer = MagicMock()
                mock_get_tokenizer.return_value = mock_tokenizer

                result = adapter._load_tokenizer(trust_remote_code=True)

                self.assertIs(result, mock_tokenizer)
                mock_get_tokenizer.assert_called_once_with(
                    model_path=str(self.model_path),
                    use_fast=False,
                    legacy=False,
                    padding_side='left',
                    pad_token='<|extra_0|>',
                    eos_token='<|endoftext|>',
                    trust_remote_code=True,
                )

    def test_load_tokenizer_with_trust_remote_code_false_when_called_then_pass_false(self):
        """边界：trust_remote_code默认False时应传递False"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.model_path = self.model_path

            with patch(
                'msmodelslim.model.qwen2.model_adapter.SafeGenerator.get_tokenizer_from_pretrained'
            ) as mock_get_tokenizer:
                mock_tokenizer = MagicMock()
                mock_get_tokenizer.return_value = mock_tokenizer

                result = adapter._load_tokenizer()

                self.assertIs(result, mock_tokenizer)
                mock_get_tokenizer.assert_called_once_with(
                    model_path=str(self.model_path),
                    use_fast=False,
                    legacy=False,
                    padding_side='left',
                    pad_token='<|extra_0|>',
                    eos_token='<|endoftext|>',
                    trust_remote_code=False,
                )


class TestQwen2ModelAdapterGetAdapterConfigForSubgraph(unittest.TestCase):
    """测试Qwen2ModelAdapter的get_adapter_config_for_subgraph方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_get_adapter_config_for_subgraph_with_valid_layers_when_called_then_return_configs(self):
        """正常：每层应生成3个AdapterConfig"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig(num_hidden_layers=2)

            result = adapter.get_adapter_config_for_subgraph()

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 6)

            norm_linear_attn = result[0]
            self.assertIsInstance(norm_linear_attn, AdapterConfig)
            self.assertEqual(norm_linear_attn.subgraph_type, "norm-linear")
            self.assertIsInstance(norm_linear_attn.mapping, MappingConfig)
            self.assertEqual(norm_linear_attn.mapping.source, "model.layers.0.input_layernorm")
            self.assertEqual(
                norm_linear_attn.mapping.targets,
                [
                    "model.layers.0.self_attn.k_proj",
                    "model.layers.0.self_attn.q_proj",
                    "model.layers.0.self_attn.v_proj",
                ],
            )

            norm_linear_mlp = result[1]
            self.assertEqual(norm_linear_mlp.subgraph_type, "norm-linear")
            self.assertEqual(norm_linear_mlp.mapping.source, "model.layers.0.post_attention_layernorm")
            self.assertEqual(
                norm_linear_mlp.mapping.targets,
                [
                    "model.layers.0.mlp.gate_proj",
                    "model.layers.0.mlp.up_proj",
                ],
            )

            up_down_mapping = result[2]
            self.assertEqual(up_down_mapping.subgraph_type, "up-down")
            self.assertEqual(up_down_mapping.mapping.source, "model.layers.0.mlp.up_proj")
            self.assertEqual(up_down_mapping.mapping.targets, ["model.layers.0.mlp.down_proj"])

            layer1_norm_attn = result[3]
            self.assertEqual(layer1_norm_attn.mapping.source, "model.layers.1.input_layernorm")

    def test_get_adapter_config_for_subgraph_with_zero_layers_when_called_then_return_empty_list(self):
        """边界：num_hidden_layers为0时应返回空列表"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig(num_hidden_layers=0)

            result = adapter.get_adapter_config_for_subgraph()

            self.assertEqual(result, [])

    def test_get_adapter_config_for_subgraph_with_single_layer_when_called_then_return_three_configs(self):
        """边界：单层模型应返回3个AdapterConfig"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            adapter.config = DummyConfig(num_hidden_layers=1)

            result = adapter.get_adapter_config_for_subgraph()

            self.assertEqual(len(result), 3)
            subgraph_types = [cfg.subgraph_type for cfg in result]
            self.assertEqual(subgraph_types, ["norm-linear", "norm-linear", "up-down"])


class TestQwen2ModelAdapterRaCompressInterface(unittest.TestCase):
    """测试Qwen2ModelAdapter实现的RA Compress分析接口"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('.')

    def test_get_proj_names_when_called_then_return_qkv_projection_names(self):
        """正常：应返回 q/k/qkv 三路投影层名，供指标侧按层名匹配"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

            result = adapter.get_proj_names()

        self.assertEqual(result, {"q": "q_proj", "k": "k_proj", "qkv": "qkv_proj"})
        # 与指标侧空串语义约定一致：每个模式都必须非空，否则会命中全部 nn.Linear
        self.assertTrue(all(result.values()))

    def test_get_proj_names_when_called_then_adapter_is_ra_compress_interface(self):
        """正常：适配器应实现 RaCompressAnalysisInterface，使指标侧 isinstance 判定成立"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)

        self.assertIsInstance(adapter, RaCompressAnalysisInterface)

    def test_get_tokenizer_when_called_then_delegate_to_load_tokenizer(self):
        """正常：get_tokenizer 应委托 _load_tokenizer 并返回同一对象"""
        with patch('msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', return_value=None):
            adapter = Qwen2ModelAdapter(model_type=self.model_type, model_path=self.model_path)
            mock_tokenizer = MagicMock()

            with patch.object(adapter, '_load_tokenizer', return_value=mock_tokenizer) as mock_load:
                result = adapter.get_tokenizer()

        self.assertIs(result, mock_tokenizer)
        mock_load.assert_called_once_with()
