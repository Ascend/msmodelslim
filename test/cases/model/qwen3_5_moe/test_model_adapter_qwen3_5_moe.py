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

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from torch import nn

from msmodelslim.core.graph import AdapterConfig
from msmodelslim.model.qwen3_5_moe.model_adapter import (
    Qwen3_5ModelAdapter,
    remove_zero_and_shift,
    default_dtype,
)
from msmodelslim.utils.exception import UnsupportedError, InvalidModelError

_scipy_mock = types.ModuleType("scipy")
_scipy_mock.__spec__ = importlib.util.spec_from_loader("scipy", loader=None)
_scipy_mock.__path__ = []
_scipy_linalg_mock = types.ModuleType("scipy.linalg")
_scipy_linalg_mock.__spec__ = importlib.util.spec_from_loader("scipy.linalg", loader=None)
_scipy_linalg_mock.qr = MagicMock()
_scipy_optimize_mock = types.ModuleType("scipy.optimize")
_scipy_optimize_mock.__spec__ = importlib.util.spec_from_loader("scipy.optimize", loader=None)
_scipy_optimize_mock.linear_sum_assignment = MagicMock()
_scipy_mock.linalg = _scipy_linalg_mock
_scipy_mock.optimize = _scipy_optimize_mock
sys.modules.setdefault("scipy", _scipy_mock)
sys.modules.setdefault("scipy.linalg", _scipy_linalg_mock)
sys.modules.setdefault("scipy.optimize", _scipy_optimize_mock)


class DummyTextConfig:
    def __init__(self):
        self.num_hidden_layers = 4
        self.hidden_size = 256
        self.full_attention_interval = 2
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.num_experts = 4
        self.moe_intermediate_size = 128
        self.shared_expert_intermediate_size = 128
        self.hidden_act = "silu"
        self.num_experts_per_tok = 2
        self.rms_norm_eps = 1e-6
        self._attn_implementation = 'eager'


class DummyVisionConfig:
    def __init__(self):
        self.depth = 4


class DummyConfig:
    def __init__(self):
        self.text_config = DummyTextConfig()
        self.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        self.use_cache = False
        self.image_token_id = 151655
        self.vision_config = DummyVisionConfig()


class TestRemoveZeroAndShift(unittest.TestCase):
    """测试remove_zero_and_shift函数的功能"""

    def test_shifts_first_zero_when_single_zero_in_row(self):
        """测试单行含一个零时：应移除该零并在末尾补零"""
        matrix = torch.tensor([[1, 2, 0, 3, 4]])
        result = remove_zero_and_shift(matrix)
        expected = torch.tensor([[1, 2, 3, 4, 0]])
        self.assertTrue(torch.equal(result, expected))

    def test_shifts_leading_zero_when_zero_at_start(self):
        """测试零在行首时：应移除行首零并在末尾补零"""
        matrix = torch.tensor([[0, 1, 2, 3]])
        result = remove_zero_and_shift(matrix)
        expected = torch.tensor([[1, 2, 3, 0]])
        self.assertTrue(torch.equal(result, expected))

    def test_removes_first_element_when_no_zero_in_row(self):
        """测试行中无零时：应移除第一个元素并在末尾补零"""
        matrix = torch.tensor([[1, 2, 3, 4]])
        result = remove_zero_and_shift(matrix)
        expected = torch.tensor([[2, 3, 4, 0]])
        self.assertTrue(torch.equal(result, expected))

    def test_handles_multiple_rows_when_batch_input(self):
        """测试多行输入时：应逐行处理"""
        matrix = torch.tensor([[1, 2, 0, 3], [4, 0, 5, 6]])
        result = remove_zero_and_shift(matrix)
        expected = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
        self.assertTrue(torch.equal(result, expected))

    def test_removes_only_first_zero_when_multiple_zeros(self):
        """测试行中多个零时：应仅移除第一个零"""
        matrix = torch.tensor([[1, 0, 0, 3]])
        result = remove_zero_and_shift(matrix)
        expected = torch.tensor([[1, 0, 3, 0]])
        self.assertTrue(torch.equal(result, expected))

    def test_preserves_dtype_when_input_is_long(self):
        """测试输入为long类型时：应保持数据类型不变"""
        matrix = torch.tensor([[1, 2, 0, 3]], dtype=torch.long)
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.dtype, torch.long)

    def test_all_zeros_when_entire_row_is_zero(self):
        """测试整行为零时：应返回全零矩阵且形状不变"""
        matrix = torch.tensor([[0, 0, 0, 0]])
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.shape, matrix.shape)
        self.assertTrue((result == 0).all())

    def test_shape_preserved_when_called(self):
        """测试输出形状与输入一致"""
        matrix = torch.tensor([[1, 2, 0, 3, 4, 5]])
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.shape, matrix.shape)


class TestDefaultDtype(unittest.TestCase):
    """测试default_dtype上下文管理器的功能"""

    def test_sets_dtype_when_entering_context(self):
        """测试进入上下文时：应设置指定dtype"""
        original = torch.get_default_dtype()
        with default_dtype(torch.float16):
            self.assertEqual(torch.get_default_dtype(), torch.float16)
        self.assertEqual(torch.get_default_dtype(), original)

    def test_restores_dtype_when_exiting_context(self):
        """测试退出上下文时：应恢复原始dtype"""
        original = torch.get_default_dtype()
        with default_dtype(torch.bfloat16):
            pass
        self.assertEqual(torch.get_default_dtype(), original)

    def test_restores_dtype_when_exception_raised(self):
        """测试异常退出时：应恢复原始dtype"""
        original = torch.get_default_dtype()
        try:
            with default_dtype(torch.float16):
                raise ValueError("test error")
        except ValueError:
            pass
        self.assertEqual(torch.get_default_dtype(), original)

    def test_nested_context_when_multiple_levels(self):
        """测试嵌套上下文时：应正确恢复各层dtype"""
        original = torch.get_default_dtype()
        with default_dtype(torch.float16):
            self.assertEqual(torch.get_default_dtype(), torch.float16)
            with default_dtype(torch.bfloat16):
                self.assertEqual(torch.get_default_dtype(), torch.bfloat16)
            self.assertEqual(torch.get_default_dtype(), torch.float16)
        self.assertEqual(torch.get_default_dtype(), original)


class TestQwen3_5ModelAdapterGetModelPedigree(unittest.TestCase):
    """测试Qwen3_5ModelAdapter的get_model_pedigree方法"""

    def test_returns_qwen3_5_moe_when_called(self):
        """测试get_model_pedigree方法：应返回'qwen3_5_moe'"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            result = adapter.get_model_pedigree()
            self.assertEqual(result, 'qwen3_5_moe')


class TestQwen3_5ModelAdapterGetModelType(unittest.TestCase):
    """测试Qwen3_5ModelAdapter的get_model_type方法"""

    def test_returns_model_type_when_called(self):
        """测试get_model_type方法：应返回初始化时传入的model_type"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter.model_type = 'Qwen3_5_MoE'
            result = adapter.get_model_type()
            self.assertEqual(result, 'Qwen3_5_MoE')


class TestQwen3_5ModelAdapterInit(unittest.TestCase):
    """测试Qwen3_5ModelAdapter的初始化"""

    def test_processor_none_when_initialized(self):
        """测试初始化后：_processor应为None"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter._processor = None
            self.assertIsNone(adapter._processor)

    def test_tokenizer_none_when_initialized(self):
        """测试初始化后：_tokenizer应为None"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter._tokenizer = None
            self.assertIsNone(adapter._tokenizer)


class TestQwen3_5ModelAdapterEnableKvCache(unittest.TestCase):
    """测试Qwen3_5ModelAdapter的enable_kv_cache方法"""

    def test_use_cache_true_when_need_kv_cache_is_true(self):
        """测试need_kv_cache为True时：应设置use_cache为True"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            adapter.enable_kv_cache(mock_model, need_kv_cache=True)
            self.assertTrue(mock_model.config.use_cache)

    def test_use_cache_false_when_need_kv_cache_is_false(self):
        """测试need_kv_cache为False时：应设置use_cache为False"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            adapter.enable_kv_cache(mock_model, need_kv_cache=False)
            self.assertFalse(mock_model.config.use_cache)


class TestQwen3_5ModelAdapterGetAdapterConfigForSubgraph(unittest.TestCase):
    """测试Qwen3_5ModelAdapter的get_adapter_config_for_subgraph方法"""

    def test_returns_list_when_called(self):
        """测试调用后：应返回列表"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter.config = DummyConfig()
            result = adapter.get_adapter_config_for_subgraph()
            self.assertIsInstance(result, list)

    def test_correct_config_count_when_full_attention_interval_2(self):
        """测试full_attention_interval为2且4层时：应返回2个配置"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter.config = DummyConfig()
            adapter.config.text_config.full_attention_interval = 2
            adapter.config.text_config.num_hidden_layers = 4
            result = adapter.get_adapter_config_for_subgraph()
            self.assertEqual(len(result), 2)

    def test_empty_list_when_no_full_attention_layers(self):
        """测试full_attention_interval大于层数时：应返回空列表"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter.config = DummyConfig()
            adapter.config.text_config.full_attention_interval = 10
            adapter.config.text_config.num_hidden_layers = 4
            result = adapter.get_adapter_config_for_subgraph()
            self.assertEqual(len(result), 0)

    def test_norm_linear_type_when_config_returned(self):
        """测试返回配置的subgraph_type：应为norm-linear"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter.config = DummyConfig()
            adapter.config.text_config.full_attention_interval = 2
            adapter.config.text_config.num_hidden_layers = 4
            result = adapter.get_adapter_config_for_subgraph()
            for cfg in result:
                self.assertIsInstance(cfg, AdapterConfig)
                self.assertEqual(cfg.subgraph_type, "norm-linear")

    def test_mapping_source_correct_when_config_returned(self):
        """测试返回配置的mapping.source：应包含input_layernorm"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter.config = DummyConfig()
            adapter.config.text_config.full_attention_interval = 2
            adapter.config.text_config.num_hidden_layers = 4
            result = adapter.get_adapter_config_for_subgraph()
            for cfg in result:
                self.assertIn("input_layernorm", cfg.mapping.source)

    def test_mapping_targets_contain_qkv_when_config_returned(self):
        """测试返回配置的mapping.targets：应包含q_proj、k_proj、v_proj"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter.config = DummyConfig()
            adapter.config.text_config.full_attention_interval = 2
            adapter.config.text_config.num_hidden_layers = 4
            result = adapter.get_adapter_config_for_subgraph()
            for cfg in result:
                target_names = cfg.mapping.targets
                self.assertTrue(any("q_proj" in n for n in target_names))
                self.assertTrue(any("k_proj" in n for n in target_names))
                self.assertTrue(any("v_proj" in n for n in target_names))


class TestQwen3_5ModelAdapterHandleDataset(unittest.TestCase):
    """测试Qwen3_5ModelAdapter的handle_dataset方法"""

    def test_raises_unsupported_when_missing_image(self):
        """测试缺少image时：应抛出UnsupportedError"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter._model_path = Path('/fake/path')
            adapter._trust_remote_code = False
            from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample

            dataset = [VlmCalibSample(text="hello", image=None)]
            with patch('msmodelslim.model.qwen3_5_moe.model_adapter.AutoProcessor') as mock_ap:
                mock_ap.from_pretrained.return_value = MagicMock()
                with self.assertRaises(UnsupportedError):
                    adapter.handle_dataset(dataset)

    def test_raises_unsupported_when_missing_text(self):
        """测试缺少text时：应抛出UnsupportedError"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter._model_path = Path('/fake/path')
            adapter._trust_remote_code = False
            from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample

            dataset = [VlmCalibSample(text=None, image="/fake/image.png")]
            with patch('msmodelslim.model.qwen3_5_moe.model_adapter.AutoProcessor') as mock_ap:
                mock_ap.from_pretrained.return_value = MagicMock()
                with self.assertRaises(UnsupportedError):
                    adapter.handle_dataset(dataset)

    def test_raises_unsupported_when_dict_missing_image(self):
        """测试字典格式缺少image时：应抛出UnsupportedError"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter._model_path = Path('/fake/path')
            adapter._trust_remote_code = False
            dataset = [{"text": "hello"}]
            with patch('msmodelslim.model.qwen3_5_moe.model_adapter.AutoProcessor') as mock_ap:
                mock_ap.from_pretrained.return_value = MagicMock()
                with self.assertRaises(UnsupportedError):
                    adapter.handle_dataset(dataset)

    def test_raises_unsupported_when_dict_missing_text(self):
        """测试字典格式缺少text时：应抛出UnsupportedError"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter._model_path = Path('/fake/path')
            adapter._trust_remote_code = False
            dataset = [{"image": "/fake/path"}]
            with patch('msmodelslim.model.qwen3_5_moe.model_adapter.AutoProcessor') as mock_ap:
                mock_ap.from_pretrained.return_value = MagicMock()
                with self.assertRaises(UnsupportedError):
                    adapter.handle_dataset(dataset)


class TestQwen3_5ModelAdapterInitModel(unittest.TestCase):
    """测试Qwen3_5ModelAdapter的init_model方法"""

    def test_raises_invalid_model_when_unknown_architecture(self):
        """测试未知架构时：应抛出InvalidModelError"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter._model_path = Path('/fake/path')
            adapter.config = DummyConfig()
            adapter.config.architectures = ["UnknownArchitecture"]
            with patch('msmodelslim.model.qwen3_5_moe.model_adapter.get_valid_read_path', return_value='/fake/path'):
                with self.assertRaises(InvalidModelError):
                    adapter.init_model(device=MagicMock())


class TestQwen3_5ModelAdapterHasMtp(unittest.TestCase):
    """测试Qwen3_5ModelAdapter的_has_mtp方法"""

    def test_returns_true_when_mtp_keys_present(self):
        """测试权重映射包含mtp键时：应返回True"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter._get_weight_map = MagicMock(
                return_value={
                    'mtp.layers.0.self_attn.q_proj.weight': 'model-00001.safetensors',
                    'model.layers.0.self_attn.q_proj.weight': 'model-00001.safetensors',
                }
            )
            self.assertTrue(adapter._has_mtp())

    def test_returns_false_when_no_mtp_keys(self):
        """测试权重映射不包含mtp键时：应返回False"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            adapter._get_weight_map = MagicMock(
                return_value={
                    'model.layers.0.self_attn.q_proj.weight': 'model-00001.safetensors',
                }
            )
            self.assertFalse(adapter._has_mtp())


class TestQwen3_5ModelAdapterGetWeightMap(unittest.TestCase):
    """测试Qwen3_5ModelAdapter的_get_weight_map方法"""

    def test_returns_dict_when_called(self):
        """测试调用后：应返回字典"""
        import json
        import tempfile
        import os

        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            weight_map = {"model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors"}
            index_data = {"weight_map": weight_map}

            with tempfile.TemporaryDirectory() as tmpdir:
                index_path = os.path.join(tmpdir, "model.safetensors.index.json")
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(index_data, f)
                adapter.model_path = tmpdir
                adapter._get_weight_map.cache_clear()
                result = adapter._get_weight_map()
                self.assertIsInstance(result, dict)
                self.assertIn("model.layers.0.self_attn.q_proj.weight", result)

    def test_cached_when_called_twice(self):
        """测试多次调用时：应返回缓存的同一对象"""
        import json
        import tempfile
        import os

        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            weight_map = {"key1": "file1.safetensors"}
            index_data = {"weight_map": weight_map}

            with tempfile.TemporaryDirectory() as tmpdir:
                index_path = os.path.join(tmpdir, "model.safetensors.index.json")
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(index_data, f)
                adapter.model_path = tmpdir
                adapter._get_weight_map.cache_clear()
                result1 = adapter._get_weight_map()
                result2 = adapter._get_weight_map()
                self.assertIs(result1, result2)


class TestQwen3_5ModelAdapterGetStateDict(unittest.TestCase):
    """测试Qwen3_5ModelAdapter的_get_state_dict方法"""

    def test_returns_dict_when_module_has_params(self):
        """测试模块有参数时：应返回字典"""
        with patch.object(Qwen3_5ModelAdapter.__bases__[0], '__init__', return_value=None):
            adapter = Qwen3_5ModelAdapter(model_type='Qwen3_5_MoE', model_path=Path('/fake/path'))
            weight_map = {"model.layers.0.input_layernorm.weight": "model-00001.safetensors"}
            adapter._get_weight_map = MagicMock(return_value=weight_map)

            mock_module = nn.Linear(4, 4)
            with patch('msmodelslim.model.qwen3_5_moe.model_adapter.safe_open') as mock_safe_open:
                mock_file = MagicMock()
                mock_file.get_tensor.return_value = torch.randn(4, 4)
                mock_safe_open.return_value.__enter__ = MagicMock(return_value=mock_file)
                mock_safe_open.return_value.__exit__ = MagicMock(return_value=False)

                result = adapter._get_state_dict(mock_module, prefix="model.layers.0")
                self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
