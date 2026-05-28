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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from msmodelslim.core.const import DeviceType
from msmodelslim.model.glm_5.model_adapter import GLM5ModelAdapter


class DummyGLM5Config:
    def __init__(self):
        self.num_hidden_layers = 2
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.qk_nope_head_dim = 64
        self.v_head_dim = 64
        self.first_k_dense_replace = 1
        self.n_routed_experts = 8
        self.n_shared_experts = 1


class TestGLM5ModelAdapter(unittest.TestCase):
    def setUp(self):
        self.model_path = Path('.')
        self.model_type = 'GLM-5'
        self.dummy_config = DummyGLM5Config()
        self.adapter_patcher = patch.object(GLM5ModelAdapter, '__init__', lambda x, model_path, model_type: None)

    def create_adapter(self, **kwargs):
        with self.adapter_patcher:
            adapter = GLM5ModelAdapter(model_path=self.model_path, model_type=self.model_type)
            adapter.config = self.dummy_config
            adapter.model_path = self.model_path
            adapter.model_type = self.model_type
            for key, value in kwargs.items():
                setattr(adapter, key, value)
            return adapter

    # Verify model pedigree returns the expected GLM-5 identifier.
    def test_get_model_pedigree_returns_glm_5_when_adapter_created(self):
        adapter = self.create_adapter()
        self.assertEqual(adapter.get_model_pedigree(), 'glm_5')

    # Verify get_model_type returns the adapter's configured model type.
    def test_get_model_type_returns_model_type_when_set(self):
        adapter = self.create_adapter(model_type=self.model_type)
        self.assertEqual(adapter.get_model_type(), self.model_type)

    # Verify handle_dataset delegates dataset processing to the tokenizer helper.
    def test_handle_dataset_calls_tokenizer_when_handling_dataset(self):
        adapter = self.create_adapter()
        adapter._get_tokenized_data = Mock(return_value=['tokenized'])

        result = adapter.handle_dataset('dataset')
        adapter._get_tokenized_data.assert_called_once_with('dataset', DeviceType.NPU)
        self.assertEqual(result, ['tokenized'])

        adapter._get_tokenized_data.reset_mock()
        result = adapter.handle_dataset('dataset', device=DeviceType.CPU)
        adapter._get_tokenized_data.assert_called_once_with('dataset', DeviceType.CPU)
        self.assertEqual(result, ['tokenized'])

    # Verify enable_kv_cache can be called without raising an exception.
    def test_enable_kv_cache_does_not_raise_when_called(self):
        adapter = self.create_adapter()
        adapter.enable_kv_cache(Mock(), True)
        self.assertIsNone(adapter.enable_kv_cache(Mock(), True))

    # Verify adapter configuration generation includes the expected mapping sources.
    def test_get_adapter_config_for_subgraph_returns_expected_mapping_sources_when_config_present(self):
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2
        adapter.config.first_k_dense_replace = 1

        configs = adapter.get_adapter_config_for_subgraph()

        self.assertGreaterEqual(len(configs), 3)
        self.assertEqual(configs[0].mapping.source, 'model.layers.0.self_attn.kv_b_proj')
        self.assertIn('model.layers.0.self_attn.q_a_proj', configs[1].mapping.targets)

    # Verify generate_model_visit calls the generic visit function with the model.
    def test_generate_model_visit_calls_visit_func_when_model_passed(self):
        adapter = self.create_adapter()
        dummy_model = Mock()
        mock_blocks = [('model.layers.0', Mock()), ('model.layers.1', Mock())]
        adapter.generate_decoder_layer = Mock(return_value=mock_blocks)

        with patch('msmodelslim.model.glm_5.model_adapter.generated_decoder_layer_visit_func') as mock_visit_func:
            mock_visit_func.return_value = iter([])
            result = adapter.generate_model_visit(dummy_model)

            mock_visit_func.assert_called_once()
            self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
