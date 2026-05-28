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
from unittest.mock import patch

import torch
from torch import nn

import msmodelslim.model.glm_5.convert_fp8_to_bf16 as mod


class TestGLM5ConvertFP8ToBF16(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    # Verify weight dequantization scales the tensor correctly when block size matches.
    def test_weight_dequant_returns_scaled_tensor_when_block_size_matches(self):
        weight = torch.arange(4 * 4, dtype=torch.float32).view(4, 4)
        scale = torch.full((1, 1), 0.5, dtype=torch.float32)

        out = mod.weight_dequant(weight.clone(), scale, block_size=4)

        expected = (weight * 0.5).to(torch.bfloat16)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(out.float(), expected.float(), atol=0, rtol=0))

    # Verify inverse weight mapping removes .weight_scale_inv suffix entries from the index.
    def test_get_inv_weight_map_filters_weight_scale_inv_when_index_contains_it(self):
        index_content = {
            'weight_map': {
                'model.layer.weight_scale_inv': 'chunk.safetensors',
                'model.layer.weight': 'chunk.safetensors',
            }
        }

        mod.get_inv_weight_map.cache_clear()
        with patch.object(mod, 'json_safe_load', return_value=index_content):
            result = mod.get_inv_weight_map('dummy_path')

        self.assertEqual(result, {'model.layer': 'chunk.safetensors'})

    # Verify get_inv_tensor opens the safetensors file on CPU when distributed is not initialized.
    def test_get_inv_tensor_uses_cpu_when_not_initialized(self):
        called = {}

        class DummySafeOpen:
            def __init__(self, tensor):
                self.tensor = tensor

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def get_tensor(self, name):
                called['tensor_name'] = name
                return self.tensor

        def fake_safe_open(file_path, framework, device):
            called['framework'] = framework
            called['device'] = device
            return DummySafeOpen(torch.ones(2, 2))

        with (
            patch.object(mod, 'get_valid_read_path', return_value='/tmp/ignored.safetensors'),
            patch.object(mod.dist, 'is_initialized', return_value=False),
            patch.object(mod, 'safe_open', side_effect=fake_safe_open),
        ):
            result = mod.get_inv_tensor('model.layer', 'ignored', {'model.layer': 'ignored.safetensors'})

        self.assertEqual(called['framework'], 'pt')
        self.assertEqual(called['device'], 'cpu')
        self.assertEqual(called['tensor_name'], 'model.layer.weight_scale_inv')
        self.assertTrue(torch.equal(result, torch.ones(2, 2)))

    # Verify auto conversion returns immediately when there is no inverse weight map.
    def test_auto_convert_module_fp8_to_bf16_returns_early_when_weight_map_empty(self):
        module = nn.Linear(2, 2, bias=False)
        original_weight = module.weight.data.clone()
        with patch.object(mod, 'get_inv_weight_map', return_value={}):
            mod.auto_convert_module_fp8_to_bf16('', module, 'ignored')

        self.assertTrue(torch.equal(module.weight.data, original_weight))

    # Verify convert_module_fp8_to_bf16 updates module weights using the provided scale tensor.
    def test_convert_module_fp8_to_bf16_updates_weight_when_scale_available(self):
        module = nn.Linear(2, 2, bias=False)
        module.weight.data = torch.ones(2, 2, dtype=torch.float32)
        fake_scale = torch.full((2, 2), 0.5, dtype=torch.float32)

        with patch.object(mod, 'get_inv_tensor', return_value=fake_scale):
            mod.convert_module_fp8_to_bf16('', module, 'ignored', {'': 'chunk.safetensors'})

        expected = (torch.ones(2, 2, dtype=torch.float32) * 0.5).to(torch.bfloat16)
        self.assertTrue(torch.allclose(module.weight.float(), expected.float(), atol=0, rtol=0))


if __name__ == '__main__':
    unittest.main()
