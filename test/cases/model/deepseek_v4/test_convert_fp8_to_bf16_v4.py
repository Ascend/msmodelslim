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
from unittest.mock import patch

import torch

import msmodelslim.model.deepseek_v4.convert_fp8_to_bf16 as mod


class TestConvertFP8ToBF16V4(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    # Verify `decode_fp8` outputs a bfloat16 tensor with correct shape and scaled values.
    def test_decode_fp8_returns_expected_bfloat16_when_input_is_float32(self):
        m, n = 256, 256
        weight = torch.randn(m, n, dtype=torch.float32)
        scale = torch.full((m // 128, n // 128), 0.5, dtype=torch.float32)

        expected = (
            (weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * scale[:, None, :, None].float())
            .flatten(2, 3)
            .flatten(0, 1)
            .to(torch.bfloat16)
        )

        out = mod.decode_fp8(weight, scale)

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, (m, n))
        self.assertTrue(torch.allclose(out.float(), expected.float(), atol=0, rtol=0))

    # Verify `decode_fp4` decodes packed fp4 values into bfloat16 with the expected LUT scaling.
    def test_decode_fp4_returns_expected_bfloat16_when_packed_data_is_max_values(self):
        packed_data = torch.tensor([[0x1F] * 16], dtype=torch.uint8)
        block_scales = torch.tensor([[2.0]], dtype=torch.float32)

        out = mod.decode_fp4(packed_data, block_scales)

        uint8 = packed_data.view(torch.uint8)
        low = uint8 & 0x0F
        high = (uint8 >> 4) & 0x0F
        indices = torch.stack([low, high], dim=-1).flatten(-2)
        sign = 1.0 - 2.0 * ((indices >> 3) & 1).float()
        abs_idx = indices & 0x07
        lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
        expected = (sign * lut[abs_idx.long()] * block_scales.float().repeat_interleave(32, dim=-1)).to(torch.bfloat16)

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(out, expected, atol=0, rtol=0))

    # Verify the real tensor name is built correctly with and without a prefix.
    def test_get_real_name_returns_name_with_prefix_when_given_prefix(self):
        self.assertEqual(mod.get_real_name('linear.weight', 'model.layers.43'), 'model.layers.43.linear.weight')
        self.assertEqual(mod.get_real_name('linear.weight', ''), 'linear.weight')

    # Verify inverse weight map removes scale entries and keeps only base weight keys.
    def test_get_inv_weight_map_filters_scale_suffix_when_index_contains_scale_and_weight(self):
        index_content = {
            'weight_map': {
                'model.layer1.weight': 'model-00001.safetensors',
                'model.layer1.scale': 'model-00001.safetensors',
                'model.layer2.weight': 'model-00002.safetensors',
                'model.layer2.scale': 'model-00002.safetensors',
            }
        }

        mod.get_inv_weight_map.cache_clear()
        with patch.object(mod, 'json_safe_load', return_value=index_content):
            result = mod.get_inv_weight_map('dummy_path')

        self.assertEqual(result, {'model.layer1': 'model-00001.safetensors', 'model.layer2': 'model-00002.safetensors'})

    # Verify that `get_inv_tensor` opens tensors on CPU when the distributed environment is not initialized.
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
            result = mod.get_inv_tensor('model.layer1', 'ignored', {'model.layer1': 'ignored.safetensors'})

        self.assertEqual(called['framework'], 'pt')
        self.assertEqual(called['device'], 'cpu')
        self.assertEqual(called['tensor_name'], 'model.layer1.scale')
        self.assertTrue(torch.equal(result, torch.ones(2, 2)))

    # Verify `auto_dequant_state_dict` leaves state_dict unchanged when there is no weight map.
    def test_auto_dequant_state_dict_skips_when_weight_map_empty(self):
        state_dict = {'linear.weight': torch.randn(2, 2)}
        original_tensor = state_dict['linear.weight']

        with patch.object(mod, 'get_inv_weight_map', return_value={}):
            mod.auto_dequant_state_dict('model.layers.43.', state_dict, 'ignored')

        self.assertIs(state_dict['linear.weight'], original_tensor)

    # Verify `auto_dequant_state_dict` calls dequant_state_dict for matching weight map entries.
    def test_auto_dequant_state_dict_calls_dequant_state_dict_when_matching_keys_present(self):
        state_dict = {'mtp.0.linear.weight': torch.randn(2, 2)}
        weight_map = {'mtp.0.linear': 'chunk.safetensors'}

        with (
            patch.object(mod, 'get_inv_weight_map', return_value=weight_map),
            patch.object(mod, 'dequant_state_dict') as mock_dequant,
        ):
            mod.auto_dequant_state_dict('', state_dict, 'ignored')

        mock_dequant.assert_called_once()
        called_layer_prefix, called_state_dict, called_path = mock_dequant.call_args[0]
        called_kwargs = mock_dequant.call_args[1]

        self.assertEqual(called_layer_prefix, '')
        self.assertIs(called_state_dict, state_dict)
        self.assertEqual(called_path, 'ignored')
        self.assertEqual(called_kwargs['weight_map'], weight_map)
        self.assertEqual(called_kwargs['mtp_layer_prefix'], 'model.layers.43.')

    # Verify warning logging occurs when `auto_dequant_state_dict` hits a missing tensor KeyError.
    def test_auto_dequant_state_dict_logs_warning_when_dequant_raises_keyerror(self):
        state_dict = {'mtp.0.linear.weight': torch.randn(2, 2)}

        with (
            patch.object(mod, 'get_inv_weight_map', return_value={'mtp.0.linear': 'chunk.safetensors'}),
            patch.object(mod, 'dequant_state_dict', side_effect=KeyError('missing tensor')),
            patch.object(mod, 'get_logger') as mock_get_logger,
        ):
            mod.auto_dequant_state_dict('', state_dict, 'ignored')

        self.assertGreaterEqual(mock_get_logger.return_value.warning.call_count, 2)

    # Verify `dequant_state_dict` decodes and replaces the fp4 weight when a matching key exists.
    def test_dequant_state_dict_applies_decode_fp4_and_replaces_weight_when_tensor_key_matches(self):
        original_weight = torch.randn(2, 2, dtype=torch.float32)
        state_dict = {'linear.weight': original_weight}
        fake_scale = torch.full((1, 1), 0.25, dtype=torch.float32)
        expected_output = torch.full((2, 2), 1.0, dtype=torch.bfloat16)

        with (
            patch.object(mod, 'get_inv_tensor', return_value=fake_scale) as mock_get_tensor,
            patch.object(mod, 'decode_fp4', return_value=expected_output) as mock_decode,
        ):
            mod.dequant_state_dict(
                '', state_dict, 'ignored', {'linear': 'chunk.safetensors'}, mtp_layer_prefix='model.layers.43.'
            )

        mock_get_tensor.assert_called_once_with('linear', 'ignored', {'linear': 'chunk.safetensors'})
        mock_decode.assert_called_once_with(original_weight, fake_scale.to(original_weight.device))
        self.assertTrue(torch.equal(state_dict['linear.weight'], expected_output))

    # Verify `dequant_state_dict` does not touch unrelated keys when no weight map entry matches.
    def test_dequant_state_dict_skips_unknown_keys_when_no_matching_weight_map_entry(self):
        original_weight = torch.randn(2, 2, dtype=torch.float32)
        state_dict = {'other.weight': original_weight.clone()}

        with patch.object(mod, 'get_inv_tensor') as mock_get_tensor, patch.object(mod, 'decode_fp4') as mock_decode:
            mod.dequant_state_dict(
                '', state_dict, 'ignored', {'linear': 'chunk.safetensors'}, mtp_layer_prefix='model.layers.43.'
            )

        mock_get_tensor.assert_not_called()
        mock_decode.assert_not_called()
        self.assertTrue(torch.equal(state_dict['other.weight'], original_weight))
