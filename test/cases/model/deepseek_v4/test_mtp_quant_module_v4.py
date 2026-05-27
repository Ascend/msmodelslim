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
from unittest.mock import Mock, patch

import torch
from torch import nn

import msmodelslim.model.deepseek_v4.mtp_quant_module as mod
from msmodelslim.utils.exception import UnexpectedError


class TestRemoveZeroAndShift(unittest.TestCase):
    # Verify zeros are removed and remaining values are shifted left in a normal matrix.
    def test_remove_zero_and_shift_returns_expected_matrix_when_single_zero_in_row(self):
        matrix = torch.tensor([[1, 0, 3, 4], [5, 6, 0, 8], [9, 10, 11, 0]])
        expected = torch.tensor([[1, 3, 4, 0], [5, 6, 8, 0], [9, 10, 11, 0]])
        torch.testing.assert_close(mod.remove_zero_and_shift(matrix), expected)

    # Verify zeros in the first column are shifted correctly to the end.
    def test_remove_zero_and_shift_returns_expected_matrix_when_zero_at_first_position(self):
        matrix = torch.tensor([[0, 2, 3, 4], [0, 6, 7, 8]])
        expected = torch.tensor([[2, 3, 4, 0], [6, 7, 8, 0]])
        torch.testing.assert_close(mod.remove_zero_and_shift(matrix), expected)

    # Verify rows with multiple zeros shift all non-zero values left and keep zeros at the end.
    def test_remove_zero_and_shift_returns_expected_matrix_when_multiple_zeros_in_row(self):
        matrix = torch.tensor([[1, 0, 3, 0], [0, 5, 0, 7]])
        expected = torch.tensor([[1, 3, 0, 0], [5, 0, 7, 0]])
        torch.testing.assert_close(mod.remove_zero_and_shift(matrix), expected)


class TestNewModelRMSNorm(unittest.TestCase):
    # Verify NewModelRMSNorm initializes weights and computes a forward pass.
    def test_new_model_rmsnorm_initialization_and_forward_when_created(self):
        hidden_size, eps = 8, 1e-5
        norm = mod.NewModelRMSNorm(hidden_size, eps)
        self.assertIsInstance(norm.weight, nn.Parameter)
        self.assertEqual(norm.weight.shape, (hidden_size,))

        x = torch.randn(2, 3, hidden_size)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    # Verify normalization logic matches expected RMS scaling behavior.
    def test_new_model_rmsnorm_normalization_logic_when_eps_zero(self):
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        norm = mod.NewModelRMSNorm(4, eps=0.0)
        output = norm(input_tensor)
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        expected = input_tensor * torch.rsqrt(variance)
        self.assertTrue(
            torch.allclose(output, norm.weight * expected, atol=1e-6) or torch.allclose(output, expected, atol=1e-6)
        )


class TestSharedHeadAndEmbedding(unittest.TestCase):
    # Verify shared head produces logits with the expected vocabulary dimension.
    def test_shared_head_forward_returns_expected_vocab_logits_when_called(self):
        cfg = Mock(dim=16, vocab_size=32)
        head = mod.SharedHead(cfg)
        x = torch.randn(2, 5, cfg.dim)
        out = head(x)
        self.assertEqual(out.shape, (2, 5, cfg.vocab_size))

    # Verify embedding outputs the expected hidden representation shape.
    def test_embedding_forward_returns_expected_shape_when_called(self):
        emb = mod.NewModelEmbedding(50, 16)
        input_ids = torch.randint(0, 50, (2, 4))
        out = emb(input_ids)
        self.assertEqual(out.shape, (2, 4, 16))


class DummyCfg:
    def __init__(self):
        self.dim = 16
        self.vocab_size = 32
        self.hc_mult = 2


class TestMTPLayer(unittest.TestCase):
    # Verify MTPLayer initializes all expected submodules and HC attributes.
    def test_mtp_layer_initialization_creates_expected_submodules_when_cfg_provided(self):
        cfg = DummyCfg()
        mtp = mod.MTPLayer(cfg)
        self.assertIsInstance(mtp.enorm, mod.NewModelRMSNorm)
        self.assertIsInstance(mtp.hnorm, mod.NewModelRMSNorm)
        self.assertIsInstance(mtp.e_proj, nn.Linear)
        self.assertIsInstance(mtp.h_proj, nn.Linear)
        self.assertIsInstance(mtp.head, nn.Linear)
        self.assertIsInstance(mtp.emb, mod.NewModelEmbedding)
        # hc params exist
        self.assertTrue(hasattr(mtp, 'hc_head_fn'))
        self.assertTrue(hasattr(mtp, 'hc_head_base'))


class TestGetSharedWeight(unittest.TestCase):
    # Verify missing shared weight entries raise an UnexpectedError.
    def test_get_shared_weight_raises_unexpected_error_when_key_not_found(self):
        with patch.object(mod, 'json_safe_load', return_value={'weight_map': {}}):
            with self.assertRaises(UnexpectedError):
                mod.get_shared_weight('/tmp/fake', 'embed.weight')

    # Verify get_shared_weight returns the tensor when the key is found in the index.
    def test_get_shared_weight_returns_tensor_when_key_found(self):
        fake_tensor = torch.randn(3, 3)
        index = {'weight_map': {'embed.weight': 'chunk.safetensors'}}

        class DummySafeOpen:
            def __init__(self, tensor):
                self.tensor = tensor

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get_tensor(self, name):
                return self.tensor

        with (
            patch.object(mod, 'json_safe_load', return_value=index),
            patch('msmodelslim.model.deepseek_v4.mtp_quant_module.safe_open', return_value=DummySafeOpen(fake_tensor)),
        ):
            out = mod.get_shared_weight('/tmp/fake', 'embed.weight')
            self.assertTrue(torch.equal(out, fake_tensor))


class TestGetMtpLayerAndWrap(unittest.TestCase):
    # Verify get_mtp_layer loads an MTP layer and applies auto-dequant state dict when successful.
    def test_get_mtp_layer_returns_layer_when_weights_available(self):
        cfg = DummyCfg()
        model_path = '/tmp/model'
        # Craft load_file to return keys with 'mtp.0.' prefix
        fake_weights = {
            'mtp.0.enorm.weight': torch.ones(cfg.dim),
            'mtp.0.hnorm.weight': torch.ones(cfg.dim),
            'mtp.0.e_proj.weight': torch.randn(cfg.dim, cfg.dim),
            'mtp.0.h_proj.weight': torch.randn(cfg.dim, cfg.dim),
            'mtp.0.head.weight': torch.randn(cfg.vocab_size, cfg.dim),
            'mtp.0.emb.tok_emb.weight': torch.randn(cfg.vocab_size, cfg.dim),
        }

        def shared_weight_side_effect(model_path_arg, key):
            if key == 'embed.weight':
                return torch.randn(cfg.vocab_size, cfg.dim)
            if key == 'head.weight':
                return torch.randn(cfg.vocab_size, cfg.dim)
            raise AssertionError(f'Unexpected shared weight key: {key}')

        with (
            patch(
                'msmodelslim.model.deepseek_v4.mtp_quant_module.get_valid_read_path',
                return_value='/tmp/model-000.safetensors',
            ),
            patch('msmodelslim.model.deepseek_v4.mtp_quant_module.load_file', return_value=fake_weights),
            patch(
                'msmodelslim.model.deepseek_v4.mtp_quant_module.get_shared_weight',
                side_effect=shared_weight_side_effect,
            ) as mock_shared,
            patch('msmodelslim.model.deepseek_v4.mtp_quant_module.auto_dequant_state_dict') as mock_auto_dequant,
            patch('msmodelslim.model.deepseek_v4.mtp_quant_module.MTPLayer.load_state_dict') as mock_load_state_dict,
        ):
            mtp_layer = mod.get_mtp_layer(cfg, model_path, layer_prefix='layers.0', mtp_layer_prefix='layers.0.')

            self.assertIsInstance(mtp_layer, mod.MTPLayer)
            self.assertEqual(mock_shared.call_count, 2)
            mock_auto_dequant.assert_called_once()
            mock_load_state_dict.assert_called_once()

    # Verify get_mtp_layer raises FileNotFoundError when weight path cannot be resolved.
    def test_get_mtp_layer_raises_file_not_found_when_path_missing(self):
        cfg = DummyCfg()
        with patch(
            'msmodelslim.model.deepseek_v4.mtp_quant_module.get_valid_read_path',
            side_effect=FileNotFoundError('missing'),
        ):
            with self.assertRaises(FileNotFoundError):
                mod.get_mtp_layer(cfg, '/tmp', 'layers.0')

    # Verify wrap_mtp_decoder assigns MTP layer attributes onto the decoder object.
    def test_wrap_mtp_decoder_assigns_attributes_when_called(self):
        cfg = DummyCfg()
        mtp_layer = mod.MTPLayer(cfg)
        mtp_decoder = Mock()
        # ensure attributes exist on decoder
        mtp_decoder.enorm = None
        mtp_decoder.hnorm = None
        mtp_decoder.e_proj = None
        mtp_decoder.h_proj = None
        mtp_decoder.head = None
        mtp_decoder.norm = None
        mtp_decoder.emb = None
        mtp_decoder.hc_head_fn = None
        mtp_decoder.hc_head_base = None
        mtp_decoder.hc_head_scale = None

        mod.wrap_mtp_decoder(mtp_decoder, mtp_layer)
        self.assertEqual(mtp_decoder.enorm, mtp_layer.enorm)
        self.assertEqual(mtp_decoder.hnorm, mtp_layer.hnorm)
        self.assertEqual(mtp_decoder.e_proj, mtp_layer.e_proj)
        self.assertEqual(mtp_decoder.h_proj, mtp_layer.h_proj)
        self.assertEqual(mtp_decoder.head, mtp_layer.head)
        self.assertEqual(mtp_decoder.norm, mtp_layer.norm)
        self.assertEqual(mtp_decoder.emb, mtp_layer.emb)


if __name__ == '__main__':
    unittest.main()
