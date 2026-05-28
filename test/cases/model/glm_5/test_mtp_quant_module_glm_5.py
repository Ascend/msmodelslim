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
from unittest.mock import Mock

import torch

from msmodelslim.model.glm_5.mtp_quant_module import (
    remove_zero_and_shift,
    SharedHead,
    GLM5RMSNorm,
    MTPLayer,
    wrap_mtp_decoder,
)


class DummyConfig:
    def __init__(self):
        self.hidden_size = 16
        self.vocab_size = 32
        self.rms_norm_eps = 1e-6


class TestGLM5MTPQuantModule(unittest.TestCase):
    # Verify remove_zero_and_shift removes the first zero in each row and shifts values left.
    def test_remove_zero_and_shift_returns_expected_matrix_when_single_zero_in_row(self):
        matrix = torch.tensor([[1, 0, 3, 4], [5, 6, 0, 8], [9, 10, 11, 0]])
        expected = torch.tensor([[1, 3, 4, 0], [5, 6, 8, 0], [9, 10, 11, 0]])
        torch.testing.assert_close(remove_zero_and_shift(matrix), expected)

    # Verify remove_zero_and_shift handles rows with multiple zeros correctly.
    def test_remove_zero_and_shift_returns_expected_matrix_when_multiple_zeros_in_row(self):
        matrix = torch.tensor([[1, 0, 3, 0], [0, 5, 0, 7]])
        expected = torch.tensor([[1, 3, 0, 0], [5, 0, 7, 0]])
        torch.testing.assert_close(remove_zero_and_shift(matrix), expected)

    # Verify SharedHead produces logits with the expected vocabulary dimension.
    def test_shared_head_forward_returns_expected_vocab_logits_when_called(self):
        cfg = DummyConfig()
        head = SharedHead(cfg)
        x = torch.randn(2, 5, cfg.hidden_size)
        out = head(x)
        self.assertEqual(out.shape, (2, 5, cfg.vocab_size))

    # Verify GLM5RMSNorm normalizes inputs correctly when epsilon is zero.
    def test_glm5_rmsnorm_normalization_logic_when_eps_zero(self):
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        norm = GLM5RMSNorm(4, eps=0.0)
        output = norm(input_tensor)
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        expected = input_tensor * torch.rsqrt(variance)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    # Verify MTPLayer initializes its required submodules correctly.
    def test_mtp_layer_initialization_creates_expected_submodules_when_cfg_provided(self):
        cfg = DummyConfig()
        mtp = MTPLayer(cfg)
        self.assertIsNotNone(mtp.enorm)
        self.assertIsNotNone(mtp.hnorm)
        self.assertIsNotNone(mtp.shared_head)
        self.assertIsNotNone(mtp.eh_proj)
        self.assertIsNotNone(mtp.embed_tokens)

    # Verify wrap_mtp_decoder copies MTP layer attributes onto the decoder object.
    def test_wrap_mtp_decoder_assigns_attributes_when_called(self):
        cfg = DummyConfig()
        mtp_layer = MTPLayer(cfg)
        mtp_decoder = Mock()
        mtp_decoder.enorm = None
        mtp_decoder.hnorm = None
        mtp_decoder.shared_head = None
        mtp_decoder.eh_proj = None
        mtp_decoder.embed_tokens = None

        wrap_mtp_decoder(mtp_decoder, mtp_layer)

        self.assertEqual(mtp_decoder.enorm, mtp_layer.enorm)
        self.assertEqual(mtp_decoder.hnorm, mtp_layer.hnorm)
        self.assertEqual(mtp_decoder.shared_head, mtp_layer.shared_head)
        self.assertEqual(mtp_decoder.eh_proj, mtp_layer.eh_proj)
        self.assertEqual(mtp_decoder.embed_tokens, mtp_layer.embed_tokens)


if __name__ == '__main__':
    unittest.main()
