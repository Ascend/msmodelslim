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
from unittest.mock import Mock
import torch
from torch import nn

from msmodelslim.model.glm_5.mtp_quant_module import (
    remove_zero_and_shift,
    SharedHead,
    GLM5RMSNorm,
    MTPLayer,
    wrap_mtp_decoder,
)


class TestRemoveZeroAndShift(unittest.TestCase):
    def test_remove_zero_and_shift_output_shouldEqualExpected_when_eachRowHasSingleZero(self):
        matrix = torch.tensor([[1, 0, 3, 4], [5, 6, 0, 8], [9, 10, 11, 0]])
        expected = torch.tensor([[1, 3, 4, 0], [5, 6, 8, 0], [9, 10, 11, 0]])
        torch.testing.assert_close(remove_zero_and_shift(matrix), expected)

    def test_remove_zero_and_shift_shouldRemoveFirstZero_when_zeroAtFirstPosition(self):
        matrix = torch.tensor([[0, 2, 3, 4], [0, 6, 7, 8]])
        expected = torch.tensor([[2, 3, 4, 0], [6, 7, 8, 0]])
        torch.testing.assert_close(remove_zero_and_shift(matrix), expected)

    def test_remove_zero_and_shift_shouldOnlyRemoveFirstZero_when_multipleZerosExist(self):
        matrix = torch.tensor([[1, 0, 3, 0], [0, 5, 0, 7]])
        expected = torch.tensor([[1, 3, 0, 0], [5, 0, 7, 0]])
        torch.testing.assert_close(remove_zero_and_shift(matrix), expected)

    def test_remove_zero_and_shift_shouldHandleAllZeros_when_everyElementIsZero(self):
        matrix = torch.tensor([[0, 0], [0, 0]])
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.shape, (2, 2))

    def test_remove_zero_and_shift_shouldPreserveDtype_when_inputIsFloat(self):
        matrix = torch.tensor([[1.0, 0.0, 3.0], [4.0, 5.0, 0.0]])
        result = remove_zero_and_shift(matrix)
        self.assertEqual(result.dtype, matrix.dtype)


class TestGLM5RMSNorm(unittest.TestCase):
    def test_GLM5RMSNorm_weight_shouldBeOnes_when_init(self):
        hidden_size, eps = 16, 1e-5
        norm = GLM5RMSNorm(hidden_size, eps)

        self.assertIsInstance(norm.weight, nn.Parameter)
        self.assertEqual(norm.weight.shape, (hidden_size,))
        self.assertTrue(torch.allclose(norm.weight.data, torch.ones(hidden_size)))
        self.assertEqual(norm.variance_epsilon, eps)

    def test_GLM5RMSNorm_output_shouldKeepShape_when_givenMultiDimInput(self):
        norm = GLM5RMSNorm(32)
        test_cases = [
            (torch.randn(10, 32), (10, 32)),
            (torch.randn(2, 10, 32), (2, 10, 32)),
            (torch.randn(5, 8, 10, 32), (5, 8, 10, 32)),
        ]

        for input_tensor, expected_shape in test_cases:
            self.assertEqual(norm(input_tensor).shape, expected_shape)

    def test_GLM5RMSNorm_output_shouldMatchManualCalculation_when_givenInput(self):
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
        norm = GLM5RMSNorm(4, eps=0.0)
        output = norm(input_tensor)

        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        expected = input_tensor * torch.rsqrt(variance)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_GLM5RMSNorm_output_shouldApplyWeight_when_weightIsSet(self):
        norm = GLM5RMSNorm(4)
        norm.weight.data = torch.tensor([0.5, 1.5, 2.0, 0.8])

        input_tensor = torch.ones(2, 4)
        output = norm(input_tensor)

        expected = norm.weight.data.repeat(2, 1)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_GLM5RMSNorm_output_shouldBeStable_when_inputHasLargeValues(self):
        input_tensor = torch.tensor([[1000.0, 2000.0, 3000.0, 4000.0], [5000.0, 6000.0, 7000.0, 8000.0]])
        output = GLM5RMSNorm(4)(input_tensor)
        self.assertTrue(torch.all(torch.abs(output) < 10.0))

    def test_GLM5RMSNorm_shouldNotCrash_when_epsIsZero(self):
        norm = GLM5RMSNorm(4, eps=0.0)
        input_tensor = torch.randn(2, 4)
        output = norm(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

    def test_GLM5RMSNorm_shouldNotCrash_when_inputIsAllZeros(self):
        norm = GLM5RMSNorm(4, eps=1e-6)
        input_tensor = torch.zeros(2, 4)
        output = norm(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)


class TestSharedHead(unittest.TestCase):
    def setUp(self):
        self.config = Mock(hidden_size=32, vocab_size=1000, rms_norm_eps=1e-6)
        self.shared_head = SharedHead(self.config)

    def test_SharedHead_output_shouldHaveVocabShape_when_givenHiddenStates(self):
        input_tensor = torch.randn(2, 5, self.config.hidden_size)
        output = self.shared_head(input_tensor)
        self.assertEqual(output.shape, (2, 5, self.config.vocab_size))

    def test_SharedHead_shouldHandleSingleToken_when_givenBatchOfOne(self):
        input_tensor = torch.randn(1, 1, self.config.hidden_size)
        output = self.shared_head(input_tensor)
        self.assertEqual(output.shape, (1, 1, self.config.vocab_size))


class TestMTPLayer(unittest.TestCase):
    def setUp(self):
        self.config = Mock(hidden_size=64, vocab_size=2000, rms_norm_eps=1e-6)

    def test_MTPLayer_components_shouldHaveCorrectTypesAndDims_when_init(self):
        mtp_layer = MTPLayer(self.config)

        self.assertIsInstance(mtp_layer.enorm, GLM5RMSNorm)
        self.assertIsInstance(mtp_layer.hnorm, GLM5RMSNorm)
        self.assertIsInstance(mtp_layer.shared_head, SharedHead)
        self.assertIsInstance(mtp_layer.eh_proj, nn.Linear)
        self.assertIsInstance(mtp_layer.embed_tokens, nn.Embedding)

        self.assertEqual(mtp_layer.eh_proj.in_features, self.config.hidden_size * 2)
        self.assertEqual(mtp_layer.eh_proj.out_features, self.config.hidden_size)
        self.assertEqual(mtp_layer.embed_tokens.num_embeddings, self.config.vocab_size)
        self.assertEqual(mtp_layer.embed_tokens.embedding_dim, self.config.hidden_size)


class TestWrapMtpDecoder(unittest.TestCase):
    def test_wrap_mtp_decoder_properties_shouldBeReplaced_when_called(self):
        mtp_decoder = Mock()
        mtp_layer = Mock()

        orig_enorm = mtp_decoder.enorm
        orig_hnorm = mtp_decoder.hnorm

        wrap_mtp_decoder(mtp_decoder, mtp_layer)

        self.assertEqual(mtp_decoder.enorm, mtp_layer.enorm)
        self.assertEqual(mtp_decoder.hnorm, mtp_layer.hnorm)
        self.assertEqual(mtp_decoder.shared_head, mtp_layer.shared_head)
        self.assertEqual(mtp_decoder.eh_proj, mtp_layer.eh_proj)
        self.assertEqual(mtp_decoder.embed_tokens, mtp_layer.embed_tokens)

        self.assertNotEqual(mtp_decoder.enorm, orig_enorm)
        self.assertNotEqual(mtp_decoder.hnorm, orig_hnorm)


if __name__ == '__main__':
    unittest.main()
