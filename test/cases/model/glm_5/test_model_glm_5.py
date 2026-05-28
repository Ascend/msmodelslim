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

import sys
import types
import unittest
from unittest.mock import patch

import torch

from msmodelslim.model.glm_5.model import (
    ModelArgs,
    ParallelEmbedding,
    linear,
    RMSNorm,
    LayerNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    hadamard_transform_ref,
    rotate_activation,
    weight_dequant,
)


class DummyArgs:
    def __init__(self):
        self.qk_rope_head_dim = 4
        self.max_seq_len = 8
        self.original_seq_len = 8
        self.beta_fast = 32
        self.beta_slow = 1
        self.rope_theta = 10000.0
        self.rope_factor = 1.0
        self.max_batch_size = 1
        self.vocab_size = 16
        self.dim = 16
        self.hidden_size = 16
        self.intermediate_size = 32
        self.num_hidden_layers = 1


class TestGLM5Model(unittest.TestCase):
    # Verify ModelArgs initializes with expected GLM-5 defaults.
    def test_model_args_defaults_when_created(self):
        args = ModelArgs()
        self.assertEqual(args.max_batch_size, 8)
        self.assertEqual(args.dtype, 'bf16')
        self.assertEqual(args.vocab_size, 154880)
        self.assertEqual(args.num_hidden_layers, 78)

    # Verify ParallelEmbedding returns the expected tensor shape for a single rank.
    def test_parallel_embedding_forward_returns_expected_shape_when_single_rank(self):
        args = DummyArgs()
        embedding = ParallelEmbedding(args.vocab_size, args.dim)
        input_ids = torch.randint(0, args.vocab_size, (1, 2), dtype=torch.long)
        out = embedding(input_ids)
        self.assertEqual(out.shape, (1, 2, args.dim))

    # Verify the linear helper computes output of the expected shape.
    def test_linear_function_returns_expected_shape_when_given_input_weight_and_bias(self):
        x = torch.randn(2, 3, 4)
        weight = torch.randn(5, 4)
        bias = torch.randn(5)
        out = linear(x, weight, bias)
        self.assertEqual(out.shape, (2, 3, 5))

    # Verify RMSNorm preserves the input shape after normalization.
    def test_rmsnorm_forward_returns_same_shape_when_applied(self):
        norm = RMSNorm(8, eps=1e-6)
        x = torch.randn(1, 2, 8)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    # Verify LayerNorm preserves the input shape after normalization.
    def test_layernorm_forward_returns_same_shape_when_applied(self):
        norm = LayerNorm(8, eps=1e-6)
        x = torch.randn(1, 2, 8)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    # Verify precompute_freqs_cis returns a complex tensor with expected sequence length.
    def test_precompute_freqs_cis_returns_complex_tensor_when_parameters_provided(self):
        args = DummyArgs()
        freqs = precompute_freqs_cis(args)
        self.assertEqual(freqs.shape[0], args.max_seq_len)
        self.assertTrue(torch.is_complex(freqs))

    # Verify apply_rotary_emb returns the expected output tensor shape.
    def test_apply_rotary_emb_preserves_shape_when_called(self):
        x = torch.randn(1, 2, 4)
        args = DummyArgs()
        args.max_seq_len = 2
        args.original_seq_len = 2
        freqs = precompute_freqs_cis(args)[: x.size(1)]
        out = apply_rotary_emb(x, freqs)
        self.assertEqual(out.shape, torch.Size([1, 2, 2, 4]))

    # Verify hadamard_transform_ref works with a mocked scipy implementation.
    def test_hadamard_transform_ref_with_fake_scipy_returns_same_shape_when_fake_scipy_present(self):
        x = torch.randn(1, 1, 4)
        fake_scipy = types.ModuleType('scipy')
        fake_linalg = types.ModuleType('scipy.linalg')

        def fake_hadamard(n, dtype=None):
            return torch.eye(n, dtype=torch.float32).numpy()

        fake_linalg.hadamard = fake_hadamard
        fake_scipy.linalg = fake_linalg
        with patch.dict(sys.modules, {'scipy': fake_scipy, 'scipy.linalg': fake_linalg}):
            out = hadamard_transform_ref(x, scale=1.0)
            self.assertEqual(out.shape, x.shape)

    # Verify rotate_activation calls hadamard_transform_ref internally.
    def test_rotate_activation_uses_hadamard_transform_when_called(self):
        x = torch.randn(1, 2, 4)
        with patch(
            'msmodelslim.model.glm_5.model.hadamard_transform_ref', return_value=torch.zeros_like(x)
        ) as mock_hadamard:
            out = rotate_activation(x)
            self.assertEqual(out.shape, x.shape)
            mock_hadamard.assert_called_once()

    # Verify weight_dequant preserves the source weight tensor shape after dequantization.
    def test_weight_dequant_returns_same_shape_when_block_size_used(self):
        weight = torch.ones(128, 128, dtype=torch.float32)
        scale = torch.full((1,), 0.5, dtype=torch.float32)
        out = weight_dequant(weight.clone(), scale)
        self.assertEqual(out.shape, weight.shape)


if __name__ == '__main__':
    unittest.main()
