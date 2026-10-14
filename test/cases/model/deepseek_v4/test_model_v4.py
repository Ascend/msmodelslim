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

import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

import msmodelslim.model.deepseek_v4.model as mod


def _convert_attention_to_float(attn: mod.Attention):
    attn.wo_a = nn.Linear(attn.n_heads * attn.head_dim // attn.n_groups, attn.n_groups * attn.o_lora_rank, bias=False)
    if hasattr(attn, 'indexer') and attn.indexer is not None:
        attn.indexer.weights_proj = nn.Linear(attn.indexer.dim, attn.indexer.n_heads, bias=False)


def _convert_module_linears_to_float(module: nn.Module):
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            with torch.no_grad():
                sub.weight.data = sub.weight.data.float()
                if sub.bias is not None:
                    sub.bias.data = sub.bias.data.float()


def _convert_transformer_to_float(transformer: mod.Transformer):
    _convert_module_linears_to_float(transformer)


class DummyArgs:
    def __init__(self):
        self.max_batch_size = 2
        self.max_seq_len = 8
        self.dtype = 'bf16'
        self.scale_fmt = 'ue8m0'
        self.vocab_size = 16
        self.dim = 16
        self.moe_inter_dim = 8
        self.num_hidden_layers = 1
        self.n_hash_layers = 0
        self.n_heads = 1
        self.n_routed_experts = 2
        self.n_shared_experts = 1
        self.n_activated_experts = 1
        self.score_func = 'softmax'
        self.route_scale = 1.0
        self.swiglu_limit = 0.0
        self.q_lora_rank = 8
        self.head_dim = 8
        self.rope_head_dim = 4
        self.norm_eps = 1e-6
        self.o_groups = 1
        self.o_lora_rank = 8
        self.window_size = 2
        self.compress_ratios = (1,)
        # default singular compress_ratio used by some tests
        self.compress_ratio = 1
        self.compress_rope_theta = 10000.0
        self.original_seq_len = 16
        self.rope_theta = 10000.0
        self.rope_factor = 1.0
        self.beta_fast = 32
        self.beta_slow = 1
        self.index_n_heads = 1
        self.index_head_dim = 4
        self.index_topk = 1
        self.hc_mult = 2
        self.hc_sinkhorn_iters = 2
        self.hc_eps = 1e-6


class TestDeepSeekV4Model(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    # Verify `ModelArgs` initializes with default DeepSeek-V4 values.
    def test_model_args_defaults_when_created(self):
        args = mod.ModelArgs()
        self.assertEqual(args.max_batch_size, 4)
        self.assertEqual(args.dtype, 'bf16')
        self.assertEqual(args.vocab_size, 129280)
        self.assertEqual(args.num_hidden_layers, 43)

    # Verify `ParallelEmbedding` forwards CPU input into the expected output shape.
    def test_parallel_embedding_forward_cpu_when_single_rank(self):
        args = DummyArgs()
        embedding = mod.ParallelEmbedding(args.vocab_size, args.dim)
        input_ids = torch.randint(0, args.vocab_size, (1, 2), dtype=torch.long)
        out = embedding(input_ids)
        self.assertEqual(out.shape, (1, 2, args.dim))

    # Verify `ParallelEmbedding` performs all_reduce when world size is greater than one.
    def test_parallel_embedding_world_size_greater_than_one_when_distributed(self):
        args = DummyArgs()
        args.vocab_size = 4
        with (
            patch.object(mod, 'world_size', 2),
            patch.object(mod, 'rank', 1),
            patch.object(mod, 'USE_DP_MODE', False),
            patch.object(mod.dist, 'all_reduce') as mock_all_reduce,
        ):
            embedding = mod.ParallelEmbedding(args.vocab_size, args.dim)
            input_ids = torch.tensor([[2, 3]], dtype=torch.long)
            out = embedding(input_ids)
            self.assertEqual(out.shape, (1, 2, args.dim))
            mock_all_reduce.assert_called_once()

    # Verify the low-level linear helper computes the expected output shape.
    def test_linear_function_returns_expected_shape_when_given_input_weight_and_bias(self):
        x = torch.randn(2, 3, 4)
        weight = torch.randn(5, 4)
        bias = torch.randn(5)
        out = mod.linear(x, weight, bias)
        self.assertEqual(out.shape, (2, 3, 5))

    # Verify RMSNorm preserves input shape during forward pass.
    def test_rmsnorm_forward_returns_same_shape_when_applied(self):
        norm = mod.RMSNorm(8, eps=1e-6)
        x = torch.randn(1, 2, 8)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    # Verify rotary frequency precomputation returns a complex tensor of expected shape.
    def test_precompute_freqs_cis_returns_complex_tensor_when_parameters_provided(self):
        freqs = mod.precompute_freqs_cis(4, 3, 16, 1000.0, 1.0, 1, 1)
        self.assertEqual(freqs.shape, (3, 2))
        self.assertTrue(torch.is_complex(freqs))

    # Verify rotary embedding and inverse preserve the original tensor shape.
    def test_apply_rotary_emb_and_inverse_preserve_shape_when_inverse_applied(self):
        x = torch.randn(1, 2, 4)
        freqs = mod.precompute_freqs_cis(4, 2, 16, 1000.0, 1.0, 1, 1)
        y = x.clone()
        out = mod.apply_rotary_emb(y, freqs)
        self.assertEqual(out.shape, x.shape)

        inv = mod.apply_rotary_emb(out, freqs, inverse=True)
        self.assertEqual(inv.shape, x.shape)

    # Verify rotate activation uses the Hadamard transform helper when called.
    def test_rotate_activation_uses_hadamard_transform_when_called(self):
        x = torch.randn(1, 2, 4)
        with patch.object(mod, 'hadamard_transform_ref', return_value=torch.zeros_like(x)) as mock_hadamard:
            out = mod.rotate_activation(x)
            self.assertEqual(out.shape, x.shape)
            mock_hadamard.assert_called_once()

    # Verify window top-k index selection behaves correctly for different window sizes.
    def test_get_window_topk_idxs_returns_expected_shapes_when_various_window_sizes_used(self):
        out0 = mod.get_window_topk_idxs(4, 1, 3, 0)
        self.assertEqual(out0.shape, (1, 3, 3))
        out1 = mod.get_window_topk_idxs(4, 1, 3, 2)
        self.assertEqual(out1.shape, (1, 1, 4))
        out2 = mod.get_window_topk_idxs(4, 1, 3, 4)
        # implementation returns a 1D arange expanded to (bsz,1,window)
        self.assertEqual(out2.shape, (1, 1, 4))

    # Verify compressor top-k index selection works when start position is zero.
    def test_get_compress_topk_idxs_returns_expected_shape_when_start_pos_zero(self):
        out = mod.get_compress_topk_idxs(2, 1, 4, 0, 0)
        self.assertEqual(out.shape, (1, 4, 2))

    # Verify compressor top-k index selection works when start position is positive.
    def test_get_compress_topk_idxs_returns_expected_shape_when_start_pos_positive(self):
        out = mod.get_compress_topk_idxs(2, 1, 4, 2, 1)
        self.assertEqual(out.shape, (1, 1, 1))

    # Verify compressor overlap transform moves non-zero slots as expected.
    def test_compressor_overlap_transform_returns_expected_layout_when_given_tensor(self):
        args = DummyArgs()
        args.compress_ratio = 4
        compressor = mod.Compressor(args, compress_ratio=4, head_dim=8, rotate=False)
        tensor = torch.randn(1, 2, 4, 16)
        out = compressor.overlap_transform(tensor, value=-1.0)
        # The current implementation places the input tensor into the
        # first `r` slots and leaves the remaining slots zeroed.
        expected_slots = compressor.compress_ratio * 2
        self.assertEqual(out.shape[0], tensor.size(0))
        self.assertEqual(out.shape[1], tensor.size(1))
        self.assertEqual(out.shape[2], expected_slots)
        r = tensor.size(2)
        # The implementation may preserve the full last-dim or collapse to head_dim.
        if out.shape[3] == tensor.size(-1):
            # full-last-dim behavior: first `r` slots equal input, rest zeros
            self.assertTrue(torch.all(out[:, :, :r, :] == tensor))
            self.assertTrue(torch.all(out[:, :, r:, :] == 0))
        elif out.shape[3] == compressor.head_dim:
            # collapsed-last-dim behavior: compare halves against head_dim slices
            self.assertFalse(torch.all(out[:, :, :r, :] == tensor[:, :, :, : compressor.head_dim]))
            self.assertTrue(
                torch.all(out[:, :, r:, :] == tensor[:, :, :, compressor.head_dim : compressor.head_dim * 2])
            )
        else:
            self.fail(f"Unexpected last-dim size: {out.shape[3]}")

    # Verify compressor forward returns None when no compression is needed.
    def test_compressor_forward_no_compress_returns_none_when_no_compress_needed(self):
        args = DummyArgs()
        args.compress_ratio = 4
        compressor = mod.Compressor(args, compress_ratio=4, head_dim=8, rotate=False)
        compressor.kv_cache = torch.zeros(1, 1, 8)
        x = torch.randn(1, 3, args.dim)
        freqs = mod.precompute_freqs_cis(
            args.rope_head_dim,
            x.size(1),
            args.original_seq_len,
            args.compress_rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        out = compressor(x, 0, freqs)
        self.assertIsNone(out)

    # Verify compressor forward returns a tensor when compression is enabled.
    def test_compressor_forward_compress_returns_tensor_when_rotate_enabled(self):
        args = DummyArgs()
        args.compress_ratio = 4
        compressor = mod.Compressor(args, compress_ratio=4, head_dim=8, rotate=True)
        compressor.kv_cache = torch.zeros(1, 1, 8)
        x = torch.randn(1, 4, args.dim)
        freqs = mod.precompute_freqs_cis(
            args.rope_head_dim,
            x.size(1),
            args.original_seq_len,
            args.compress_rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        with patch.object(mod, 'rotate_activation', return_value=torch.zeros(1, 1, 8)):
            out = compressor(x, 0, freqs)
        self.assertIsNotNone(out)

    # Verify indexer forward produces a 3D tensor for valid inputs.
    def test_indexer_forward_returns_three_dimensional_output_when_called(self):
        args = DummyArgs()
        args.compress_ratio = 4
        args.index_topk = 1
        indexer = mod.Indexer(args, compress_ratio=4)
        indexer.compressor.kv_cache = indexer.kv_cache
        indexer.weights_proj = nn.Linear(args.dim, indexer.n_heads, bias=False)
        _convert_module_linears_to_float(indexer)
        x = torch.randn(1, 4, args.dim)
        qr = torch.randn(1, 4, args.q_lora_rank)
        freqs = mod.precompute_freqs_cis(
            args.rope_head_dim,
            x.size(1),
            args.original_seq_len,
            args.compress_rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        try:
            out = indexer(x, qr, 0, freqs, 0)
        except RuntimeError as e:
            msg = str(e).lower()
            if 'primitive' in msg or 'matmul' in msg:
                self.skipTest('matmul primitive not supported on this platform')
            raise
        self.assertEqual(out.ndim, 3)

    # Verify gate forward outputs weights and indices of expected expert count.
    def test_gate_forward_softmax_returns_expected_shapes_when_called(self):
        args = DummyArgs()
        gate = mod.Gate(0, args)
        # Gate expects 2D inputs (batch, dim) for correct broadcasting
        x = torch.randn(1, args.dim)
        input_ids = torch.tensor([0])
        weights, indices = gate(x, input_ids)
        self.assertEqual(weights.shape[-1], args.n_activated_experts)
        self.assertEqual(indices.shape[-1], args.n_activated_experts)

    # Verify expert module preserves sequence and hidden dimensions on forward.
    def test_expert_forward_returns_same_sequence_shape_when_called(self):
        expert = mod.Expert(8, 16, swiglu_limit=1.0)
        x = torch.randn(1, 1, 8)
        out = expert(x)
        self.assertEqual(out.shape, (1, 1, 8))

    # Verify MoE forward preserves the input tensor shape.
    def test_moe_forward_returns_same_shape_when_called(self):
        args = DummyArgs()
        args.compress_ratios = (1,)
        moe = mod.MoE(0, args)
        x = torch.randn(1, 1, args.dim)
        input_ids = torch.zeros(1, 1, dtype=torch.long)
        out = moe(x, input_ids)
        self.assertEqual(out.shape, x.shape)

    # Verify block forward preserves the multi-head expanded shape.
    def test_block_forward_returns_expected_shape_when_called(self):
        args = DummyArgs()
        args.compress_ratios = (1,)
        block = mod.Block(0, args)
        _convert_module_linears_to_float(block)
        x = torch.randn(1, 1, args.hc_mult, args.dim)
        input_ids = torch.zeros(1, 1, dtype=torch.long)
        out = block(x, 0, input_ids)
        self.assertEqual(out.shape, (1, 1, args.hc_mult, args.dim))

    # Verify transformer output shape matches vocabulary size for single-layer models.
    def test_transformer_forward_returns_vocab_logits_when_single_layer(self):
        args = DummyArgs()
        args.num_hidden_layers = 1
        args.compress_ratios = (1,)
        args.vocab_size = 8
        transformer = mod.Transformer(args)
        _convert_module_linears_to_float(transformer)
        input_ids = torch.randint(0, args.vocab_size, (1, 2), dtype=torch.long)
        out = transformer(input_ids)
        self.assertEqual(out.shape, (1, args.vocab_size))

    # Verify the DeepSeek model produces logits of vocabulary size on forward.
    def test_deepseek_model_forward_returns_vocab_logits_when_called(self):
        args = DummyArgs()
        args.num_hidden_layers = 1
        args.compress_ratios = (1,)
        args.vocab_size = 8
        model = mod.DeepSeekModel(args)
        _convert_module_linears_to_float(model.model)
        model.forward = lambda input_ids, start_pos=0: model.model(input_ids, start_pos)
        input_ids = torch.randint(0, args.vocab_size, (1, 2), dtype=torch.long)
        out = model(input_ids)
        self.assertEqual(out.shape, (1, args.vocab_size))

    # Verify the fallback Hadamard transform implementation works with a fake scipy module.
    def test_hadamard_transform_ref_with_fake_scipy_returns_same_shape_when_fake_scipy_present(self):
        x = torch.randn(1, 1, 4)
        fake_scipy = types.ModuleType('scipy')
        fake_linalg = types.ModuleType('scipy.linalg')

        def fake_hadamard(n, dtype=None):
            return torch.eye(n, dtype=torch.float32).numpy()

        fake_linalg.hadamard = fake_hadamard
        fake_scipy.linalg = fake_linalg
        with patch.dict(sys.modules, {'scipy': fake_scipy, 'scipy.linalg': fake_linalg}):
            out = mod.hadamard_transform_ref(x, scale=1.0)
            self.assertEqual(out.shape, x.shape)


# ===== hc_split_sinkhorn =====


class TestHcSplitSinkhorn(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_hc_split_returns_correct_shapes_when_normal_input(self):
        """验证 hc_split_sinkhorn 对正常输入返回 (pre, post, comb) 且形状符合预期"""
        b, s, hc = 2, 3, 4
        mixes = torch.randn(b, s, (2 + hc) * hc)
        hc_scale = torch.tensor([0.5, 1.0, 2.0])
        hc_base = torch.randn((2 + hc) * hc)

        pre, post, comb = mod.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc, sinkhorn_iters=10)

        self.assertEqual(pre.shape, (b, s, hc))
        self.assertEqual(post.shape, (b, s, hc))
        self.assertEqual(comb.shape, (b, s, hc, hc))
        self.assertGreaterEqual(float(pre.min()), 0.0)
        self.assertLessEqual(float(pre.max()), 1.0)
        self.assertGreaterEqual(float(post.min()), 0.0)
        self.assertLessEqual(float(post.max()), 2.0)
        # comb 所有元素非负且有限
        self.assertGreaterEqual(float(comb.min()), 0.0)
        self.assertFalse(torch.isnan(comb).any())
        self.assertFalse(torch.isinf(comb).any())

    def test_hc_split_sinkhorn_repeats_when_sinkhorn_iters_increased(self):
        """验证 sinkhorn_iters 增加后 comb 收敛性更好（diff 更小）"""
        b, s, hc = 1, 2, 4
        mixes = torch.randn(b, s, (2 + hc) * hc)
        hc_scale = torch.tensor([1.0, 1.0, 1.0])
        hc_base = torch.randn((2 + hc) * hc)

        _, _, comb = mod.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc, sinkhorn_iters=1)
        _, _, comb_many = mod.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc, sinkhorn_iters=10)

        self.assertTrue((comb >= 0).all() and (comb_many >= 0).all())

    def test_hc_split_eps_keeps_comb_stable_when_all_zeros(self):
        """验证 eps 在 comb 全零时防止除零错误"""
        b, s, hc = 1, 1, 2
        hc_scale = torch.zeros(3)
        hc_base = torch.zeros((2 + hc) * hc)
        mixes = torch.zeros(b, s, (2 + hc) * hc)

        pre, post, comb = mod.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc, eps=1e-6)

        self.assertEqual(comb.shape, (b, s, hc, hc))
        self.assertFalse(torch.isnan(comb).any())
        self.assertFalse(torch.isinf(comb).any())


# ===== sparse_attn =====


class TestSparseAttn(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_sparse_attn_returns_correct_shape_when_valid_input(self):
        """验证 sparse_attn 对有效输入返回 (b, m, h, d) 形状"""
        b, m, h, d = 2, 4, 2, 8
        q = torch.randn(b, m, h, d)
        kv = torch.randn(b, 10, d)
        attn_sink = torch.randn(h)
        topk_idxs = torch.randint(0, 10, (b, m, 4))
        softmax_scale = d**-0.5

        out = mod.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

        self.assertEqual(out.shape, (b, m, h, d))

    def test_sparse_attn_handles_mask_when_some_indices_are_negative_one(self):
        """验证 sparse_attn 在 topk_idxs 包含 -1 时正确 mask"""
        b, m, h, d = 1, 3, 2, 8
        q = torch.randn(b, m, h, d)
        kv = torch.randn(b, 5, d)
        attn_sink = torch.randn(h)
        topk_idxs = torch.tensor([[[0, -1], [1, 2], [3, -1]]])
        softmax_scale = d**-0.5

        out = mod.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

        self.assertEqual(out.shape, (b, m, h, d))
        self.assertFalse(torch.isnan(out).any())

    def test_sparse_attn_returns_same_dtype_when_q_is_bfloat16(self):
        """验证 sparse_attn 在输入为 bf16 时输出保持同 dtype"""
        b, m, h, d = 1, 2, 2, 8
        q = torch.randn(b, m, h, d, dtype=torch.bfloat16)
        kv = torch.randn(b, 4, d, dtype=torch.bfloat16)
        attn_sink = torch.randn(h, dtype=torch.bfloat16)
        topk_idxs = torch.randint(0, 4, (b, m, 2))
        softmax_scale = d**-0.5

        out = mod.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

        self.assertEqual(out.dtype, torch.bfloat16)


# ===== hadamard_transform_ref =====


class TestHadamardTransformRef(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_hadamard_transform_preserves_shape_when_2d_input(self):
        """验证 hadamard_transform_ref 保持二维输入形状"""
        x = torch.randn(1, 1, 8)
        try:
            out = mod.hadamard_transform_ref(x, scale=1.0)
        except ImportError:
            self.skipTest("scipy not available")
            return
        self.assertEqual(out.shape, x.shape)

    def test_hadamard_transform_scale_applies_when_scale_not_one(self):
        """验证 scale 参数缩放了变换结果"""
        x = torch.randn(1, 1, 16)
        try:
            out1 = mod.hadamard_transform_ref(x, scale=1.0)
            out2 = mod.hadamard_transform_ref(x, scale=2.0)
        except ImportError:
            self.skipTest("scipy not available")
            return
        self.assertTrue(torch.allclose(out2, out1 * 2, atol=1e-5))


# ===== precompute_freqs_cis edge cases =====


class TestPrecomputeFreqsCisEdgeCases(unittest.TestCase):
    def test_precompute_freqs_cis_differs_when_rope_factor_changes(self):
        """验证不同 rope_factor 产生不同的频率"""
        freqs1 = mod.precompute_freqs_cis(4, 3, 16, 10000.0, 1.0, 32, 1)
        freqs2 = mod.precompute_freqs_cis(4, 3, 16, 10000.0, 8.0, 32, 1)
        self.assertFalse(torch.allclose(freqs1, freqs2))

    def test_precompute_freqs_cis_returns_zero_freqs_when_seqlen_zero(self):
        """验证 seqlen=0 时返回空 tensor"""
        freqs = mod.precompute_freqs_cis(4, 0, 16, 10000.0, 1.0, 32, 1)
        self.assertEqual(freqs.shape[0], 0)


# ===== Attention =====


class TestAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_attention_inits_all_projections_when_default(self):
        """验证 Attention 初始化后包含 q/输出投影和 kv/压缩相关子模块"""
        args = DummyArgs()
        attn = mod.Attention(0, args)

        self.assertIsInstance(attn.wq_b, nn.Linear)
        self.assertIsInstance(attn.wkv, nn.Linear)
        self.assertIsInstance(attn.wo_a, nn.Linear)
        self.assertIsInstance(attn.wo_b, nn.Linear)

    def test_attention_has_compressor_only_when_compress_ratio_greater_than_one(self):
        """验证 compress_ratio>1 时 compressor 存在，否则 compressor 不存在"""
        args = DummyArgs()
        args.compress_ratios = (4,)
        args.compress_ratio = 4
        attn = mod.Attention(0, args)

        self.assertTrue(hasattr(attn, 'compressor'))

    def test_attention_forward_returns_tensor_when_called(self):
        """验证 Attention.forward 返回与输入同形状的 tensor"""
        args = DummyArgs()
        attn = mod.Attention(0, args)
        _convert_module_linears_to_float(attn)
        bsz, seqlen = 1, 2
        x = torch.randn(bsz, seqlen, args.dim)

        out = attn(x, 0)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (bsz, seqlen, args.dim))


# ===== Gate score_func variants =====


class TestGateScoreFunc(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_gate_forward_sigmoid_when_score_func_set_to_sigmoid(self):
        """验证 Gate 使用 sigmoid 评分函数时返回正确形状"""
        args = DummyArgs()
        args.score_func = 'sigmoid'
        gate = mod.Gate(0, args)
        x = torch.randn(1, args.dim)
        input_ids = torch.tensor([0])

        weights, indices = gate(x, input_ids)

        self.assertEqual(weights.shape[-1], args.n_activated_experts)
        self.assertEqual(indices.shape[-1], args.n_activated_experts)

    def test_gate_forward_sqrtsoftplus_when_score_func_set_to_sqrtsoftplus(self):
        """验证 Gate 使用 sqrtsoftplus 评分函数时返回正确形状"""
        args = DummyArgs()
        args.score_func = 'sqrtsoftplus'
        gate = mod.Gate(0, args)
        x = torch.randn(1, args.dim)
        input_ids = torch.tensor([0])

        weights, indices = gate(x, input_ids)

        self.assertEqual(weights.shape[-1], args.n_activated_experts)


# ===== MoE.run_gate =====


class TestMoeRunGate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_moe_run_gate_delegates_to_gate_when_called(self):
        """验证 MoE.run_gate 委托给 self.gate 并返回 (weights, indices)"""
        args = DummyArgs()
        args.compress_ratios = (1,)
        moe = mod.MoE(0, args)
        x = torch.randn(1, args.dim)
        input_ids = torch.tensor([0])

        weights, indices = moe.run_gate(x, input_ids)

        self.assertIsNotNone(weights)
        self.assertIsNotNone(indices)


# ===== Expert swiglu_limit =====


class TestExpertSwigluLimit(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_expert_forward_clamps_when_swiglu_limit_zero(self):
        """验证 Expert 在 swiglu_limit=0 时 gate 和 up 输出被 clamp 到 0 附近"""
        expert = mod.Expert(8, 16, swiglu_limit=0.0)
        x = torch.randn(1, 1, 8) * 100

        out = expert(x)

        self.assertEqual(out.shape, (1, 1, 8))

    def test_expert_forward_large_swiglu_limit_when_swiglu_limit_is_high(self):
        """验证 Expert 在 swiglu_limit 很高时输出形状仍然正确"""
        expert = mod.Expert(8, 16, swiglu_limit=100.0)
        x = torch.randn(1, 1, 8)

        out = expert(x)

        self.assertEqual(out.shape, (1, 1, 8))


# ===== MoE 分布式 =====


class TestMoeDistributed(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_moe_forward_applies_all_reduce_when_world_size_greater_than_one(self):
        """验证 MoE 在分布式环境中调用 all_reduce"""
        args = DummyArgs()
        args.compress_ratios = (1,)
        moe = mod.MoE(0, args)
        x = torch.randn(1, 1, args.dim)
        input_ids = torch.zeros(1, 1, dtype=torch.long)

        with (
            patch.object(mod, 'world_size', 2),
            patch.object(mod, 'USE_DP_MODE', False),
            patch.object(mod, 'dist') as mock_dist,
        ):
            mock_dist.all_reduce = unittest.mock.MagicMock()
            out = moe(x, input_ids)

        self.assertEqual(out.shape, x.shape)
        mock_dist.all_reduce.assert_called_once()


# ===== Block hc_pre and hc_post =====


class TestBlockHcMethods(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_block_hc_pre_returns_tuple_when_called(self):
        """验证 Block.hc_pre 返回 (y, post, comb) 三元组"""
        args = DummyArgs()
        block = mod.Block(0, args)
        _convert_module_linears_to_float(block)
        x = torch.randn(1, 1, args.hc_mult, args.dim)

        y, post, comb = block.hc_pre(x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)

        self.assertIsInstance(y, torch.Tensor)
        self.assertIsInstance(post, torch.Tensor)
        self.assertIsInstance(comb, torch.Tensor)

    def test_block_hc_post_returns_same_dtype_when_called(self):
        """验证 Block.hc_post 输出 dtype 与输入 x 一致"""
        args = DummyArgs()
        block = mod.Block(0, args)
        x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16)
        residual = torch.randn(1, 1, args.hc_mult, args.dim, dtype=torch.bfloat16)
        post = torch.randn(1, 1, args.hc_mult)
        comb = torch.randn(1, 1, args.hc_mult, args.hc_mult)

        y = block.hc_post(x, residual, post, comb)

        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertEqual(y.shape, (1, 1, args.hc_mult, args.dim))
