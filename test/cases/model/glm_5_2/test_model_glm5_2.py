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
#  Test cases for GLM-5.2 model

import unittest

import torch

from msmodelslim.model.glm_5_2.model import (
    ModelArgs,
    ParallelEmbedding,
    RMSNorm,
    LayerNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    rotate_activation,
    fp8_index,
    Indexer,
    weight_dequant,
    MLA,
    MLP,
    Gate,
    Expert,
    MoE,
    Block,
    GLMMOEModel,
    Transformer,
    BLOCK_SIZE,
    _normalize_indexer_type,
    get_indexer_type,
    has_indexer,
)


# Add npu method to torch.Tensor for CPU testing
def _npu_to_cpu(self):
    return self.cpu()


torch.Tensor.npu = _npu_to_cpu


def add_scale_attribute_to_model(model):
    """Add scale attribute to all MLA layers in the model to avoid AttributeError"""
    for module in model.modules():
        if isinstance(module, MLA):
            module.kv_b_proj.scale = None


def create_small_model_args():
    """Create a small ModelArgs configuration for CPU testing."""
    args = ModelArgs()
    args.max_batch_size = 2
    args.max_seq_len = 128
    args.hidden_size = 256
    args.num_attention_heads = 8
    args.q_lora_rank = 128
    args.kv_lora_rank = 64
    args.qk_nope_head_dim = 32
    args.qk_rope_head_dim = 32
    args.v_head_dim = 32
    args.index_n_heads = 4
    args.index_head_dim = 128  # Must be multiple of 128
    args.index_topk = 64
    return args


def create_model_args_with_vocab(vocab_size=1000):
    """Create ModelArgs for full model testing with vocabulary."""
    args = create_small_model_args()
    args.vocab_size = vocab_size
    args.intermediate_size = 512
    args.moe_intermediate_size = 256
    args.num_hidden_layers = 4
    args.first_k_dense_replace = 2
    args.n_routed_experts = 8
    args.num_experts_per_tok = 2
    args.n_shared_experts = 1
    args.n_group = 1
    return args


class TestModelArgs(unittest.TestCase):
    def test_ModelArgs_values_shouldMatchDefault_when_initWithoutArgs(self):
        args = ModelArgs()
        self.assertEqual(args.max_batch_size, 8)
        self.assertEqual(args.max_seq_len, 4096 * 4)
        self.assertEqual(args.dtype, "bf16")
        self.assertEqual(args.vocab_size, 154880)
        self.assertIsNone(args.indexer_types)
        self.assertIsNone(args.index_topk_pattern)
        self.assertEqual(args.index_topk_freq, 1)


class TestIndexerTypeFunctions(unittest.TestCase):
    def test_normalizeIndexerType_shouldHandle_when_variousInputs(self):
        self.assertEqual(_normalize_indexer_type("F"), "full")
        self.assertEqual(_normalize_indexer_type("s"), "shared")
        self.assertEqual(_normalize_indexer_type("Full"), "full")
        self.assertEqual(_normalize_indexer_type(None), "full")

    def test_getIndexerType_shouldFollowPatternAndFreq_when_noExplicitTypes(self):
        args = ModelArgs()
        args.num_hidden_layers = 4
        args.index_topk_pattern = "FSSF"
        self.assertEqual([get_indexer_type(args, i) for i in range(4)], ["full", "shared", "shared", "full"])
        self.assertEqual(args.indexer_types, ["full", "shared", "shared", "full"])

        args2 = ModelArgs()
        args2.num_hidden_layers = 4
        args2.index_topk_freq = 2
        self.assertEqual(get_indexer_type(args2, 2), "shared")

    def test_getIndexerType_shouldReturnFull_when_mtpLayer(self):
        args = ModelArgs()
        args.num_hidden_layers = 79
        args.indexer_types = ["shared"] * 78
        self.assertEqual(get_indexer_type(args, 78), "full")

    def test_hasIndexer_shouldReturnCorrect_when_givenTypes(self):
        args = ModelArgs()
        args.num_hidden_layers = 4
        args.indexer_types = ["full", "shared", "full", "shared"]
        self.assertTrue(has_indexer(args, 0))
        self.assertFalse(has_indexer(args, 1))


class TestUtilityFunctions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_fp8_index_shape_shouldMatch_when_givenRandomTensors(self):
        q = torch.randn(2, 8, 4, 16)
        q_s = torch.randn(2, 8, 4, 1).abs()
        k = torch.randn(2, 10, 1, 16)
        result = fp8_index(q, q_s, k)
        self.assertEqual(result.shape, (2, 8, 10))

    def test_rotateActivation_shouldKeepShape_when_givenInput(self):
        x = torch.randn(2, 4, 32)
        self.assertEqual(rotate_activation(x).shape, x.shape)

    def test_weightDequant_shouldKeepShape_when_givenQuantizedWeight(self):
        weight = torch.randn(256, 256)
        scale = torch.randn(256 * 256 // (BLOCK_SIZE * BLOCK_SIZE))
        self.assertEqual(weight_dequant(weight, scale).shape, weight.shape)


class TestNormalizationLayers(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_RMSNorm_shouldKeepShape_when_withAndWithoutResidual(self):
        dim = 128
        x = torch.randn(2, 4, dim)
        residual = torch.randn(2, 4, dim)
        norm = RMSNorm(dim, eps=1e-6)
        self.assertEqual(norm(x).shape, x.shape)
        output, new_residual = norm(x, residual)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(new_residual.shape, x.shape)

    def test_LayerNorm_shouldKeepShape_when_givenInput(self):
        dim = 128
        norm = LayerNorm(dim, eps=1e-6)
        self.assertEqual(norm(torch.randn(2, 4, dim)).shape, (2, 4, dim))


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_precomputeFreqsCis_shouldHandle_when_seqLenAroundOriginal(self):
        args = ModelArgs()
        args.max_seq_len = 2048
        args.original_seq_len = 4096
        self.assertEqual(precompute_freqs_cis(args).shape, (2048, args.qk_rope_head_dim // 2))

        args.max_seq_len = 8192
        args.original_seq_len = 4096
        self.assertEqual(precompute_freqs_cis(args).shape, (8192, args.qk_rope_head_dim // 2))

    def test_applyRotaryEmb_shouldKeepShape_when_givenInput(self):
        batch_size, seq_len, n_heads, head_dim = 2, 4, 8, 64
        x = torch.randn(batch_size, seq_len, n_heads, head_dim)
        args = ModelArgs()
        args.max_seq_len = seq_len
        freqs_cis = precompute_freqs_cis(args)
        self.assertEqual(apply_rotary_emb(x, freqs_cis).shape, x.shape)


class TestParallelEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_ParallelEmbedding_output_shouldHaveCorrectShape_when_givenTokenIds(self):
        vocab_size, dim = 1000, 128
        embedding = ParallelEmbedding(vocab_size, dim)
        x = torch.randint(0, vocab_size, (2, 10))
        self.assertEqual(embedding(x).shape, (2, 10, dim))


class TestMLPExpertGate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_MLPAndExpert_shouldKeepShape_when_givenInput(self):
        x = torch.randn(2, 4, 128)
        self.assertEqual(MLP(128, 256)(x).shape, x.shape)
        self.assertEqual(Expert(128, 256)(x).shape, x.shape)

    def test_Gate_weightsAndIndices_shouldMatchShape_when_variousConfigs(self):
        for scoring_func, n_group in [("sigmoid", 1), ("softmax", 1), ("sigmoid", 4)]:
            args = ModelArgs()
            args.hidden_size = 128
            args.n_routed_experts = 16
            args.num_experts_per_tok = 4
            args.n_group = n_group
            args.topk_group = 2
            args.scoring_func = scoring_func
            weights, indices = Gate(args)(torch.randn(8, 128))
            self.assertEqual(weights.shape, (8, 4))
            self.assertEqual(indices.shape, (8, 4))

    def test_MoE_shouldKeepShape_when_givenInput(self):
        args = ModelArgs()
        args.hidden_size = 128
        args.moe_intermediate_size = 256
        args.n_routed_experts = 8
        args.num_experts_per_tok = 2
        args.n_shared_experts = 1
        args.n_group = 1
        args.scoring_func = "sigmoid"
        x = torch.randn(2, 4, args.hidden_size)
        self.assertEqual(MoE(args)(x).shape, x.shape)


class TestMLALayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.args = create_small_model_args()

    def test_MLA_output_shouldKeepShape_when_prefillAndDecode(self):
        mla = MLA(self.args)
        mla.kv_b_proj.scale = None

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, self.args.hidden_size)
        freqs_cis = precompute_freqs_cis(self.args)[:seq_len]
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)

        output = mla(x, start_pos=0, freqs_cis=freqs_cis, mask=mask)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.any(mla.kv_cache[:batch_size, :seq_len] != 0))

        # decode：使用无 indexer 的 shared 层并复用前一层的 topk 结果
        args_shared = create_small_model_args()
        args_shared.num_hidden_layers = 2
        args_shared.indexer_types = ["full", "shared"]
        mla_shared = MLA(args_shared, layer_id=1)
        mla_shared.kv_b_proj.scale = None
        mla_shared.kv_cache[:batch_size, :seq_len] = mla.kv_cache[:batch_size, :seq_len]
        mla_shared.pe_cache[:batch_size, :seq_len] = mla.pe_cache[:batch_size, :seq_len]

        x_decode = torch.randn(batch_size, 1, self.args.hidden_size)
        freqs_cis_decode = precompute_freqs_cis(self.args)[seq_len : seq_len + 1]
        prev_topk = torch.randint(0, seq_len, (batch_size, 1, seq_len))
        output = mla_shared(
            x_decode,
            start_pos=seq_len,
            freqs_cis=freqs_cis_decode,
            mask=None,
            prev_topk_indices=prev_topk,
        )
        self.assertEqual(output.shape, x_decode.shape)

    def test_MLA_hasIndexer_shouldFollow_when_layerIdAndIndexerTypes(self):
        args = create_small_model_args()
        args.num_hidden_layers = 2
        args.indexer_types = ["full", "shared"]

        mla_full = MLA(args, layer_id=0)
        self.assertTrue(mla_full.has_indexer)
        self.assertIsNotNone(mla_full.indexer)
        self.assertTrue(mla_full.next_skip_topk)

        mla_shared = MLA(args, layer_id=1)
        self.assertFalse(mla_shared.has_indexer)
        self.assertIsNone(mla_shared.indexer)

    def test_MLA_shouldRaise_when_noIndexerAndPrevMissing(self):
        args = create_small_model_args()
        args.num_hidden_layers = 2
        args.indexer_types = ["full", "shared"]
        mla = MLA(args, layer_id=1)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, args.hidden_size)
        freqs_cis = precompute_freqs_cis(args)[:seq_len]
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)
        with self.assertRaises(RuntimeError):
            mla(x, start_pos=0, freqs_cis=freqs_cis, mask=mask)

    def test_MLA_shouldReusePrevTopk_when_noIndexerAndPrevProvided(self):
        args = create_small_model_args()
        args.num_hidden_layers = 2
        args.indexer_types = ["full", "shared"]
        mla = MLA(args, layer_id=1)
        mla.kv_b_proj.scale = None

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, args.hidden_size)
        freqs_cis = precompute_freqs_cis(args)[:seq_len]
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)
        topk = min(args.index_topk, seq_len)
        prev_topk = torch.randint(0, seq_len, (batch_size, seq_len, topk))

        output = mla(x, start_pos=0, freqs_cis=freqs_cis, mask=mask, prev_topk_indices=prev_topk)
        self.assertEqual(output.shape, x.shape)


class TestIndexer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.args = create_small_model_args()
        self.args.index_topk = 32

    def test_Indexer_indicesShape_shouldMatch_when_givenMask(self):
        indexer = Indexer(self.args)
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, self.args.hidden_size)
        qr = torch.randn(batch_size, seq_len, self.args.q_lora_rank)
        freqs_cis = precompute_freqs_cis(self.args)[:seq_len]
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)
        indices = indexer(x, qr, start_pos=0, freqs_cis=freqs_cis, mask=mask)
        self.assertEqual(indices.shape[0], batch_size)
        self.assertEqual(indices.shape[1], seq_len)


class TestBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.args = create_model_args_with_vocab()

    def test_Block_outputShape_shouldMatch_when_usingMLPAndMoE(self):
        for layer_id in (0, 3):
            block = Block(layer_id, self.args)
            add_scale_attribute_to_model(block)
            x = torch.randn(2, 4, self.args.hidden_size)
            freqs_cis = precompute_freqs_cis(self.args)[:4]
            mask = torch.full((4, 4), float("-inf")).triu_(1)
            output, residual = block(x, residual=None, start_pos=0, freqs_cis=freqs_cis, mask=mask)
            self.assertEqual(output.shape, x.shape)
            self.assertEqual(residual.shape, x.shape)


class TestGLMMOEModelAndTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.args = create_model_args_with_vocab(vocab_size=1000)

    def test_GLMMOEModel_outputShape_shouldMatch_when_givenTokens(self):
        model = GLMMOEModel(self.args)
        add_scale_attribute_to_model(model)
        tokens = torch.randint(0, self.args.vocab_size, (2, 8))
        output, residual = model(tokens, start_pos=0)
        self.assertEqual(output.shape, (2, 8, self.args.hidden_size))

    def test_Transformer_logitsShape_shouldMatch_when_givenTokens(self):
        model = Transformer(self.args)
        add_scale_attribute_to_model(model)
        tokens = torch.randint(0, self.args.vocab_size, (2, 8))
        logits = model(tokens, start_pos=0)
        self.assertEqual(logits.shape, (2, self.args.vocab_size))

    def test_Transformer_shouldRun_when_alternatingIndexerTypes(self):
        self.args.num_hidden_layers = 4
        self.args.indexer_types = ["full", "shared", "full", "shared"]
        model = Transformer(self.args)
        add_scale_attribute_to_model(model)
        tokens = torch.randint(0, self.args.vocab_size, (2, 8))
        logits = model(tokens, start_pos=0)
        self.assertEqual(logits.shape, (2, self.args.vocab_size))
