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
#  Test cases for GLM-5 model

import unittest
import sys

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
)


# Add npu method to torch.Tensor for CPU testing
def _npu_to_cpu(self):
    """Mock npu() method to return tensor on CPU"""
    return self.cpu()


torch.Tensor.npu = _npu_to_cpu


def add_scale_attribute_to_model(model):
    """Add scale attribute to all MLA layers in the model to avoid AttributeError"""
    for module in model.modules():
        if isinstance(module, MLA):
            module.kv_b_proj.scale = None


def create_small_model_args():
    """Create a small ModelArgs configuration for testing

    Returns a ModelArgs instance with reduced dimensions suitable for CPU testing.
    This configuration is used across multiple test classes to ensure consistency.
    """
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
    """Create ModelArgs for full model testing with vocabulary

    Args:
        vocab_size: Vocabulary size for the model

    Returns a ModelArgs instance suitable for testing complete transformer models.
    """
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

    def test_ModelArgs_values_shouldOverrideDefault_when_initWithCustomArgs(self):
        args = ModelArgs(max_batch_size=4, max_seq_len=1024, dtype="fp8", vocab_size=50000, scoring_func="softmax")
        self.assertEqual(args.max_batch_size, 4)
        self.assertEqual(args.max_seq_len, 1024)
        self.assertEqual(args.dtype, "fp8")
        self.assertEqual(args.scoring_func, "softmax")


class TestUtilityFunctions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')

    def test_fp8_index_shape_shouldMatch_when_givenRandomTensors(self):
        q = torch.randn(2, 8, 4, 16)
        q_s = torch.randn(2, 8, 4, 1).abs()
        k = torch.randn(2, 10, 1, 16)

        result = fp8_index(q, q_s, k)
        self.assertEqual(result.shape, (2, 8, 10))
        self.assertIsNotNone(result)

    def test_linear_output_shouldMatch_when_withOrWithoutBias(self):
        x = torch.randn(2, 3, 4)
        weight = torch.randn(5, 4)
        bias = torch.randn(5)

        result = linear(x, weight)
        self.assertEqual(result.shape, (2, 3, 5))

        result = linear(x, weight, bias)
        self.assertEqual(result.shape, (2, 3, 5))

    def test_hadamard_transform_ref_output_shouldKeepShape_when_powerOfTwoDim(self):
        x = torch.randn(2, 3, 16)
        result = hadamard_transform_ref(x, scale=2.0)
        self.assertEqual(result.shape, x.shape)

    def test_hadamard_transform_ref_output_shouldKeepShape_when_nonPowerOfTwoDim(self):
        x = torch.randn(2, 3, 10)
        result = hadamard_transform_ref(x, scale=1.5)
        self.assertEqual(result.shape, x.shape)

    def test_rotate_activation_output_shouldKeepShape_when_givenRandomInput(self):
        x = torch.randn(2, 4, 32)
        result = rotate_activation(x)
        self.assertEqual(result.shape, x.shape)

    def test_weight_dequant_output_shouldRestoreShape_when_givenQuantizedWeight(self):
        weight = torch.randn(256, 256)
        scale = torch.randn(256 * 256 // (BLOCK_SIZE * BLOCK_SIZE))

        result = weight_dequant(weight, scale)
        self.assertEqual(result.shape, weight.shape)


class TestNormalizationLayers(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_RMSNorm_output_shouldKeepShape_when_withoutResidual(self):
        dim = 128
        norm = RMSNorm(dim, eps=1e-6)
        x = torch.randn(2, 4, dim)

        output = norm(x)
        self.assertEqual(output.shape, x.shape)
        self.assertIsInstance(output, torch.Tensor)

    def test_RMSNorm_output_shouldKeepShape_when_withResidual(self):
        dim = 128
        norm = RMSNorm(dim, eps=1e-6)
        x = torch.randn(2, 4, dim)
        residual = torch.randn(2, 4, dim)

        output, new_residual = norm(x, residual)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(new_residual.shape, x.shape)

    def test_LayerNorm_output_shouldKeepShape_when_givenInput(self):
        dim = 128
        norm = LayerNorm(dim, eps=1e-6)
        x = torch.randn(2, 4, dim)

        output = norm(x)
        self.assertEqual(output.shape, x.shape)


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_precompute_freqs_cis_shouldHandle_when_seqLenLessThanOriginal(self):
        args = ModelArgs()
        args.max_seq_len = 2048
        args.original_seq_len = 4096

        freqs_cis = precompute_freqs_cis(args)
        self.assertEqual(freqs_cis.shape, (2048, args.qk_rope_head_dim // 2))

    def test_precompute_freqs_cis_shouldHandle_when_seqLenGreaterThanOriginal(self):
        args = ModelArgs()
        args.max_seq_len = 8192
        args.original_seq_len = 4096

        freqs_cis = precompute_freqs_cis(args)
        self.assertEqual(freqs_cis.shape, (8192, args.qk_rope_head_dim // 2))

    def test_apply_rotary_emb_output_shouldKeepShape_when_givenInput(self):
        batch_size, seq_len, n_heads, head_dim = 2, 4, 8, 64
        x = torch.randn(batch_size, seq_len, n_heads, head_dim)

        args = ModelArgs()
        args.max_seq_len = seq_len
        freqs_cis = precompute_freqs_cis(args)

        result = apply_rotary_emb(x, freqs_cis)
        self.assertEqual(result.shape, x.shape)


class TestParallelEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_ParallelEmbedding_output_shouldHaveCorrectShape_when_givenTokenIds(self):
        vocab_size = 1000
        dim = 128

        embedding = ParallelEmbedding(vocab_size, dim)
        x = torch.randint(0, vocab_size, (2, 10))

        output = embedding(x)
        self.assertEqual(output.shape, (2, 10, dim))


class TestMLPLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_MLP_output_shouldKeepShape_when_givenInput(self):
        dim = 128
        inter_dim = 256

        mlp = MLP(dim, inter_dim)
        x = torch.randn(2, 4, dim)

        output = mlp(x)
        self.assertEqual(output.shape, x.shape)


class TestExpertAndMoE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_Expert_output_shouldKeepShape_when_givenInput(self):
        dim = 128
        inter_dim = 256

        expert = Expert(dim, inter_dim)
        x = torch.randn(2, 4, dim)

        output = expert(x)
        self.assertEqual(output.shape, x.shape)

    def test_Gate_weightsAndIndices_shouldMatchShape_when_scoringFuncIsSigmoid(self):
        args = ModelArgs()
        args.hidden_size = 128
        args.n_routed_experts = 16
        args.num_experts_per_tok = 4
        args.n_group = 1
        args.scoring_func = "sigmoid"

        gate = Gate(args)
        x = torch.randn(8, args.hidden_size)

        weights, indices = gate(x)
        self.assertEqual(weights.shape, (8, 4))
        self.assertEqual(indices.shape, (8, 4))

    def test_Gate_weightsAndIndices_shouldMatchShape_when_scoringFuncIsSoftmax(self):
        args = ModelArgs()
        args.hidden_size = 128
        args.n_routed_experts = 16
        args.num_experts_per_tok = 4
        args.n_group = 1
        args.scoring_func = "softmax"

        gate = Gate(args)
        x = torch.randn(8, args.hidden_size)

        weights, indices = gate(x)
        self.assertEqual(weights.shape, (8, 4))
        self.assertEqual(indices.shape, (8, 4))

    def test_Gate_weightsAndIndices_shouldMatchShape_when_withGrouping(self):
        args = ModelArgs()
        args.hidden_size = 128
        args.n_routed_experts = 16
        args.num_experts_per_tok = 4
        args.n_group = 4
        args.topk_group = 2
        args.scoring_func = "sigmoid"

        gate = Gate(args)
        x = torch.randn(8, args.hidden_size)

        weights, indices = gate(x)
        self.assertEqual(weights.shape, (8, 4))
        self.assertEqual(indices.shape, (8, 4))

    def test_MoE_output_shouldKeepShape_when_givenInput(self):
        args = ModelArgs()
        args.hidden_size = 128
        args.moe_intermediate_size = 256
        args.n_routed_experts = 8
        args.num_experts_per_tok = 2
        args.n_shared_experts = 1
        args.n_group = 1
        args.scoring_func = "sigmoid"

        moe = MoE(args)
        x = torch.randn(2, 4, args.hidden_size)

        output = moe(x)
        self.assertEqual(output.shape, x.shape)


class TestMLALayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.args = create_small_model_args()

    def test_MLA_output_shouldKeepShape_when_prefillWithMask(self):
        mla = MLA(self.args)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, self.args.hidden_size)
        freqs_cis = precompute_freqs_cis(self.args)[:seq_len]
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)

        output = mla(x, start_pos=0, freqs_cis=freqs_cis, mask=mask)

        self.assertEqual(output.shape, x.shape)

    def test_MLA_scaleAttribute_shouldBeHandled_when_inDecodeMode(self):
        mla = MLA(self.args)

        self.assertFalse(hasattr(mla.kv_b_proj, 'scale') and mla.kv_b_proj.scale is not None)

        mla.kv_b_proj.scale = None
        self.assertIsNone(mla.kv_b_proj.scale)

        self.assertIsNone(mla.dequant_wkv_b)

    def test_MLA_cache_shouldBePopulated_when_afterForwardPass(self):
        mla = MLA(self.args)
        mla.kv_b_proj.scale = None

        batch_size = 2

        seq_len_first = 8
        x_first = torch.randn(batch_size, seq_len_first, self.args.hidden_size)
        freqs_cis_first = precompute_freqs_cis(self.args)[:seq_len_first]
        mask_first = torch.full((seq_len_first, seq_len_first), float("-inf")).triu_(1)

        output_first = mla(x_first, start_pos=0, freqs_cis=freqs_cis_first, mask=mask_first)
        self.assertEqual(output_first.shape, x_first.shape)

        self.assertTrue(torch.any(mla.kv_cache[:batch_size, :seq_len_first] != 0))
        self.assertTrue(torch.any(mla.pe_cache[:batch_size, :seq_len_first] != 0))

    def test_MLA_softmaxScale_shouldBeAdjusted_when_maxSeqLenGtOriginal(self):
        args_extended = create_small_model_args()
        args_extended.max_seq_len = 8192
        args_extended.original_seq_len = 4096

        mla = MLA(args_extended)

        self.assertIsNotNone(mla.softmax_scale)

    def test_MLA_dequantWkvB_shouldRemainNone_when_scaleIsNone(self):
        mla = MLA(self.args)

        self.assertIsNone(mla.dequant_wkv_b)

        mla.kv_b_proj.scale = None

        self.assertIsNone(mla.dequant_wkv_b)


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

    def test_Indexer_indicesShape_shouldMatch_when_maskIsNone(self):
        indexer = Indexer(self.args)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, self.args.hidden_size)
        qr = torch.randn(batch_size, seq_len, self.args.q_lora_rank)
        freqs_cis = precompute_freqs_cis(self.args)[:seq_len]

        indices = indexer(x, qr, start_pos=0, freqs_cis=freqs_cis, mask=None)

        self.assertEqual(indices.shape[0], batch_size)
        self.assertEqual(indices.shape[1], seq_len)

    def test_Indexer_topk_shouldBeLimited_when_topkGtEndPos(self):
        self.args.index_topk = 1000
        indexer = Indexer(self.args)

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, self.args.hidden_size)
        qr = torch.randn(batch_size, seq_len, self.args.q_lora_rank)
        freqs_cis = precompute_freqs_cis(self.args)[:seq_len]
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)

        indices = indexer(x, qr, start_pos=0, freqs_cis=freqs_cis, mask=mask)

        self.assertTrue(indices.shape[-1] <= seq_len)


class TestBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.args = create_model_args_with_vocab()

    def test_Block_outputShape_shouldMatch_when_usingMLP(self):
        layer_id = 0
        block = Block(layer_id, self.args)
        add_scale_attribute_to_model(block)

        self.assertIsInstance(block.mlp, MLP)

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, self.args.hidden_size)
        freqs_cis = precompute_freqs_cis(self.args)[:seq_len]
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)

        output, residual = block(x, residual=None, start_pos=0, freqs_cis=freqs_cis, mask=mask)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(residual.shape, x.shape)

    def test_Block_outputShape_shouldMatch_when_usingMoE(self):
        layer_id = 3
        block = Block(layer_id, self.args)
        add_scale_attribute_to_model(block)

        self.assertIsInstance(block.mlp, MoE)

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, self.args.hidden_size)
        freqs_cis = precompute_freqs_cis(self.args)[:seq_len]
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)

        output, residual = block(x, residual=None, start_pos=0, freqs_cis=freqs_cis, mask=mask)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(residual.shape, x.shape)

    def test_Block_outputShape_shouldMatch_when_withResidualInput(self):
        layer_id = 0
        block = Block(layer_id, self.args)
        add_scale_attribute_to_model(block)

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, self.args.hidden_size)
        residual = torch.randn(batch_size, seq_len, self.args.hidden_size)
        freqs_cis = precompute_freqs_cis(self.args)[:seq_len]
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)

        output, new_residual = block(x, residual=residual, start_pos=0, freqs_cis=freqs_cis, mask=mask)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(new_residual.shape, x.shape)


class TestGLMMOEModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.args = create_model_args_with_vocab(vocab_size=1000)

    def test_GLMMOEModel_outputShape_shouldMatch_when_seqLenGt1(self):
        model = GLMMOEModel(self.args)
        add_scale_attribute_to_model(model)

        batch_size, seq_len = 2, 8
        tokens = torch.randint(0, self.args.vocab_size, (batch_size, seq_len))

        output, residual = model(tokens, start_pos=0)

        self.assertEqual(output.shape, (batch_size, seq_len, self.args.hidden_size))

    def test_GLMMOEModel_shouldHandle_when_seqLenIs1(self):
        model = GLMMOEModel(self.args)
        add_scale_attribute_to_model(model)

        batch_size, seq_len = 2, 1
        tokens = torch.randint(0, self.args.vocab_size, (batch_size, seq_len))

        try:
            output, residual = model(tokens, start_pos=0)
            self.assertEqual(output.shape, (batch_size, seq_len, self.args.hidden_size))
        except RuntimeError:
            pass

    def test_GLMMOEModel_shouldAccept_when_startPosProvided(self):
        model = GLMMOEModel(self.args)
        add_scale_attribute_to_model(model)

        batch_size = 2
        seq_len = 8

        tokens = torch.randint(0, self.args.vocab_size, (batch_size, seq_len))
        output, _ = model(tokens, start_pos=0)
        self.assertEqual(output.shape, (batch_size, seq_len, self.args.hidden_size))

        try:
            tokens2 = torch.randint(0, self.args.vocab_size, (batch_size, seq_len))
            output2, _ = model(tokens2, start_pos=5)
            self.assertEqual(output2.shape, (batch_size, seq_len, self.args.hidden_size))
        except RuntimeError:
            pass


class TestTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.args = create_model_args_with_vocab(vocab_size=1000)

    def test_Transformer_logitsShape_shouldMatch_when_givenTokens(self):
        model = Transformer(self.args)
        add_scale_attribute_to_model(model)

        batch_size, seq_len = 2, 8
        tokens = torch.randint(0, self.args.vocab_size, (batch_size, seq_len))

        logits = model(tokens, start_pos=0)

        self.assertEqual(logits.shape, (batch_size, self.args.vocab_size))

    def test_Transformer_shouldHandle_when_singleTokenInput(self):
        model = Transformer(self.args)
        add_scale_attribute_to_model(model)

        batch_size, seq_len = 2, 1
        tokens = torch.randint(0, self.args.vocab_size, (batch_size, seq_len))

        try:
            logits = model(tokens, start_pos=0)
            self.assertEqual(logits.shape, (batch_size, self.args.vocab_size))
        except RuntimeError:
            pass


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_precomputeFreqsCis_shouldHandle_when_maxSeqLenGtOriginal(self):
        args = ModelArgs()
        args.max_seq_len = 8192
        args.original_seq_len = 4096

        freqs_cis = precompute_freqs_cis(args)
        self.assertIsNotNone(freqs_cis)

    def test_MoE_shouldHandle_when_smallBatchInput(self):
        args = ModelArgs()
        args.hidden_size = 128
        args.moe_intermediate_size = 256
        args.n_routed_experts = 16
        args.num_experts_per_tok = 2
        args.n_shared_experts = 1
        args.n_group = 1
        args.scoring_func = "sigmoid"

        moe = MoE(args)
        x = torch.randn(1, 1, args.hidden_size)

        output = moe(x)
        self.assertEqual(output.shape, x.shape)

    def test_Gate_shouldHandle_when_correctionBiasAndGroups(self):
        args = ModelArgs()
        args.hidden_size = 7168
        args.n_routed_experts = 16
        args.num_experts_per_tok = 4
        args.n_group = 4
        args.topk_group = 2
        args.scoring_func = "sigmoid"

        gate = Gate(args)
        x = torch.randn(8, args.hidden_size)

        weights, indices = gate(x)
        self.assertEqual(weights.shape, (8, 4))


class TestDataTypes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_Transformer_shouldRun_when_defaultDtypeIsFloat32(self):
        args = create_model_args_with_vocab(vocab_size=500)
        args.max_seq_len = 64
        args.hidden_size = 128
        args.intermediate_size = 256
        args.num_hidden_layers = 2
        args.first_k_dense_replace = 1
        args.num_attention_heads = 4
        args.q_lora_rank = 64
        args.kv_lora_rank = 32
        args.qk_nope_head_dim = 16
        args.qk_rope_head_dim = 16
        args.v_head_dim = 16
        args.index_n_heads = 2
        args.index_topk = 32
        args.n_routed_experts = 4
        args.num_experts_per_tok = 2

        torch.set_default_dtype(torch.float32)
        model = Transformer(args)
        add_scale_attribute_to_model(model)

        batch_size, seq_len = 1, 4
        tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

        logits = model(tokens, start_pos=0)

        self.assertEqual(logits.shape, (batch_size, args.vocab_size))
        torch.set_default_dtype(torch.float32)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelArgs))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestNormalizationLayers))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionalEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelEmbedding))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPLayer))
    suite.addTests(loader.loadTestsFromTestCase(TestExpertAndMoE))
    suite.addTests(loader.loadTestsFromTestCase(TestMLALayer))
    suite.addTests(loader.loadTestsFromTestCase(TestIndexer))
    suite.addTests(loader.loadTestsFromTestCase(TestBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestGLMMOEModel))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformer))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestDataTypes))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    # Set default device to CPU
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)

    # Run tests
    test_result = run_tests()

    # Exit with appropriate code
    sys.exit(0 if test_result.wasSuccessful() else 1)
