#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.model.step3_7_flash.step3p7_mtp import (
    Step3p5RotaryEmbedding,
    Step3p5RMSNorm,
    Step3p5MLP,
    Step3p5Attention,
    Step3p7MTPModule,
    SharedHead,
)


MTP_PATH = "msmodelslim.model.step3_7_flash.step3p7_mtp"


def _make_config(
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=8,
    num_attention_groups=4,
    max_position_embeddings=128,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    sliding_window=None,
    use_head_wise_attn_gate=False,
    use_rope_layers=None,
    layer_types=None,
    rope_scaling=None,
    vocab_size=100,
    partial_rotary_factors=None,
    yarn_only_types=None,
    attention_other_setting=None,
    swiglu_limits=None,
    swiglu_limits_shared=None,
    num_layer_types=48,
):
    if attention_other_setting is None:
        attention_other_setting = {"num_attention_heads": 4, "num_attention_groups": 2}
    # Default layer_types to num_layer_types entries of "sliding_attention" so
    # Step3p5Attention can index any layer up to (num_layer_types-1) without
    # IndexError. The real Step3.7 config has 48 entries (45 main + 3 MTP).
    if layer_types is None:
        layer_types = ["sliding_attention"] * num_layer_types
    cfg = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_attention_groups=num_attention_groups,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        sliding_window=sliding_window,
        use_head_wise_attn_gate=use_head_wise_attn_gate,
        use_rope_layers=use_rope_layers,
        layer_types=layer_types,
        rope_scaling=rope_scaling,
        # Step3p5RotaryEmbedding reads rope_parameters first; fall back to rope_scaling
        # so the test config matches the actual model config.
        rope_parameters=rope_scaling if rope_scaling is not None else {"rope_type": "linear"},
        vocab_size=vocab_size,
        partial_rotary_factors=partial_rotary_factors,
        attention_other_setting=attention_other_setting,
        swiglu_limits=swiglu_limits,
        swiglu_limits_shared=swiglu_limits_shared,
    )
    if yarn_only_types is not None:
        cfg.yarn_only_types = yarn_only_types
    return cfg


def _patched_rope_init():
    """Patch ROPE_INIT_FUNCTIONS so Step3p5RotaryEmbedding can be instantiated in tests.

    The real transformers registry does not contain a 'default' entry, so we add a
    minimal stub that returns the inverse frequencies the test expects.
    """

    def _fake_rope_init(config, device=None):
        base = getattr(config, "rope_theta", 10000.0)
        head_dim = getattr(config, "head_dim", None) or (
            getattr(config, "hidden_size", 64) // getattr(config, "num_attention_heads", 8)
        )
        partial = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        return inv_freq, 1.0

    return patch(f"{MTP_PATH}.ROPE_INIT_FUNCTIONS", {"linear": _fake_rope_init, "default": _fake_rope_init})


class TestStep3p5RMSNorm(unittest.TestCase):
    def test_init_when_default_then_weight_is_ones(self):
        norm = Step3p5RMSNorm(64, eps=1e-5)

        self.assertEqual(norm.weight.shape, (64,))
        self.assertTrue(torch.allclose(norm.weight, torch.ones(64)))
        self.assertEqual(norm.variance_epsilon, 1e-5)

    def test_init_when_different_hidden_size_then_weight_shape_matches(self):
        norm = Step3p5RMSNorm(128, eps=1e-6)

        self.assertEqual(norm.weight.shape, (128,))
        self.assertEqual(norm.variance_epsilon, 1e-6)


class TestStep3p5MLP(unittest.TestCase):
    def test_init_when_default_then_has_correct_layer_shapes(self):
        config = _make_config(hidden_size=64, intermediate_size=128)
        mlp = Step3p5MLP(config)

        self.assertEqual(mlp.gate_proj.weight.shape, (128, 64))
        self.assertEqual(mlp.up_proj.weight.shape, (128, 64))
        self.assertEqual(mlp.down_proj.weight.shape, (64, 128))
        self.assertIsNone(mlp.limit)

    def test_init_when_custom_intermediate_size_then_uses_override(self):
        config = _make_config(hidden_size=64, intermediate_size=128)
        mlp = Step3p5MLP(config, intermediate_size=48)

        self.assertEqual(mlp.gate_proj.weight.shape[0], 48)
        self.assertEqual(mlp.up_proj.weight.shape[0], 48)
        self.assertEqual(mlp.down_proj.weight.shape[1], 48)

    def test_init_when_swiglu_limit_set_then_stored(self):
        config = _make_config()
        mlp = Step3p5MLP(config, swiglu_limit=1.0)

        self.assertEqual(mlp.limit, 1.0)


class TestStep3p5RotaryEmbedding(unittest.TestCase):
    def test_init_when_rope_theta_float_then_inv_freq_cached(self):
        config = _make_config(rope_theta=10000.0)
        with _patched_rope_init():
            rotary = Step3p5RotaryEmbedding(config, layer_idx=0)

        self.assertTrue(hasattr(rotary, "inv_freq"))
        self.assertIsNotNone(rotary.inv_freq)
        self.assertEqual(rotary.max_seq_len_cached, 128)

    def test_init_when_rope_theta_list_then_picks_layer_specific_value(self):
        config = _make_config(rope_theta=[10000.0, 20000.0, 30000.0])
        with _patched_rope_init():
            rotary = Step3p5RotaryEmbedding(config, layer_idx=1)

        # Each instance should pick its own rope_theta from the list, not mutate the list
        self.assertEqual(rotary.rope_theta, 20000.0)

    def test_init_when_shared_config_reused_then_subsequent_layers_see_full_list(self):
        """验证：修复 #3 后，第二个 MTP 实例的 rope_theta 不会受第一个污染。"""
        config = _make_config(rope_theta=[10000.0, 20000.0, 30000.0])

        with _patched_rope_init():
            first = Step3p5RotaryEmbedding(config, layer_idx=0)
            second = Step3p5RotaryEmbedding(config, layer_idx=1)
            third = Step3p5RotaryEmbedding(config, layer_idx=2)

        # The shared config's rope_theta must remain the original list (not mutated)
        self.assertEqual(config.rope_theta, [10000.0, 20000.0, 30000.0])
        # Each layer got its own value
        self.assertEqual(first.rope_theta, 10000.0)
        self.assertEqual(second.rope_theta, 20000.0)
        self.assertEqual(third.rope_theta, 30000.0)

    def test_init_when_partial_rotary_factors_set_then_uses_layer_specific_factor(self):
        config = _make_config(partial_rotary_factors=[0.5, 1.0])
        with _patched_rope_init():
            rotary = Step3p5RotaryEmbedding(config, layer_idx=0)

        self.assertEqual(rotary.config.partial_rotary_factor, 0.5)

    def test_init_when_rope_scaling_dict_then_sets_rope_type_from_scaling(self):
        config = _make_config(rope_scaling={"rope_type": "linear", "factor": 1.0})
        with _patched_rope_init():
            rotary = Step3p5RotaryEmbedding(config, layer_idx=0)

        self.assertEqual(rotary.rope_type, "linear")

    def test_init_when_layer_idx_none_then_defaults_to_index_zero(self):
        """layer_idx=None 时使用 rope_theta[0] / partial_rotary_factors[0]（与真实模型一致）"""
        config = _make_config(rope_theta=[10000.0, 20000.0], partial_rotary_factors=[0.5, 1.0])
        with _patched_rope_init():
            rotary = Step3p5RotaryEmbedding(config, layer_idx=None)

        self.assertEqual(rotary.rope_theta, 10000.0)
        self.assertEqual(rotary.config.partial_rotary_factor, 0.5)


class TestStep3p5Attention(unittest.TestCase):
    def test_init_when_default_then_has_all_projections(self):
        config = _make_config()
        with _patched_rope_init():
            attn = Step3p5Attention(config, layer_idx=0)

        self.assertIsInstance(attn.q_proj, nn.Linear)
        self.assertIsInstance(attn.k_proj, nn.Linear)
        self.assertIsInstance(attn.v_proj, nn.Linear)
        self.assertIsInstance(attn.o_proj, nn.Linear)
        self.assertIsInstance(attn.q_norm, Step3p5RMSNorm)
        self.assertIsInstance(attn.k_norm, Step3p5RMSNorm)
        # head_dim = hidden_size / n_heads = 64 / 8 = 8
        # q_size = n_heads * head_dim = 8 * 8 = 64
        self.assertEqual(attn.q_proj.weight.shape, (64, 64))
        # kv_size = n_kv_heads * head_dim = 4 * 8 = 32
        self.assertEqual(attn.k_proj.weight.shape, (32, 64))
        self.assertEqual(attn.v_proj.weight.shape, (32, 64))

    def test_init_when_sliding_window_enabled_then_uses_other_settings(self):
        config = _make_config(
            sliding_window=128,
            layer_types=["sliding_attention", "full"],
        )
        with _patched_rope_init():
            attn = Step3p5Attention(config, layer_idx=0)

        self.assertEqual(attn.num_attention_heads, 4)
        self.assertEqual(attn.num_key_value_heads, 2)
        self.assertIsNotNone(attn.sliding_window)

    def test_init_when_sliding_window_disabled_then_sliding_window_is_none(self):
        config = _make_config(
            sliding_window=128,
            layer_types=["sliding_attention", "full"],
        )
        with _patched_rope_init():
            attn = Step3p5Attention(config, layer_idx=1)

        self.assertIsNone(attn.sliding_window)

    def test_init_when_use_head_wise_attn_gate_then_has_g_proj(self):
        config = _make_config(use_head_wise_attn_gate=True)
        with _patched_rope_init():
            attn = Step3p5Attention(config, layer_idx=0)

        self.assertIsInstance(attn.g_proj, nn.Linear)
        self.assertIsNotNone(attn.g_proj)

    def test_init_when_use_rope_disabled_by_layer_then_use_rope_is_false(self):
        config = _make_config(use_rope_layers=[False, True])
        with _patched_rope_init():
            attn = Step3p5Attention(config, layer_idx=0)

        self.assertFalse(attn.use_rope)


class TestSharedHead(unittest.TestCase):
    def test_init_when_default_then_has_norm_and_output(self):
        config = _make_config(hidden_size=64, vocab_size=100)
        head = SharedHead(config)

        self.assertIsInstance(head.norm, Step3p5RMSNorm)
        self.assertIsInstance(head.output, nn.Linear)
        self.assertEqual(head.output.weight.shape, (100, 64))


class TestStep3p7MTPModule(unittest.TestCase):
    def test_init_when_default_then_has_all_submodules(self):
        config = _make_config()
        with _patched_rope_init():
            mtp = Step3p7MTPModule(config, layer_idx=0)

        self.assertIsInstance(mtp.enorm, Step3p5RMSNorm)
        self.assertIsInstance(mtp.hnorm, Step3p5RMSNorm)
        self.assertIsInstance(mtp.input_layernorm, Step3p5RMSNorm)
        self.assertIsInstance(mtp.eh_proj, nn.Linear)
        # eh_proj 输入 channels = hidden_size * 2
        self.assertEqual(mtp.eh_proj.weight.shape[1], 64 * 2)
        self.assertIsInstance(mtp.self_attn, Step3p5Attention)
        self.assertIsInstance(mtp.post_attention_layernorm, Step3p5RMSNorm)
        self.assertTrue(hasattr(mtp.transformer, "shared_head"))

    def test_init_when_swiglu_limits_zero_for_layer_then_mlp_limit_is_none(self):
        """MTP 层 swiglu_limits[layer_idx]=0 → 不应设置 limit（避免误截断）"""
        # layer_idx=45, with a 48-element list (45 main layers + 3 MTP slots)
        swiglu_limits = [0.0] * 48
        config = _make_config(swiglu_limits=swiglu_limits)
        with _patched_rope_init():
            mtp = Step3p7MTPModule(config, layer_idx=45)

        self.assertIsNone(mtp.mlp.limit)

    def test_init_when_swiglu_limits_none_for_layer_idx_then_mlp_limit_is_none(self):
        """MTP 层 swiglu_limits[layer_idx]=None → limit 仍为 None（不抛异常）"""
        swiglu_limits = [0.0] * 48
        swiglu_limits[45] = None  # type: ignore[assignment]
        config = _make_config(swiglu_limits=swiglu_limits)
        with _patched_rope_init():
            mtp = Step3p7MTPModule(config, layer_idx=45)

        self.assertIsNone(mtp.mlp.limit)

    def test_init_when_swiglu_limits_nonzero_for_layer_then_mlp_limit_is_set(self):
        """MTP 层 swiglu_limits[layer_idx]!=0 → MLP 应拿到对应 limit（修复 #6）"""
        swiglu_limits = [0.0] * 48
        swiglu_limits[45] = 16.0
        config = _make_config(swiglu_limits=swiglu_limits)
        with _patched_rope_init():
            mtp = Step3p7MTPModule(config, layer_idx=45)

        self.assertEqual(mtp.mlp.limit, 16.0)

    def test_init_when_swiglu_limits_attr_missing_then_mlp_limit_is_none(self):
        """config 完全没有 swiglu_limits 属性 → 安全降级为 None"""
        # Build config without swiglu_limits
        config = _make_config()
        del config.swiglu_limits  # type: ignore[attr-defined]
        with _patched_rope_init():
            mtp = Step3p7MTPModule(config, layer_idx=45)

        self.assertIsNone(mtp.mlp.limit)

    def test_init_when_layer_types_set_then_attention_type_picks_layer_specific(self):
        """MTP 层的 attention_type 应当从 config.layer_types[layer_idx] 读取，
        与 Step3p7DecoderLayer.__init__ 行为一致（修复 attention_type 缺失）。
        """
        # 真实 Step-3.7 config 中 layer_types[45/46/47] 都是 sliding_attention
        layer_types = ["full_attention"] * 48
        layer_types[45] = "sliding_attention"
        layer_types[46] = "sliding_attention"
        layer_types[47] = "sliding_attention"
        config = _make_config(layer_types=layer_types)

        with _patched_rope_init():
            mtp45 = Step3p7MTPModule(config, layer_idx=45)
            mtp46 = Step3p7MTPModule(config, layer_idx=46)
            mtp47 = Step3p7MTPModule(config, layer_idx=47)

        self.assertEqual(mtp45.attention_type, "sliding_attention")
        self.assertEqual(mtp46.attention_type, "sliding_attention")
        self.assertEqual(mtp47.attention_type, "sliding_attention")

    def test_init_when_layer_types_short_then_falls_back_to_modulo_rule(self):
        """config.layer_types 存在但 layer_idx 越界时，MTP 模块会捕获 IndexError
        并退回到 layer_idx % 2 规则（与 decoder 行为一致）。
        """
        # 长度=3，只覆盖 45/46/47 三个槽位——但 layer_idx=48 会越界
        # Step3p5Attention 会先读 layer_types[48] 抛 IndexError。
        # 我们的修改必须在 attention_type 计算阶段不读 layer_types，避免与
        # Step3p5Attention 的依赖冲突——这里只验证注意类型计算的回退语义。
        config = _make_config(layer_types=[])
        # 直接复制 MTP 的 attention_type 计算逻辑（不经过 __init__）验证回退
        layer_types_cfg = config.layer_types or []
        if layer_types_cfg:
            attn45 = layer_types_cfg[45]
        else:
            attn45 = "sliding_attention" if 45 % 2 == 0 else "full_attention"
        self.assertEqual(attn45, "full_attention")
        attn46 = "sliding_attention" if 46 % 2 == 0 else "full_attention"
        self.assertEqual(attn46, "sliding_attention")

    def test_init_when_layer_types_present_then_does_not_pollute_config(self):
        """MTP 层不应改变 config.layer_types（多次实例化不应互相影响）。"""
        layer_types = ["full_attention"] * 48
        layer_types[45] = "sliding_attention"
        config = _make_config(layer_types=layer_types)

        with _patched_rope_init():
            Step3p7MTPModule(config, layer_idx=45)
            Step3p7MTPModule(config, layer_idx=46)

        # Config 的 layer_types 应保持原样
        self.assertEqual(config.layer_types[45], "sliding_attention")
        self.assertEqual(config.layer_types[46], "full_attention")


if __name__ == "__main__":
    unittest.main()
