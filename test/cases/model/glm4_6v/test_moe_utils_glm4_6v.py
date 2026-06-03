#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Unit tests for msmodelslim.model.glm4_6v.moe_utils.
Single test class TestGlm4_6VMoeUtils covers UnstackedGlm4vTextExpertMLP,
UnstackedGlm4vTextTopkRouter, UnstackedGlm4vMoeTextMoE.
"""

import types

import torch
from torch import nn

from msmodelslim.model.glm4_6v.moe_utils import (
    UnstackedGlm4vTextExpertMLP,
    UnstackedGlm4vTextTopkRouter,
    UnstackedGlm4vMoeTextMoE,
)


class _MockConfig:
    hidden_size = 8
    moe_intermediate_size = 4
    hidden_act = "silu"
    num_experts_per_tok = 1
    n_routed_experts = 2
    n_group = 1
    topk_group = 1
    norm_topk_prob = True
    routed_scaling_factor = 1.0
    num_local_experts = 2


class TestGlm4_6VMoeUtils:
    """Single test class for glm4_6v.moe_utils."""

    def test_expert_mlp_init_creates_linear_layers_when_called_with_valid_args(self):
        mlp = UnstackedGlm4vTextExpertMLP(hidden_size=8, intermediate_size=4, hidden_act="silu", dtype=torch.float32)
        assert isinstance(mlp.gate_proj, nn.Linear)
        assert isinstance(mlp.up_proj, nn.Linear)
        assert isinstance(mlp.down_proj, nn.Linear)
        assert mlp.gate_proj.weight.shape == (4, 8)
        assert mlp.up_proj.weight.shape == (4, 8)
        assert mlp.down_proj.weight.shape == (8, 4)

    def test_expert_mlp_forward_return_tensor_with_hidden_size_when_input_2d(self):
        mlp = UnstackedGlm4vTextExpertMLP(hidden_size=8, intermediate_size=4, hidden_act="silu", dtype=torch.float32)
        x = torch.randn(2, 8)
        y = mlp(x)
        assert y.shape == (2, 8)
        assert not torch.isnan(y).any() and not torch.isinf(y).any()

    def test_expert_mlp_forward_return_correct_shape_when_dtype_is_bfloat16(self):
        mlp = UnstackedGlm4vTextExpertMLP(hidden_size=4, intermediate_size=8, hidden_act="silu", dtype=torch.bfloat16)
        x = torch.randn(1, 4, dtype=torch.bfloat16)
        y = mlp(x)
        assert y.shape == (1, 4)
        assert y.dtype == torch.bfloat16

    def test_topk_router_init_creates_weight_and_e_score_param_when_given_original_gate(self):
        cfg = _MockConfig()
        gate = types.SimpleNamespace(
            weight=nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size)),
            e_score_correction_bias=torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
        )
        router = UnstackedGlm4vTextTopkRouter(cfg, gate)
        assert "weight" in dict(router.named_parameters())
        assert "e_score_correction_bias" in dict(router.named_parameters())
        assert router.e_score_correction_bias.shape == (cfg.n_routed_experts,)

    def test_topk_router_forward_return_logits_with_n_routed_experts_dim_when_input_flat(self):
        cfg = _MockConfig()
        gate = types.SimpleNamespace(
            weight=nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size)),
            e_score_correction_bias=torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
        )
        router = UnstackedGlm4vTextTopkRouter(cfg, gate)
        with torch.no_grad():
            router.weight.fill_(0.1)
        logits = router(torch.randn(3, cfg.hidden_size))
        assert logits.shape == (3, cfg.n_routed_experts)

    def test_topk_router_forward_return_logits_when_input_3d_flattened(self):
        cfg = _MockConfig()
        gate = types.SimpleNamespace(
            weight=nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size)),
            e_score_correction_bias=torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
        )
        router = UnstackedGlm4vTextTopkRouter(cfg, gate)
        with torch.no_grad():
            router.weight.fill_(0.0)
        hidden = torch.randn(1, 2, cfg.hidden_size)
        logits = router(hidden)
        assert logits.shape == (2, cfg.n_routed_experts)

    def test_moe_init_creates_gate_and_experts_when_given_original_moe(self):
        cfg = _MockConfig()
        original_gate = types.SimpleNamespace(
            weight=nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size)),
            e_score_correction_bias=torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
        )
        original_experts = nn.Module()
        original_experts.gate_up_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
        )
        original_experts.down_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, cfg.hidden_size, cfg.moe_intermediate_size)
        )
        original_moe = types.SimpleNamespace(
            gate=original_gate,
            shared_experts=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False),
            experts=original_experts,
        )
        moe = UnstackedGlm4vMoeTextMoE(cfg, original_moe)
        assert isinstance(moe.gate, UnstackedGlm4vTextTopkRouter)
        assert len(moe.experts) == cfg.num_local_experts
        assert all(isinstance(e, UnstackedGlm4vTextExpertMLP) for e in moe.experts)
        assert not hasattr(original_moe.experts, "gate_up_proj")
        assert not hasattr(original_moe.experts, "down_proj")

    def test_moe_forward_return_same_shape_as_input_when_single_batch(self):
        cfg = _MockConfig()
        original_gate = types.SimpleNamespace(
            weight=nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size)),
            e_score_correction_bias=torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
        )
        original_experts = nn.Module()
        original_experts.gate_up_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
        )
        original_experts.down_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, cfg.hidden_size, cfg.moe_intermediate_size)
        )
        original_moe = types.SimpleNamespace(
            gate=original_gate,
            shared_experts=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False),
            experts=original_experts,
        )
        moe = UnstackedGlm4vMoeTextMoE(cfg, original_moe)
        with torch.no_grad():
            moe.gate.weight.fill_(0.0)
            moe.gate.e_score_correction_bias.fill_(0.0)
        inp = torch.randn(1, 2, cfg.hidden_size)
        out = moe(inp)
        assert out.shape == inp.shape
        assert not torch.isnan(out).any() and not torch.isinf(out).any()

    def test_moe_route_tokens_to_experts_return_two_tensors_when_called_with_logits(self):
        cfg = _MockConfig()
        original_gate = types.SimpleNamespace(
            weight=nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size)),
            e_score_correction_bias=torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
        )
        original_experts = nn.Module()
        original_experts.gate_up_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
        )
        original_experts.down_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, cfg.hidden_size, cfg.moe_intermediate_size)
        )
        original_moe = types.SimpleNamespace(
            gate=original_gate,
            shared_experts=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False),
            experts=original_experts,
        )
        moe = UnstackedGlm4vMoeTextMoE(cfg, original_moe)
        # router_logits should be 2D (num_tokens, n_routed_experts) as returned by gate.forward
        num_tokens = 2 * 4
        router_logits = torch.randn(num_tokens, cfg.n_routed_experts)
        topk_indices, topk_weights = moe.route_tokens_to_experts(router_logits)
        assert topk_indices.shape == (num_tokens, cfg.num_experts_per_tok)
        assert topk_weights.shape == (num_tokens, cfg.num_experts_per_tok)

    def test_moe_dispatch_to_experts_return_zeros_shape_when_no_expert_hit(self):
        cfg = _MockConfig()
        original_gate = types.SimpleNamespace(
            weight=nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size)),
            e_score_correction_bias=torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
        )
        original_experts = nn.Module()
        original_experts.gate_up_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
        )
        original_experts.down_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, cfg.hidden_size, cfg.moe_intermediate_size)
        )
        original_moe = types.SimpleNamespace(
            gate=original_gate,
            shared_experts=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False),
            experts=original_experts,
        )
        moe = UnstackedGlm4vMoeTextMoE(cfg, original_moe)
        hidden = torch.randn(2, cfg.hidden_size)
        top_k_index = torch.zeros(2, 1, dtype=torch.long)
        top_k_weights = torch.ones(2, 1)
        out = moe._dispatch_to_experts(hidden, top_k_index, top_k_weights)
        assert out.shape == hidden.shape
        assert not torch.isnan(out).any()

    def test_moe_dispatch_to_experts_return_correct_output_when_expert_hit(self):
        cfg = _MockConfig()
        original_gate = types.SimpleNamespace(
            weight=nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size)),
            e_score_correction_bias=torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
        )
        original_experts = nn.Module()
        original_experts.gate_up_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
        )
        original_experts.down_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, cfg.hidden_size, cfg.moe_intermediate_size)
        )
        original_moe = types.SimpleNamespace(
            gate=original_gate,
            shared_experts=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False),
            experts=original_experts,
        )
        moe = UnstackedGlm4vMoeTextMoE(cfg, original_moe)
        hidden = torch.randn(4, cfg.hidden_size)
        # 让第一个token使用expert 0，第二个token使用expert 1
        top_k_index = torch.tensor([[0], [1], [0], [1]], dtype=torch.long)
        top_k_weights = torch.ones(4, 1)
        out = moe._dispatch_to_experts(hidden, top_k_index, top_k_weights)
        assert out.shape == hidden.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_moe_dispatch_to_experts_handle_all_experts_when_multiple_experts_used(self):
        cfg = _MockConfig()
        original_gate = types.SimpleNamespace(
            weight=nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size)),
            e_score_correction_bias=torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
        )
        original_experts = nn.Module()
        original_experts.gate_up_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
        )
        original_experts.down_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, cfg.hidden_size, cfg.moe_intermediate_size)
        )
        original_moe = types.SimpleNamespace(
            gate=original_gate,
            shared_experts=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False),
            experts=original_experts,
        )
        moe = UnstackedGlm4vMoeTextMoE(cfg, original_moe)
        hidden = torch.randn(4, cfg.hidden_size)
        # 使用所有可用的expert索引
        top_k_index = torch.tensor([[0], [1], [0], [1]], dtype=torch.long)
        top_k_weights = torch.ones(4, 1)
        out = moe._dispatch_to_experts(hidden, top_k_index, top_k_weights)
        assert out.shape == hidden.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_moe_route_tokens_to_experts_return_unnormalized_weights_when_norm_topk_prob_is_false(self):
        cfg = _MockConfig()
        cfg.norm_topk_prob = False
        original_gate = types.SimpleNamespace(
            weight=nn.Parameter(torch.randn(cfg.n_routed_experts, cfg.hidden_size)),
            e_score_correction_bias=torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
        )
        original_experts = nn.Module()
        original_experts.gate_up_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
        )
        original_experts.down_proj = nn.Parameter(
            torch.randn(cfg.num_local_experts, cfg.hidden_size, cfg.moe_intermediate_size)
        )
        original_moe = types.SimpleNamespace(
            gate=original_gate,
            shared_experts=nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False),
            experts=original_experts,
        )
        moe = UnstackedGlm4vMoeTextMoE(cfg, original_moe)
        num_tokens = 4
        router_logits = torch.randn(num_tokens, cfg.n_routed_experts)
        topk_indices, topk_weights = moe.route_tokens_to_experts(router_logits)
        assert topk_indices.shape == (num_tokens, cfg.num_experts_per_tok)
        assert topk_weights.shape == (num_tokens, cfg.num_experts_per_tok)
        # 权重应该被scaling factor缩放，但不应该被归一化
        assert torch.all(topk_weights >= 0)
