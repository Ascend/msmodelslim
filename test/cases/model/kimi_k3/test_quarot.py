# -*- coding: UTF-8 -*-
"""Unit tests for msmodelslim.model.kimi_k3.quarot."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from msmodelslim.model.kimi_k3 import quarot as target
from msmodelslim.model.kimi_k3.quarot import get_ln_fuse_map, get_rotate_map


def _make_config(
    num_hidden_layers=3,
    first_k_dense_replace=1,
    hidden_size=16,
    q_lora_rank=8,
    kv_lora_rank=8,
    qk_rope_head_dim=4,
    qk_nope_head_dim=4,
    v_head_dim=4,
    num_experts=4,
    linear_attn_config=None,
    attn_res_block_size=None,
    routed_expert_hidden_size=None,
    latent_moe_use_norm=False,
    mla_use_output_gate=False,
    is_kda_layer_fn=None,
):
    text = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        first_k_dense_replace=first_k_dense_replace,
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        num_experts=num_experts,
        linear_attn_config=linear_attn_config,
        attn_res_block_size=attn_res_block_size,
        routed_expert_hidden_size=routed_expert_hidden_size,
        latent_moe_use_norm=latent_moe_use_norm,
        mla_use_output_gate=mla_use_output_gate,
    )
    if is_kda_layer_fn is not None:
        text.is_kda_layer = is_kda_layer_fn
    return SimpleNamespace(text_config=text)


def _eye_rotate(size, mode, block_size):
    return torch.eye(size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_is_kda_layer_given_method_when_called_then_use_method():
    cfg = _make_config(is_kda_layer_fn=lambda idx: idx == 1).text_config
    assert target.is_kda_layer(cfg, 1) is True
    assert target.is_kda_layer(cfg, 0) is False


def test_is_kda_layer_given_list_config_when_called_then_one_based():
    # (layer_idx + 1) in kda_layers
    cfg = _make_config(linear_attn_config={"kda_layers": [2]}).text_config
    assert target.is_kda_layer(cfg, 1) is True
    assert target.is_kda_layer(cfg, 0) is False


def test_is_kda_layer_given_none_config_when_called_then_false():
    cfg = _make_config(linear_attn_config=None).text_config
    assert target.is_kda_layer(cfg, 0) is False


def test_is_dense_mlp_given_first_k_when_called_then_expected():
    cfg = _make_config(first_k_dense_replace=2).text_config
    assert target.is_dense_mlp(cfg, 0) is True
    assert target.is_dense_mlp(cfg, 1) is True
    assert target.is_dense_mlp(cfg, 2) is False


def test_use_attn_residuals_and_kda_full_rank_and_latent_flags():
    cfg_off = _make_config().text_config
    assert target.use_attn_residuals(cfg_off) is False
    assert target.kda_use_full_rank_gate(cfg_off) is False
    assert target.has_latent_moe(cfg_off) is False
    assert target.latent_moe_use_norm(cfg_off) is False

    cfg_on = _make_config(
        attn_res_block_size=4,
        linear_attn_config={"use_full_rank_gate": True},
        routed_expert_hidden_size=32,
        latent_moe_use_norm=True,
    ).text_config
    assert target.use_attn_residuals(cfg_on) is True
    assert target.kda_use_full_rank_gate(cfg_on) is True
    assert target.has_latent_moe(cfg_on) is True
    assert target.latent_moe_use_norm(cfg_on) is True


def test_layer_attn_moe_prefix_helpers():
    assert target.layer_prefix(3) == "language_model.model.layers.3"
    assert target.attn_prefix(3) == "language_model.model.layers.3.self_attn"
    assert target.moe_prefix(3) == "language_model.model.layers.3.block_sparse_moe"


def test_input_layernorm_targets_given_mla_when_called_then_q_kv_and_optional_gate():
    cfg = _make_config(mla_use_output_gate=False).text_config
    assert target.input_layernorm_targets(cfg, 0) == [
        "language_model.model.layers.0.self_attn.q_a_proj",
        "language_model.model.layers.0.self_attn.kv_a_proj_with_mqa",
    ]
    cfg_g = _make_config(mla_use_output_gate=True).text_config
    assert "language_model.model.layers.0.self_attn.g_proj" in target.input_layernorm_targets(cfg_g, 0)


def test_input_layernorm_targets_given_kda_when_called_then_kda_projs():
    cfg = _make_config(linear_attn_config={"kda_layers": [1], "use_full_rank_gate": False}).text_config
    targets = target.input_layernorm_targets(cfg, 0)
    assert "language_model.model.layers.0.self_attn.q_proj" in targets
    assert "language_model.model.layers.0.self_attn.g_a_proj" in targets
    assert "language_model.model.layers.0.self_attn.g_proj" not in targets

    cfg_fr = _make_config(linear_attn_config={"kda_layers": [1], "use_full_rank_gate": True}).text_config
    targets_fr = target.input_layernorm_targets(cfg_fr, 0)
    assert "language_model.model.layers.0.self_attn.g_proj" in targets_fr


def test_q_kv_a_layernorm_targets_given_kda_when_called_then_none():
    cfg = _make_config(linear_attn_config={"kda_layers": [1]}).text_config
    assert target.q_a_layernorm_targets(cfg, 0) is None
    assert target.kv_a_layernorm_targets(cfg, 0) is None
    cfg_mla = _make_config().text_config
    assert target.q_a_layernorm_targets(cfg_mla, 0) == ["language_model.model.layers.0.self_attn.q_b_proj"]
    assert target.kv_a_layernorm_targets(cfg_mla, 0) == ["language_model.model.layers.0.self_attn.kv_b_proj"]


def test_post_attention_layernorm_targets_dense_and_moe():
    dense = _make_config(first_k_dense_replace=2).text_config
    assert target.post_attention_layernorm_targets(dense, 0) == [
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
    ]
    moe = _make_config(first_k_dense_replace=1).text_config
    targets = target.post_attention_layernorm_targets(moe, 1)
    assert targets == [
        "language_model.model.layers.1.block_sparse_moe.gate",
        "language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj",
        "language_model.model.layers.1.block_sparse_moe.shared_experts.up_proj",
        "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj",
    ]


def test_routed_expert_norm_targets_flags():
    dense = _make_config(first_k_dense_replace=2).text_config
    assert target.routed_expert_norm_targets(dense, 0) is None

    moe_off = _make_config(first_k_dense_replace=1, latent_moe_use_norm=False).text_config
    assert target.routed_expert_norm_targets(moe_off, 1) is None

    moe_flag = _make_config(first_k_dense_replace=1, latent_moe_use_norm=True).text_config
    assert target.routed_expert_norm_targets(moe_flag, 1) == [
        "language_model.model.layers.1.block_sparse_moe.routed_expert_up_proj"
    ]

    # require_latent_hidden=True needs both flag and routed_expert_hidden_size
    moe_latent_off = _make_config(
        first_k_dense_replace=1, latent_moe_use_norm=True, routed_expert_hidden_size=None
    ).text_config
    assert target.routed_expert_norm_targets(moe_latent_off, 1, require_latent_hidden=True) is None

    moe_latent_on = _make_config(
        first_k_dense_replace=1, latent_moe_use_norm=True, routed_expert_hidden_size=32
    ).text_config
    assert target.routed_expert_norm_targets(moe_latent_on, 1, require_latent_hidden=True) is not None


def test_attn_res_norm_pairs_and_hidden_rot():
    cfg_off = _make_config().text_config
    assert not target.attn_res_norm_pairs(cfg_off, 0)

    cfg = _make_config(
        first_k_dense_replace=1,
        attn_res_block_size=4,
        mla_use_output_gate=False,
    ).text_config
    pairs = target.attn_res_norm_pairs(cfg, 0)
    assert pairs == [
        (
            "language_model.model.layers.0.self_attention_res_norm",
            "language_model.model.layers.0.self_attention_res_proj",
        ),
        (
            "language_model.model.layers.0.mlp_res_norm",
            "language_model.model.layers.0.mlp_res_proj",
        ),
    ]

    right = target.hidden_rot_right(cfg, 0)
    assert "language_model.model.layers.0.mlp.gate_proj" in right
    assert "language_model.model.layers.0.self_attention_res_proj" in right
    left = target.hidden_rot_left(cfg, 0)
    assert "language_model.model.layers.0.self_attn.o_proj" in left
    assert "language_model.model.layers.0.mlp.down_proj" in left

    moe_cfg = _make_config(first_k_dense_replace=1, attn_res_block_size=4).text_config
    right_moe = target.hidden_rot_right(moe_cfg, 1)
    assert "language_model.model.layers.1.block_sparse_moe.gate" in right_moe
    left_moe = target.hidden_rot_left(moe_cfg, 1)
    assert "language_model.model.layers.1.block_sparse_moe.shared_experts.down_proj" in left_moe
    assert "language_model.model.layers.1.block_sparse_moe.routed_expert_up_proj" in left_moe


# ---------------------------------------------------------------------------
# get_ln_fuse_map — returns (pre_run_fused_ln, ln_linear_map)
# ---------------------------------------------------------------------------


def test_get_ln_fuse_map_returns_two_values_with_expected_keys():
    config = _make_config(num_hidden_layers=3, first_k_dense_replace=1)
    pre_run, ln_map = get_ln_fuse_map(config, num_hidden_layers=2)
    assert isinstance(pre_run, dict) and isinstance(ln_map, dict)
    for i in range(2):
        assert f"language_model.model.layers.{i}.input_layernorm" in ln_map
        assert f"language_model.model.layers.{i}.post_attention_layernorm" in ln_map
    assert pre_run["language_model.model.norm"] == ["language_model.lm_head"]
    # MLA layers get q/kv a layernorm
    assert "language_model.model.layers.0.self_attn.q_a_layernorm" in ln_map
    assert "language_model.model.layers.0.self_attn.kv_a_layernorm" in ln_map


def test_get_ln_fuse_map_uses_config_num_hidden_layers_when_none():
    config = _make_config(num_hidden_layers=2)
    _, ln_map = get_ln_fuse_map(config, num_hidden_layers=None)
    assert "language_model.model.layers.1.input_layernorm" in ln_map


def test_get_ln_fuse_map_dense_and_moe_and_kda_and_residuals():
    config = _make_config(
        num_hidden_layers=3,
        first_k_dense_replace=1,
        linear_attn_config={"kda_layers": [1]},  # layer 0 is KDA
        attn_res_block_size=4,
        latent_moe_use_norm=True,
        routed_expert_hidden_size=32,
        mla_use_output_gate=True,
    )
    pre_run, ln_map = get_ln_fuse_map(config, num_hidden_layers=2)

    # Dense layer 0 is KDA → no q/kv a layernorm keys
    assert "language_model.model.layers.0.self_attn.q_a_layernorm" not in ln_map
    assert "language_model.model.layers.0.self_attn.kv_a_layernorm" not in ln_map
    assert "language_model.model.layers.0.self_attn.q_proj" in ln_map["language_model.model.layers.0.input_layernorm"]

    # Dense post-attn
    assert ln_map["language_model.model.layers.0.post_attention_layernorm"] == [
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
    ]

    # MoE layer 1 uses block_sparse_moe
    moe_targets = ln_map["language_model.model.layers.1.post_attention_layernorm"]
    assert "language_model.model.layers.1.block_sparse_moe.gate" in moe_targets
    assert "language_model.model.layers.1.block_sparse_moe.routed_expert_norm" in ln_map
    assert ln_map["language_model.model.layers.1.block_sparse_moe.routed_expert_norm"] == [
        "language_model.model.layers.1.block_sparse_moe.routed_expert_up_proj"
    ]

    # Residual pairs in map + pre_run
    assert "language_model.model.layers.0.self_attention_res_norm" in ln_map
    assert "language_model.model.output_attn_res_norm" in pre_run
    assert pre_run["language_model.model.output_attn_res_norm"] == ["language_model.model.output_attn_res_proj"]


# ---------------------------------------------------------------------------
# get_rotate_map — returns (pre_run_pairs, list[RotatePair])
# ---------------------------------------------------------------------------


def test_get_rotate_map_returns_list_of_pairs_when_all_enabled():
    config = _make_config(
        num_hidden_layers=2,
        first_k_dense_replace=1,
        routed_expert_hidden_size=8,
        attn_res_block_size=4,
    )
    with patch.object(target.QuaRotInterface, "get_rotate_command", side_effect=_eye_rotate):
        with patch.object(target, "_get_expert_range", return_value=(0, 2)):
            pre_run, rot_list = get_rotate_map(config, block_size=4, num_hidden_layers=2)

    assert isinstance(pre_run, list) and len(pre_run) == 1
    assert isinstance(rot_list, list) and len(rot_list) == 4  # rot, b, kv, latent
    assert "language_model.model.embed_tokens" in pre_run[0].right_rot
    assert "mm_projector.rot_proj" in pre_run[0].left_rot
    assert "language_model.lm_head" in pre_run[0].right_rot
    assert "language_model.model.output_attn_res_proj" in pre_run[0].right_rot


def test_get_rotate_map_uses_config_layers_when_none():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1)
    with patch.object(target.QuaRotInterface, "get_rotate_command", side_effect=_eye_rotate):
        pre_run, rot_list = get_rotate_map(config, block_size=4, num_hidden_layers=None)
    assert pre_run and rot_list
    rot = rot_list[0]
    assert rot.right_rot.get("language_model.model.layers.1.self_attn.q_a_proj") is not None


def test_get_rotate_map_enable_flags_off_when_called_then_empty_or_partial():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1, routed_expert_hidden_size=8)
    with patch.object(target.QuaRotInterface, "get_rotate_command", side_effect=_eye_rotate):
        with patch.object(target, "_get_expert_range", return_value=(0, 1)):
            pre_run, rot_list = get_rotate_map(
                config,
                block_size=4,
                num_hidden_layers=2,
                enable_rot=False,
                enable_rot_b_proj=False,
                enable_rot_kv_b_proj=False,
                enable_rot_latent=False,
            )
    assert not pre_run
    assert not rot_list


def test_get_rotate_map_kda_skips_b_and_kv_proj():
    config = _make_config(
        num_hidden_layers=2,
        first_k_dense_replace=1,
        linear_attn_config={"kda_layers": [1]},  # layer 0 KDA
    )
    with patch.object(target.QuaRotInterface, "get_rotate_command", side_effect=_eye_rotate):
        _, rot_list = get_rotate_map(config, block_size=4, num_hidden_layers=2)

    # Find rot_b_proj / rot_kv by inspecting left_rot keys
    rot_b = next(p for p in rot_list if any("q_b_proj" in k for k in p.right_rot))
    rot_kv = next(p for p in rot_list if any("kv_b_proj" in k for k in p.right_rot))
    assert "language_model.model.layers.0.self_attn.q_a_proj" not in rot_b.left_rot
    assert "language_model.model.layers.1.self_attn.q_a_proj" in rot_b.left_rot
    assert "language_model.model.layers.0.self_attn.kv_b_proj" not in rot_kv.right_rot
    assert "language_model.model.layers.1.self_attn.kv_b_proj" in rot_kv.right_rot


def test_get_rotate_map_rot_kv_left_is_list_with_eye():
    config = _make_config(num_hidden_layers=1, first_k_dense_replace=1, qk_rope_head_dim=4)
    with patch.object(target.QuaRotInterface, "get_rotate_command", side_effect=_eye_rotate):
        _, rot_list = get_rotate_map(config, block_size=4, num_hidden_layers=1)
    rot_kv = next(p for p in rot_list if any("kv_b_proj" in k for k in p.right_rot))
    left_val = rot_kv.left_rot["language_model.model.layers.0.self_attn.kv_a_proj_with_mqa"]
    assert isinstance(left_val, list) and len(left_val) == 2
    assert left_val[1].shape == (4, 4)


def test_get_rotate_map_latent_moe_ep_expert_range():
    config = _make_config(
        num_hidden_layers=2,
        first_k_dense_replace=1,
        routed_expert_hidden_size=8,
        num_experts=4,
    )
    with patch.object(target.QuaRotInterface, "get_rotate_command", side_effect=_eye_rotate):
        with patch.object(target, "_get_expert_range", return_value=(1, 3)) as mock_range:
            _, rot_list = get_rotate_map(
                config,
                block_size=4,
                num_hidden_layers=2,
                enable_rot=False,
                enable_rot_b_proj=False,
                enable_rot_kv_b_proj=False,
            )
    assert mock_range.called
    assert len(rot_list) == 1
    latent = rot_list[0]
    moe = "language_model.model.layers.1.block_sparse_moe"
    assert f"{moe}.routed_expert_down_proj" in latent.left_rot
    assert f"{moe}.routed_expert_up_proj" in latent.right_rot
    assert f"{moe}.experts.1.w1" in latent.right_rot
    assert f"{moe}.experts.2.w2" in latent.left_rot
    assert f"{moe}.experts.0.w1" not in latent.right_rot
    # dense layer skipped
    assert "language_model.model.layers.0.block_sparse_moe.experts.1.w1" not in latent.right_rot


def test_get_rotate_map_dense_and_moe_hidden_rot_targets():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1)
    with patch.object(target.QuaRotInterface, "get_rotate_command", side_effect=_eye_rotate):
        _, rot_list = get_rotate_map(
            config,
            block_size=4,
            num_hidden_layers=2,
            enable_rot_b_proj=False,
            enable_rot_kv_b_proj=False,
            enable_rot_latent=False,
        )
    rot = rot_list[0]
    assert rot.right_rot.get("language_model.model.layers.0.mlp.gate_proj") is not None
    assert rot.left_rot.get("language_model.model.layers.0.mlp.down_proj") is not None
    assert rot.right_rot.get("language_model.model.layers.1.block_sparse_moe.gate") is not None
    assert rot.left_rot.get("language_model.model.layers.1.block_sparse_moe.shared_experts.down_proj") is not None


def test_get_rotate_map_no_latent_when_h_lat_none():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1, routed_expert_hidden_size=None)
    with patch.object(target.QuaRotInterface, "get_rotate_command", side_effect=_eye_rotate):
        _, rot_list = get_rotate_map(
            config,
            block_size=4,
            num_hidden_layers=2,
            enable_rot=False,
            enable_rot_b_proj=False,
            enable_rot_kv_b_proj=False,
            enable_rot_latent=True,
        )
    assert not rot_list
