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

from types import SimpleNamespace
from unittest.mock import patch

import torch

from msmodelslim.model.kimi_k2_5 import quarot as target
from msmodelslim.model.kimi_k2_5.quarot import get_ln_fuse_map, get_rotate_map


def _make_config(
    num_hidden_layers=3,
    first_k_dense_replace=1,
    n_routed_experts=2,
    n_shared_experts=1,
    hidden_size=16,
    q_lora_rank=8,
    kv_lora_rank=8,
    qk_nope_head_dim=4,
    v_head_dim=4,
    qk_rope_head_dim=4,
):
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            first_k_dense_replace=first_k_dense_replace,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
        ),
    )


def test_get_ln_fuse_map_returns_expected_keys_when_explicit_num_hidden_layers():
    config = _make_config(num_hidden_layers=3, first_k_dense_replace=1)
    result = get_ln_fuse_map(config, num_hidden_layers=2)
    for i in range(2):
        assert f"language_model.model.layers.{i}.input_layernorm" in result
        assert f"language_model.model.layers.{i}.self_attn.q_a_layernorm" in result
        assert f"language_model.model.layers.{i}.self_attn.kv_a_layernorm" in result
        assert f"language_model.model.layers.{i}.post_attention_layernorm" in result
    assert "language_model.model.norm" in result
    assert result["language_model.model.norm"] == ["language_model.lm_head"]


def test_get_ln_fuse_map_uses_config_num_hidden_layers_when_none():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1)
    result = get_ln_fuse_map(config, num_hidden_layers=None)
    assert f"language_model.model.layers.{config.text_config.num_hidden_layers - 1}.input_layernorm" in result


def test_get_ln_fuse_map_targets_mlp_gate_up_when_dense_layer():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=2)
    result = get_ln_fuse_map(config, num_hidden_layers=1)
    assert result["language_model.model.layers.0.post_attention_layernorm"] == [
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
    ]


def test_get_ln_fuse_map_targets_moe_experts_shared_and_gate_when_moe_layer():
    config = _make_config(num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=2, n_shared_experts=1)
    result = get_ln_fuse_map(config, num_hidden_layers=2)
    targets = result["language_model.model.layers.1.post_attention_layernorm"]
    assert "language_model.model.layers.1.mlp.experts.0.gate_proj" in targets
    assert "language_model.model.layers.1.mlp.experts.0.up_proj" in targets
    assert "language_model.model.layers.1.mlp.experts.1.gate_proj" in targets
    assert "language_model.model.layers.1.mlp.experts.1.up_proj" in targets
    assert "language_model.model.layers.1.mlp.shared_experts.gate_proj" in targets
    assert "language_model.model.layers.1.mlp.shared_experts.up_proj" in targets
    assert "language_model.model.layers.1.mlp.gate" in targets


def test_get_ln_fuse_map_contains_q_a_and_kv_a_for_input_layernorm():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1)
    result = get_ln_fuse_map(config, num_hidden_layers=2)
    assert result["language_model.model.layers.0.input_layernorm"] == [
        "language_model.model.layers.0.self_attn.q_a_proj",
        "language_model.model.layers.0.self_attn.kv_a_proj_with_mqa",
    ]
    assert result["language_model.model.layers.0.self_attn.q_a_layernorm"] == [
        "language_model.model.layers.0.self_attn.q_b_proj"
    ]
    assert result["language_model.model.layers.0.self_attn.kv_a_layernorm"] == [
        "language_model.model.layers.0.self_attn.kv_b_proj"
    ]


def test_get_ln_fuse_map_moe_post_attention_layernorm_count_matches_formula():
    n_routed = 3
    config = _make_config(num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=n_routed, n_shared_experts=1)
    result = get_ln_fuse_map(config, num_hidden_layers=2)
    targets = result["language_model.model.layers.1.post_attention_layernorm"]
    expected_count = n_routed * 2 + 2 + 1
    assert len(targets) == expected_count


def test_get_rotate_map_returns_three_components_when_explicit_num_hidden_layers():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1)
    with patch.object(
        target.QuaRotInterface, "get_rotate_command", side_effect=lambda size, mode, block_size: torch.eye(size)
    ):
        pre_run, rot_pairs, rotate_matrix = get_rotate_map(config, block_size=4, num_hidden_layers=1)

    assert pre_run is not None
    assert "rot" in rot_pairs
    assert "rot_b_proj" in rot_pairs
    assert "rot_uv" in rot_pairs
    assert "rot_kv_b_proj" in rot_pairs
    assert "rot" in rotate_matrix
    assert "rot_b_proj" in rotate_matrix
    assert "rot_uv" in rotate_matrix
    assert "rot_kv_b_proj" in rotate_matrix


def test_get_rotate_map_uses_config_num_hidden_layers_when_none():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1)
    with patch.object(
        target.QuaRotInterface, "get_rotate_command", side_effect=lambda size, mode, block_size: torch.eye(size)
    ):
        pre_run, rot_pairs, _ = get_rotate_map(config, block_size=4, num_hidden_layers=None)

    assert pre_run is not None
    last_layer = config.text_config.num_hidden_layers - 1
    assert rot_pairs["rot"].right_rot.get(f"language_model.model.layers.{last_layer}.self_attn.q_a_proj") is not None


def test_get_rotate_map_targets_mlp_gate_up_down_when_dense_layer():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=2, n_routed_experts=2, n_shared_experts=1)
    with patch.object(
        target.QuaRotInterface, "get_rotate_command", side_effect=lambda size, mode, block_size: torch.eye(size)
    ):
        _, rot_pairs, _ = get_rotate_map(config, block_size=4, num_hidden_layers=1)

    rot = rot_pairs["rot"]
    assert rot.right_rot.get("language_model.model.layers.0.mlp.gate_proj") is not None
    assert rot.right_rot.get("language_model.model.layers.0.mlp.up_proj") is not None
    assert rot.left_rot.get("language_model.model.layers.0.mlp.down_proj") is not None


def test_get_rotate_map_targets_moe_experts_shared_and_gate_when_moe_layer():
    config = _make_config(num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=2, n_shared_experts=1)
    with patch.object(
        target.QuaRotInterface, "get_rotate_command", side_effect=lambda size, mode, block_size: torch.eye(size)
    ):
        _, rot_pairs, _ = get_rotate_map(config, block_size=4, num_hidden_layers=2)

    rot = rot_pairs["rot"]
    assert rot.right_rot.get("language_model.model.layers.1.mlp.experts.0.gate_proj") is not None
    assert rot.right_rot.get("language_model.model.layers.1.mlp.experts.0.up_proj") is not None
    assert rot.left_rot.get("language_model.model.layers.1.mlp.experts.0.down_proj") is not None
    assert rot.right_rot.get("language_model.model.layers.1.mlp.shared_experts.gate_proj") is not None
    assert rot.right_rot.get("language_model.model.layers.1.mlp.shared_experts.up_proj") is not None
    assert rot.left_rot.get("language_model.model.layers.1.mlp.shared_experts.down_proj") is not None
    assert rot.right_rot.get("language_model.model.layers.1.mlp.gate") is not None


def test_get_rotate_map_pre_run_targets_embed_tokens_and_mm_projector():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1)
    with patch.object(
        target.QuaRotInterface, "get_rotate_command", side_effect=lambda size, mode, block_size: torch.eye(size)
    ):
        pre_run, _, _ = get_rotate_map(config, block_size=4, num_hidden_layers=1)

    assert "language_model.model.embed_tokens" in pre_run.right_rot
    assert "mm_projector.proj.2" in pre_run.left_rot


def test_get_rotate_map_rot_pair_targets_lm_head():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1)
    with patch.object(
        target.QuaRotInterface, "get_rotate_command", side_effect=lambda size, mode, block_size: torch.eye(size)
    ):
        _, rot_pairs, _ = get_rotate_map(config, block_size=4, num_hidden_layers=1)

    assert "language_model.lm_head" in rot_pairs["rot"].right_rot


def test_get_rotate_map_rot_b_proj_pair_contains_q_a_and_q_b_proj():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1)
    with patch.object(
        target.QuaRotInterface, "get_rotate_command", side_effect=lambda size, mode, block_size: torch.eye(size)
    ):
        _, rot_pairs, _ = get_rotate_map(config, block_size=4, num_hidden_layers=1)

    rot_b = rot_pairs["rot_b_proj"]
    assert "language_model.model.layers.0.self_attn.q_a_proj" in rot_b.left_rot
    assert "language_model.model.layers.0.self_attn.q_b_proj" in rot_b.right_rot


def test_get_rotate_map_rot_uv_pair_kv_b_proj_split_and_o_proj_present():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1, qk_nope_head_dim=4, v_head_dim=4)
    with patch.object(
        target.QuaRotInterface, "get_rotate_command", side_effect=lambda size, mode, block_size: torch.eye(size)
    ):
        _, rot_pairs, _ = get_rotate_map(config, block_size=4, num_hidden_layers=1)

    rot_uv = rot_pairs["rot_uv"]
    left_value = rot_uv.left_rot.get("language_model.model.layers.0.self_attn.kv_b_proj")
    assert left_value is not None
    assert isinstance(left_value, list)
    assert len(left_value) == 2
    assert "language_model.model.layers.0.self_attn.o_proj" in rot_uv.right_rot


def test_get_rotate_map_rot_kv_b_proj_pair_kv_a_and_kv_b_split():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1, qk_rope_head_dim=4)
    with patch.object(
        target.QuaRotInterface, "get_rotate_command", side_effect=lambda size, mode, block_size: torch.eye(size)
    ):
        _, rot_pairs, _ = get_rotate_map(config, block_size=4, num_hidden_layers=1)

    rot_kv = rot_pairs["rot_kv_b_proj"]
    left_value = rot_kv.left_rot.get("language_model.model.layers.0.self_attn.kv_a_proj_with_mqa")
    assert left_value is not None
    assert isinstance(left_value, list)
    assert len(left_value) == 2
    assert "language_model.model.layers.0.self_attn.kv_b_proj" in rot_kv.right_rot


def test_get_rotate_map_skips_expert_rotation_when_zero_experts():
    config = _make_config(num_hidden_layers=2, first_k_dense_replace=1, n_routed_experts=0, n_shared_experts=1)
    with patch.object(
        target.QuaRotInterface, "get_rotate_command", side_effect=lambda size, mode, block_size: torch.eye(size)
    ):
        _, rot_pairs, _ = get_rotate_map(config, block_size=4, num_hidden_layers=2)

    rot = rot_pairs["rot"]
    assert "language_model.model.layers.1.mlp.experts.0.gate_proj" not in rot.right_rot
