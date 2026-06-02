#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
from unittest.mock import patch, MagicMock

import torch

from msmodelslim.model.glm_5.quarot import get_ln_fuse_map, get_rotate_map


class MockConfig:
    def __init__(
        self,
        num_hidden_layers=4,
        first_k_dense_replace=2,
        n_routed_experts=8,
        hidden_size=256,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.first_k_dense_replace = first_k_dense_replace
        self.n_routed_experts = n_routed_experts
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim


class TestGetLnFuseMap(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig(num_hidden_layers=4, first_k_dense_replace=2, n_routed_experts=8)

    def test_get_ln_fuse_map_shouldContainExpectedKeys_when_defaultNumLayers(self):
        result = get_ln_fuse_map(self.config)

        self.assertIn("model.layers.0.input_layernorm", result)
        self.assertIn("model.layers.0.self_attn.q_a_layernorm", result)
        self.assertIn("model.layers.0.self_attn.kv_a_layernorm", result)
        self.assertIn("model.layers.0.post_attention_layernorm", result)
        self.assertIn("model.norm", result)

    def test_get_ln_fuse_map_shouldHaveCorrectTargets_when_denseLayer(self):
        result = get_ln_fuse_map(self.config)

        dense_layer_key = "model.layers.0.post_attention_layernorm"
        targets = result[dense_layer_key]
        self.assertIn("model.layers.0.mlp.gate_proj", targets)
        self.assertIn("model.layers.0.mlp.up_proj", targets)

        for target in targets:
            self.assertNotIn("experts", target)

    def test_get_ln_fuse_map_shouldIncludeMtpKeys_when_numHiddenLayersProvided(self):
        num_layers = 3
        result = get_ln_fuse_map(self.config, num_hidden_layers=num_layers)

        mtp_key1 = (f"model.layers.{num_layers - 1}.enorm", f"model.layers.{num_layers - 1}.hnorm")
        self.assertIn(mtp_key1, result)
        self.assertIn(result[mtp_key1], [["model.layers.2.eh_proj"], ["model.layers.2.eh_proj"]])

        self.assertIn(f"model.layers.{num_layers - 1}.shared_head.norm", result)

    def test_get_ln_fuse_map_shouldIncludeLmHead_when_modelNormKey(self):
        result = get_ln_fuse_map(self.config)

        self.assertIn("model.norm", result)
        self.assertEqual(result["model.norm"], ["lm_head"])

    def test_get_ln_fuse_map_shouldRespect_when_firstKDenseReplaceIsZero(self):
        config = MockConfig(num_hidden_layers=3, first_k_dense_replace=0)
        result = get_ln_fuse_map(config)

        for layer_idx in range(3):
            key = f"model.layers.{layer_idx}.post_attention_layernorm"
            targets = result[key]
            self.assertIn(f"model.layers.{layer_idx}.mlp.gate", targets)

    def test_get_ln_fuse_map_shouldRespect_when_numHiddenLayersIsOne(self):
        config = MockConfig(num_hidden_layers=1, first_k_dense_replace=0)
        result = get_ln_fuse_map(config)

        self.assertEqual(len([k for k in result if "model.layers." in str(k)]), 10)


class TestGetRotateMap(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig(
            num_hidden_layers=3,
            first_k_dense_replace=1,
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=32,
            hidden_size=128,
        )
        self.block_size = 64

    @patch("msmodelslim.model.glm_5.quarot.QuaRotInterface.get_rotate_command")
    @patch("msmodelslim.model.glm_5.quarot.QuaRotInterface.RotatePair")
    def test_get_rotate_map_shouldReturnThreeComponents_when_called(self, mock_rot_pair, mock_get_rot):
        mock_get_rot.return_value = torch.eye(self.config.hidden_size)
        mock_rot_pair.side_effect = lambda left_rot=None, right_rot=None: MagicMock(
            left_rot=left_rot or {}, right_rot=right_rot or {}
        )

        pre_run, rot_pairs, rotate_matrix = get_rotate_map(self.config, self.block_size)

        self.assertIsNotNone(pre_run)
        self.assertIsInstance(rot_pairs, dict)
        self.assertIsInstance(rotate_matrix, dict)
        self.assertIn("rot", rotate_matrix)
        self.assertIn("rot_b_proj", rotate_matrix)
        self.assertIn("rot_uv", rotate_matrix)
        self.assertIn("rot_kv_b_proj", rotate_matrix)

    @patch("msmodelslim.model.glm_5.quarot.QuaRotInterface.get_rotate_command")
    @patch("msmodelslim.model.glm_5.quarot.QuaRotInterface.RotatePair")
    def test_get_rotate_map_preRun_shouldContainEmbedTokens_when_called(self, mock_rot_pair, mock_get_rot):
        mock_get_rot.return_value = torch.eye(self.config.hidden_size)
        mock_rot_pair.side_effect = lambda left_rot=None, right_rot=None: MagicMock(
            left_rot=left_rot or {}, right_rot=right_rot or {}
        )

        pre_run, rot_pairs, rotate_matrix = get_rotate_map(self.config, self.block_size)

        self.assertIn("model.embed_tokens", pre_run.right_rot)

    @patch("msmodelslim.model.glm_5.quarot.QuaRotInterface.get_rotate_command")
    @patch("msmodelslim.model.glm_5.quarot.QuaRotInterface.RotatePair")
    def test_get_rotate_map_shouldHandle_when_numHiddenLayersIsTwo(self, mock_rot_pair, mock_get_rot):
        mock_get_rot.return_value = torch.eye(self.config.hidden_size)
        mock_rot_pair.side_effect = lambda left_rot=None, right_rot=None: MagicMock(
            left_rot=left_rot or {}, right_rot=right_rot or {}
        )

        pre_run, rot_pairs, rotate_matrix = get_rotate_map(self.config, self.block_size, num_hidden_layers=2)

        self.assertIsNotNone(pre_run)
        self.assertIn("rot", rot_pairs)
        self.assertIn("rot_b_proj", rot_pairs)
        self.assertIn("rot_uv", rot_pairs)
        self.assertIn("rot_kv_b_proj", rot_pairs)

    @patch("msmodelslim.model.glm_5.quarot.QuaRotInterface.get_rotate_command")
    @patch("msmodelslim.model.glm_5.quarot.QuaRotInterface.RotatePair")
    def test_get_rotate_map_rotPairs_shouldContainFourPairs_when_called(self, mock_rot_pair, mock_get_rot):
        mock_get_rot.return_value = torch.eye(self.config.hidden_size)
        mock_rot_pair.side_effect = lambda left_rot=None, right_rot=None: MagicMock(
            left_rot=left_rot or {}, right_rot=right_rot or {}
        )

        _, rot_pairs, _ = get_rotate_map(self.config, self.block_size)

        self.assertEqual(len(rot_pairs), 4)
        self.assertIn("rot", rot_pairs)
        self.assertIn("rot_b_proj", rot_pairs)
        self.assertIn("rot_uv", rot_pairs)
        self.assertIn("rot_kv_b_proj", rot_pairs)


if __name__ == '__main__':
    unittest.main()
