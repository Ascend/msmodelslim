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
from unittest.mock import Mock, patch
import torch
from msmodelslim.model.deepseek_v3.quarot import get_ln_fuse_map, get_rotate_map


class TestQuaRot(unittest.TestCase):
    def setUp(self):
        # Create a mock config object
        self.config = Mock()
        self.config.num_hidden_layers = 2
        self.config.first_k_dense_replace = 1
        self.config.n_routed_experts = 2
        self.config.hidden_size = 256
        self.config.q_lora_rank = 128
        self.config.v_head_dim = 64
        self.config.kv_lora_rank = 64
        self.config.qk_nope_head_dim = 32
        self.config.qk_rope_head_dim = 32

    def test_get_ln_fuse_map_default_layers(self):
        """Test get_ln_fuse_map with default number of layers"""
        result = get_ln_fuse_map(self.config)

        # Check basic structure for each layer
        for layer_idx in range(3):  # 0, 1, 2
            # Check input layernorm mapping exists
            input_ln_key = f"model.layers.{layer_idx}.input_layernorm"
            self.assertIn(input_ln_key, result)
            self.assertIn(f"model.layers.{layer_idx}.self_attn.q_a_proj", result[input_ln_key])
            self.assertIn(f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa", result[input_ln_key])

            # Check q_a_layernorm mapping exists
            q_a_ln_key = f"model.layers.{layer_idx}.self_attn.q_a_layernorm"
            self.assertIn(q_a_ln_key, result)
            self.assertIn(f"model.layers.{layer_idx}.self_attn.q_b_proj", result[q_a_ln_key])

            # Check kv_a_layernorm mapping exists
            kv_a_ln_key = f"model.layers.{layer_idx}.self_attn.kv_a_layernorm"
            self.assertIn(kv_a_ln_key, result)
            self.assertIn(f"model.layers.{layer_idx}.self_attn.kv_b_proj", result[kv_a_ln_key])

            # Check post_attention_layernorm mapping based on layer index
            post_ln_key = f"model.layers.{layer_idx}.post_attention_layernorm"
            self.assertIn(post_ln_key, result)

            if layer_idx < self.config.first_k_dense_replace:
                # Dense layer - should have gate_proj and up_proj
                self.assertEqual(len(result[post_ln_key]), 2)
                self.assertIn(f"model.layers.{layer_idx}.mlp.gate_proj", result[post_ln_key])
                self.assertIn(f"model.layers.{layer_idx}.mlp.up_proj", result[post_ln_key])
            else:
                # Expert layer - should have experts, shared_experts and gate
                # 2 experts * 2 projections + 2 shared experts projections + 1 gate = 7 items
                self.assertEqual(len(result[post_ln_key]), 7)

                # Check for routed experts
                expert_gate_proj_found = any(
                    f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj" in result[post_ln_key]
                    for i in range(self.config.n_routed_experts)
                )
                expert_up_proj_found = any(
                    f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj" in result[post_ln_key]
                    for i in range(self.config.n_routed_experts)
                )
                self.assertTrue(expert_gate_proj_found)
                self.assertTrue(expert_up_proj_found)

                # Check for shared experts
                shared_gate_proj_found = any(
                    f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj" in item for item in result[post_ln_key]
                )
                shared_up_proj_found = any(
                    f"model.layers.{layer_idx}.mlp.shared_experts.up_proj" in item for item in result[post_ln_key]
                )
                self.assertTrue(shared_gate_proj_found)
                self.assertTrue(shared_up_proj_found)

                # Check for gate - fixed the iteration issue
                gate_found = False
                for item in result[post_ln_key]:
                    if f"model.layers.{layer_idx}.mlp.gate" in item:
                        gate_found = True
                        break
                self.assertTrue(gate_found)

        # Check MTP mappings exist
        mtp_key = ("model.layers.2.enorm", "model.layers.2.hnorm")  # Last layer index should be 2
        self.assertIn(mtp_key, result)
        self.assertIn("model.layers.2.eh_proj", result[mtp_key])

        # Check final shared head norm
        shared_head_key = "model.layers.2.shared_head.norm"
        self.assertIn(shared_head_key, result)
        self.assertIn("model.layers.2.shared_head.head", result[shared_head_key])

        # Check model.norm mapping exists
        self.assertIn("model.norm", result)
        self.assertIn("lm_head", result["model.norm"])

    def test_get_ln_fuse_map_custom_layers(self):
        """Test get_ln_fuse_map with custom number of layers"""
        custom_num_layers = 1
        result = get_ln_fuse_map(self.config, custom_num_layers)

        # With custom_num_layers=1, we should have:
        # Layers 0 (regular) and MTP at layer 1

        # Check regular layer
        layer_0_key = "model.layers.0.post_attention_layernorm"
        self.assertIn(layer_0_key, result)
        # Layer 0 should be dense layer (< first_k_dense_replace=1)
        self.assertIn("model.layers.0.mlp.gate_proj", result[layer_0_key])
        self.assertIn("model.layers.0.mlp.up_proj", result[layer_0_key])
        self.assertEqual(len(result[layer_0_key]), 2)

        # Check other required keys for layer 0
        input_ln_key = "model.layers.0.input_layernorm"
        self.assertIn(input_ln_key, result)
        self.assertIn("model.layers.0.self_attn.q_a_proj", result[input_ln_key])
        self.assertIn("model.layers.0.self_attn.kv_a_proj_with_mqa", result[input_ln_key])

        q_a_ln_key = "model.layers.0.self_attn.q_a_layernorm"
        self.assertIn(q_a_ln_key, result)
        self.assertIn("model.layers.0.self_attn.q_b_proj", result[q_a_ln_key])

        kv_a_ln_key = "model.layers.0.self_attn.kv_a_layernorm"
        self.assertIn(kv_a_ln_key, result)
        self.assertIn("model.layers.0.self_attn.kv_b_proj", result[kv_a_ln_key])

        # Check model.norm mapping exists
        self.assertIn("model.norm", result)
        self.assertIn("lm_head", result["model.norm"])

    def test_get_rotate_map_default_layers(self):
        """Test get_rotate_map with default number of layers"""
        block_size = 32

        with (
            patch('msmodelslim.model.interface_hub.QuaRotInterface.get_rotate_command') as mock_get_rotate,
            patch('msmodelslim.model.interface_hub.QuaRotInterface.RotatePair') as mock_rotate_pair,
        ):
            # Setup mock return values
            mock_rotation_matrix = torch.eye(256)
            mock_get_rotate.return_value = mock_rotation_matrix
            mock_rotate_pair_instance = Mock()
            mock_rotate_pair.return_value = mock_rotate_pair_instance

            pre_run, rot_pairs, rotate_matrix = get_rotate_map(self.config, block_size)

            # Verify get_rotate_command was called with correct parameters
            self.assertEqual(mock_get_rotate.call_count, 4)

            # Verify RotatePair was called for each rotation type
            # pre_run + rot + rot_b_proj + rot_uv + rot_kv_b_proj = 5 calls total
            self.assertEqual(mock_rotate_pair.call_count, 5)

            # Check all rotation matrices are returned
            self.assertIn('rot', rotate_matrix)
            self.assertIn('rot_b_proj', rotate_matrix)
            self.assertIn('rot_uv', rotate_matrix)
            self.assertIn('rot_kv_b_proj', rotate_matrix)

            # Check all rotation pairs are created
            self.assertIn('rot', rot_pairs)
            self.assertIn('rot_b_proj', rot_pairs)
            self.assertIn('rot_uv', rot_pairs)
            self.assertIn('rot_kv_b_proj', rot_pairs)

    def test_get_rotate_map_custom_layers(self):
        """Test get_rotate_map with custom number of layers"""
        block_size = 32
        custom_num_layers = 1

        with (
            patch('msmodelslim.model.interface_hub.QuaRotInterface.get_rotate_command') as mock_get_rotate,
            patch('msmodelslim.model.interface_hub.QuaRotInterface.RotatePair') as mock_rotate_pair,
        ):
            # Setup mock return values
            mock_rotation_matrix = torch.eye(256)
            mock_get_rotate.return_value = mock_rotation_matrix
            mock_rotate_pair_instance = Mock()
            mock_rotate_pair.return_value = mock_rotate_pair_instance

            pre_run, rot_pairs, rotate_matrix = get_rotate_map(self.config, block_size, custom_num_layers)

            # Should still work correctly with custom layer count
            self.assertIsNotNone(pre_run)
            self.assertIsNotNone(rot_pairs)
            self.assertIsNotNone(rotate_matrix)


if __name__ == '__main__':
    unittest.main()
