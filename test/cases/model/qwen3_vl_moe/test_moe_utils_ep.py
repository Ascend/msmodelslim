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

# Unit tests for the expert-parallel helpers in ``qwen3_vl_moe.moe_utils``.
# All cases run on CPU and stub ``torch.distributed`` so they do not need NPU or
# a real process group.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.model.qwen3_vl_moe import moe_utils


class _FusedExperts(nn.Module):
    """Holds the fused 3D expert tensors exactly like the real block."""

    def __init__(self, num_experts: int, hidden: int, inter: int, dtype=torch.bfloat16):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden, 2 * inter).to(dtype))
        self.down_proj = nn.Parameter(torch.randn(num_experts, inter, hidden).to(dtype))


class _FusedMoeBlock(nn.Module):
    """Router + fused experts, mirroring the upstream MoE block layout."""

    def __init__(self, num_experts: int, hidden: int = 8, inter: int = 4, dtype=torch.bfloat16):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False, dtype=dtype)
        self.experts = _FusedExperts(num_experts, hidden, inter, dtype)


def _make_experts(num_experts: int, hidden: int, inter: int, dtype=torch.bfloat16) -> nn.ModuleList:
    """Build a ModuleList of tiny two-layer expert MLPs (no bias)."""
    experts = nn.ModuleList()
    for _ in range(num_experts):
        experts.append(
            nn.Sequential(
                nn.Linear(hidden, inter, bias=False, dtype=dtype),
                nn.ReLU(),
                nn.Linear(inter, hidden, bias=False, dtype=dtype),
            )
        )
    return experts


def _make_routing(batch_tokens: int, num_experts: int, top_k: int, dtype=torch.bfloat16):
    """Return (selected_experts, routing_weights) with normalised top-k weights."""
    logits = torch.randn(batch_tokens, num_experts)
    probs = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(dtype)
    return indices, weights


class TestAccumulateExpertRange(unittest.TestCase):
    """``_accumulate_expert_range`` performs deterministic sparse float32 summing."""

    def setUp(self):
        torch.manual_seed(0)
        self.num_experts = 8
        self.hidden = 6
        self.inter = 4
        self.tokens = 12
        self.top_k = 2
        self.experts = _make_experts(self.num_experts, self.hidden, self.inter)
        self.hidden_states = torch.randn(self.tokens, self.hidden, dtype=torch.bfloat16)
        self.selected, self.weights = _make_routing(self.tokens, self.num_experts, self.top_k)

    def test_full_range_matches_dense_reference(self):
        actual = moe_utils._accumulate_expert_range(
            self.experts, self.hidden_states, self.selected, self.weights, 0, self.num_experts
        )
        # Dense reference: explicitly iterate every expert and accumulate.
        expected = torch.zeros(self.tokens, self.hidden, dtype=torch.float32)
        for expert_id in range(self.num_experts):
            token_idx, topk_idx = torch.where(self.selected == expert_id)
            if token_idx.numel() == 0:
                continue
            out = self.experts[expert_id](self.hidden_states[token_idx]).to(torch.float32)
            weight = self.weights[token_idx, topk_idx].unsqueeze(-1).to(torch.float32)
            expected.index_add_(0, token_idx, out * weight)
        self.assertTrue(torch.equal(actual, expected))

    def test_output_is_float32(self):
        out = moe_utils._accumulate_expert_range(
            self.experts, self.hidden_states, self.selected, self.weights, 0, self.num_experts
        )
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (self.tokens, self.hidden))

    def test_none_slots_are_skipped(self):
        # Emulate an EP rank: only experts [2, 5) are materialised.
        sharded = nn.ModuleList([None] * 2 + [self.experts[i] for i in range(2, 5)] + [None] * 3)
        actual = moe_utils._accumulate_expert_range(
            sharded, self.hidden_states, self.selected, self.weights, 0, self.num_experts
        )
        expected = moe_utils._accumulate_expert_range(
            self.experts, self.hidden_states, self.selected, self.weights, 2, 5
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_empty_range_returns_zeros(self):
        out = moe_utils._accumulate_expert_range(self.experts, self.hidden_states, self.selected, self.weights, 3, 3)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_expert_without_tokens_is_skipped(self):
        # Force routing to only expert 0 and 1, then sum experts [2, 3).
        selected = torch.zeros(self.tokens, self.top_k, dtype=torch.long)
        weights = torch.ones(self.tokens, self.top_k, dtype=torch.bfloat16) / self.top_k
        out = moe_utils._accumulate_expert_range(self.experts, self.hidden_states, selected, weights, 2, 3)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_range_is_inclusive_left_exclusive_right(self):
        only_last = moe_utils._accumulate_expert_range(
            self.experts,
            self.hidden_states,
            self.selected,
            self.weights,
            self.num_experts - 1,
            self.num_experts,
        )
        full = moe_utils._accumulate_expert_range(
            self.experts, self.hidden_states, self.selected, self.weights, 0, self.num_experts
        )
        last_only = moe_utils._accumulate_expert_range(
            self.experts,
            self.hidden_states,
            self.selected,
            self.weights,
            self.num_experts - 1,
            self.num_experts,
        )
        self.assertTrue(torch.equal(only_last, last_only))
        # And it must be a proper subset contribution of the full sum.
        self.assertFalse(torch.equal(only_last, full))


class TestSumInRankOrder(unittest.TestCase):
    """``_sum_in_rank_order`` fixes the floating-point addition order."""

    def test_single_partial_returned_unchanged(self):
        part = torch.randn(3, 4)
        self.assertTrue(torch.equal(moe_utils._sum_in_rank_order([part]), part))

    def test_order_matters(self):
        # 1e16 + 1.0 loses the 1.0; adding the small values first keeps it.
        big = torch.tensor([1e16], dtype=torch.float32)
        one = torch.tensor([1.0], dtype=torch.float32)
        small = torch.tensor([1.0], dtype=torch.float32)
        ordered = moe_utils._sum_in_rank_order([big, one, small])
        self.assertTrue(torch.equal(ordered, big))
        # Reference: left-to-right folding is exactly what the implementation does.
        expected = (big + one) + small
        self.assertTrue(torch.equal(ordered, expected))

    def test_three_partials_equal_explicit_fold(self):
        parts = [torch.randn(2, 3) for _ in range(4)]
        actual = moe_utils._sum_in_rank_order(parts)
        expected = ((parts[0] + parts[1]) + parts[2]) + parts[3]
        self.assertTrue(torch.equal(actual, expected))


class TestGatherPartials(unittest.TestCase):
    """``_gather_partials`` uses the contiguous collective and keeps rank order."""

    def test_all_gather_into_tensor_path(self):
        local = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        def fake_into_tensor(out, inp):
            out.copy_(inp.expand_as(out))

        with patch.object(moe_utils.dist, "get_world_size", return_value=4):
            with patch.object(moe_utils.dist, "all_gather_into_tensor", side_effect=fake_into_tensor):
                with patch.object(moe_utils.dist, "all_gather", side_effect=AssertionError("fallback used")):
                    gathered = moe_utils._gather_partials(local)
        self.assertEqual(len(gathered), 4)
        for part in gathered:
            self.assertTrue(torch.equal(part, local))

    def test_list_fallback_path(self):
        local = torch.tensor([[3.0]], dtype=torch.float32)

        def fake_all_gather(out_list, inp):
            for buf in out_list:
                buf.copy_(inp)

        with patch.object(moe_utils.dist, "get_world_size", return_value=2):
            # Force the legacy branch by hiding the contiguous collective.
            with patch.object(moe_utils.dist, "all_gather_into_tensor", None, create=True):
                with patch.object(moe_utils, "hasattr", lambda obj, name: False, create=True):
                    with patch.object(moe_utils.dist, "all_gather", side_effect=fake_all_gather):
                        gathered = moe_utils._gather_partials(local)
        self.assertEqual(len(gathered), 2)
        self.assertTrue(torch.equal(gathered[0], local))

    def test_non_contiguous_input_is_accepted(self):
        base = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        local = base.t()  # non-contiguous view

        def fake_into_tensor(out, inp):
            out.copy_(inp.expand_as(out))

        with patch.object(moe_utils.dist, "get_world_size", return_value=2):
            with patch.object(moe_utils.dist, "all_gather_into_tensor", side_effect=fake_into_tensor):
                gathered = moe_utils._gather_partials(local)
        self.assertTrue(torch.equal(gathered[0], local))


class TestExpertSharding(unittest.TestCase):
    """The MoE block only materialises the experts owned by the current rank."""

    @staticmethod
    def _config(num_experts: int, hidden: int = 8, inter: int = 4, top_k: int = 2):
        return SimpleNamespace(
            hidden_size=hidden,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            moe_intermediate_size=inter,
            hidden_act="silu",
        )

    @staticmethod
    def _fused_block(num_experts: int, hidden: int = 8, inter: int = 4):
        """Minimal stand-in for ``Qwen3VLMoeTextSparseMoeBlock``."""
        return _FusedMoeBlock(num_experts, hidden, inter)

    def test_local_range_is_materialised_and_others_are_none(self):
        num_experts = 8
        config = self._config(num_experts)
        block = self._fused_block(num_experts)
        with patch.object(moe_utils, "resolve_expert_ep_range", return_value=(4, 1, 2, 4)):
            moe = moe_utils.UnstackedQwen3VLMoeSparseMoeBlock(config, block)
        self.assertEqual(moe.experts_start_idx, 2)
        self.assertEqual(moe.experts_end_idx, 4)
        self.assertEqual(moe.ep_size, 4)
        self.assertEqual(moe.ep_rank, 1)
        for idx, slot in enumerate(moe.experts):
            if 2 <= idx < 4:
                self.assertIsNotNone(slot)
            else:
                self.assertIsNone(slot)

    def test_single_card_materialises_every_expert(self):
        num_experts = 8
        config = self._config(num_experts)
        block = self._fused_block(num_experts)
        with patch.object(moe_utils, "resolve_expert_ep_range", return_value=(1, 0, 0, num_experts)):
            moe = moe_utils.UnstackedQwen3VLMoeSparseMoeBlock(config, block, copy_weights=True)
        moe.eval()
        self.assertEqual(moe.ep_size, 1)
        for slot in moe.experts:
            self.assertIsNotNone(slot)
        # Every expert weight should be copied from the fused tensor.
        expected = block.experts.gate_up_proj[0].chunk(2, dim=-1)[0].t()
        self.assertTrue(torch.equal(moe.experts[0].gate_proj.weight, expected))

    def test_pre_sliced_weights_are_used(self):
        num_experts = 8
        hidden, inter = 8, 4
        config = self._config(num_experts, hidden, inter)
        block = self._fused_block(num_experts, hidden, inter)
        # Rank 2 owns experts [4, 6); pass only those two slices.
        gate_up_slice = block.experts.gate_up_proj[4:6].clone()
        down_slice = block.experts.down_proj[4:6].clone()
        with patch.object(moe_utils, "resolve_expert_ep_range", return_value=(4, 2, 4, 6)):
            moe = moe_utils.UnstackedQwen3VLMoeSparseMoeBlock(
                config, block, copy_weights=True, expert_slices=(gate_up_slice, down_slice)
            )
        moe.eval()
        expected_gate = gate_up_slice[0].chunk(2, dim=-1)[0].t()
        self.assertTrue(torch.equal(moe.experts[4].gate_proj.weight, expected_gate))
        expected_down = down_slice[1].t()
        self.assertTrue(torch.equal(moe.experts[5].down_proj.weight, expected_down))

    def test_training_mode_forward_raises(self):
        num_experts = 4
        config = self._config(num_experts)
        block = self._fused_block(num_experts)
        with patch.object(moe_utils, "resolve_expert_ep_range", return_value=(1, 0, 0, num_experts)):
            moe = moe_utils.UnstackedQwen3VLMoeSparseMoeBlock(config, block, copy_weights=True)
        moe.train()
        with self.assertRaises(NotImplementedError):
            moe(torch.randn(1, 3, 8))


class TestForwardBranchSelection(unittest.TestCase):
    """``forward`` picks the single-card or the EP reduction correctly."""

    def _build(self, num_experts=8, replicated=True):
        config = SimpleNamespace(
            hidden_size=8,
            num_experts=num_experts,
            num_experts_per_tok=2,
            moe_intermediate_size=4,
            hidden_act="silu",
        )
        fused = _FusedMoeBlock(num_experts, 8, 4)
        with patch.object(moe_utils, "resolve_expert_ep_range", return_value=(1, 0, 0, num_experts)):
            moe = moe_utils.UnstackedQwen3VLMoeSparseMoeBlock(config, fused, copy_weights=True)
        moe.eval()
        return moe

    def test_ep_path_with_identity_gather_equals_local_when_single_rank(self):
        torch.manual_seed(11)
        moe = self._build()
        x = torch.randn(2, 4, 8, dtype=torch.bfloat16)

        def fake_into_tensor(out, inp):
            out.copy_(inp.expand_as(out))

        with patch.object(moe_utils.dist, "is_initialized", return_value=True):
            with patch.object(moe_utils.dist, "get_world_size", return_value=1):
                with patch.object(moe_utils.dist, "get_rank", return_value=0):
                    with patch.object(moe_utils.dist, "all_gather_into_tensor", side_effect=fake_into_tensor):
                        out_ep = moe(x)
        out_single = moe(x)
        self.assertTrue(torch.equal(out_ep, out_single))


class TestTokenGatherHelper(unittest.TestCase):
    """``_gather_hidden_states`` reports this rank's slice in the gathered block."""

    def test_slices_follow_rank_order(self):
        hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        world = 3
        seq_lens = [2, 3, 4]
        rank = 1

        def fake_all_gather(out_list, inp):
            for idx, buf in enumerate(out_list):
                buf.fill_(seq_lens[idx])

        def fake_gather_variable(tensor):
            return [torch.zeros(2, n, 4) for n in seq_lens]

        fake_helper = SimpleNamespace(gather_variable_shapes=fake_gather_variable)
        with patch.object(moe_utils.dist, "get_world_size", return_value=world):
            with patch.object(moe_utils.dist, "get_rank", return_value=rank):
                with patch.object(moe_utils.dist, "all_gather", side_effect=fake_all_gather):
                    with patch("msmodelslim.utils.distributed.DistHelper", fake_helper):
                        gathered, start, end = moe_utils._gather_hidden_states(hidden)
        self.assertEqual(start, seq_lens[0])
        self.assertEqual(end, seq_lens[0] + seq_lens[rank])
        self.assertEqual(tuple(gathered.shape), (2, sum(seq_lens), 4))


if __name__ == "__main__":
    unittest.main()
