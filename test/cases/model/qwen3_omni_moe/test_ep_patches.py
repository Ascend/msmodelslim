#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Unit tests for Qwen3-Omni-MoE expert-parallel patches."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from msmodelslim.model.common.utils import resolve_expert_ep_range
from msmodelslim.model.qwen3_omni_moe import ep_patches
from msmodelslim.model.qwen3_omni_moe.ep_patches import moe_infer_local_experts
from msmodelslim.utils.exception import SchemaValidateError


class _ScaleExpert(nn.Module):
    def __init__(self, config=None, intermediate_size=None, scale=1.0):
        super().__init__()
        del intermediate_size
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.hidden_size = getattr(config, "hidden_size", None)

    def forward(self, hidden_states):
        return hidden_states * self.scale


def _make_modeling():
    class FakeSparseMoeBlock(nn.Module):  # pylint: disable=abstract-method
        pass

    class FakeExpert(_ScaleExpert):
        pass

    return SimpleNamespace(
        Qwen3OmniMoeThinkerTextSparseMoeBlock=FakeSparseMoeBlock,
        Qwen3OmniMoeThinkerTextMLP=FakeExpert,
    )


def _make_config(num_experts=4):
    return SimpleNamespace(
        hidden_size=4,
        num_experts=num_experts,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        moe_intermediate_size=8,
    )


class TestResolveExpertEpRange:
    def test_range_equals_all_experts_when_single_process(self):
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            assert resolve_expert_ep_range(128) == (1, 0, 0, 128)

    def test_ranges_equal_contiguous_shards_when_four_ranks_and_128_experts(self):
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 4
            ranges = []
            for rank in range(4):
                mock_dist.get_rank.return_value = rank
                ranges.append(resolve_expert_ep_range(128)[2:])

        assert ranges == [(0, 32), (32, 64), (64, 96), (96, 128)]

    def test_range_raises_when_experts_are_not_divisible(self):
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 3
            mock_dist.get_rank.return_value = 0
            with pytest.raises(SchemaValidateError):
                resolve_expert_ep_range(128)


class TestMoeInferLocalExperts:
    def test_output_equals_full_inference_when_partial_outputs_are_summed(self):
        experts = nn.ModuleList([_ScaleExpert(scale=index + 1) for index in range(4)])
        hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        selected_experts = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
        routing_weights = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4]])

        full = moe_infer_local_experts(experts, hidden_states, selected_experts, routing_weights, 0, 4)
        rank_zero = moe_infer_local_experts(experts, hidden_states, selected_experts, routing_weights, 0, 2)
        rank_one = moe_infer_local_experts(experts, hidden_states, selected_experts, routing_weights, 2, 4)

        assert torch.allclose(full, rank_zero + rank_one)

    def test_output_equals_materialized_experts_when_slots_contain_none(self):
        expert_zero = _ScaleExpert(scale=1)
        expert_two = _ScaleExpert(scale=3)
        experts = nn.ModuleList([expert_zero, None, expert_two, None])
        hidden_states = torch.ones(3, 4)
        selected_experts = torch.tensor([[0, 1], [1, 2], [2, 3]])
        routing_weights = torch.ones(3, 2)

        output = moe_infer_local_experts(experts, hidden_states, selected_experts, routing_weights, 0, 4)

        expected = torch.tensor([[1.0] * 4, [3.0] * 4, [3.0] * 4])
        assert torch.equal(output, expected)


class TestGatherHiddenStates:
    def test_gather_returns_rank_slice_when_sequence_lengths_differ(self):
        hidden_states = torch.ones(1, 3, 2)

        def fake_all_gather(outputs, _local_length):
            outputs[0].fill_(2)
            outputs[1].fill_(3)

        gathered_parts = [torch.zeros(1, 2, 2), hidden_states]
        with (
            patch.object(ep_patches.dist, "get_world_size", return_value=2),
            patch.object(ep_patches.dist, "get_rank", return_value=1),
            patch.object(ep_patches.dist, "all_gather", side_effect=fake_all_gather),
            patch(
                "msmodelslim.utils.distributed.DistHelper.gather_variable_shapes",
                return_value=gathered_parts,
            ) as mock_gather,
        ):
            gathered, start, end = ep_patches._gather_hidden_states(hidden_states)

        assert gathered.shape == (1, 5, 2)
        assert (start, end) == (2, 5)
        mock_gather.assert_called_once_with(hidden_states)


class TestPatchSparseMoeBlock:
    def test_init_materializes_rank_shard_when_four_ranks_and_128_experts(self):
        modeling = _make_modeling()
        assert ep_patches._patch_sparse_moe_block(modeling) is True

        with patch(
            "msmodelslim.model.qwen3_omni_moe.ep_patches.resolve_expert_ep_range",
            return_value=(4, 2, 64, 96),
        ):
            moe = modeling.Qwen3OmniMoeThinkerTextSparseMoeBlock(_make_config(128))

        assert len(moe.experts) == 128
        assert sum(expert is not None for expert in moe.experts) == 32
        assert all(moe.experts[index] is not None for index in range(64, 96))
        assert moe.experts[63] is None and moe.experts[96] is None

    def test_patch_returns_false_when_required_classes_are_missing(self):
        assert ep_patches._patch_sparse_moe_block(SimpleNamespace()) is False

    def test_patch_returns_false_when_class_is_already_patched(self):
        modeling = _make_modeling()
        assert ep_patches._patch_sparse_moe_block(modeling) is True
        assert ep_patches._patch_sparse_moe_block(modeling) is False


class TestApplyQwen3OmniMoeEpPatches:
    def test_apply_returns_one_then_zero_when_called_twice(self):
        modeling = _make_modeling()
        assert ep_patches.apply_qwen3_omni_moe_ep_patches(modeling) == 1
        assert ep_patches.is_ep_patch_applied(modeling) is True
        assert ep_patches.apply_qwen3_omni_moe_ep_patches(modeling) == 0
        assert ep_patches.is_ep_patch_applied(modeling) is True


class TestPatchedSparseMoeForward:
    def test_forward_raises_when_module_is_training(self):
        modeling = _make_modeling()
        ep_patches._patch_sparse_moe_block(modeling)
        moe = modeling.Qwen3OmniMoeThinkerTextSparseMoeBlock(_make_config())

        with pytest.raises(NotImplementedError, match="Training mode"):
            moe(torch.randn(1, 2, 4))

    def test_forward_uses_upstream_impl_for_single_process(self):
        modeling = _make_modeling()
        block_cls = modeling.Qwen3OmniMoeThinkerTextSparseMoeBlock
        calls = []

        def upstream(self, hidden_states):
            calls.append(hidden_states)
            return "upstream", None

        block_cls.forward = upstream
        ep_patches._patch_sparse_moe_block(modeling)
        moe = block_cls(_make_config())
        moe.eval()

        with patch.object(ep_patches.dist, "is_initialized", return_value=False):
            output = moe(torch.randn(1, 2, 4))

        assert output[0] == "upstream"
        assert len(calls) == 1

    def test_forward_uses_aligned_gather_by_default(self):
        modeling = _make_modeling()
        ep_patches._patch_sparse_moe_block(modeling)
        moe = modeling.Qwen3OmniMoeThinkerTextSparseMoeBlock(_make_config())
        moe.eval()
        hidden_states = torch.randn(1, 2, 4)
        gathered_states = torch.cat([torch.randn(1, 1, 4), hidden_states], dim=1)

        def fake_all_gather_into_tensor(output, local, *args, **kwargs):
            del args, kwargs
            output[0].copy_(local)
            output[1].copy_(local)

        with (
            patch.object(ep_patches.dist, "is_initialized", return_value=True),
            patch.object(ep_patches.dist, "get_world_size", return_value=2),
            patch.object(
                ep_patches,
                "_gather_hidden_states",
                return_value=(gathered_states, 1, 3),
            ) as mock_gather,
            patch.object(
                ep_patches.dist,
                "all_gather_into_tensor",
                side_effect=fake_all_gather_into_tensor,
            ) as mock_gather_partials,
        ):
            output, router_logits = moe(hidden_states)

        assert output.shape == hidden_states.shape
        assert router_logits.shape == (2, 4)
        mock_gather.assert_called_once_with(hidden_states)
        mock_gather_partials.assert_called_once()
