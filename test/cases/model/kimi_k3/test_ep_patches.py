#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=no-member
"""Accuracy tests for Kimi-K3 Expert Parallelism helpers / patches."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from msmodelslim.model.common.utils import _get_expert_range
from msmodelslim.model.kimi_k3.ep_patches import (
    count_local_experts,
    moe_infer_local_experts,
    resolve_expert_ep_range,
)
from msmodelslim.utils.exception import SchemaValidateError


class _TinyExpert(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _reference_moe_infer_sort(
    experts: nn.ModuleList,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
) -> torch.Tensor:
    """Mirror of upstream Kimi ``moe_infer`` (full ModuleList, ep_rank=0)."""
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + int(num_tokens)
        if num_tokens == 0:
            continue
        expert = experts[i]
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        outputs.append(expert(tokens_for_this_expert))
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


class TestResolveExpertEpRange:
    def test_single_process_full_range(self):
        ep_size, ep_rank, start, end = resolve_expert_ep_range(896)
        assert (ep_size, ep_rank, start, end) == (1, 0, 0, 896)

    def test_kimi_896_on_8_ranks(self):
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 8
            ranges = []
            for rank in range(8):
                mock_dist.get_rank.return_value = rank
                ranges.append(resolve_expert_ep_range(896)[2:])
            assert ranges == [(r * 112, (r + 1) * 112) for r in range(8)]
            # Contiguous, disjoint, cover all
            flat = [i for s, e in ranges for i in range(s, e)]
            assert flat == list(range(896))

    def test_not_divisible_raises(self):
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 3
            mock_dist.get_rank.return_value = 0
            with pytest.raises(SchemaValidateError):
                resolve_expert_ep_range(896)

    def test_matches_get_expert_range(self):
        """P0-3: adapter helper and EP helper share the same sharding math."""
        cfg = SimpleNamespace(num_experts=896)
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 8
            mock_dist.get_rank.return_value = 3
            start, end = _get_expert_range(cfg)
            _, _, ep_start, ep_end = resolve_expert_ep_range(896)
            assert (start, end) == (ep_start, ep_end) == (336, 448)


class TestGetExpertRangeAdapterHelper:
    def test_matches_ep_helper_for_text_config(self):
        cfg = SimpleNamespace(num_experts=896)
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 4
            mock_dist.get_rank.return_value = 2
            start, end = _get_expert_range(cfg)
            assert (start, end) == (448, 672)


class TestMoeInferLocalExpertsAccuracy:  # pylint: disable=attribute-defined-outside-init
    def setup_method(self):
        torch.manual_seed(0)
        self.hidden = 16
        self.num_experts = 4
        self.top_k = 2
        self.tokens = 12
        self.experts = nn.ModuleList([_TinyExpert(self.hidden) for _ in range(self.num_experts)])
        self.x = torch.randn(self.tokens, self.hidden)
        # Ensure every expert is selected at least once for a strong check.
        self.topk_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [0, 3], [1, 2], [2, 0], [3, 1], [0, 1], [2, 3]],
            dtype=torch.long,
        )
        self.topk_weight = torch.softmax(torch.randn(self.tokens, self.top_k), dim=-1)

    def test_matches_upstream_sort_path_full_experts(self):
        y_ref = _reference_moe_infer_sort(self.experts, self.x, self.topk_ids, self.topk_weight)
        y_new = moe_infer_local_experts(self.experts, self.x, self.topk_ids, self.topk_weight, 0, self.num_experts)
        assert torch.allclose(y_ref, y_new, rtol=1e-4, atol=1e-5)

    def test_ep_partial_sum_equals_full(self):
        y_full = moe_infer_local_experts(self.experts, self.x, self.topk_ids, self.topk_weight, 0, self.num_experts)
        y0 = moe_infer_local_experts(self.experts, self.x, self.topk_ids, self.topk_weight, 0, 2)
        y1 = moe_infer_local_experts(self.experts, self.x, self.topk_ids, self.topk_weight, 2, 4)
        assert torch.allclose(y_full, y0 + y1, rtol=1e-4, atol=1e-5)

    def test_none_slots_match_partial_range(self):
        # Rank0 view: only experts 0,1 materialized.
        experts_r0 = nn.ModuleList([self.experts[0], self.experts[1], None, None])
        y_slots = moe_infer_local_experts(experts_r0, self.x, self.topk_ids, self.topk_weight, 0, 2)
        y_range = moe_infer_local_experts(self.experts, self.x, self.topk_ids, self.topk_weight, 0, 2)
        assert torch.allclose(y_slots, y_range, rtol=1e-5, atol=1e-6)
        assert count_local_experts(SimpleNamespace(experts=experts_r0)) == 2


class TestDistHelperLocalOnlyWithNoneSlots:
    def test_experts_are_local_only_when_none_slotted(self):
        """Simulate 2-rank DistHelper module-name exchange with EP None slots."""

        class ToyMoE(nn.Module):
            def __init__(self, local_ids):
                super().__init__()
                self.gate = nn.Linear(4, 4, bias=False)
                self.shared = nn.Linear(4, 4, bias=False)
                self.experts = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) if i in local_ids else None for i in range(4)]
                )

        rank0 = ToyMoE({0, 1})
        rank1 = ToyMoE({2, 3})

        def names(m: nn.Module):
            return {n for n, mod in m.named_modules() if mod is not None}

        n0, n1 = names(rank0), names(rank1)
        shared = n0 & n1
        local0 = n0 - shared
        local1 = n1 - shared

        assert "gate" in shared
        assert "shared" in shared
        assert any(x.startswith("experts.0") or x == "experts.0" for x in local0) or "experts.0" in local0
        # named_modules for ModuleList children: "experts.0", "experts.1", ...
        assert "experts.0" in local0 and "experts.1" in local0
        assert "experts.2" in local1 and "experts.3" in local1
        assert "experts.0" not in shared
        assert "experts.2" not in shared


class TestEpInitPatchBuildsNoneSlots:
    def test_patched_init_respects_mocked_world(self):
        """Build a minimal stand-in modeling module and apply EP init patch."""
        from msmodelslim.model.kimi_k3 import ep_patches

        class FakeRMS(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class FakeGate(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))

        class FakeMLP(nn.Module):
            def __init__(self, config, intermediate_size=None):
                super().__init__()
                self.down_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        class FakeExpert(nn.Module):
            def __init__(self, config, hidden_size=None, intermediate_size=None):
                super().__init__()
                self.w2 = nn.Linear(hidden_size or config.hidden_size, hidden_size or config.hidden_size, bias=False)

        class FakeMoe(nn.Module):
            pass

        modeling = SimpleNamespace(
            KimiSparseMoeBlock=FakeMoe,
            KimiBlockSparseMLP=FakeExpert,
            KimiMoEGate=FakeGate,
            KimiMLP=FakeMLP,
            KimiRMSNorm=FakeRMS,
        )

        assert ep_patches._patch_kimi_sparse_moe_block(modeling)

        cfg = SimpleNamespace(
            hidden_size=8,
            num_experts=8,
            num_experts_per_token=2,
            moe_renormalize=True,
            moe_intermediate_size=16,
            num_shared_experts=1,
            routed_expert_hidden_size=None,
            latent_moe_use_norm=False,
            rms_norm_eps=1e-5,
            hidden_act="silu",
        )

        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 4
            mock_dist.get_rank.return_value = 1  # experts [2, 4)
            moe = modeling.KimiSparseMoeBlock(cfg)

        assert len(moe.experts) == 8
        assert count_local_experts(moe) == 2
        assert moe.experts_start_idx == 2 and moe.experts_end_idx == 4
        assert moe.experts[2] is not None and moe.experts[3] is not None
        assert moe.experts[0] is None and moe.experts[7] is None
        # Only local expert parameters appear in state_dict
        sd_keys = list(moe.state_dict().keys())
        assert any(k.startswith("experts.2.") for k in sd_keys)
        assert not any(k.startswith("experts.0.") for k in sd_keys)
        assert any(k.startswith("gate.") for k in sd_keys)
        assert any(k.startswith("shared_experts.") for k in sd_keys)


def _make_real_kimi_modeling_module(name="fake_modeling_kimi"):
    """Build a module-like object with the three required nn.Module subclasses."""
    import types

    class KimiSparseMoeBlock(nn.Module):
        pass

    class KimiBlockSparseMLP(nn.Module):
        def __init__(self, config, hidden_size=None, intermediate_size=None):
            super().__init__()
            h = hidden_size or config.hidden_size
            self.w2 = nn.Linear(h, h, bias=False)

        def forward(self, x):
            return self.w2(x)

    class KimiMoEGate(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))

        def forward(self, hidden_states):
            # return topk ids/weights shaped [B*S, K] after flatten path — match ep_forward
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            scores = flat @ self.weight.T
            topk_weight, topk_idx = torch.topk(scores, k=min(2, self.weight.shape[0]), dim=-1)
            topk_weight = torch.softmax(topk_weight, dim=-1)
            return topk_idx, topk_weight

    class KimiMLP(nn.Module):
        def __init__(self, config, intermediate_size=None):
            super().__init__()
            self.down_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        def forward(self, x):
            return self.down_proj(x)

    class KimiRMSNorm(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x

    mod = types.ModuleType(name)
    mod.KimiSparseMoeBlock = KimiSparseMoeBlock
    mod.KimiBlockSparseMLP = KimiBlockSparseMLP
    mod.KimiMoEGate = KimiMoEGate
    mod.KimiMLP = KimiMLP
    mod.KimiRMSNorm = KimiRMSNorm
    return mod


class TestIsRealKimiModelingModule:
    def test_none_is_false(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        assert ep._is_real_kimi_modeling_module(None) is False

    def test_torch_ops_name_is_false(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        assert ep._is_real_kimi_modeling_module(SimpleNamespace(__name__="torch.ops.aten")) is False
        assert ep._is_real_kimi_modeling_module(SimpleNamespace(__name__="torch._ops")) is False

    def test_missing_attrs_is_false(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        assert ep._is_real_kimi_modeling_module(SimpleNamespace(__name__="m", KimiSparseMoeBlock=nn.Linear)) is False

    def test_real_fake_module_is_true(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        mod = _make_real_kimi_modeling_module()
        assert ep._is_real_kimi_modeling_module(mod) is True

    def test_non_type_attrs_is_false(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        mod = SimpleNamespace(
            __name__="m",
            KimiSparseMoeBlock=object(),
            KimiBlockSparseMLP=object(),
            KimiMoEGate=object(),
        )
        assert ep._is_real_kimi_modeling_module(mod) is False


class TestIsEpAlreadyPatched:
    def test_false_before_flag(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        class B(nn.Module):
            pass

        assert ep._is_ep_already_patched(B) is False

    def test_true_when_flag_set(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        class B(nn.Module):
            _msmodelslim_ep_patched = True

        assert ep._is_ep_already_patched(B) is True


class TestClassAndModelingFilesForModelPath:
    def test_prefers_existing_files(self, tmp_path):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        (tmp_path / "modeling_kimi_linear.py").write_text("# stub\n", encoding="utf-8")
        (tmp_path / "modeling_kimi.py").write_text("# stub\n", encoding="utf-8")
        assert ep._class_paths_for_model_path(str(tmp_path)) == (
            "modeling_kimi_linear.KimiSparseMoeBlock",
            "modeling_kimi.KimiSparseMoeBlock",
        )
        assert ep._modeling_files_for_model_path(str(tmp_path)) == (
            "modeling_kimi_linear.py",
            "modeling_kimi.py",
        )

    def test_falls_back_to_defaults_when_missing(self, tmp_path):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        assert ep._class_paths_for_model_path(str(tmp_path)) == ep._KIMI_SPARSE_MOE_CLASS_PATHS
        assert ep._modeling_files_for_model_path(str(tmp_path)) == ep._KIMI_TEXT_MODELING_FILES

    def test_only_legacy_file(self, tmp_path):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        (tmp_path / "modeling_kimi.py").write_text("# stub\n", encoding="utf-8")
        assert ep._class_paths_for_model_path(str(tmp_path)) == ("modeling_kimi.KimiSparseMoeBlock",)
        assert ep._modeling_files_for_model_path(str(tmp_path)) == ("modeling_kimi.py",)


class TestLoadModelingFromWeightDir:
    def test_loads_minimal_valid_stub(self, tmp_path):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        stub = tmp_path / "modeling_kimi_linear.py"
        stub.write_text(
            "\n".join(
                [
                    "from torch import nn",
                    "class KimiSparseMoeBlock(nn.Module):",
                    "    pass",
                    "class KimiBlockSparseMLP(nn.Module):",
                    "    pass",
                    "class KimiMoEGate(nn.Module):",
                    "    pass",
                    "class KimiMLP(nn.Module):",
                    "    pass",
                    "class KimiRMSNorm(nn.Module):",
                    "    pass",
                ]
            ),
            encoding="utf-8",
        )
        mod = ep._load_modeling_from_weight_dir(str(tmp_path), stub)
        assert mod is not None
        assert ep._is_real_kimi_modeling_module(mod)

    def test_bad_file_returns_none(self, tmp_path):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        bad = tmp_path / "modeling_kimi.py"
        bad.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        assert ep._load_modeling_from_weight_dir(str(tmp_path), bad) is None


class TestIterModelingKimiModules:
    def test_yields_injected_sys_modules(self):
        import sys

        from msmodelslim.model.kimi_k3 import ep_patches as ep

        key = "msmodelslim_test_fake_kimi_modeling_iter"
        mod = _make_real_kimi_modeling_module(name=key)
        sys.modules[key] = mod
        try:
            found = [m for m in ep._iter_modeling_kimi_modules(None) if m is mod]
            assert found == [mod]
        finally:
            sys.modules.pop(key, None)


class TestApplyKimiK3EpPatches:
    def test_patches_fake_modeling_in_sys_modules(self):
        import sys

        from msmodelslim.model.kimi_k3 import ep_patches as ep

        key = "msmodelslim_test_fake_kimi_modeling_apply"
        mod = _make_real_kimi_modeling_module(name=key)
        sys.modules[key] = mod
        ep._EP_PATCHED_MODULES.clear()
        try:
            n = ep.apply_kimi_k3_ep_patches()
            assert n >= 1
            assert getattr(mod.KimiSparseMoeBlock, "_msmodelslim_ep_patched", False) is True
            # second call should not re-patch
            n2 = ep.apply_kimi_k3_ep_patches()
            assert n2 == 0
        finally:
            sys.modules.pop(key, None)
            ep._EP_PATCHED_MODULES.discard(id(mod))


class TestEpForwardAndTrainingRaise:
    def test_ep_forward_raises_in_training(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        modeling = _make_real_kimi_modeling_module(name="ep_fwd_train")
        assert ep._patch_kimi_sparse_moe_block(modeling)

        cfg = SimpleNamespace(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=2,
            moe_renormalize=True,
            moe_intermediate_size=16,
            num_shared_experts=1,
            routed_expert_hidden_size=None,
            latent_moe_use_norm=False,
            rms_norm_eps=1e-5,
            hidden_act="silu",
        )
        moe = modeling.KimiSparseMoeBlock(cfg)
        moe.train()
        with pytest.raises(NotImplementedError, match="Training mode"):
            moe.forward(torch.randn(1, 3, 8))

    def test_ep_forward_and_moe_infer_eval_path(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        modeling = _make_real_kimi_modeling_module(name="ep_fwd_eval")
        assert ep._patch_kimi_sparse_moe_block(modeling)

        cfg = SimpleNamespace(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=2,
            moe_renormalize=True,
            moe_intermediate_size=16,
            num_shared_experts=1,
            routed_expert_hidden_size=None,
            latent_moe_use_norm=False,
            rms_norm_eps=1e-5,
            hidden_act="silu",
        )
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            moe = modeling.KimiSparseMoeBlock(cfg)

        moe.eval()
        with patch.object(ep.dist, "is_initialized", return_value=False):
            out = moe.forward(torch.randn(2, 5, 8))
        assert out.shape == (2, 5, 8)

        # moe_infer directly
        x = torch.randn(6, 8)
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [0, 3]], dtype=torch.long)
        topk_weight = torch.softmax(torch.randn(6, 2), dim=-1)
        y = moe.moe_infer(x, topk_ids, topk_weight)
        assert y.shape == x.shape

    def test_ep_forward_with_latent_moe(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        modeling = _make_real_kimi_modeling_module(name="ep_fwd_latent")
        assert ep._patch_kimi_sparse_moe_block(modeling)

        cfg = SimpleNamespace(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=2,
            moe_renormalize=True,
            moe_intermediate_size=16,
            num_shared_experts=None,
            routed_expert_hidden_size=4,
            latent_moe_use_norm=True,
            rms_norm_eps=1e-5,
            hidden_act="silu",
        )
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            moe = modeling.KimiSparseMoeBlock(cfg)

        moe.eval()
        with patch.object(ep.dist, "is_initialized", return_value=False):
            out = moe.forward(torch.randn(1, 4, 8))
        assert out.shape == (1, 4, 8)


class TestMoeInferNoneExpertContinue:
    def test_none_expert_inside_range_is_skipped(self):
        torch.manual_seed(0)
        hidden = 8
        experts = nn.ModuleList([_TinyExpert(hidden), None, _TinyExpert(hidden), None])
        x = torch.randn(4, hidden)
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [3, 1]], dtype=torch.long)
        topk_weight = torch.softmax(torch.randn(4, 2), dim=-1)
        # Range covers None slots 1 and (outside) — ensure line 76 continue is hit.
        y = moe_infer_local_experts(experts, x, topk_ids, topk_weight, 0, 4)
        assert y.shape == x.shape


class TestIsRealKimiModelingModuleEdgeCases:
    def test_non_nn_module_types_is_false(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        class NotModule:
            pass

        mod = SimpleNamespace(
            __name__="m",
            KimiSparseMoeBlock=NotModule,
            KimiBlockSparseMLP=NotModule,
            KimiMoEGate=NotModule,
        )
        assert ep._is_real_kimi_modeling_module(mod) is False

    def test_issubclass_type_error_is_false(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        mod = SimpleNamespace(
            __name__="m",
            KimiSparseMoeBlock=nn.Linear,
            KimiBlockSparseMLP=nn.Linear,
            KimiMoEGate=nn.Linear,
        )
        with patch("msmodelslim.model.kimi_k3.ep_patches.issubclass", side_effect=TypeError("boom")):
            assert ep._is_real_kimi_modeling_module(mod) is False


class TestIsEpAlreadyPatchedException:
    def test_exception_from_dict_get_returns_false(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        class Boom:
            def __getattribute__(self, name):
                if name == "__dict__":
                    raise RuntimeError("no dict")
                return object.__getattribute__(self, name)

        assert ep._is_ep_already_patched(Boom()) is False  # type: ignore[arg-type]


class TestLoadModelingSpecNone:
    def test_spec_none_returns_none(self, tmp_path):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        stub = tmp_path / "modeling_kimi.py"
        stub.write_text("x = 1\n", encoding="utf-8")
        with patch("importlib.util.spec_from_file_location", return_value=None):
            assert ep._load_modeling_from_weight_dir(str(tmp_path), stub) is None


class TestPreloadModelingFromModelPath:
    def test_dynamic_import_success(self, tmp_path):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        fake_cls = type("KimiSparseMoeBlock", (nn.Module,), {})
        fake_mod = _make_real_kimi_modeling_module(name="preload_dyn")
        fake_cls.__module__ = fake_mod.__name__
        seen = set()

        with (
            patch(
                "transformers.dynamic_module_utils.get_class_from_dynamic_module",
                return_value=fake_cls,
            ),
            patch("msmodelslim.model.kimi_k3.ep_patches.inspect.getmodule", return_value=fake_mod),
        ):
            mods = list(ep._preload_modeling_from_model_path(str(tmp_path), seen))
        assert mods == [fake_mod]
        assert id(fake_mod) in seen

        # Already in seen → empty
        with (
            patch(
                "transformers.dynamic_module_utils.get_class_from_dynamic_module",
                return_value=fake_cls,
            ),
            patch("msmodelslim.model.kimi_k3.ep_patches.inspect.getmodule", return_value=fake_mod),
        ):
            assert not list(ep._preload_modeling_from_model_path(str(tmp_path), seen))

    def test_fallback_to_weight_dir_file(self, tmp_path):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        stub = tmp_path / "modeling_kimi_linear.py"
        stub.write_text(
            "\n".join(
                [
                    "from torch import nn",
                    "class KimiSparseMoeBlock(nn.Module):",
                    "    pass",
                    "class KimiBlockSparseMLP(nn.Module):",
                    "    pass",
                    "class KimiMoEGate(nn.Module):",
                    "    pass",
                    "class KimiMLP(nn.Module):",
                    "    pass",
                    "class KimiRMSNorm(nn.Module):",
                    "    pass",
                ]
            ),
            encoding="utf-8",
        )
        seen = set()
        with patch(
            "transformers.dynamic_module_utils.get_class_from_dynamic_module",
            side_effect=ImportError("no hf"),
        ):
            mods = list(ep._preload_modeling_from_model_path(str(tmp_path), seen))
        assert len(mods) == 1
        assert ep._is_real_kimi_modeling_module(mods[0])

    def test_warns_when_all_fail_and_seen_empty(self, tmp_path):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        seen = set()
        with patch(
            "transformers.dynamic_module_utils.get_class_from_dynamic_module",
            side_effect=ImportError("no hf"),
        ):
            assert not list(ep._preload_modeling_from_model_path(str(tmp_path), seen))


class TestIterWithModelPath:
    def test_iter_yields_from_preload(self, tmp_path):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        fake_mod = _make_real_kimi_modeling_module(name="iter_preload_only")
        with patch.object(ep, "_preload_modeling_from_model_path", return_value=iter([fake_mod])):
            # Force no sys.modules hits by filtering — preload still yields.
            found = list(ep._iter_modeling_kimi_modules(str(tmp_path)))
        assert fake_mod in found


class TestPatchKimiSparseMoeBlockGuards:
    def test_not_real_module_returns_false(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        assert ep._patch_kimi_sparse_moe_block(SimpleNamespace(__name__="x")) is False

    def test_already_patched_returns_false(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        modeling = _make_real_kimi_modeling_module(name="already_patched_guard")
        assert ep._patch_kimi_sparse_moe_block(modeling) is True
        assert ep._patch_kimi_sparse_moe_block(modeling) is False


class TestEpInitWithWorldSizeLog:
    def test_logs_when_ep_size_gt_one(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        modeling = _make_real_kimi_modeling_module(name="ep_init_log")
        assert ep._patch_kimi_sparse_moe_block(modeling)
        cfg = SimpleNamespace(
            hidden_size=8,
            num_experts=8,
            num_experts_per_token=2,
            moe_renormalize=True,
            moe_intermediate_size=16,
            num_shared_experts=1,
            routed_expert_hidden_size=None,
            latent_moe_use_norm=False,
            rms_norm_eps=1e-5,
            hidden_act="silu",
        )
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 4
            mock_dist.get_rank.return_value = 0
            moe = modeling.KimiSparseMoeBlock(cfg)
        assert moe.ep_size == 4
        assert moe.experts_start_idx == 0 and moe.experts_end_idx == 2


class TestEpForwardDistributedPath:
    def test_dp_ep_gather_all_reduce_and_slice(self):
        from msmodelslim.model.kimi_k3 import ep_patches as ep

        modeling = _make_real_kimi_modeling_module(name="ep_fwd_dp")
        assert ep._patch_kimi_sparse_moe_block(modeling)
        cfg = SimpleNamespace(
            hidden_size=8,
            num_experts=4,
            num_experts_per_token=2,
            moe_renormalize=True,
            moe_intermediate_size=16,
            num_shared_experts=1,
            routed_expert_hidden_size=4,
            latent_moe_use_norm=True,
            rms_norm_eps=1e-5,
            hidden_act="silu",
        )
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            moe = modeling.KimiSparseMoeBlock(cfg)

        moe.eval()
        hs = torch.randn(1, 3, 8)

        def fake_all_gather(out_list, tensor):
            for i, o in enumerate(out_list):
                o.copy_(tensor)

        with (
            patch.object(ep.dist, "is_initialized", return_value=True),
            patch.object(ep.dist, "get_world_size", return_value=2),
            patch.object(ep.dist, "get_rank", return_value=0),
            patch.object(ep.dist, "all_gather", side_effect=fake_all_gather),
            patch.object(ep.dist, "all_reduce", side_effect=lambda t: None),
            patch(
                "msmodelslim.utils.distributed.DistHelper.gather_variable_shapes",
                return_value=[hs, hs],
            ),
        ):
            out = moe.forward(hs)
        assert out.shape == hs.shape


class TestApplySkipsAlreadyTracked:
    def test_skips_when_mod_id_already_in_set(self):
        import sys

        from msmodelslim.model.kimi_k3 import ep_patches as ep

        key = "msmodelslim_test_fake_kimi_already_tracked"
        mod = _make_real_kimi_modeling_module(name=key)
        sys.modules[key] = mod
        ep._EP_PATCHED_MODULES.clear()
        ep._EP_PATCHED_MODULES.add(id(mod))
        try:
            assert ep.apply_kimi_k3_ep_patches() == 0
        finally:
            sys.modules.pop(key, None)
            ep._EP_PATCHED_MODULES.discard(id(mod))


class TestCountLocalExperts:
    def test_none_experts_attr_returns_zero(self):
        assert count_local_experts(SimpleNamespace()) == 0
        assert count_local_experts(SimpleNamespace(experts=None)) == 0
