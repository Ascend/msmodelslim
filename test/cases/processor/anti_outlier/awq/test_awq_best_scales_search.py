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

Tests for AWQAlphaSearcher.search() interface.
"""
# pylint: disable=attribute-defined-outside-init

import copy

import pytest
import torch
from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal.qbase import QDType, QScope
from msmodelslim.processor.anti_outlier.awq.best_scales_search import AWQBestScalesSearcher
from msmodelslim.processor.anti_outlier.awq.common import AWQContext, offload


def _make_qconfig() -> QConfig:
    return QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, method="minmax", symmetric=True)


class SimpleMLP(nn.Module):
    """A tiny MLP used as module2inspect in tests."""

    def __init__(self, in_features: int = 64, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


def _build_context(
    in_features: int = 64,
    n_samples: int = 2,
    seq_len: int = 8,
    inspect_module: nn.Module = None,
) -> AWQContext:
    """Build an AWQContext with random act_mean and block kwargs."""
    act_mean = torch.rand(in_features).clamp(min=1e-3)
    cache = []
    for _ in range(n_samples):
        cache.append(offload({"hidden_states": torch.randn(1, seq_len, in_features)}))
    if inspect_module is None:
        inspect_module = SimpleMLP(in_features=in_features)
    return AWQContext(act_mean=act_mean, inspect_module=inspect_module, inspect_module_args=cache)


class TestSearchInputValidation:
    def test_empty_linears_raises(self):
        searcher = AWQBestScalesSearcher(weight_qconfig=_make_qconfig(), n_grid=5)
        module = SimpleMLP()
        ctx = _build_context(inspect_module=module)

        with pytest.raises(ValueError, match="linears2scale must not be empty"):
            searcher.search(linears2scale=[], context=ctx)

    def test_empty_block_kwargs_raises(self):
        searcher = AWQBestScalesSearcher(weight_qconfig=_make_qconfig(), n_grid=5)
        module = SimpleMLP()
        empty_ctx = AWQContext(
            act_mean=torch.rand(64),
            inspect_module=module,
            inspect_module_args=[],
        )

        with pytest.raises(ValueError, match="block_kwargs_cache is empty"):
            searcher.search(
                linears2scale=[module.fc1],
                context=empty_ctx,
            )


class TestSearchReturnsScales:
    def setup_method(self):
        self.in_features = 64
        self.module = SimpleMLP(in_features=self.in_features)
        self.ctx = _build_context(in_features=self.in_features, inspect_module=self.module)
        self.searcher = AWQBestScalesSearcher(weight_qconfig=_make_qconfig(), n_grid=5)

    def test_returns_valid_scales(self):
        scales = self.searcher.search(
            linears2scale=[self.module.fc1],
            context=self.ctx,
        )
        assert isinstance(scales, torch.Tensor)
        assert scales.shape == (self.in_features,)
        assert torch.isfinite(scales).all()


class TestSearchPreservesState:
    def setup_method(self):
        self.module = SimpleMLP()
        self.ctx = _build_context(inspect_module=self.module)
        self.searcher = AWQBestScalesSearcher(weight_qconfig=_make_qconfig(), n_grid=5)

    def test_weights_restored_after_search(self):
        original_sd = copy.deepcopy(self.module.state_dict())
        fc1_weight_before = self.module.fc1.weight.data.clone()
        self.searcher.search(
            linears2scale=[self.module.fc1],
            context=self.ctx,
        )
        for key in original_sd:
            assert torch.equal(original_sd[key], self.module.state_dict()[key]), (
                f"Weight '{key}' was not restored after search"
            )
        assert torch.equal(fc1_weight_before, self.module.fc1.weight.data)


class TestSearchGridBehavior:
    def setup_method(self):
        self.in_features = 64
        self.module = SimpleMLP(in_features=self.in_features)
        self.ctx = _build_context(in_features=self.in_features, inspect_module=self.module)

    def test_n_grid_1_returns_identity_scales(self):
        """n_grid=1 → only ratio=0 is tried → scales = act_mean^0 = 1 → normalised ones."""
        searcher = AWQBestScalesSearcher(weight_qconfig=_make_qconfig(), n_grid=1)
        scales = searcher.search(
            linears2scale=[self.module.fc1],
            context=self.ctx,
        )
        assert scales is not None
        assert torch.allclose(scales, torch.ones_like(scales))


class TestSearchWithMultipleLinears:
    def test_two_linears_same_in_features(self):
        """Search with two linears sharing the same in_features succeeds."""
        in_features = 64

        class TwoLinearBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc_a = nn.Linear(in_features, 128, bias=False)
                self.fc_b = nn.Linear(in_features, 128, bias=False)

            def forward(self, hidden_states):
                return self.fc_a(hidden_states) + self.fc_b(hidden_states)

        module = TwoLinearBlock()
        ctx = _build_context(in_features=in_features, inspect_module=module)
        searcher = AWQBestScalesSearcher(weight_qconfig=_make_qconfig(), n_grid=5)

        scales = searcher.search(
            linears2scale=[module.fc_a, module.fc_b],
            context=ctx,
        )
        assert scales is not None
        assert scales.shape == (in_features,)

    def test_multiple_samples_in_cache(self):
        """Search succeeds with multiple batches in cache."""
        module = SimpleMLP()
        ctx = _build_context(n_samples=4, inspect_module=module)
        searcher = AWQBestScalesSearcher(weight_qconfig=_make_qconfig(), n_grid=5)

        scales = searcher.search(
            linears2scale=[module.fc1],
            context=ctx,
        )
        assert scales is not None
        assert torch.isfinite(scales).all()
