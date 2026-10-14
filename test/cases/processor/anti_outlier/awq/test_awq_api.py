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

Tests for awq() dispatch API.
"""

from typing import Optional

from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.ir.norm_bias import RMSNormBias
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    UpDownSubgraph,
)
from msmodelslim.processor.anti_outlier.awq.api import awq
from msmodelslim.processor.anti_outlier.awq.common import AWQConfig, AWQContext, offload
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
)

IN_FEATURES = 32
HIDDEN = 64


class SimpleBlock(nn.Module):
    """Minimal block used as module2inspect."""

    def __init__(self, in_f: int = IN_FEATURES, out_f: int = HIDDEN):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden_states)


def _make_context(
    in_features: int = IN_FEATURES,
    n_samples: int = 2,
    inspect_module: nn.Module = None,
) -> AWQContext:
    cache = []
    for _ in range(n_samples):
        cache.append(offload({"hidden_states": torch.randn(1, 4, in_features)}))
    if inspect_module is None:
        inspect_module = SimpleBlock(in_features)
    return AWQContext(
        act_mean=torch.rand(in_features).clamp(min=1e-3),
        inspect_module=inspect_module,
        inspect_module_args=cache,
    )


def _mock_searcher(return_value: Optional[torch.Tensor] = None):
    """Create a mock AWQSearcher whose search() returns the given tensor."""
    searcher = MagicMock()
    searcher.search = MagicMock(return_value=return_value)
    return searcher


def _make_config(searcher) -> AWQConfig:
    return AWQConfig(version=1, awq_searcher=searcher)


def _non_trivial_scales(size: int = IN_FEATURES) -> torch.Tensor:
    """Return scales != 1 so fusion actually changes weights."""
    return torch.full((size,), 2.0)


class TestDispatchBySubgraphType:
    def test_dispatch_norm_linear(self):
        norm = RMSNormBias(IN_FEATURES)
        fc1 = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        fc2 = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        block = SimpleBlock(IN_FEATURES, HIDDEN)

        scales = _non_trivial_scales()
        searcher = _mock_searcher(return_value=scales)
        subgraph = NormLinearSubgraph(norm=norm, linears=[fc1, fc2])
        awq(subgraph, _make_config(searcher), _make_context(inspect_module=block))

        searcher.search.assert_called_once()
        call_args = searcher.search.call_args
        assert call_args[0][0] is subgraph.linears  # linears2scale

    def test_dispatch_linear_linear(self):
        linear1 = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        linear2 = nn.Linear(HIDDEN, HIDDEN, bias=False)
        block = SimpleBlock(HIDDEN, HIDDEN)

        scales = _non_trivial_scales(HIDDEN)
        searcher = _mock_searcher(return_value=scales)
        subgraph = LinearLinearSubgraph(linear1=linear1, linear2=linear2)
        awq(subgraph, _make_config(searcher), _make_context(in_features=HIDDEN, inspect_module=block))

        searcher.search.assert_called_once()
        call_args = searcher.search.call_args
        assert call_args[0][0] == [linear2]

    def test_dispatch_ov(self):
        o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        v_proj = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        block = SimpleBlock(HIDDEN, HIDDEN)

        scales = _non_trivial_scales(HIDDEN)
        searcher = _mock_searcher(return_value=scales)
        subgraph = OVSubgraph(
            o_proj=o_proj,
            v_proj=v_proj,
            num_attention_heads=4,
            key_value_heads=4,
        )
        awq(subgraph, _make_config(searcher), _make_context(in_features=HIDDEN, inspect_module=block))

        searcher.search.assert_called_once()
        call_args = searcher.search.call_args
        assert call_args[0][0] == [o_proj]  # linears2scale = [o_proj]

    def test_dispatch_up_down(self):
        up_proj = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        down_proj = nn.Linear(HIDDEN, IN_FEATURES, bias=False)
        gate_proj = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        block = SimpleBlock(HIDDEN, IN_FEATURES)

        scales = _non_trivial_scales(HIDDEN)
        searcher = _mock_searcher(return_value=scales)
        subgraph = UpDownSubgraph(
            up_proj=up_proj,
            down_proj=down_proj,
            gate_proj=gate_proj,
        )
        awq(subgraph, _make_config(searcher), _make_context(in_features=HIDDEN, inspect_module=block))

        searcher.search.assert_called_once()
        call_args = searcher.search.call_args
        assert call_args[0][0] == [down_proj]  # linears2scale = [down_proj]

    def test_dispatch_non_fusion(self):
        fc1 = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        fc2 = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        block = SimpleBlock(IN_FEATURES, HIDDEN)

        scales = _non_trivial_scales()
        searcher = _mock_searcher(return_value=scales)
        subgraph = NonFusionSubgraph(linears=[fc1, fc2])
        awq(subgraph, _make_config(searcher), _make_context(inspect_module=block))

        searcher.search.assert_called_once()
        call_args = searcher.search.call_args
        assert call_args[0][0] is subgraph.linears  # linears2scale


class TestSkipWhenEmptyLinears:
    def test_skip_non_fusion_empty_linears(self):
        searcher = _mock_searcher()
        subgraph = NonFusionSubgraph(linears=[])
        awq(subgraph, _make_config(searcher), _make_context())
        searcher.search.assert_not_called()


class TestSkipWhenSearchReturnsNone:
    def test_no_fusion_norm_linear_when_search_returns_none(self):
        norm = RMSNormBias(IN_FEATURES)
        fc = nn.Linear(IN_FEATURES, HIDDEN, bias=False)

        norm_w_before = norm.weight.data.clone()
        fc_w_before = fc.weight.data.clone()

        searcher = _mock_searcher(return_value=None)
        subgraph = NormLinearSubgraph(norm=norm, linears=[fc])
        awq(subgraph, _make_config(searcher), _make_context())

        assert torch.equal(norm.weight.data, norm_w_before)
        assert torch.equal(fc.weight.data, fc_w_before)

    def test_no_fusion_up_down_when_search_returns_none(self):
        up = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        down = nn.Linear(HIDDEN, IN_FEATURES, bias=False)
        gate = nn.Linear(IN_FEATURES, HIDDEN, bias=False)

        up_w = up.weight.data.clone()
        down_w = down.weight.data.clone()

        searcher = _mock_searcher(return_value=None)
        subgraph = UpDownSubgraph(up_proj=up, down_proj=down, gate_proj=gate)
        awq(subgraph, _make_config(searcher), _make_context(in_features=HIDDEN))

        assert torch.equal(up.weight.data, up_w)
        assert torch.equal(down.weight.data, down_w)

    def test_no_fusion_non_fusion_when_search_returns_none(self):
        fc1 = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        fc2 = nn.Linear(IN_FEATURES, HIDDEN, bias=False)

        fc1_w_before = fc1.weight.data.clone()
        fc2_w_before = fc2.weight.data.clone()

        searcher = _mock_searcher(return_value=None)
        subgraph = NonFusionSubgraph(linears=[fc1, fc2])
        awq(subgraph, _make_config(searcher), _make_context())

        assert torch.equal(fc1.weight.data, fc1_w_before)
        assert torch.equal(fc2.weight.data, fc2_w_before)


class TestFusionApplied:
    def test_norm_linear_weights_modified(self):
        norm = RMSNormBias(IN_FEATURES)
        fc = nn.Linear(IN_FEATURES, HIDDEN, bias=False)

        norm_w_before = norm.weight.data.clone()
        fc_w_before = fc.weight.data.clone()

        scales = _non_trivial_scales()
        searcher = _mock_searcher(return_value=scales)
        subgraph = NormLinearSubgraph(norm=norm, linears=[fc])
        awq(subgraph, _make_config(searcher), _make_context())

        assert not torch.equal(norm.weight.data, norm_w_before), "norm weight should be modified by fusion"
        assert not torch.equal(fc.weight.data, fc_w_before), "linear weight should be modified by fusion"

    def test_ov_weights_modified(self):
        o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        v_proj = nn.Linear(IN_FEATURES, HIDDEN, bias=False)

        o_w_before = o_proj.weight.data.clone()
        v_w_before = v_proj.weight.data.clone()

        scales = _non_trivial_scales(HIDDEN)
        searcher = _mock_searcher(return_value=scales)
        subgraph = OVSubgraph(
            o_proj=o_proj,
            v_proj=v_proj,
            num_attention_heads=4,
            key_value_heads=4,
        )
        awq(subgraph, _make_config(searcher), _make_context(in_features=HIDDEN))

        assert not torch.equal(o_proj.weight.data, o_w_before), "o_proj weight should be modified"
        assert not torch.equal(v_proj.weight.data, v_w_before), "v_proj weight should be modified"

    def test_non_fusion_weights_modified(self):
        fc1 = nn.Linear(IN_FEATURES, HIDDEN, bias=False)
        fc2 = nn.Linear(IN_FEATURES, HIDDEN, bias=False)

        fc1_w_before = fc1.weight.data.clone()
        fc2_w_before = fc2.weight.data.clone()

        scales = _non_trivial_scales()
        searcher = _mock_searcher(return_value=scales)
        subgraph = NonFusionSubgraph(linears=[fc1, fc2])
        awq(subgraph, _make_config(searcher), _make_context())

        assert not torch.equal(fc1.weight.data, fc1_w_before), "fc1 weight should be modified by fusion"
        assert not torch.equal(fc2.weight.data, fc2_w_before), "fc2 weight should be modified by fusion"
