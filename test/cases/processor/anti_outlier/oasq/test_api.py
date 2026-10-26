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

OASQ API 单元测试：各子图类型分发、shift 约束与 NonFusion 返回值。
"""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.ir.norm_bias import RMSNormBias
from msmodelslim.processor.anti_outlier.common import (
    OASQConfig,
    OASQContext,
    SubgraphFusionFactory,
)
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
)
from msmodelslim.processor.anti_outlier.oasq.api import (
    _oasq_calculator,
    _require_shift_tensor,
    oasq,
)


def _make_context(hidden: int, with_shift: bool = True) -> OASQContext:
    return OASQContext(
        version=1,
        a_smooth_scale=torch.ones(hidden),
        shift=torch.zeros(hidden) if with_shift else None,
    )


class TestOasqCalculatorFactory(unittest.TestCase):
    """场景：根据 OASQConfig.max_iters 构造 calculator。"""

    def test_oasq_calculator_when_max_iters_none_then_uses_default_calculator(self):
        """给定 max_iters=None，期望返回默认 OASQScaleCalculator（max_iters=8）。"""
        calc = _oasq_calculator(OASQConfig(max_iters=None))
        self.assertEqual(calc.max_iters, 8)

    def test_oasq_calculator_when_max_iters_set_then_propagates_to_calculator(self):
        """给定 max_iters=3，期望 calculator.max_iters 为 3。"""
        calc = _oasq_calculator(OASQConfig(max_iters=3))
        self.assertEqual(calc.max_iters, 3)


class TestRequireShiftTensor(unittest.TestCase):
    """场景：asymmetric 路径要求 context.shift 存在。"""

    def test_require_shift_tensor_when_shift_missing_then_raises_value_error(self):
        """给定 shift=True 但 context.shift=None，期望抛出 ValueError。"""
        cfg = OASQConfig(shift=True, version=1)
        ctx = _make_context(8, with_shift=False)
        with self.assertRaises(ValueError) as err:
            _require_shift_tensor(cfg, ctx)
        self.assertIn("context.shift", str(err.exception))

    def test_require_shift_tensor_when_shift_present_then_returns_tensor(self):
        """给定 context.shift 存在，期望原样返回该张量。"""
        cfg = OASQConfig(shift=True, version=1)
        ctx = _make_context(8, with_shift=True)
        out = _require_shift_tensor(cfg, ctx)
        self.assertIs(out, ctx.shift)


class TestOasqApiNormLinear(unittest.TestCase):
    """场景：norm-linear 子图 OASQ。"""

    def setUp(self):
        self.hidden = 16
        self.norm = RMSNormBias(self.hidden)
        self.linear = nn.Linear(self.hidden, self.hidden)
        self.subgraph = NormLinearSubgraph(norm=self.norm, linears=[self.linear])
        self.context = _make_context(self.hidden)

    def test_oasq_norm_linear_when_symmetric_then_applies_fusion_and_returns_none(self):
        """给定 symmetric（shift=False），期望调用融合写回且 API 返回 None。"""
        cfg = OASQConfig(max_iters=2, shift=False, version=1)
        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            result = oasq(self.subgraph, cfg, self.context)
            mock_fusion.assert_called_once()
            self.assertIsNone(result)
            kwargs = mock_fusion.call_args.kwargs
            self.assertIn("scales", kwargs["scales"])
            self.assertIsNone(kwargs.get("shifts"))

    def test_oasq_norm_linear_when_shift_true_then_passes_linear_and_norm_shifts(self):
        """给定 shift=True，期望融合参数包含 linear_shifts 与 norm_shift。"""
        cfg = OASQConfig(max_iters=2, shift=True, version=1)
        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            oasq(self.subgraph, cfg, self.context)
            shifts = mock_fusion.call_args.kwargs["shifts"]
            self.assertIn("linear_shifts", shifts)
            self.assertIn("norm_shift", shifts)
            self.assertEqual(len(shifts["linear_shifts"]), 1)


class TestOasqApiLinearLinear(unittest.TestCase):
    """场景：linear-linear 子图 OASQ。"""

    def setUp(self):
        self.hidden = 16
        self.linear1 = nn.Linear(self.hidden, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.hidden)
        self.subgraph = LinearLinearSubgraph(linear1=self.linear1, linear2=self.linear2)
        self.context = _make_context(self.hidden)

    def test_oasq_linear_linear_when_shift_false_then_applies_fusion_without_shifts(self):
        """给定 shift=False，期望融合写回且 shifts 为 None。"""
        cfg = OASQConfig(max_iters=2, shift=False, version=1)
        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            self.assertIsNone(oasq(self.subgraph, cfg, self.context))
            self.assertIsNone(mock_fusion.call_args.kwargs.get("shifts"))

    def test_oasq_linear_linear_when_shift_true_then_includes_linear1_and_linear2_shifts(self):
        """给定 shift=True，期望 shifts 含 linear1_shift / linear2_shift。"""
        cfg = OASQConfig(max_iters=2, shift=True, version=1)
        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            oasq(self.subgraph, cfg, self.context)
            shifts = mock_fusion.call_args.kwargs["shifts"]
            self.assertIn("linear1_shift", shifts)
            self.assertIn("linear2_shift", shifts)


class TestOasqApiUpDown(unittest.TestCase):
    """场景：up-down 子图 OASQ。"""

    def setUp(self):
        self.hidden = 16
        self.up = nn.Linear(self.hidden, self.hidden * 2)
        self.down = nn.Linear(self.hidden * 2, self.hidden)
        self.gate = nn.Linear(self.hidden, self.hidden * 2)
        self.subgraph = UpDownSubgraph(up_proj=self.up, down_proj=self.down, gate_proj=self.gate)
        # down_proj 输入维是 hidden*2，a_smooth_scale 应对齐 down 的 in_features
        self.context = OASQContext(
            version=1,
            a_smooth_scale=torch.ones(self.hidden * 2),
            shift=torch.zeros(self.hidden * 2),
        )

    def test_oasq_up_down_when_shift_false_then_applies_fusion_and_returns_none(self):
        """给定 up-down + shift=False，期望融合成功且返回 None。"""
        cfg = OASQConfig(max_iters=2, shift=False, version=1)
        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            self.assertIsNone(oasq(self.subgraph, cfg, self.context))
            mock_fusion.assert_called_once()

    def test_oasq_up_down_when_shift_true_then_includes_up_and_down_shifts(self):
        """给定 shift=True，期望 shifts 含 up_shift / down_shift。"""
        cfg = OASQConfig(max_iters=2, shift=True, version=1)
        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            oasq(self.subgraph, cfg, self.context)
            shifts = mock_fusion.call_args.kwargs["shifts"]
            self.assertIn("up_shift", shifts)
            self.assertIn("down_shift", shifts)


class TestOasqApiOv(unittest.TestCase):
    """场景：OV 子图 OASQ。"""

    def setUp(self):
        self.hidden = 64
        self.n_heads = 8
        self.o_proj = nn.Linear(self.hidden, self.hidden)
        self.v_proj = nn.Linear(self.hidden, self.hidden)
        self.subgraph = OVSubgraph(
            o_proj=self.o_proj,
            v_proj=self.v_proj,
            num_attention_heads=self.n_heads,
            key_value_heads=self.n_heads,
        )
        self.context = _make_context(self.hidden)

    def test_oasq_ov_when_shift_false_then_applies_fusion_with_o_and_v_scales(self):
        """给定 OV + shift=False，期望融合参数含 o_scales / v_scales。"""
        cfg = OASQConfig(max_iters=2, shift=False, version=1)
        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            self.assertIsNone(oasq(self.subgraph, cfg, self.context))
            scales = mock_fusion.call_args.kwargs["scales"]
            self.assertIn("o_scales", scales)
            self.assertIn("v_scales", scales)

    def test_oasq_ov_when_shift_true_then_includes_o_and_v_shifts(self):
        """给定 OV + shift=True，期望 shifts 含 o_shift / v_shift。"""
        cfg = OASQConfig(max_iters=2, shift=True, version=1)
        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            oasq(self.subgraph, cfg, self.context)
            shifts = mock_fusion.call_args.kwargs["shifts"]
            self.assertIn("o_shift", shifts)
            self.assertIn("v_shift", shifts)


class TestOasqApiNonFusion(unittest.TestCase):
    """场景：NonFusion 子图 OASQ。"""

    def test_oasq_non_fusion_when_empty_linears_then_raises_value_error(self):
        """给定空 linears，期望抛出 ValueError。"""
        subgraph = NonFusionSubgraph(linears=[])
        cfg = OASQConfig(max_iters=2, shift=False, version=1)
        ctx = _make_context(8)
        with self.assertRaises(ValueError) as err:
            oasq(subgraph, cfg, ctx)
        self.assertIn("at least one linear", str(err.exception))

    def test_oasq_non_fusion_when_single_linear_then_returns_scales_tensor(self):
        """给定单个 linear，期望返回 scales 张量并触发融合。"""
        hidden = 8
        linear = nn.Linear(hidden, 4)
        subgraph = NonFusionSubgraph(linears=[linear])
        cfg = OASQConfig(max_iters=2, shift=False, version=1)
        ctx = _make_context(hidden)
        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            scales = oasq(subgraph, cfg, ctx)
            mock_fusion.assert_called_once()
        self.assertIsInstance(scales, torch.Tensor)
        self.assertEqual(tuple(scales.shape), (hidden,))

    def test_oasq_non_fusion_when_shift_true_then_ignores_shifts_and_still_returns_scales(self):
        """给定 NonFusion + shift=True，期望忽略 shift 仍返回 scales（并打 warning）。"""
        hidden = 8
        linear = nn.Linear(hidden, 4)
        subgraph = NonFusionSubgraph(linears=[linear])
        cfg = OASQConfig(max_iters=2, shift=True, version=1)
        ctx = _make_context(hidden, with_shift=True)
        with patch.object(SubgraphFusionFactory, "apply_fusion_to_subgraph") as mock_fusion:
            scales = oasq(subgraph, cfg, ctx)
            # NonFusion 忽略 shift：传给 fusion 的 shifts 为空 dict 时被转为 None
            self.assertIsNone(mock_fusion.call_args.kwargs.get("shifts"))
        self.assertIsInstance(scales, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
