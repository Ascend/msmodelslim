#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for RoundTuneOp: STE rounding, value clamping and the INT round path."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QParam, QScope, QScheme
from msmodelslim.processor.trainable_linear_quant.core.ops import (
    RoundTuneOp,
    RoundTuneOpConfig,
)
from msmodelslim.processor.trainable_linear_quant.core.ops.round_tune import (
    _align_v_to_normed,
    _clamp_v,
    _round_ste,
)
from msmodelslim.processor.trainable_linear_quant.core.wrapper import (
    TrainableLinearQuantWrapper,
)


def _int8_QConfig() -> QConfig:
    return QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax")


def _int8_wrapper() -> TrainableLinearQuantWrapper:
    qconfig = LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=_int8_QConfig(),
    )
    wrapper = TrainableLinearQuantWrapper(nn.Linear(4, 4), qconfig)
    wrapper.layer_path = "block0.fc"
    return wrapper


class TestRoundSte(unittest.TestCase):
    def test_matches_round_and_passes_gradient(self):
        x = torch.tensor([0.4, 1.6, -1.5], requires_grad=True)

        y = _round_ste(x)

        self.assertTrue(torch.equal(y, torch.tensor([0.0, 2.0, -2.0])))
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones(3)))


class TestClampV(unittest.TestCase):
    def test_int4_clamps_to_half(self):
        out = _clamp_v(torch.tensor([-2.0, 0.25, 2.0]), QDType.INT4, 7)
        self.assertTrue(torch.equal(out, torch.tensor([-0.5, 0.25, 0.5])))

    def test_int8_clamps_to_unit(self):
        out = _clamp_v(torch.tensor([-2.0, 0.25, 2.0]), QDType.INT8, 127)
        self.assertTrue(torch.equal(out, torch.tensor([-1.0, 0.25, 1.0])))

    def test_non_tensor_passes_through(self):
        self.assertEqual(_clamp_v(3.5, QDType.INT8, 127), 3.5)

    def test_other_dtype_passes_through(self):
        tensor = torch.tensor([-5.0, 5.0])
        self.assertIs(_clamp_v(tensor, QDType.MXFP4, 7), tensor)


class TestAlignVToNormed(unittest.TestCase):
    def test_matching_shape_returns_original(self):
        v = torch.zeros(2, 3)
        self.assertIs(_align_v_to_normed(v, torch.zeros(2, 3), torch.zeros(2, 3), None), v)

    def test_non_tensor_passes_through(self):
        self.assertEqual(_align_v_to_normed(1.0, torch.zeros(2, 3), torch.zeros(2, 3), None), 1.0)


class TestRoundTuneIntPath(unittest.TestCase):
    def setUp(self):
        self.op = RoundTuneOp(RoundTuneOpConfig(), "block0.fc", _int8_wrapper())

    def _ctx(self, normed, train_tensors):
        qparam = QParam(
            scheme=QScheme(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True),
            ext={"scale": torch.ones(1), "offset": torch.zeros(1)},
        )
        return SimpleNamespace(
            config=_int8_QConfig(),
            q_param=qparam,
            train_tensors=train_tensors,
            normed=normed,
            quantized=None,
        )

    def test_config_and_param_key(self):
        self.assertEqual(RoundTuneOpConfig().type, "round_tune")
        self.assertEqual(RoundTuneOp.PARAM_KEY, "value")

    def test_bind_creates_zero_value_tensor(self):
        self.op.bind()
        try:
            params = self.op.train_params
            self.assertEqual(set(params), {"value"})
            self.assertEqual(tuple(params["value"].shape), (4, 4))
            self.assertTrue(torch.equal(params["value"], torch.zeros(4, 4)))
        finally:
            self.op.unbind()

    def test_pre_round_adds_value_before_rounding(self):
        ctx = self._ctx(torch.tensor([3.4, 3.4]), {"value": torch.tensor([0.2, -0.2])})

        self.op._on_pre_round_int(ctx)

        self.assertTrue(torch.equal(ctx.quantized, torch.tensor([4.0, 3.0])))

    def test_pre_round_without_value_rounds_normed(self):
        ctx = self._ctx(torch.tensor([3.4, 3.4]), {})

        self.op._on_pre_round_int(ctx)

        self.assertTrue(torch.equal(ctx.quantized, torch.tensor([3.0, 3.0])))

    def test_pre_round_clamps_quantized_to_int_range(self):
        ctx = self._ctx(torch.tensor([1000.0, -1000.0]), {})

        self.op._on_pre_round_int(ctx)

        self.assertTrue(torch.equal(ctx.quantized, torch.tensor([127.0, -128.0])))

    def test_pre_round_requires_qparam(self):
        ctx = self._ctx(torch.tensor([1.0]), {})
        ctx.q_param = None
        with self.assertRaises(RuntimeError):
            self.op._on_pre_round_int(ctx)


if __name__ == "__main__":
    unittest.main()
