#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for MinmaxTuneOp: scale clamping and INT/MX range tuning."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.core.ops import (
    MinmaxTuneOp,
    MinmaxTuneOpConfig,
)
from msmodelslim.processor.trainable_linear_quant.core.wrapper import (
    TrainableLinearQuantWrapper,
)


def _weight_qconfig(dtype: QDType, scope: QScope, ext=None) -> QConfig:
    return QConfig(dtype=dtype, scope=scope, symmetric=True, method="minmax", ext=ext or {})


def _linear_qconfig(dtype: QDType, scope: QScope, ext=None) -> LinearQConfig:
    return LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=_weight_qconfig(dtype, scope, ext),
    )


def _make_op(qconfig: LinearQConfig) -> MinmaxTuneOp:
    wrapper = TrainableLinearQuantWrapper(nn.Linear(4, 4), qconfig)
    wrapper.layer_path = "block0.fc"
    return MinmaxTuneOp(MinmaxTuneOpConfig(), "block0.fc", wrapper)


class TestMinmaxTuneConfig(unittest.TestCase):
    def test_param_keys_and_config_type(self):
        self.assertEqual(MinmaxTuneOp.MIN_KEY, "min_scale")
        self.assertEqual(MinmaxTuneOp.MAX_KEY, "max_scale")
        self.assertEqual(MinmaxTuneOpConfig().type, "minmax_tune")


class TestClampedScale(unittest.TestCase):
    def test_missing_key_returns_default(self):
        ctx = SimpleNamespace(train_tensors={})
        self.assertEqual(MinmaxTuneOp._clamped_scale(ctx, "max_scale", 0.75), 0.75)

    def test_tensor_is_clamped_without_mutating_source(self):
        raw = torch.tensor([-1.0, 0.5, 2.0])
        ctx = SimpleNamespace(train_tensors={"max_scale": raw})
        out = MinmaxTuneOp._clamped_scale(ctx, "max_scale", 1.0)

        self.assertTrue(torch.equal(out, torch.tensor([0.0, 0.5, 1.0])))
        self.assertTrue(torch.equal(raw, torch.tensor([-1.0, 0.5, 2.0])))

    def test_parameter_is_clamped_in_place(self):
        param = nn.Parameter(torch.tensor([-1.0, 0.5, 2.0]))
        ctx = SimpleNamespace(train_tensors={"max_scale": param})
        out = MinmaxTuneOp._clamped_scale(ctx, "max_scale", 1.0)

        self.assertIs(out, param)
        self.assertTrue(torch.equal(param.data, torch.tensor([0.0, 0.5, 1.0])))

    def test_scalar_value_passes_through(self):
        ctx = SimpleNamespace(train_tensors={"max_scale": 0.25})
        self.assertEqual(MinmaxTuneOp._clamped_scale(ctx, "max_scale", 1.0), 0.25)


class TestMinmaxTuneIntPath(unittest.TestCase):
    def setUp(self):
        self.op = _make_op(_linear_qconfig(QDType.INT8, QScope.PER_CHANNEL))

    def test_bind_creates_min_and_max_scales(self):
        self.op.bind()
        try:
            params = self.op.train_params
            self.assertEqual(set(params), {"min_scale", "max_scale"})
            self.assertTrue(torch.equal(params["min_scale"], torch.ones(4)))
            self.assertTrue(torch.equal(params["max_scale"], torch.ones(4)))
        finally:
            self.op.unbind()

    def test_before_qparam_scales_abs_range(self):
        ctx = SimpleNamespace(
            train_tensors={"min_scale": torch.tensor([0.5]), "max_scale": torch.tensor([2.0])},
            min_val=torch.tensor([-4.0]),
            max_val=torch.tensor([8.0]),
            min_abs=None,
            max_abs=None,
        )

        self.op._on_before_qparam_int(ctx)

        self.assertTrue(torch.equal(ctx.min_abs, torch.tensor([2.0])))
        self.assertTrue(torch.equal(ctx.max_abs, torch.tensor([8.0])))

    def test_before_qparam_is_noop_without_range(self):
        ctx = SimpleNamespace(train_tensors={}, min_val=None, max_val=None, min_abs=None, max_abs=None)

        self.op._on_before_qparam_int(ctx)

        self.assertIsNone(ctx.min_abs)
        self.assertIsNone(ctx.max_abs)

    def test_forward_keeps_shape_and_is_finite(self):
        self.op.bind()
        try:
            wrapper = self.op.target_modules["block0.fc"]
            out = wrapper(torch.randn(2, 4))
            self.assertEqual(tuple(out.shape), (2, 4))
            self.assertTrue(bool(torch.isfinite(out).all()))
        finally:
            self.op.unbind()


class TestMinmaxTuneMxPath(unittest.TestCase):
    def setUp(self):
        self.op = _make_op(_linear_qconfig(QDType.MXFP4, QScope.PER_BLOCK, {"group_size": 32}))

    def test_bind_exposes_only_max_scale(self):
        self.op.bind()
        try:
            self.assertEqual(set(self.op.train_params), {"max_scale"})
        finally:
            self.op.unbind()

    def test_before_qparam_scales_block_max(self):
        ctx = SimpleNamespace(
            block_max=torch.tensor([[4.0], [8.0]]),
            train_tensors={"max_scale": torch.tensor([0.5, 0.25])},
        )

        self.op._on_before_qparam_mx(ctx)

        self.assertTrue(torch.equal(ctx.block_max, torch.tensor([[2.0], [2.0]])))

    def test_before_qparam_is_noop_without_block_max(self):
        ctx = SimpleNamespace(block_max=None, train_tensors={})

        self.op._on_before_qparam_mx(ctx)

        self.assertIsNone(ctx.block_max)


if __name__ == "__main__":
    unittest.main()
