#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for TrainableLinearQuantWrapper and its pure helpers."""

# pylint: disable=unnecessary-lambda-assignment

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.core.wrapper import (
    TrainableLinearQuantWrapper,
    _apply_pre_hook_to_tensor,
    _needs_act_fake_quant,
)


def _qconfig(dtype: QDType, scope: QScope, method: str = "minmax") -> QConfig:
    return QConfig(dtype=dtype, scope=scope, symmetric=True, method=method)


def _int8_qconfig(act_dtype: QDType = QDType.FLOAT) -> LinearQConfig:
    return LinearQConfig(
        act=_qconfig(act_dtype, QScope.PER_TOKEN, "none" if act_dtype == QDType.FLOAT else "minmax"),
        weight=_qconfig(QDType.INT8, QScope.PER_CHANNEL),
    )


class TestWrapperHelpers(unittest.TestCase):
    def test_needs_act_fake_quant_excludes_float_and_none(self):
        self.assertFalse(_needs_act_fake_quant(_qconfig(QDType.FLOAT, QScope.PER_TENSOR, "none")))
        self.assertFalse(_needs_act_fake_quant(_qconfig(QDType.INT8, QScope.PER_TENSOR, "none")))
        self.assertTrue(_needs_act_fake_quant(_qconfig(QDType.INT8, QScope.PER_TENSOR, "minmax")))

    def test_apply_pre_hook_returns_tensor(self):
        module = nn.Identity()
        x = torch.tensor([1.0, 2.0])
        self.assertTrue(torch.equal(_apply_pre_hook_to_tensor(lambda m, args: args[0] + 1, module, x), x + 1))
        self.assertTrue(torch.equal(_apply_pre_hook_to_tensor(lambda m, args: (args[0] + 2,), module, x), x + 2))

    def test_apply_pre_hook_none_keeps_input(self):
        module = nn.Identity()
        x = torch.tensor([3.0])
        self.assertIs(_apply_pre_hook_to_tensor(lambda m, args: None, module, x), x)


class TestTrainableLinearQuantWrapper(unittest.TestCase):
    def test_act_kernel_is_created_only_when_requested(self):
        layer = nn.Linear(4, 3)
        wrapper = TrainableLinearQuantWrapper(layer, _int8_qconfig(QDType.INT8), train_with_act_quant=True)
        self.assertIsNotNone(wrapper.w_kernel)
        self.assertIsNotNone(wrapper.x_kernel)

        plain = TrainableLinearQuantWrapper(layer, _int8_qconfig(), train_with_act_quant=True)
        self.assertIsNone(plain.x_kernel)

    def test_register_train_op_deduplicates_and_respects_side(self):
        wrapper = TrainableLinearQuantWrapper(nn.Linear(4, 3), _int8_qconfig())
        act_param = torch.tensor([1.0])
        weight_param = torch.tensor([2.0])
        act_op = MagicMock()
        act_op.train_params = {"act": act_param}
        weight_op = MagicMock()
        weight_op.train_params = {"weight": weight_param}

        wrapper.register_train_op(act_op, side="act")
        wrapper.register_train_op(act_op, side="act")
        wrapper.register_train_op(weight_op, side="weight")
        wrapper.register_train_op(weight_op, side="weight")

        self.assertEqual(wrapper._collect_train_params("act"), {"act": act_param})
        self.assertEqual(wrapper._collect_train_params("weight"), {"weight": weight_param})

    def test_forward_applies_pre_hook_and_transform(self):
        layer = nn.Linear(4, 3, bias=False)
        wrapper = TrainableLinearQuantWrapper(layer, _int8_qconfig())
        layer.register_forward_pre_hook(lambda module, inputs: (inputs[0] + 1.0,))
        seen = []

        def transform(x):
            seen.append(x)
            return x * 2.0

        wrapper.register_forward_act_transform(transform)
        wrapper.register_forward_act_transform(transform)
        out = wrapper(torch.ones(2, 4))
        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0], torch.full((2, 4), 2.0)))

    def test_unregister_transform_is_idempotent(self):
        wrapper = TrainableLinearQuantWrapper(nn.Linear(4, 3), _int8_qconfig())
        fn = lambda x: x  # noqa: E731
        wrapper.register_forward_act_transform(fn)
        wrapper.unregister_forward_act_transform(fn)
        wrapper.unregister_forward_act_transform(fn)
        self.assertEqual(wrapper._forward_act_transforms, [])

    def test_unwrapper_writes_quant_metadata(self):
        layer = nn.Linear(4, 3)
        config = _int8_qconfig()
        wrapper = TrainableLinearQuantWrapper(layer, config)

        result = wrapper.unwrapper()

        self.assertIs(result, layer)
        self.assertEqual(tuple(layer.scale.shape), (3,))
        self.assertEqual(tuple(layer.zp.shape), (3,))
        self.assertTrue(torch.equal(layer.zp, torch.zeros(3)))
        self.assertIs(layer.weight_qconfig, config.weight)
        self.assertIs(layer.act_qconfig, config.act)


if __name__ == "__main__":
    unittest.main()
