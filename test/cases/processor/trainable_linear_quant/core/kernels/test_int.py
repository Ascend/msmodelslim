#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for INT TLQ contexts, bounds, and fake-quant kernel."""

# pylint: disable=no-member

import unittest

import torch

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.core.kernels.int import (
    _default_max_bound,
    _parse_int_ext,
    build_int_context,
    create_int_kernel,
    int_max_bound,
)


def _qconfig(dtype: QDType, scope: QScope = QScope.PER_CHANNEL, **ext) -> QConfig:
    return QConfig(dtype=dtype, scope=scope, symmetric=True, method="minmax", ext=ext)


class TestIntExtParsing(unittest.TestCase):
    def test_default_ext_values(self):
        q_scale_thresh, scale_dtype = _parse_int_ext(_qconfig(QDType.INT8))
        self.assertEqual(q_scale_thresh, 1e-5)
        self.assertIs(scale_dtype, torch.bfloat16)

    def test_override_ext_values(self):
        q_scale_thresh, scale_dtype = _parse_int_ext(_qconfig(QDType.INT8, q_scale_thresh=2e-4, scale_dtype="float32"))
        self.assertEqual(q_scale_thresh, 2e-4)
        self.assertIs(scale_dtype, torch.float32)

    def test_invalid_scale_dtype_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_int_ext(_qconfig(QDType.INT8, scale_dtype="float64"))
        self.assertIn("Unsupported scale dtype", str(ctx.exception))


class TestIntBounds(unittest.TestCase):
    def test_default_bounds_for_supported_dtypes(self):
        self.assertEqual(_default_max_bound(_qconfig(QDType.INT4)), 7)
        self.assertEqual(_default_max_bound(_qconfig(QDType.INT8)), 127)

    def test_int_max_bound_uses_config_dtype(self):
        self.assertEqual(int_max_bound(None, _qconfig(QDType.INT4)), 7)
        self.assertEqual(int_max_bound(None, _qconfig(QDType.INT8)), 127)

    def test_unsupported_dtype_raises_type_error(self):
        with self.assertRaises(TypeError) as ctx:
            _default_max_bound(_qconfig(QDType.FLOAT, QScope.PER_TENSOR))
        self.assertIn("not supported by int fake-quant", str(ctx.exception))


class TestIntContextAndKernel(unittest.TestCase):
    def test_build_context_copies_train_tensors(self):
        train_tensors = {"value": torch.tensor([1.0])}
        ctx = build_int_context(_qconfig(QDType.INT8), torch.zeros(2, 3), train_tensors)
        self.assertIsNot(ctx.train_tensors, train_tensors)
        self.assertIs(ctx.train_tensors["value"], train_tensors["value"])

    def test_int8_kernel_rounds_and_clamps_to_symmetric_range(self):
        config = _qconfig(QDType.INT8)
        kernel = create_int_kernel(config)
        x = torch.tensor([[-1.0, -0.25, 0.0, 0.25, 1.0]])

        result = kernel.fake_quantize(x, return_quantized_weight=True)

        self.assertEqual(result.quantized.tolist(), [[-127.0, -32.0, 0.0, 32.0, 127.0]])
        self.assertEqual(tuple(result.scale.shape), (1,))
        self.assertEqual(result.q_param.scheme.dtype, QDType.INT8)
        self.assertEqual(result.tensor.dtype, torch.float32)

    def test_int4_kernel_uses_int4_bound(self):
        kernel = create_int_kernel(_qconfig(QDType.INT4))
        result = kernel.fake_quantize(torch.tensor([[-1.0, 0.0, 1.0]]), return_quantized_weight=True)
        self.assertEqual(result.quantized.tolist(), [[-7.0, 0.0, 7.0]])


if __name__ == "__main__":
    unittest.main()
