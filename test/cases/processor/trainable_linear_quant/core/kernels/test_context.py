#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for shared TLQ kernel context helpers."""

import unittest

import torch

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QParam, QScheme, QScope
from msmodelslim.processor.trainable_linear_quant.core.kernels.context import (
    FakeQuantPipelineContext,
    FakeQuantResult,
    kernel_family_from_config,
)


def _qconfig(dtype: QDType, scope: QScope = QScope.PER_TENSOR) -> QConfig:
    return QConfig(dtype=dtype, scope=scope, symmetric=True, method="minmax")


class TestKernelFamily(unittest.TestCase):
    def test_int_dtypes_map_to_int_family(self):
        self.assertEqual(kernel_family_from_config(_qconfig(QDType.INT4)), "int")
        self.assertEqual(kernel_family_from_config(_qconfig(QDType.INT8)), "int")

    def test_mx_dtypes_map_to_mxfp_family(self):
        self.assertEqual(kernel_family_from_config(_qconfig(QDType.MXFP4, QScope.PER_BLOCK)), "mxfp")
        self.assertEqual(kernel_family_from_config(_qconfig(QDType.MXFP8, QScope.PER_BLOCK)), "mxfp")

    def test_unsupported_dtype_raises_type_error(self):
        with self.assertRaises(TypeError) as ctx:
            kernel_family_from_config(_qconfig(QDType.FLOAT))
        self.assertIn("unsupported TLQ kernel dtype", str(ctx.exception))


class TestFakeQuantPipelineContext(unittest.TestCase):
    def test_working_tensor_can_be_replaced_by_listener(self):
        config = _qconfig(QDType.INT8)
        original = torch.tensor([1.0, 2.0])
        ctx = FakeQuantPipelineContext(config=config, float_tensor=original)
        self.assertIs(ctx.working_tensor, original)

        replacement = torch.tensor([3.0, 4.0])
        ctx.set_working_tensor(replacement)

        self.assertIs(ctx.working_tensor, replacement)
        self.assertEqual(ctx.normed, None)
        self.assertEqual(ctx.train_tensors, {})

    def test_train_tensors_are_kept_as_given(self):
        value = torch.tensor([5.0])
        ctx = FakeQuantPipelineContext(
            config=_qconfig(QDType.INT8),
            float_tensor=torch.zeros(1),
            train_tensors={"value": value},
        )
        self.assertIs(ctx.train_tensors["value"], value)


class TestFakeQuantResult(unittest.TestCase):
    def test_scale_and_offset_properties(self):
        scale = torch.tensor([0.25, 0.5])
        offset = torch.tensor([1.0, -1.0])
        q_param = QParam(
            scheme=QScheme(QScope.PER_CHANNEL, QDType.INT8, True),
            ext={"scale": scale, "offset": offset},
        )
        result = FakeQuantResult(tensor=torch.ones(2), q_param=q_param)
        self.assertTrue(torch.equal(result.scale, scale))
        self.assertTrue(torch.equal(result.offset, offset))

    def test_missing_offset_returns_zeros(self):
        scale = torch.tensor([0.25, 0.5])
        q_param = QParam(
            scheme=QScheme(QScope.PER_CHANNEL, QDType.INT8, True),
            ext={"scale": scale},
        )
        result = FakeQuantResult(tensor=torch.ones(2), q_param=q_param)
        self.assertEqual(result.offset.tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
