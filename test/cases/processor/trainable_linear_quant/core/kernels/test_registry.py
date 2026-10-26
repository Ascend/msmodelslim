#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for TLQ kernel dispatch registration."""

import unittest

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.core.kernels import int as int_kernel
from msmodelslim.processor.trainable_linear_quant.core.kernels.registry import (
    create_tlq_kernel,
    ensure_tlq_kernel_registered,
    register_tlq_kernel,
    registered_tlq_dispatch_keys,
    tlq_kernel_dispatch_key,
)
from msmodelslim.utils.exception import UnsupportedError


def _qconfig(dtype: QDType, scope: QScope) -> QConfig:
    return QConfig(dtype=dtype, scope=scope, symmetric=True, method="minmax")


class TestDispatchKey(unittest.TestCase):
    def test_key_uses_config_dtype_for_storage(self):
        key = tlq_kernel_dispatch_key(_qconfig(QDType.INT8, QScope.PER_CHANNEL))
        self.assertEqual(key, (QDType.INT8, QDType.INT8, QScope.PER_CHANNEL, True))

    def test_registered_keys_include_int_and_mxfp(self):
        keys = registered_tlq_dispatch_keys()
        self.assertIn((QDType.INT8, QDType.INT8, QScope.PER_CHANNEL, True), keys)
        self.assertIn((QDType.MXFP4, QDType.MXFP4, QScope.PER_BLOCK, True), keys)


class TestRegistration(unittest.TestCase):
    def test_register_without_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            register_tlq_kernel()
        self.assertIn("at least one dispatch_key", str(ctx.exception))

    def test_malformed_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            register_tlq_kernel((QDType.INT8, QScope.PER_CHANNEL, True))(lambda config: None)
        self.assertIn("must be (q_dtype, storage_dtype, scope, symmetric)", str(ctx.exception))

    def test_duplicate_key_with_different_factory_raises(self):
        key = (QDType.INT8, QDType.INT8, QScope.PER_CHANNEL, True)
        with self.assertRaises(ValueError) as ctx:
            register_tlq_kernel(key)(lambda config: None)
        self.assertIn("already registered", str(ctx.exception))

    def test_re_registering_same_factory_is_idempotent(self):
        key = (QDType.INT8, QDType.INT8, QScope.PER_CHANNEL, True)
        registered = register_tlq_kernel(key)(int_kernel.create_int_kernel)
        self.assertIs(registered, int_kernel.create_int_kernel)


class TestCreateAndEnsureKernel(unittest.TestCase):
    def test_create_returns_registered_kernel(self):
        kernel = create_tlq_kernel(_qconfig(QDType.INT8, QScope.PER_CHANNEL))
        self.assertIs(kernel.config.dtype, QDType.INT8)

    def test_create_unregistered_raises_unsupported(self):
        with self.assertRaises(UnsupportedError) as ctx:
            create_tlq_kernel(_qconfig(QDType.INT8, QScope.PER_TENSOR))
        self.assertIn("No TLQ kernel for dispatch key", str(ctx.exception))

    def test_ensure_registered_returns_none_when_act_role_supported(self):
        self.assertIsNone(ensure_tlq_kernel_registered(_qconfig(QDType.INT4, QScope.PER_TOKEN), role="act"))

    def test_ensure_registered_raises_for_unsupported_role(self):
        with self.assertRaises(UnsupportedError) as ctx:
            ensure_tlq_kernel_registered(_qconfig(QDType.INT4, QScope.PER_TENSOR), role="weight")
        self.assertIn("No TLQ kernel for weight config", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
