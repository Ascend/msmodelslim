#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for pipeline finalize: scale export, fake-quant creation and block finalize."""

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.core.wrapper import (
    TrainableLinearQuantWrapper,
)
from msmodelslim.processor.trainable_linear_quant.pipeline.finalize import (
    _apply_hook_ir_to_fake_quantizer,
    _export_scale_offset_for_fake_quant,
    _is_mx_scheme,
    _resolve_per_block_size,
    create_fake_quantizer,
    finalize_block,
)
from msmodelslim.processor.trainable_linear_quant.pipeline.runtime import BlockTLQContext


def _int8_weight_qconfig() -> QConfig:
    return QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax")


def _float_act_qconfig() -> QConfig:
    return QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none")


def _mx_weight_qconfig() -> QConfig:
    return QConfig(
        dtype=QDType.MXFP4,
        scope=QScope.PER_BLOCK,
        symmetric=True,
        method="minmax",
        ext={"group_size": 32},
    )


def _int8_qconfig() -> LinearQConfig:
    return LinearQConfig(act=_float_act_qconfig(), weight=_int8_weight_qconfig())


class TestSchemeHelpers(unittest.TestCase):
    def test_resolve_per_block_size_prefers_mx_block_size(self):
        self.assertEqual(_resolve_per_block_size(_mx_weight_qconfig().to_scheme(), 64), 32)

    def test_resolve_per_block_size_falls_back_to_group_size(self):
        self.assertEqual(_resolve_per_block_size(_int8_weight_qconfig().to_scheme(), 16), 16)

    def test_resolve_per_block_size_returns_minus_one_when_not_applicable(self):
        self.assertEqual(_resolve_per_block_size(_int8_weight_qconfig().to_scheme(), -1), -1)

    def test_is_mx_scheme(self):
        self.assertTrue(_is_mx_scheme(_mx_weight_qconfig().to_scheme()))
        self.assertFalse(_is_mx_scheme(_int8_weight_qconfig().to_scheme()))


class TestExportScaleOffset(unittest.TestCase):
    def test_mx_per_channel_scale_expands_to_block_layout(self):
        scale = torch.ones(8)
        offset = torch.zeros(8)

        out_scale, out_offset = _export_scale_offset_for_fake_quant(
            scale,
            offset,
            out_features=8,
            in_features=64,
            block_size=32,
            is_mx=True,
        )

        self.assertEqual(tuple(out_scale.shape), (8, 2, 1))
        self.assertTrue(torch.equal(out_scale, torch.ones(8, 2, 1)))
        self.assertEqual(tuple(out_offset.shape), (8, 2, 1))

    def test_mx_block_scale_keeps_block_layout(self):
        scale = torch.arange(16, dtype=torch.float32).reshape(8, 2, 1)
        offset = torch.zeros_like(scale)

        out_scale, out_offset = _export_scale_offset_for_fake_quant(
            scale,
            offset,
            out_features=8,
            in_features=64,
            block_size=32,
            is_mx=True,
        )

        self.assertTrue(torch.equal(out_scale, scale))
        self.assertTrue(torch.equal(out_offset, offset))

    def test_non_mx_group_scale_expands_along_blocks(self):
        scale = torch.ones(8)
        offset = torch.zeros(8)

        out_scale, out_offset = _export_scale_offset_for_fake_quant(
            scale,
            offset,
            out_features=8,
            in_features=64,
            block_size=32,
            is_mx=False,
        )

        self.assertEqual(tuple(out_scale.shape), (8, 2))
        self.assertTrue(torch.equal(out_scale, torch.ones(8, 2)))
        self.assertEqual(tuple(out_offset.shape), (8, 2))

    def test_non_mx_channel_scale_squeezes_to_1d(self):
        out_scale, out_offset = _export_scale_offset_for_fake_quant(
            torch.ones(8),
            torch.zeros(8),
            out_features=8,
            in_features=64,
            block_size=-1,
            is_mx=False,
        )

        self.assertEqual(tuple(out_scale.shape), (8,))
        self.assertEqual(tuple(out_offset.shape), (8,))


class TestCreateFakeQuantizer(unittest.TestCase):
    def test_per_channel_scale_is_1d(self):
        layer = nn.Linear(4, 8)
        layer.weight_qconfig = _int8_weight_qconfig()
        layer.act_qconfig = _float_act_qconfig()
        layer.scale = torch.ones(8)
        layer.zp = torch.zeros(8)

        fake = create_fake_quantizer(layer)

        self.assertEqual(tuple(fake.weight_scale.shape), (8,))

    def test_mxfp4_scale_exports_ir_broadcastable_shape(self):
        in_features, out_features, block_size = 64, 8, 32
        n_blocks = in_features // block_size
        layer = nn.Linear(in_features, out_features)
        layer.weight_qconfig = _mx_weight_qconfig()
        layer.act_qconfig = _mx_weight_qconfig()
        layer.scale = torch.ones(out_features, n_blocks, 1)
        layer.zp = torch.zeros(out_features, n_blocks, 1)

        fake = create_fake_quantizer(layer)

        self.assertEqual(tuple(fake.weight_scale.shape), (out_features, n_blocks, 1))
        y = fake(torch.randn(2, in_features))
        self.assertEqual(y.shape, (2, out_features))

    def test_mxfp4_per_channel_init_expands_to_ir_shape(self):
        in_features, out_features, block_size = 64, 8, 32
        n_blocks = in_features // block_size
        layer = nn.Linear(in_features, out_features)
        layer.weight_qconfig = _mx_weight_qconfig()
        layer.act_qconfig = _mx_weight_qconfig()
        layer.scale = torch.ones(out_features)
        layer.zp = torch.zeros(out_features)

        fake = create_fake_quantizer(layer)

        self.assertEqual(tuple(fake.weight_scale.shape), (out_features, n_blocks, 1))
        y = fake(torch.randn(2, in_features))
        self.assertEqual(y.shape, (2, out_features))


class TestApplyHookIr(unittest.TestCase):
    def test_failed_hook_is_skipped_with_warning(self):
        layer = nn.Linear(4, 4)
        bad_hook = MagicMock(spec=qir.HookIR)
        bad_hook.wrapper_module.side_effect = RuntimeError("hook broken")
        layer.register_forward_pre_hook(bad_hook)

        fake = MagicMock()
        result, count = _apply_hook_ir_to_fake_quantizer(layer, fake)

        self.assertIs(result, fake)
        self.assertEqual(count, 0)
        bad_hook.wrapper_module.assert_called_once()

    def test_non_hookir_pre_hook_is_ignored(self):
        layer = nn.Linear(4, 4)
        layer.register_forward_pre_hook(lambda module, inputs: None)

        result, count = _apply_hook_ir_to_fake_quantizer(layer, MagicMock())

        self.assertEqual(count, 0)
        self.assertIsNotNone(result)


class TestFinalizeBlock(unittest.TestCase):
    def test_finalize_replaces_wrapper_with_fake_quant_module(self):
        qconfig = _int8_qconfig()
        orig = nn.Linear(4, 4)
        wrapper = TrainableLinearQuantWrapper(orig, linear_qconfig=qconfig)
        wrapper.layer_path = "block0.fc"
        model = nn.Module()
        model.block0 = nn.Module()
        model.block0.fc = wrapper
        block = model.block0

        op = MagicMock()
        op.load_best_params = MagicMock()
        op.unbind = MagicMock()
        ctx = BlockTLQContext(block_name="block0")
        ctx.ops = [op]

        count = finalize_block("block0", block, ctx, model=model)

        self.assertEqual(count, 1)
        self.assertNotIsInstance(model.block0.fc, TrainableLinearQuantWrapper)
        self.assertIsInstance(model.block0.fc, qir.AutoFakeQuantLinear)
        op.load_best_params.assert_called_once()
        op.unbind.assert_called_once()

    def test_finalize_without_ops_returns_zero(self):
        model = nn.Module()
        model.block0 = nn.Sequential(nn.Linear(4, 4))
        ctx = BlockTLQContext(block_name="block0")

        count = finalize_block("block0", model.block0, ctx, model=model)

        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
