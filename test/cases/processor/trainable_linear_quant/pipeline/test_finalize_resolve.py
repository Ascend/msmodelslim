#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.config import (
    QuantStrategyConfig,
)
from msmodelslim.processor.trainable_linear_quant.pipeline.finalize import (
    _apply_hook_ir_to_fake_quantizer,
    create_fake_quantizer,
    finalize_block,
)
from msmodelslim.processor.trainable_linear_quant.pipeline.resolve import (
    StrategyResolver,
    resolve_layer_qconfigs,
)
from msmodelslim.processor.trainable_linear_quant.pipeline.runtime import BlockTLQContext
from msmodelslim.utils.exception import SchemaValidateError


def _int8_qconfig() -> LinearQConfig:
    return LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
    )


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


class TestResolveLayerQconfigs(unittest.TestCase):
    def test_resolve_applies_first_matching_strategy(self):
        model = TinyModel()
        layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        self.assertGreaterEqual(len(layer_names), 2)

        qconfig = _int8_qconfig()
        layer_qconfigs, _ = resolve_layer_qconfigs(
            model,
            [QuantStrategyConfig(qconfig=qconfig, include=["*"])],
        )
        for name in layer_names:
            self.assertIn(name, layer_qconfigs)
            self.assertIs(layer_qconfigs[name], qconfig)

    def test_resolve_raises_when_no_layer_matched(self):
        model = TinyModel()
        qconfig = _int8_qconfig()
        with self.assertRaises(SchemaValidateError) as ctx:
            resolve_layer_qconfigs(
                model,
                [QuantStrategyConfig(qconfig=qconfig, include=["nonexistent_layer_*"])],
            )
        self.assertIn("No supported linear layer matched", str(ctx.exception))

    def test_resolve_warns_on_strategy_conflict(self):
        model = TinyModel()
        qconfig1 = _int8_qconfig()
        qconfig2 = _int8_qconfig()
        with self.assertLogs("msmodelslim", level="WARNING") as logs:
            layer_qconfigs, _ = resolve_layer_qconfigs(
                model,
                [
                    QuantStrategyConfig(qconfig=qconfig1, include=["*"]),
                    QuantStrategyConfig(qconfig=qconfig2, include=["*"]),
                ],
            )
        self.assertGreater(len(layer_qconfigs), 0)
        self.assertTrue(
            any("configuration already exists" in msg for msg in logs.output),
            logs.output,
        )
        for qc in layer_qconfigs.values():
            self.assertIs(qc, qconfig1)

    def test_exclude_prevents_layer_match(self):
        model = TinyModel()
        qconfig = _int8_qconfig()
        target = next(n for n, m in model.named_modules() if isinstance(m, nn.Linear))
        layer_qconfigs, _ = resolve_layer_qconfigs(
            model,
            [QuantStrategyConfig(qconfig=qconfig, include=["*"], exclude=[target])],
        )
        self.assertNotIn(target, layer_qconfigs)

    def test_strategy_resolver_matches_after_module_added(self):
        """Simulates layer-wise load: modules appear after StrategyResolver is built."""
        resolver = StrategyResolver([QuantStrategyConfig(qconfig=_int8_qconfig(), include=["*mlp*"], exclude=[])])
        self.assertIsNone(resolver.match("layers.5.self_attn.q_proj"))
        self.assertIsNotNone(resolver.match("layers.5.mlp.gate_proj"))

    def test_strategy_resolver_respects_exclude(self):
        resolver = StrategyResolver(
            [
                QuantStrategyConfig(
                    qconfig=_int8_qconfig(),
                    include=["*mlp*"],
                    exclude=["*mlp.down_proj*"],
                )
            ]
        )
        self.assertIsNotNone(resolver.match("model.layers.2.mlp.up_proj"))
        self.assertIsNone(resolver.match("model.layers.2.mlp.down_proj"))


class TestFinalizeBlock(unittest.TestCase):
    def test_finalize_replaces_wrapper_with_fake_quant_module(self):
        from msmodelslim.processor.trainable_linear_quant.core.wrapper import (
            TrainableLinearQuantWrapper,
        )

        qconfig = _int8_qconfig()
        orig = nn.Linear(4, 4)
        wrapper = TrainableLinearQuantWrapper(orig, linear_qconfig=qconfig)
        wrapper.layer_path = "block0.fc"
        model = nn.Module()
        model.block0 = nn.Module()
        model.block0.fc = wrapper
        block = model.block0

        op = MagicMock()
        op.best_params = {"w": torch.ones(1)}
        op.train_params = {}
        op.load_best_params = MagicMock()
        op.unbind = MagicMock()

        ctx = BlockTLQContext(block_name="block0")
        ctx.ops = [op]

        count = finalize_block("block0", block, ctx, model=model)
        self.assertEqual(count, 1)
        self.assertNotIsInstance(model.block0.fc, TrainableLinearQuantWrapper)
        op.load_best_params.assert_called_once()
        op.unbind.assert_called_once()


class TestCreateFakeQuantizer(unittest.TestCase):
    def test_per_channel_scale_is_1d(self):
        layer = nn.Linear(4, 8)
        layer.weight_qconfig = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_CHANNEL,
            symmetric=True,
            method="minmax",
        )
        layer.act_qconfig = QConfig(
            dtype=QDType.FLOAT,
            scope=QScope.PER_TENSOR,
            symmetric=True,
            method="none",
        )
        layer.scale = torch.ones(8)
        layer.zp = torch.zeros(8)
        fake = create_fake_quantizer(layer)
        self.assertIsNotNone(fake)

    def test_mxfp4_scale_exports_ir_broadcastable_shape(self):
        """TLQ MX scale must be [out, n_blocks, 1] to broadcast with IR reshape_to_blocks."""
        in_features, out_features, block_size = 64, 8, 32
        n_blocks = in_features // block_size
        layer = nn.Linear(in_features, out_features)
        layer.weight_qconfig = QConfig(
            dtype=QDType.MXFP4,
            scope=QScope.PER_BLOCK,
            symmetric=True,
            method="minmax",
        )
        layer.act_qconfig = QConfig(
            dtype=QDType.MXFP4,
            scope=QScope.PER_BLOCK,
            symmetric=True,
            method="minmax",
        )
        # Simulate TLQ training output (keepdim on block dim) then flattened mid-shape.
        layer.scale = torch.ones(out_features, n_blocks, 1)
        layer.zp = torch.zeros(out_features, n_blocks, 1)
        fake = create_fake_quantizer(layer)
        self.assertEqual(tuple(fake.weight_scale.shape), (out_features, n_blocks, 1))

        # Forward must not raise on ndim mismatch (no IR-side squeeze shim).
        x = torch.randn(2, in_features)
        y = fake(x)
        self.assertEqual(y.shape, (2, out_features))

    def test_mxfp4_per_channel_init_expands_to_ir_shape(self):
        in_features, out_features, block_size = 64, 8, 32
        n_blocks = in_features // block_size
        layer = nn.Linear(in_features, out_features)
        layer.weight_qconfig = QConfig(
            dtype=QDType.MXFP4,
            scope=QScope.PER_BLOCK,
            symmetric=True,
            method="minmax",
        )
        layer.act_qconfig = QConfig(
            dtype=QDType.MXFP4,
            scope=QScope.PER_BLOCK,
            symmetric=True,
            method="minmax",
        )
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


if __name__ == "__main__":
    unittest.main()
