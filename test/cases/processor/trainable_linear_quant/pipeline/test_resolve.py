#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for pipeline resolve: strategy matching and per-layer qconfig resolution."""

# pylint: disable=abstract-method

import unittest

from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.config import QuantStrategyConfig
from msmodelslim.processor.trainable_linear_quant.pipeline.resolve import (
    SUPPORTED_LAYER_TYPES,
    StrategyResolver,
    resolve_layer_qconfigs,
)
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


class TestSupportedLayerTypes(unittest.TestCase):
    def test_only_linear_is_supported(self):
        self.assertEqual(SUPPORTED_LAYER_TYPES, (nn.Linear,))


class TestStrategyResolver(unittest.TestCase):
    def test_empty_strategies_raise(self):
        with self.assertRaises(SchemaValidateError):
            StrategyResolver([])

    def test_match_returns_first_matching_qconfig(self):
        first = _int8_qconfig()
        second = _int8_qconfig()
        resolver = StrategyResolver(
            [
                QuantStrategyConfig(qconfig=first, include=["*mlp*"]),
                QuantStrategyConfig(qconfig=second, include=["*"]),
            ]
        )

        self.assertIs(resolver.match("layers.5.mlp.gate_proj"), first)
        self.assertIs(resolver.match("layers.5.self_attn.q_proj"), second)

    def test_match_returns_none_when_nothing_matches(self):
        resolver = StrategyResolver([QuantStrategyConfig(qconfig=_int8_qconfig(), include=["*mlp*"])])
        self.assertIsNone(resolver.match("layers.5.self_attn.q_proj"))

    def test_exclude_wins_over_include(self):
        resolver = StrategyResolver(
            [
                QuantStrategyConfig(
                    qconfig=_int8_qconfig(),
                    include=["*mlp*"],
                    exclude=["*down_proj*"],
                )
            ]
        )

        self.assertIsNotNone(resolver.match("model.layers.2.mlp.up_proj"))
        self.assertIsNone(resolver.match("model.layers.2.mlp.down_proj"))

    def test_warn_unmatched_reports_never_matched_patterns(self):
        resolver = StrategyResolver([QuantStrategyConfig(qconfig=_int8_qconfig(), include=["*never_present*"])])

        with self.assertLogs("msmodelslim", level="WARNING") as logs:
            resolver.warn_unmatched()

        self.assertTrue(any("not matched any module" in message for message in logs.output))


class TestResolveLayerQconfigs(unittest.TestCase):
    def test_resolve_applies_first_matching_strategy(self):
        model = TinyModel()
        layer_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        qconfig = _int8_qconfig()

        layer_qconfigs, supported_count = resolve_layer_qconfigs(
            model,
            [QuantStrategyConfig(qconfig=qconfig, include=["*"])],
        )

        self.assertEqual(supported_count, len(layer_names))
        self.assertEqual(set(layer_qconfigs), set(layer_names))
        for name in layer_names:
            self.assertIs(layer_qconfigs[name], qconfig)

    def test_resolve_raises_when_no_layer_matched(self):
        model = TinyModel()

        with self.assertRaises(SchemaValidateError) as err:
            resolve_layer_qconfigs(
                model,
                [QuantStrategyConfig(qconfig=_int8_qconfig(), include=["nonexistent_layer_*"])],
            )

        self.assertIn("No supported linear layer matched", str(err.exception))

    def test_resolve_warns_on_conflict_and_keeps_first(self):
        model = TinyModel()
        first = _int8_qconfig()
        second = _int8_qconfig()

        with self.assertLogs("msmodelslim", level="WARNING") as logs:
            layer_qconfigs, _ = resolve_layer_qconfigs(
                model,
                [
                    QuantStrategyConfig(qconfig=first, include=["*"]),
                    QuantStrategyConfig(qconfig=second, include=["*"]),
                ],
            )

        self.assertTrue(any("configuration already exists" in message for message in logs.output))
        for qconfig in layer_qconfigs.values():
            self.assertIs(qconfig, first)

    def test_exclude_prevents_layer_match(self):
        model = TinyModel()
        excluded = next(name for name, module in model.named_modules() if isinstance(module, nn.Linear))

        layer_qconfigs, _ = resolve_layer_qconfigs(
            model,
            [QuantStrategyConfig(qconfig=_int8_qconfig(), include=["*"], exclude=[excluded])],
        )

        self.assertEqual(set(layer_qconfigs), {"block.1"})
        self.assertNotIn(excluded, layer_qconfigs)


if __name__ == "__main__":
    unittest.main()
