#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from unittest.mock import MagicMock

from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.core.ops import (
    MinmaxTuneOpConfig,
    RoundTuneOpConfig,
)
from msmodelslim.processor.trainable_linear_quant.core.wrapper import (
    TrainableLinearQuantWrapper,
)
from msmodelslim.processor.trainable_linear_quant.pipeline.runtime import BlockTLQContext
from msmodelslim.processor.trainable_linear_quant.pipeline.setup import (
    BlockSetup,
    OpInstallReporter,
)
from msmodelslim.utils.exception import UnsupportedError


def _int8_qconfig() -> LinearQConfig:
    weight = QConfig(
        dtype=QDType.INT8,
        scope=QScope.PER_CHANNEL,
        symmetric=True,
        method="minmax",
    )
    act = QConfig(
        dtype=QDType.FLOAT,
        scope=QScope.PER_TENSOR,
        symmetric=True,
        method="none",
    )
    return LinearQConfig(act=act, weight=weight)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)


class TestBlockLinearLayerPaths(unittest.TestCase):
    def test_block_prefix_scan_matches_strategy_resolver_paths(self):
        model = nn.Module()
        model.block0 = TinyBlock()
        block_name = "block0"
        resolver_paths = {
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name.startswith(f"{block_name}.")
        }
        wrap_paths = {
            name for name, module in model.block0.named_modules(prefix=block_name) if isinstance(module, nn.Linear)
        }
        self.assertEqual(wrap_paths, resolver_paths)


class TestOpInstallReporter(unittest.TestCase):
    def test_finish_returns_ops_and_tracks_counts(self):
        reporter = OpInstallReporter()
        ops = [MagicMock()]
        reporter.record_installed()
        reporter.record_skip("round_tune:linear:fc", "missing wrapper")
        result = reporter.finish(ops, block_name="block0")
        self.assertIs(result, ops)
        self.assertEqual(reporter.installed, 1)
        self.assertEqual(reporter.skipped, 1)

    def test_finish_raises_when_all_installs_skipped(self):
        reporter = OpInstallReporter()
        reporter.record_skip("round_tune:linear:fc", "dtype unsupported")
        with self.assertRaises(UnsupportedError) as ctx:
            reporter.finish([], block_name="block0")
        self.assertIn("all TLQ op installs failed", str(ctx.exception))
        self.assertIn("block0", str(ctx.exception))

    def test_finish_allows_empty_ops_when_nothing_was_attempted(self):
        reporter = OpInstallReporter()
        self.assertEqual(reporter.finish([], block_name="block0"), [])


class TestBlockSetup(unittest.TestCase):
    def setUp(self):
        self.model = nn.Module()
        self.model.block0 = TinyBlock()
        self.block = self.model.block0
        self.qconfig = _int8_qconfig()
        self.layer_qconfigs = {"block0.fc": self.qconfig}
        self.operations = [MinmaxTuneOpConfig(), RoundTuneOpConfig()]

    def _make_setup(self, **kwargs) -> BlockSetup:
        return BlockSetup(
            model=self.model,
            operation_configs=self.operations,
            layer_qconfigs=self.layer_qconfigs,
            **kwargs,
        )

    def test_wrap_linears_replaces_linear_with_wrapper(self):
        setup = self._make_setup()
        wrappers = setup.wrap_linears("block0", self.block)
        self.assertIn("block0.fc", wrappers)
        self.assertIsInstance(self.block.fc, TrainableLinearQuantWrapper)

    def test_wrap_linears_skips_layers_without_qconfig(self):
        setup = BlockSetup(
            model=self.model,
            operation_configs=self.operations,
            layer_qconfigs={},
        )
        wrappers = setup.wrap_linears("block0", self.block)
        self.assertEqual(wrappers, {})
        self.assertIsInstance(self.block.fc, nn.Linear)

    def test_wrap_linears_lazy_matches_strategy_resolver(self):
        from msmodelslim.processor.trainable_linear_quant.config import QuantStrategyConfig
        from msmodelslim.processor.trainable_linear_quant.pipeline.resolve import (
            StrategyResolver,
        )

        resolver = StrategyResolver([QuantStrategyConfig(qconfig=self.qconfig, include=["*mlp*"], exclude=[])])
        self.model.block1 = TinyBlock()
        # rename to look like mlp path
        self.model.block1 = nn.Module()
        self.model.block1.mlp = TinyBlock()
        # TinyBlock has .fc → path block1.mlp.fc will NOT match *mlp* wait - "block1.mlp.fc" matches *mlp*
        setup = BlockSetup(
            model=self.model,
            operation_configs=self.operations,
            layer_qconfigs={},
            strategy_resolver=resolver,
        )
        wrappers = setup.wrap_linears("block1.mlp", self.model.block1.mlp)
        self.assertIn("block1.mlp.fc", wrappers)
        self.assertIn("block1.mlp.fc", setup._layer_qconfigs)

    def test_wrap_and_install_populates_context(self):
        setup = self._make_setup()
        ctx = BlockTLQContext(block_name="block0")
        ctx.wrappers_by_path = setup.wrap_linears("block0", self.block)
        ctx.ops = setup.install_ops("block0", self.block, ctx.wrappers_by_path)
        self.assertGreater(len(ctx.wrappers_by_path), 0)
        self.assertGreater(len(ctx.ops), 0)


if __name__ == "__main__":
    unittest.main()
