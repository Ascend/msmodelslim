#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest

import torch
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
from msmodelslim.processor.trainable_linear_quant.pipeline.setup import BlockSetup


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


def _mxfp4_qconfig() -> LinearQConfig:
    weight = QConfig(
        dtype=QDType.MXFP4,
        scope=QScope.PER_BLOCK,
        symmetric=True,
        method="minmax",
        ext={"group_size": 32},
    )
    act = QConfig(
        dtype=QDType.FLOAT,
        scope=QScope.PER_TENSOR,
        symmetric=True,
        method="none",
    )
    return LinearQConfig(act=act, weight=weight)


class TinyBlock(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)


class TestLinearOpsLifecycle(unittest.TestCase):
    def _install(self, qconfig: LinearQConfig):
        model = nn.Module()
        model.block0 = TinyBlock()
        setup = BlockSetup(
            model=model,
            operation_configs=[MinmaxTuneOpConfig(), RoundTuneOpConfig()],
            layer_qconfigs={"block0.fc": qconfig},
        )
        ctx = BlockTLQContext(block_name="block0")
        ctx.wrappers_by_path = setup.wrap_linears("block0", model.block0)
        ctx.ops = setup.install_ops("block0", model.block0, ctx.wrappers_by_path)
        return model, ctx

    def test_int8_bind_forward_best_params_unbind(self):
        model, ctx = self._install(_int8_qconfig())
        self.assertGreaterEqual(len(ctx.ops), 2)

        for op in ctx.ops:
            params = op.train_params
            self.assertTrue(params)
            op.save_best_params()
            self.assertIsNotNone(op.best_params)
            op.load_best_params()

        x = torch.randn(2, 4)
        y = model.block0.fc(x)
        self.assertEqual(tuple(y.shape), (2, 4))

        for op in ctx.ops:
            op.unbind()
            op.release_cached_params()
            self.assertIsNone(op.best_params)

    def test_mxfp_minmax_round_forward(self):
        model = nn.Module()
        model.block0 = TinyBlock(in_features=32, out_features=8)
        setup = BlockSetup(
            model=model,
            operation_configs=[MinmaxTuneOpConfig(), RoundTuneOpConfig()],
            layer_qconfigs={"block0.fc": _mxfp4_qconfig()},
        )
        wrappers = setup.wrap_linears("block0", model.block0)
        ops = setup.install_ops("block0", model.block0, wrappers)
        self.assertGreaterEqual(len(ops), 2)
        y = model.block0.fc(torch.randn(2, 32))
        self.assertEqual(tuple(y.shape), (2, 8))
        for op in ops:
            op.unbind()


class TestWrapperHelpers(unittest.TestCase):
    def test_forward_act_transform_and_unwrapper(self):
        linear = nn.Linear(4, 3)
        wrapper = TrainableLinearQuantWrapper(
            linear,
            linear_qconfig=_int8_qconfig(),
            train_with_act_quant=False,
        )
        wrapper.layer_path = "fc"

        seen = []

        def _scale(x: torch.Tensor) -> torch.Tensor:
            seen.append(True)
            return x * 1.0

        wrapper.register_forward_act_transform(_scale)
        wrapper.register_forward_act_transform(_scale)  # dedupe
        out = wrapper(torch.randn(2, 4))
        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertEqual(len(seen), 1)

        wrapper.unregister_forward_act_transform(_scale)
        wrapper.unregister_forward_act_transform(_scale)  # missing is ok
        layer = wrapper.unwrapper()
        self.assertIs(layer, linear)
        self.assertTrue(hasattr(layer, "scale"))


if __name__ == "__main__":
    unittest.main()
