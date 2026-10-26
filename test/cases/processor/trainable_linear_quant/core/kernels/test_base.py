#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for the listener-driven TLQ kernel base."""

# pylint: disable=no-member

import unittest

import torch

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir.qal import QDType, QParam, QScheme, QScope
from msmodelslim.processor.trainable_linear_quant.core.kernels.base import (
    FakeQuantStage,
    TLQKernel,
)
from msmodelslim.processor.trainable_linear_quant.core.kernels.context import (
    FakeQuantPipelineContext,
    FakeQuantResult,
)


def _qconfig() -> QConfig:
    return QConfig(dtype=QDType.INT8, scope=QScope.PER_TENSOR, symmetric=True, method="minmax")


def _result() -> FakeQuantResult:
    q_param = QParam(
        scheme=QScheme(QScope.PER_TENSOR, QDType.INT8, True),
        ext={"scale": torch.tensor([0.5]), "offset": torch.tensor([0.0])},
    )
    return FakeQuantResult(tensor=torch.tensor([1.0]), q_param=q_param, quantized=torch.tensor([2.0]))


class TestFakeQuantStage(unittest.TestCase):
    def test_stage_values_are_stable(self):
        self.assertEqual(FakeQuantStage.QUANT_INPUT.value, "quant_input")
        self.assertEqual(FakeQuantStage.BEFORE_QPARAM.value, "before_qparam")
        self.assertEqual(FakeQuantStage.QPARAM_READY.value, "qparam_ready")
        self.assertEqual(FakeQuantStage.PRE_ROUND.value, "pre_round")
        self.assertEqual(FakeQuantStage.QUANTIZED.value, "quantized")
        self.assertEqual(FakeQuantStage.DEQUANT_OUTPUT.value, "dequant_output")


class TestTLQKernel(unittest.TestCase):
    def _kernel(self):
        def factory(config, tensor, train_tensors):
            return FakeQuantPipelineContext(config=config, float_tensor=tensor, train_tensors=train_tensors)

        def driver(ctx, return_quantized_weight=False):
            yield FakeQuantStage.QUANT_INPUT
            yield FakeQuantStage.BEFORE_QPARAM
            yield FakeQuantStage.QUANTIZED
            yield FakeQuantStage.DEQUANT_OUTPUT
            return _result()

        return TLQKernel(_qconfig(), driver, factory)

    def test_dispatches_all_stages_to_listeners(self):
        kernel = self._kernel()
        seen = []

        def listener(stage, ctx):
            seen.append((stage, ctx.float_tensor.tolist(), ctx.train_tensors["v"].item()))

        kernel.add_listener(listener, op_id="layer.round_tune")
        result = kernel.fake_quantize(
            torch.tensor([3.0]),
            train_tensors={"v": torch.tensor(7.0)},
            return_quantized_weight=True,
        )
        self.assertEqual(
            [item[0] for item in seen],
            [
                FakeQuantStage.QUANT_INPUT,
                FakeQuantStage.BEFORE_QPARAM,
                FakeQuantStage.QUANTIZED,
                FakeQuantStage.DEQUANT_OUTPUT,
            ],
        )
        self.assertEqual(seen[0][1:], ([3.0], 7.0))
        self.assertEqual(result.tensor.tolist(), [1.0])
        self.assertEqual(result.quantized.tolist(), [2.0])

    def test_remove_listeners_removes_only_matching_op_id(self):
        kernel = self._kernel()
        calls = []
        kernel.add_listener(lambda stage, ctx: calls.append(("a", stage)), op_id="a")
        kernel.add_listener(lambda stage, ctx: calls.append(("b", stage)), op_id="b")
        kernel.add_listener(lambda stage, ctx: calls.append(("a2", stage)), op_id="a")

        kernel.remove_listeners("a")
        kernel.fake_quantize(torch.tensor([1.0]))

        self.assertEqual([name for name, _ in calls], ["b"] * 4)

    def test_train_tensors_are_copied_per_call(self):
        kernel = self._kernel()
        original = {"v": torch.tensor(1.0)}
        observed = []

        def listener(stage, ctx):
            if stage == FakeQuantStage.QUANT_INPUT:
                ctx.train_tensors["added"] = torch.tensor(10.0)
                observed.append(sorted(ctx.train_tensors))

        kernel.add_listener(listener, op_id="mutating")
        kernel.fake_quantize(torch.tensor([1.0]), train_tensors=original)

        self.assertEqual(observed, [["added", "v"]])
        self.assertEqual(original["v"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
