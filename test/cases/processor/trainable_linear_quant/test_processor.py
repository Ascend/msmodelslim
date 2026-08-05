#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.config import (
    QuantStrategyConfig,
    TrainableLinearQuantProcessorConfig,
)
from msmodelslim.processor.trainable_linear_quant.pipeline.runtime import BlockTLQContext
from msmodelslim.processor.trainable_linear_quant.processor import TrainableLinearQuantProcessor


def _int8_qconfig() -> LinearQConfig:
    return LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
    )


class TinyBlock(nn.Module):
    def forward(self, x):
        return x


class TestTLQProcessorPropagation(unittest.TestCase):
    def _make_processor(self, *, adapter=None) -> TrainableLinearQuantProcessor:
        model = nn.Module()
        model.linear = nn.Linear(4, 4)
        config = TrainableLinearQuantProcessorConfig(
            strategies=[QuantStrategyConfig(qconfig=_int8_qconfig())],
            enable_quanted_input=True,
        )
        return TrainableLinearQuantProcessor(model, config, adapter=adapter)

    def test_pre_run_without_adapter_when_default_ops(self):
        processor = self._make_processor(adapter=None)
        processor.pre_run()
        self.assertEqual(processor._global_adapter_configs, [])
        self.assertIsNotNone(processor._block_setup)

    @patch(
        "msmodelslim.processor.trainable_linear_quant.core.train.block_trainer.TrainableLinearQuantBlockTrainer.train_block"
    )
    def test_process_injects_propagation_outputs_before_training(self, mock_train_block):
        processor = self._make_processor()
        block = TinyBlock()
        original = torch.randn(2, 4)
        propagated = torch.randn(2, 4)
        sample = [[original.clone()], {}]
        request = BatchProcessRequest(name="model.layers.0", module=block, datas=[sample])

        ctx = BlockTLQContext(block_name=request.name)
        ctx.teacher_outputs = [torch.randn(2, 4)]
        ctx.ops = [MagicMock()]
        processor._sessions[request.name] = ctx
        processor._propagation_outputs = [propagated]

        processor.process(request)

        self.assertTrue(torch.equal(sample[0][0], propagated))
        mock_train_block.assert_called_once()
        call_kwargs = mock_train_block.call_args.kwargs
        trained_sample = call_kwargs["all_datas"][0]
        self.assertTrue(torch.equal(trained_sample[0][0], propagated))

    @patch(
        "msmodelslim.processor.trainable_linear_quant.core.train.block_trainer.TrainableLinearQuantBlockTrainer.train_block"
    )
    def test_process_skips_injection_without_propagation_outputs(self, mock_train_block):
        processor = self._make_processor()
        block = TinyBlock()
        original = torch.randn(2, 4)
        sample = [[original.clone()], {}]
        request = BatchProcessRequest(name="model.layers.0", module=block, datas=[sample])

        ctx = BlockTLQContext(block_name=request.name)
        ctx.teacher_outputs = [torch.randn(2, 4)]
        ctx.ops = [MagicMock()]
        processor._sessions[request.name] = ctx

        processor.process(request)

        self.assertTrue(torch.equal(sample[0][0], original))
        mock_train_block.assert_called_once()

    @patch("msmodelslim.processor.trainable_linear_quant.processor.finalize_block")
    def test_postprocess_request_outputs_uses_teacher_when_quanted_input_disabled(self, mock_finalize):
        model = nn.Module()
        config = TrainableLinearQuantProcessorConfig(
            strategies=[QuantStrategyConfig(qconfig=_int8_qconfig())],
            enable_quanted_input=False,
        )
        processor = TrainableLinearQuantProcessor(model, config)
        block = TinyBlock()
        teacher = torch.randn(2, 4)
        request = BatchProcessRequest(
            name="model.layers.0",
            module=block,
            datas=[[torch.randn(2, 4)], {}],
        )
        ctx = BlockTLQContext(block_name=request.name)
        ctx.teacher_outputs = [teacher]
        ctx.ops = [MagicMock()]
        processor._sessions[request.name] = ctx

        processor.postprocess(request)

        self.assertEqual(len(request.outputs), 1)
        self.assertTrue(torch.equal(request.outputs[0], teacher))
        mock_finalize.assert_called_once()

    @patch("msmodelslim.processor.trainable_linear_quant.processor.finalize_block")
    @patch("msmodelslim.processor.trainable_linear_quant.processor.capture_quant_propagation")
    def test_postprocess_keeps_teacher_on_runner_when_quanted_input_enabled(
        self, mock_capture_quant_propagation, mock_finalize
    ):
        """enable_quanted_input：量化结果旁路保存；request.outputs 仍为 teacher。"""
        processor = self._make_processor()
        block = TinyBlock()
        teacher = torch.randn(2, 4)
        quantized = torch.randn(2, 4)
        mock_capture_quant_propagation.return_value = [quantized]
        request = BatchProcessRequest(
            name="model.layers.0",
            module=block,
            datas=[[torch.randn(2, 4)], {}],
        )
        ctx = BlockTLQContext(block_name=request.name)
        ctx.teacher_outputs = [teacher]
        ctx.ops = [MagicMock()]
        processor._sessions[request.name] = ctx

        processor.postprocess(request)

        self.assertEqual(len(request.outputs), 1)
        self.assertTrue(torch.equal(request.outputs[0], teacher))
        self.assertIsNotNone(processor._propagation_outputs)
        self.assertTrue(torch.equal(processor._propagation_outputs[0], quantized))
        mock_capture_quant_propagation.assert_called_once()
        mock_finalize.assert_called_once()

    @patch("msmodelslim.processor.trainable_linear_quant.processor.finalize_block")
    @patch("msmodelslim.processor.trainable_linear_quant.processor.capture_quant_propagation")
    def test_postprocess_loads_best_params_before_capture(self, mock_capture_quant_propagation, mock_finalize):
        processor = self._make_processor()
        block = TinyBlock()
        quantized = torch.randn(2, 4)
        events: list[str] = []
        op = MagicMock()
        op.load_best_params.side_effect = lambda: events.append("load_best")

        def _capture(_request):
            events.append("capture")
            return [quantized]

        mock_capture_quant_propagation.side_effect = _capture
        request = BatchProcessRequest(
            name="model.layers.0",
            module=block,
            datas=[[torch.randn(2, 4)], {}],
        )
        ctx = BlockTLQContext(block_name=request.name)
        ctx.teacher_outputs = [torch.randn(2, 4)]
        ctx.ops = [op]
        processor._sessions[request.name] = ctx

        processor.postprocess(request)

        op.load_best_params.assert_called_once()
        self.assertEqual(events, ["load_best", "capture"])

    @patch(
        "msmodelslim.processor.trainable_linear_quant.core.train.block_trainer.TrainableLinearQuantBlockTrainer.train_block"
    )
    @patch("msmodelslim.processor.trainable_linear_quant.processor.finalize_block")
    def test_empty_ops_block_skips_train_and_finalize(self, mock_finalize, mock_train_block):
        processor = self._make_processor()
        processor.config.enable_quanted_input = False
        block = TinyBlock()
        teacher = torch.randn(2, 4)
        request = BatchProcessRequest(
            name="model.visual",
            module=block,
            datas=[[[torch.randn(2, 4)], {}]],
        )
        ctx = BlockTLQContext(block_name=request.name)
        ctx.teacher_outputs = [teacher]
        ctx.ops = []
        processor._sessions[request.name] = ctx

        processor.process(request)
        processor.postprocess(request)

        mock_train_block.assert_not_called()
        mock_finalize.assert_not_called()
        self.assertTrue(torch.equal(request.outputs[0], teacher))
        self.assertNotIn(request.name, processor._sessions)

    @patch("msmodelslim.processor.trainable_linear_quant.processor.capture_float_teacher")
    def test_install_accepts_injected_teacher_outputs(self, mock_capture):
        processor = self._make_processor()
        processor.pre_run()
        block = nn.Sequential(nn.Linear(4, 4))
        teacher = torch.randn(2, 4)
        request = BatchProcessRequest(
            name="block0",
            module=block,
            datas=[[torch.randn(2, 4)], {}],
        )
        with patch.object(processor._block_setup, "wrap_linears", return_value={}):
            with patch.object(processor._block_setup, "install_ops", return_value=[]):
                ctx = processor.install(request, teacher_outputs=[teacher])

        mock_capture.assert_not_called()
        self.assertTrue(torch.equal(ctx.teacher_outputs[0], teacher))
        self.assertIn("block0", processor._sessions)


if __name__ == "__main__":
    unittest.main()
