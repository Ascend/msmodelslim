#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from unittest.mock import MagicMock, patch

import pytest
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
from msmodelslim.processor.trainable_linear_quant.pipeline.setup import BlockSetup
from msmodelslim.processor.trainable_linear_quant.processor import TrainableLinearQuantProcessor


def _int8_qconfig() -> LinearQConfig:
    return LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
    )


def _make_processor(*, enable_quanted_input: bool = True) -> TrainableLinearQuantProcessor:
    model = nn.Module()
    model.linear = nn.Linear(4, 4)
    config = TrainableLinearQuantProcessorConfig(
        strategies=[QuantStrategyConfig(qconfig=_int8_qconfig())],
        enable_quanted_input=enable_quanted_input,
    )
    return TrainableLinearQuantProcessor(model, config, adapter=None)


def _make_request(name: str, sample=None) -> BatchProcessRequest:
    module = nn.Sequential(nn.Linear(4, 4))
    if sample is None:
        sample = [[torch.randn(2, 4)], {}]
    return BatchProcessRequest(name=name, module=module, datas=[sample])


def _install(processor, request, teacher, ops):
    with (
        patch.object(BlockSetup, "wrap_linears", return_value={}),
        patch.object(BlockSetup, "install_ops", return_value=ops),
    ):
        return processor.install(request, teacher_outputs=[teacher])


def test_trainable_linear_quant_processor_returns_false_when_support_distributed_called():
    processor = _make_processor()
    assert processor.support_distributed() is False


def test_trainable_linear_quant_processor_builds_context_when_pre_run_without_adapter():
    processor = _make_processor()
    processor.pre_run()
    request = _make_request("block0")
    teacher = torch.randn(2, 4)
    ctx = _install(processor, request, teacher, [])
    assert ctx.block_name == "block0"
    assert torch.equal(ctx.teacher_outputs[0], teacher)


def test_trainable_linear_quant_processor_captures_teacher_when_install_has_no_teacher_outputs():
    processor = _make_processor()
    processor.pre_run()
    request = _make_request("block0")
    teacher = torch.randn(2, 4)
    with (
        patch.object(BlockSetup, "wrap_linears", return_value={}),
        patch.object(BlockSetup, "install_ops", return_value=[]),
        patch(
            "msmodelslim.processor.trainable_linear_quant.processor.capture_float_teacher", return_value=[teacher]
        ) as capture,
    ):
        ctx = processor.install(request)
    capture.assert_called_once_with(request)
    assert torch.equal(ctx.teacher_outputs[0], teacher)


def test_trainable_linear_quant_processor_injects_previous_quantized_output_when_process_next_block():
    processor = _make_processor()
    processor.pre_run()
    propagated = torch.randn(2, 4)
    first_request = _make_request("block0")
    _install(processor, first_request, torch.randn(2, 4), [MagicMock()])
    with (
        patch(
            "msmodelslim.processor.trainable_linear_quant.processor.capture_quant_propagation",
            return_value=[propagated],
        ),
        patch("msmodelslim.processor.trainable_linear_quant.processor.finalize_block"),
    ):
        processor.postprocess(first_request)

    original = torch.randn(2, 4)
    sample = [[original.clone()], {}]
    second_request = _make_request("block1", sample)
    _install(processor, second_request, torch.randn(2, 4), [MagicMock()])
    with patch(
        "msmodelslim.processor.trainable_linear_quant.core.train.block_trainer."
        "TrainableLinearQuantBlockTrainer.train_block"
    ) as train_block:
        processor.process(second_request)

    assert torch.equal(sample[0][0], propagated)
    assert torch.equal(train_block.call_args.kwargs["all_datas"][0][0][0], propagated)


def test_trainable_linear_quant_processor_keeps_input_when_process_has_no_propagation_outputs():
    processor = _make_processor()
    processor.pre_run()
    original = torch.randn(2, 4)
    request = _make_request("block0", [[original.clone()], {}])
    _install(processor, request, torch.randn(2, 4), [MagicMock()])
    with patch(
        "msmodelslim.processor.trainable_linear_quant.core.train.block_trainer."
        "TrainableLinearQuantBlockTrainer.train_block"
    ):
        processor.process(request)
    assert torch.equal(request.datas[0][0][0], original)


def test_trainable_linear_quant_processor_sets_teacher_output_when_quanted_input_disabled():
    processor = _make_processor(enable_quanted_input=False)
    processor.pre_run()
    teacher = torch.randn(2, 4)
    request = _make_request("block0")
    _install(processor, request, teacher, [MagicMock()])
    with patch("msmodelslim.processor.trainable_linear_quant.processor.finalize_block") as finalize:
        processor.postprocess(request)
    assert torch.equal(request.outputs[0], teacher)
    finalize.assert_called_once()


def test_trainable_linear_quant_processor_captures_quantized_output_when_quanted_input_enabled():
    processor = _make_processor(enable_quanted_input=True)
    processor.pre_run()
    teacher = torch.randn(2, 4)
    quantized = torch.randn(2, 4)
    request = _make_request("block0")
    _install(processor, request, teacher, [MagicMock()])
    with (
        patch(
            "msmodelslim.processor.trainable_linear_quant.processor.capture_quant_propagation",
            return_value=[quantized],
        ) as capture,
        patch("msmodelslim.processor.trainable_linear_quant.processor.finalize_block"),
    ):
        processor.postprocess(request)
    assert torch.equal(request.outputs[0], teacher)
    capture.assert_called_once_with(request)


def test_trainable_linear_quant_processor_loads_best_params_before_capture_when_ops_exist():
    processor = _make_processor()
    processor.pre_run()
    events = []
    op = MagicMock()
    op.load_best_params.side_effect = lambda: events.append("load_best")

    def _capture(_request):
        events.append("capture")
        return [torch.randn(2, 4)]

    request = _make_request("block0")
    _install(processor, request, torch.randn(2, 4), [op])
    with (
        patch("msmodelslim.processor.trainable_linear_quant.processor.capture_quant_propagation", side_effect=_capture),
        patch("msmodelslim.processor.trainable_linear_quant.processor.finalize_block"),
    ):
        processor.postprocess(request)
    assert events == ["load_best", "capture"]


def test_trainable_linear_quant_processor_skips_train_and_finalize_when_ops_empty():
    processor = _make_processor(enable_quanted_input=False)
    processor.pre_run()
    teacher = torch.randn(2, 4)
    request = _make_request("model.visual")
    _install(processor, request, teacher, [])
    with (
        patch(
            "msmodelslim.processor.trainable_linear_quant.core.train.block_trainer."
            "TrainableLinearQuantBlockTrainer.train_block"
        ) as train_block,
        patch("msmodelslim.processor.trainable_linear_quant.processor.finalize_block") as finalize,
    ):
        processor.process(request)
        processor.postprocess(request)
    train_block.assert_not_called()
    finalize.assert_not_called()
    assert torch.equal(request.outputs[0], teacher)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
