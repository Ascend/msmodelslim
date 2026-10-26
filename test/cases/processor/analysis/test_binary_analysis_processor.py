#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=no-member

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

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.analysis.binary_operator.processor import BinaryAnalysisProcessor
from msmodelslim.utils.exception import UnexpectedError


class TinyAttention(nn.Module):
    def forward(self, x):
        return x


class TinySubBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = TinyAttention()

    def forward(self, x):
        return self.attn2(x)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()
        self.linear = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.sub = TinySubBlock()

    def forward(self, x):
        x = self.attn(x)
        x = self.linear(x)
        x = self.sub(x)
        return self.relu(x)


class TestBinaryAnalysisProcessor(unittest.TestCase):
    """Public lifecycle tests for BinaryAnalysisProcessor."""

    def setUp(self):
        self.model = TinyBlock()
        self.adapter = MagicMock()
        self.config = SimpleNamespace(
            metrics="attention_mse",
            patterns=["block.*"],
            configs=[MagicMock(name="cfg1"), MagicMock(name="cfg2")],
        )
        self.request = BatchProcessRequest(
            name="block",
            module=self.model,
            datas=[((torch.ones(1, 4),), {})],
        )

    def _make_fake_method(self, hook=None):
        fake_method = MagicMock()
        fake_method.name = "attention_mse"
        fake_method.supports_distributed = True
        fake_method.get_target_layers.return_value = ["block.attn", "block.linear", "block.sub.attn2"]
        fake_method.filter_layers_by_patterns.return_value = ["block.attn", "block.sub.attn2"]
        fake_method.get_hook.return_value = hook or (
            lambda module, inputs, output, layer_name, stats_dict: stats_dict.setdefault(layer_name, {"outputs": []})[
                "outputs"
            ].append(output)
        )
        fake_method.compute_score.return_value = 0.5
        return fake_method

    def _make_processor(self, fake_method, quant_processors):
        create_patcher = patch(
            "msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method"
        )
        from_config_patcher = patch(
            "msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config"
        )
        mock_create_method = create_patcher.start()
        mock_from_config = from_config_patcher.start()
        self.addCleanup(create_patcher.stop)
        self.addCleanup(from_config_patcher.stop)
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = quant_processors
        return BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)

    def test_binary_analysis_processor_initializes_empty_state_when_constructed(self):
        fake_method = self._make_fake_method()
        qp1 = MagicMock()
        qp2 = MagicMock()
        processor = self._make_processor(fake_method, [qp1, qp2])

        self.assertEqual(processor.config, self.config)
        self.assertIs(processor.adapter, self.adapter)
        self.assertEqual(processor.quant_processors, [qp1, qp2])
        self.assertEqual(processor._target_layers, [])
        self.assertEqual(processor._float_layer_stats, {})
        self.assertEqual(processor._quant_layer_stats, {})
        self.assertEqual(processor._layer_scores, [])
        self.assertEqual(processor._hook_handles, {})

    @patch("msmodelslim.processor.analysis.binary_operator.processor.get_current_context")
    def test_binary_analysis_processor_pre_run_calls_quant_processors_when_context_exists(
        self, mock_get_current_context
    ):
        fake_method = self._make_fake_method()
        qp1 = MagicMock()
        qp2 = MagicMock()
        processor = self._make_processor(fake_method, [qp1, qp2])
        mock_get_current_context.return_value = {"layer_analysis": SimpleNamespace(state={})}

        processor.pre_run()

        qp1.pre_run.assert_called_once_with()
        qp2.pre_run.assert_called_once_with()

    @patch("msmodelslim.processor.analysis.binary_operator.processor.get_current_context")
    def test_binary_analysis_processor_pre_run_raises_unexpected_error_when_context_missing(
        self, mock_get_current_context
    ):
        processor = self._make_processor(self._make_fake_method(), [MagicMock(), MagicMock()])
        mock_get_current_context.return_value = None

        with self.assertRaises(UnexpectedError):
            processor.pre_run()

    def test_binary_analysis_processor_preprocess_records_float_hooks_and_removes_handles_when_targets_match(self):
        fake_method = self._make_fake_method()
        processor = self._make_processor(fake_method, [MagicMock(), MagicMock()])

        processor.preprocess(self.request)

        self.assertEqual(processor._target_layers, ["block.attn", "block.sub.attn2"])
        self.assertEqual(set(processor._float_layer_stats), {"block.attn", "block.sub.attn2"})
        self.assertEqual(processor._hook_handles, {})
        self.assertEqual(processor._layer_scores, [])
        fake_method.get_target_layers.assert_called_once_with(self.model, "block")
        fake_method.filter_layers_by_patterns.assert_called_once_with(
            ["block.attn", "block.linear", "block.sub.attn2"], self.config.patterns
        )

    def test_binary_analysis_processor_does_not_record_linear_targets_when_registering_hooks(self):
        fake_method = self._make_fake_method()
        fake_method.filter_layers_by_patterns.return_value = ["block.linear"]
        processor = self._make_processor(fake_method, [MagicMock(), MagicMock()])

        processor.preprocess(self.request)

        self.assertEqual(processor._float_layer_stats, {})
        self.assertEqual(processor._hook_handles, {})

    def test_binary_analysis_processor_process_calls_quant_processors_in_order_when_configured(self):
        qp1 = MagicMock()
        qp2 = MagicMock()
        processor = self._make_processor(self._make_fake_method(), [qp1, qp2])

        processor.process(self.request)

        self.assertEqual(
            qp1.method_calls,
            [
                call.preprocess(self.request),
                call.process(self.request),
                call.postprocess(self.request),
            ],
        )
        self.assertEqual(
            qp2.method_calls,
            [
                call.preprocess(self.request),
                call.process(self.request),
                call.postprocess(self.request),
            ],
        )

    def test_binary_analysis_processor_postprocess_computes_score_and_clears_stats_when_both_observations_exist(self):
        fake_method = self._make_fake_method()
        processor = self._make_processor(fake_method, [MagicMock(), MagicMock()])
        processor.preprocess(self.request)
        processor.request_for_test = self.request

        processor.postprocess(self.request)

        self.assertEqual(
            processor._layer_scores,
            [
                {"name": "block.attn", "score": 0.5},
                {"name": "block.sub.attn2", "score": 0.5},
            ],
        )
        self.assertEqual(processor._float_layer_stats, {})
        self.assertEqual(processor._quant_layer_stats, {})
        self.assertEqual(processor._hook_handles, {})
        self.assertEqual(fake_method.compute_score.call_count, 2)

    @patch("msmodelslim.processor.analysis.distributed_utils.get_current_context")
    def test_binary_analysis_processor_post_run_publishes_scores_when_scores_ready(self, mock_get_current_context):
        fake_method = self._make_fake_method()
        qp1 = MagicMock()
        qp2 = MagicMock()
        processor = self._make_processor(fake_method, [qp1, qp2])
        processor.preprocess(self.request)
        processor.postprocess(self.request)
        context = {"layer_analysis": SimpleNamespace(debug={})}
        mock_get_current_context.return_value = context

        processor.post_run()

        qp1.post_run.assert_called_once_with()
        qp2.post_run.assert_called_once_with()
        self.assertEqual(
            context["layer_analysis"].debug["layer_scores"],
            [{"name": "block.attn", "score": 0.5}, {"name": "block.sub.attn2", "score": 0.5}],
        )
        self.assertEqual(context["layer_analysis"].debug["method"], "attention_mse")
        self.assertEqual(context["layer_analysis"].debug["patterns"], self.config.patterns)

    @patch("msmodelslim.processor.analysis.binary_operator.processor.maybe_barrier_before_linear_quant")
    def test_binary_analysis_processor_process_waits_at_barrier_when_distributed(self, mock_barrier):
        processor = self._make_processor(self._make_fake_method(), [MagicMock(), MagicMock()])

        processor.process(self.request)

        mock_barrier.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
