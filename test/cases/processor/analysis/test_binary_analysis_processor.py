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

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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
        return self.relu(self.linear(self.attn(x)))


class TestBinaryAnalysisProcessor(unittest.TestCase):
    """测试 BinaryAnalysisProcessor。"""

    def setUp(self):
        self.model = TinyBlock()
        self.adapter = MagicMock()
        self.config = SimpleNamespace(
            metrics="attention_mse",
            patterns=["block.*"],
            configs=[MagicMock(name="cfg1"), MagicMock(name="cfg2")],
        )
        self.request = BatchProcessRequest(name="block", module=self.model, datas=[])

    def _build_fake_method(self):
        fake_method = MagicMock()
        fake_method.name = "attention_mse"
        fake_method.get_hook.return_value = lambda module, input_tensor, output_tensor, layer_name, stats_dict: None
        return fake_method

    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method")
    def test_init_set_empty_state_when_config_valid(self, mock_create_method, mock_from_config):
        fake_method = self._build_fake_method()
        qp1 = MagicMock()
        qp2 = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = [qp1, qp2]

        processor = BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)

        self.assertEqual(processor.config, self.config)
        self.assertIs(processor.adapter, self.adapter)
        self.assertEqual(processor.quant_processors, [qp1, qp2])
        mock_from_config.assert_has_calls(
            [
                call(self.model, self.config.configs[0], self.adapter),
                call(self.model, self.config.configs[1], self.adapter),
            ]
        )
        mock_create_method.assert_called_once_with("attention_mse", adapter=self.adapter)
        self.assertEqual(processor._target_layers, [])
        self.assertEqual(processor._float_layer_stats, {})
        self.assertEqual(processor._quant_layer_stats, {})
        self.assertEqual(processor._layer_scores, [])
        self.assertEqual(processor._hook_handles, {})

    @patch("msmodelslim.processor.analysis.binary_operator.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method")
    def test_pre_run_call_quant_processors_when_context_exists(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        fake_method = self._build_fake_method()
        qp1 = MagicMock()
        qp2 = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = [qp1, qp2]
        mock_get_current_context.return_value = {"layer_analysis": SimpleNamespace(state={})}

        processor = BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)
        processor.pre_run()

        qp1.pre_run.assert_called_once()
        qp2.pre_run.assert_called_once()

    @patch("msmodelslim.processor.analysis.binary_operator.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method")
    def test_pre_run_raise_unexpected_error_when_context_missing(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        fake_method = self._build_fake_method()
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = [MagicMock(), MagicMock()]
        mock_get_current_context.return_value = None

        processor = BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)

        with self.assertRaises(UnexpectedError):
            processor.pre_run()

    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method")
    def test_register_hooks_set_hook_handles_when_target_layers_matched(self, mock_create_method, mock_from_config):
        fake_method = self._build_fake_method()
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = [MagicMock(), MagicMock()]

        processor = BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)
        processor._target_layers = ["block.attn", "block.linear", "block.sub.attn2"]

        processor._register_hooks_for_request(self.request, fake_method.get_hook(), processor._float_layer_stats)

        self.assertIn("block.attn", processor._hook_handles)
        self.assertIn("block.sub.attn2", processor._hook_handles)
        self.assertNotIn("block.linear", processor._hook_handles)
        self.assertNotIn("block.relu", processor._hook_handles)

        for handle in processor._hook_handles.values():
            handle.remove()

    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.process")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method")
    def test_preprocess_call_float_forward_and_clear_hooks_when_targets_matched(
        self, mock_create_method, mock_from_config, mock_super_process
    ):
        fake_method = self._build_fake_method()
        fake_method.get_target_layers.return_value = ["block.attn", "block.linear", "block.sub.attn2"]
        fake_method.filter_layers_by_patterns.return_value = ["block.attn", "block.sub.attn2"]
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = [MagicMock(), MagicMock()]

        processor = BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)
        processor.preprocess(self.request)

        fake_method.get_target_layers.assert_called_once_with(self.model, "block")
        fake_method.filter_layers_by_patterns.assert_called_once_with(
            ["block.attn", "block.linear", "block.sub.attn2"],
            self.config.patterns,
        )
        mock_super_process.assert_called_once_with(self.request)
        self.assertEqual(processor._target_layers, ["block.attn", "block.sub.attn2"])
        self.assertEqual(processor._hook_handles, {})
        self.assertEqual(processor._layer_scores, [])

    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method")
    def test_process_call_quant_processors_in_order_when_quant_processors_exist(
        self, mock_create_method, mock_from_config
    ):
        fake_method = self._build_fake_method()
        qp1 = MagicMock()
        qp2 = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = [qp1, qp2]

        processor = BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)
        processor.process(self.request)

        qp1.preprocess.assert_called_once_with(self.request)
        qp1.process.assert_called_once_with(self.request)
        qp1.postprocess.assert_called_once_with(self.request)
        qp2.preprocess.assert_called_once_with(self.request)
        qp2.process.assert_called_once_with(self.request)
        qp2.postprocess.assert_called_once_with(self.request)

    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.process")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method")
    def test_postprocess_return_scores_and_clear_stats_when_both_stats_exist(
        self, mock_create_method, mock_from_config, mock_super_process
    ):
        fake_method = self._build_fake_method()
        fake_method.compute_score.return_value = 0.5
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = [MagicMock(), MagicMock()]

        processor = BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)
        processor._target_layers = ["block.attn"]
        processor._float_layer_stats = {"block.attn": {"float": [1]}}

        handle = MagicMock()

        def fake_register(request, hook_fn, stats_dict):
            processor._hook_handles["block.attn"] = handle
            stats_dict["block.attn"] = {"quant": [2]}

        processor._register_hooks_for_request = MagicMock(side_effect=fake_register)

        processor.postprocess(self.request)

        processor._register_hooks_for_request.assert_called_once_with(
            self.request, fake_method.get_hook.return_value, processor._quant_layer_stats
        )
        mock_super_process.assert_called_once_with(self.request)
        handle.remove.assert_called_once()
        fake_method.compute_score.assert_called_once_with({"float": [1]}, {"quant": [2]})
        self.assertEqual(processor._layer_scores, [{"name": "block.attn", "score": 0.5}])
        self.assertNotIn("block.attn", processor._float_layer_stats)
        self.assertNotIn("block.attn", processor._quant_layer_stats)
        self.assertEqual(processor._hook_handles, {})

    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method")
    def test_postprocess_return_remaining_hooks_when_request_scope_filtered(self, mock_create_method, mock_from_config):
        fake_method = self._build_fake_method()
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = [MagicMock(), MagicMock()]

        processor = BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)
        handle_block = MagicMock()
        handle_nested = MagicMock()
        handle_other = MagicMock()
        processor._hook_handles = {
            "block.attn": handle_block,
            "block.sub.attn2": handle_nested,
            "other.attn": handle_other,
        }
        processor._target_layers = []
        processor._register_hooks_for_request = MagicMock()

        with patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.process"):
            processor.postprocess(self.request)

        handle_block.remove.assert_called_once()
        handle_nested.remove.assert_called_once()
        handle_other.remove.assert_not_called()
        self.assertNotIn("block.attn", processor._hook_handles)
        self.assertNotIn("block.sub.attn2", processor._hook_handles)
        self.assertIn("other.attn", processor._hook_handles)

    @patch("msmodelslim.processor.analysis.binary_operator.processor.maybe_barrier_before_linear_quant")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method")
    def test_process_barriers_before_quant_when_distributed(self, mock_create_method, mock_from_config, mock_barrier):
        fake_method = self._build_fake_method()
        qp1 = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = [qp1, MagicMock()]

        processor = BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)
        processor.process(self.request)

        mock_barrier.assert_called_once()
        qp1.preprocess.assert_called_once_with(self.request)

    @patch("msmodelslim.processor.analysis.distributed_utils.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator.processor.BinaryAnalysisMethodFactory.create_method")
    def test_post_run_call_quant_processors_and_set_context_when_scores_ready(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        fake_method = self._build_fake_method()
        fake_method.name = "attention_mse"
        qp1 = MagicMock()
        qp2 = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.side_effect = [qp1, qp2]

        processor = BinaryAnalysisProcessor(self.model, self.config, adapter=self.adapter)
        processor._layer_scores = [{"name": "block.attn", "score": 3.0}]
        fake_ctx = {"layer_analysis": SimpleNamespace(debug={})}
        mock_get_current_context.return_value = fake_ctx

        processor.post_run()

        qp1.post_run.assert_called_once()
        qp2.post_run.assert_called_once()
        self.assertEqual(fake_ctx["layer_analysis"].debug["layer_scores"], processor._layer_scores)
        self.assertEqual(fake_ctx["layer_analysis"].debug["method"], "attention_mse")
        self.assertEqual(fake_ctx["layer_analysis"].debug["patterns"], self.config.patterns)


if __name__ == "__main__":
    unittest.main()
