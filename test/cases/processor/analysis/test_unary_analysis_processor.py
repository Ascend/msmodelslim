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
from unittest.mock import MagicMock, patch

import torch.nn as nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.analysis.unary_operator.processor import (
    UnaryAnalysisProcessor,
    UnaryAnalysisProcessorConfig,
)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)
        self.relu = nn.ReLU()


class TestUnaryAnalysisProcessor(unittest.TestCase):
    """测试 UnaryAnalysisProcessor。"""

    def setUp(self):
        self.model = TinyBlock()
        self.config = UnaryAnalysisProcessorConfig(metrics="std", patterns=["*.linear1", "*.relu"])
        self.request = BatchProcessRequest(name="block", module=self.model, datas=[])

    def _build_fake_method(self):
        fake_method = MagicMock()
        fake_method.name = "std"
        fake_method.get_hook.return_value = lambda module, input_tensor, output_tensor, layer_name, stats_dict: None
        return fake_method

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_init_return_empty_state_when_config_valid(self, mock_create_method):
        fake_method = self._build_fake_method()
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)

        mock_create_method.assert_called_once_with("std")
        self.assertEqual(processor.config, self.config)
        self.assertIs(processor._analysis_method, fake_method)
        self.assertEqual(processor._target_layers, [])
        self.assertEqual(processor._layer_stats, {})
        self.assertEqual(processor._layer_scores, [])
        self.assertEqual(processor._hook_handles, {})

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_preprocess_return_hook_handles_when_target_linear_layers_matched(self, mock_create_method):
        fake_method = self._build_fake_method()
        fake_method.get_target_layers.return_value = [
            "block.linear1",
            "block.linear2",
        ]
        fake_method.filter_layers_by_patterns.return_value = ["block.linear1"]
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        processor.preprocess(self.request)

        fake_method.get_target_layers.assert_called_once_with(self.model, "block")
        fake_method.filter_layers_by_patterns.assert_called_once_with(
            ["block.linear1", "block.linear2"],
            self.config.patterns,
        )
        self.assertEqual(processor._target_layers, ["block.linear1"])
        self.assertIn("block.linear1", processor._hook_handles)
        self.assertNotIn("block.linear2", processor._hook_handles)
        self.assertEqual(len(processor._hook_handles), 1)

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_postprocess_return_scores_and_clear_stats_when_target_stats_exist(self, mock_create_method):
        fake_method = self._build_fake_method()
        fake_method.compute_score.return_value = 0.75
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        handle_linear1 = MagicMock()
        handle_linear2 = MagicMock()

        processor._target_layers = ["block.linear1"]
        processor._hook_handles = {
            "block.linear1": handle_linear1,
            "block.linear2": handle_linear2,
        }
        processor._layer_stats = {
            "block.linear1": {"tensor": [1]},
            "block.linear2": {"tensor": [2]},
        }

        processor.postprocess(self.request)

        handle_linear1.remove.assert_called_once()
        handle_linear2.remove.assert_called_once()
        fake_method.compute_score.assert_called_once_with({"tensor": [1]})
        self.assertEqual(processor._layer_scores, [{"name": "block.linear1", "score": 0.75}])
        self.assertNotIn("block.linear1", processor._layer_stats)
        self.assertIn("block.linear2", processor._layer_stats)
        self.assertEqual(processor._hook_handles, {})

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_postprocess_return_remaining_hooks_when_request_scope_filtered(self, mock_create_method):
        fake_method = self._build_fake_method()
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        handle_block = MagicMock()
        handle_other = MagicMock()

        processor._hook_handles = {
            "block.linear1": handle_block,
            "other.linear": handle_other,
        }

        processor.postprocess(self.request)

        handle_block.remove.assert_called_once()
        handle_other.remove.assert_not_called()
        self.assertNotIn("block.linear1", processor._hook_handles)
        self.assertIn("other.linear", processor._hook_handles)

    @patch("msmodelslim.processor.analysis.unary_operator.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_post_run_set_context_state_when_scores_ready(self, mock_create_method, mock_get_current_context):
        fake_method = self._build_fake_method()
        fake_method.name = "std"
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        processor._layer_scores = [{"name": "block.linear1", "score": 1.23}]

        fake_ctx = {"layer_analysis": SimpleNamespace(debug={})}
        mock_get_current_context.return_value = fake_ctx

        processor.post_run()

        self.assertEqual(fake_ctx["layer_analysis"].debug["layer_scores"], processor._layer_scores)
        self.assertEqual(fake_ctx["layer_analysis"].debug["method"], "std")
        self.assertEqual(fake_ctx["layer_analysis"].debug["patterns"], self.config.patterns)

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_get_layer_scores_return_scores_when_called(self, mock_create_method):
        fake_method = self._build_fake_method()
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        processor._layer_scores = [{"name": "block.linear1", "score": 2.0}]

        self.assertEqual(processor.get_layer_scores(), [{"name": "block.linear1", "score": 2.0}])

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_preprocess_return_no_hooks_when_no_matching_layers(self, mock_create_method):
        fake_method = self._build_fake_method()
        fake_method.get_target_layers.return_value = ["block.linear1", "block.linear2"]
        fake_method.filter_layers_by_patterns.return_value = []
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        processor.preprocess(self.request)

        self.assertEqual(processor._target_layers, [])
        self.assertEqual(processor._hook_handles, {})


if __name__ == "__main__":
    unittest.main()
