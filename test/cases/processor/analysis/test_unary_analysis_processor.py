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

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.analysis.unary_operator.processor import (
    UnaryAnalysisProcessor,
    UnaryAnalysisProcessorConfig,
)
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear2(self.linear1(x)))


class TestUnaryAnalysisProcessor(unittest.TestCase):
    """Unit tests for UnaryAnalysisProcessor public lifecycle."""

    def setUp(self):
        self.model = TinyBlock()
        self.config = UnaryAnalysisProcessorConfig(metrics="std", patterns=["*.linear1", "*.relu"])
        self.request = BatchProcessRequest(name="block", module=self.model, datas=[])

    @staticmethod
    def build_fake_method():
        fake_method = MagicMock()
        fake_method.name = "std"
        fake_method.pack_stats_for_distributed_merge.return_value = None
        fake_method.compute_score.return_value = 0.75
        fake_method.merge_distributed_stats.side_effect = lambda values: {
            "t_max": max(value["t_max"] for value in values),
            "t_min": min(value["t_min"] for value in values),
            "std": max(value["std"] for value in values),
        }
        fake_method.set_forward_kwargs = None
        return fake_method

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_init_returns_empty_scores_when_config_valid(self, mock_create_method):
        fake_method = self.build_fake_method()
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)

        mock_create_method.assert_called_once_with("std", adapter=None)
        self.assertIs(processor.config, self.config)
        self.assertTrue(processor.support_distributed())
        self.assertEqual(processor.get_layer_scores(), [])

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_init_passes_metric_params_when_metrics_ra_compress(self, mock_create_method):
        fake_method = self.build_fake_method()
        mock_create_method.return_value = fake_method
        config = UnaryAnalysisProcessorConfig(
            metrics="ra_compress",
            patterns=["*.linear1"],
            metric_params={"induction_head_ratio": 0.2, "echo_head_ratio": 0.05},
        )

        UnaryAnalysisProcessor(self.model, config)

        mock_create_method.assert_called_once_with(
            "ra_compress",
            adapter=None,
            induction_head_ratio=0.2,
            echo_head_ratio=0.05,
        )

    def test_config_raises_schema_error_when_metric_params_used_by_other_metrics(self):
        with self.assertRaises(SchemaValidateError) as ctx:
            UnaryAnalysisProcessorConfig(metrics="std", metric_params={"induction_head_ratio": 0.2})

        self.assertIn("does not accept metric_params", str(ctx.exception))

    def test_config_raises_schema_error_when_metric_params_key_unknown(self):
        with self.assertRaises(SchemaValidateError) as ctx:
            UnaryAnalysisProcessorConfig(metrics="ra_compress", metric_params={"head_ratio": 0.2})

        self.assertIn("Unsupported metric_params", str(ctx.exception))

    def test_config_raises_schema_error_when_ratio_out_of_range(self):
        with self.assertRaises(SchemaValidateError) as ctx:
            UnaryAnalysisProcessorConfig(metrics="ra_compress", metric_params={"induction_head_ratio": 1.5})

        self.assertIn("must be a number in [0, 1]", str(ctx.exception))

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_preprocess_registers_only_filtered_linear_hooks_when_patterns_match(self, mock_create_method):
        fake_method = self.build_fake_method()
        seen_layers = []

        def hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            del module, input_tensor, output_tensor, stats_dict
            seen_layers.append(layer_name)

        fake_method.get_target_layers.return_value = ["block.linear1", "block.linear2"]
        fake_method.filter_layers_by_patterns.return_value = ["block.linear1"]
        fake_method.get_hook.return_value = hook
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        processor.preprocess(self.request)
        self.model(torch.randn(1, 4))

        fake_method.get_target_layers.assert_called_once_with(self.model, "block")
        fake_method.filter_layers_by_patterns.assert_called_once_with(
            ["block.linear1", "block.linear2"],
            self.config.patterns,
        )
        self.assertEqual(seen_layers, ["block.linear1"])

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_postprocess_returns_score_when_local_stats_are_collected(self, mock_create_method):
        fake_method = self.build_fake_method()

        def hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            del module, input_tensor, output_tensor
            stats_dict[layer_name] = {"tensor": [1]}

        fake_method.get_target_layers.return_value = ["block.linear1"]
        fake_method.filter_layers_by_patterns.return_value = ["block.linear1"]
        fake_method.get_hook.return_value = hook
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        processor.preprocess(self.request)
        self.model(torch.randn(1, 4))
        processor.postprocess(self.request)

        self.assertEqual(processor.get_layer_scores(), [{"name": "block.linear1", "score": 0.75}])
        fake_method.compute_score.assert_called_once_with({"tensor": [1]})
        self.assertEqual(fake_method.pack_stats_for_distributed_merge.call_count, 1)

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_post_run_merges_packed_stats_when_distributed_stats_pending(self, mock_create_method):
        fake_method = self.build_fake_method()
        fake_method.pack_stats_for_distributed_merge.return_value = {
            "t_max": 2.0,
            "t_min": -1.0,
            "std": 0.5,
        }
        fake_method.compute_score.return_value = 4.0

        def hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            del module, input_tensor, output_tensor
            stats_dict[layer_name] = {"t_max": 2.0, "t_min": -1.0, "std": 0.5}

        fake_method.get_target_layers.return_value = ["block.linear1"]
        fake_method.filter_layers_by_patterns.return_value = ["block.linear1"]
        fake_method.get_hook.return_value = hook
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        processor.preprocess(self.request)
        self.model(torch.randn(1, 4))
        processor.postprocess(self.request)
        self.assertEqual(processor.get_layer_scores(), [])

        with patch(
            "msmodelslim.processor.analysis.unary_operator.processor.merge_packed_layer_stats_across_ranks",
            side_effect=lambda local, merge_fn: {k: merge_fn([v]) for k, v in local.items()},
        ) as mock_merge:
            processor.post_run()

        mock_merge.assert_called_once()
        fake_method.compute_score.assert_called_once_with({"t_max": 2.0, "t_min": -1.0, "std": 0.5})
        self.assertEqual(processor.get_layer_scores(), [{"name": "block.linear1", "score": 4.0}])

    @patch("msmodelslim.processor.analysis.distributed_utils.get_current_context")
    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_post_run_writes_context_when_local_scores_ready(self, mock_create_method, mock_get_current_context):
        fake_method = self.build_fake_method()

        def hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            del module, input_tensor, output_tensor
            stats_dict[layer_name] = {"tensor": [1]}

        fake_method.get_target_layers.return_value = ["block.linear1"]
        fake_method.filter_layers_by_patterns.return_value = ["block.linear1"]
        fake_method.get_hook.return_value = hook
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        processor.preprocess(self.request)
        self.model(torch.randn(1, 4))
        processor.postprocess(self.request)
        fake_ctx = {"layer_analysis": SimpleNamespace(state={}, debug={})}
        mock_get_current_context.return_value = fake_ctx

        processor.post_run()

        expected = [{"name": "block.linear1", "score": 0.75}]
        self.assertEqual(processor.get_layer_scores(), expected)
        self.assertEqual(fake_ctx["layer_analysis"].state["layer_scores"], expected)
        self.assertEqual(fake_ctx["layer_analysis"].state["method"], "std")
        self.assertEqual(fake_ctx["layer_analysis"].state["patterns"], self.config.patterns)

    @patch("msmodelslim.processor.analysis.unary_operator.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_pre_run_raises_unexpected_error_when_context_is_none(self, mock_create_method, mock_get_current_context):
        fake_method = self.build_fake_method()
        mock_create_method.return_value = fake_method
        mock_get_current_context.return_value = None

        processor = UnaryAnalysisProcessor(self.model, self.config)
        with self.assertRaises(UnexpectedError):
            processor.pre_run()

    @patch("msmodelslim.processor.analysis.unary_operator.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_post_run_returns_empty_scores_when_context_is_none(self, mock_create_method, mock_get_current_context):
        fake_method = self.build_fake_method()
        mock_create_method.return_value = fake_method
        mock_get_current_context.return_value = None

        processor = UnaryAnalysisProcessor(self.model, self.config)
        processor.post_run()

        self.assertEqual(processor.get_layer_scores(), [])

    @patch("msmodelslim.processor.analysis.unary_operator.processor.UnaryAnalysisMethodFactory.create_method")
    def test_get_layer_scores_returns_lifecycle_scores_when_called(self, mock_create_method):
        fake_method = self.build_fake_method()
        fake_method.compute_score.return_value = 2.0

        def hook(module, input_tensor, output_tensor, layer_name, stats_dict):
            del module, input_tensor, output_tensor
            stats_dict[layer_name] = {"tensor": [1]}

        fake_method.get_target_layers.return_value = ["block.linear1"]
        fake_method.filter_layers_by_patterns.return_value = ["block.linear1"]
        fake_method.get_hook.return_value = hook
        mock_create_method.return_value = fake_method

        processor = UnaryAnalysisProcessor(self.model, self.config)
        processor.preprocess(self.request)
        self.model(torch.randn(1, 4))
        processor.postprocess(self.request)

        self.assertEqual(processor.get_layer_scores(), [{"name": "block.linear1", "score": 2.0}])


if __name__ == "__main__":
    unittest.main()
