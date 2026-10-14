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

Unit tests for msmodelslim.processor.anti_outlier.default.model_adapter.
Covers: get_adapter_config_for_subgraph.
"""
# pylint: disable=consider-using-from-import,redefined-outer-name,reimported,undefined-variable

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.anti_outlier.default.model_adapter import get_adapter_config_for_subgraph


class _NormLinearModel(nn.Module):
    """Minimal model: norm -> linear, with device/config for llm_dummy_forward."""

    def __init__(self, dim=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(is_encoder_decoder=False)  # noqa: F821

    def forward(self, x):
        return self.linear(self.norm(x))


class TestGetAdapterConfigForSubgraph(unittest.TestCase):
    """Tests for get_adapter_config_for_subgraph."""

    def test_get_adapter_config_for_subgraph_returns_empty_list_when_model_is_none(self):
        request = BatchProcessRequest(name="dummy", module=None, datas=None, outputs=None)
        result = get_adapter_config_for_subgraph(request)
        self.assertEqual(result, [])

    @patch("msmodelslim.processor.anti_outlier.default.model_adapter.collect_shared_input_modules")
    def test_get_adapter_config_for_subgraph_returns_empty_list_and_logs_when_model_not_set(self, mock_collect):
        request = BatchProcessRequest(name="dummy", module=None, datas=None, outputs=None)
        with patch("msmodelslim.processor.anti_outlier.default.model_adapter.get_logger") as mock_logger:
            result = get_adapter_config_for_subgraph(request)
        self.assertEqual(result, [])
        mock_collect.assert_not_called()
        mock_logger.return_value.warning.assert_called()

    @patch("msmodelslim.processor.anti_outlier.default.model_adapter.collect_shared_input_modules")
    def test_get_adapter_config_for_subgraph_returns_adapter_config_list_when_norm_linear_detected(self, mock_collect):
        import torch

        fake_tensor = torch.ones(2, 2)
        norm_module = MagicMock()
        norm_module.name = "norm"
        norm_module.__class__.__name__ = "LayerNorm"
        linear_module = MagicMock()
        linear_module.name = "linear"
        input_to_linear = {fake_tensor: [linear_module]}
        output_to_layernorm = {fake_tensor: norm_module}
        mock_collect.return_value = (input_to_linear, output_to_layernorm)
        request = BatchProcessRequest(name="block", module=nn.Linear(2, 2), datas=None, outputs=None)
        with patch("msmodelslim.processor.anti_outlier.default.model_adapter.is_layernorm", return_value=True):
            result = get_adapter_config_for_subgraph(request)
        self.assertGreater(len(result), 0)
        for item in result:
            self.assertIsInstance(item, AdapterConfig)
            self.assertEqual(item.subgraph_type, "norm-linear")
            self.assertIsInstance(item.mapping, MappingConfig)
            self.assertIsNotNone(item.mapping.source)
            self.assertIsInstance(item.mapping.targets, list)

    @patch("msmodelslim.processor.anti_outlier.default.model_adapter.collect_shared_input_modules")
    def test_get_adapter_config_for_subgraph_returns_empty_list_when_collect_raises_exception(self, mock_collect):
        mock_collect.side_effect = RuntimeError("collect failed")
        model = nn.Linear(4, 4)
        request = BatchProcessRequest(name="block", module=model, datas=None, outputs=None)
        with patch("msmodelslim.processor.anti_outlier.default.model_adapter.get_logger"):
            result = get_adapter_config_for_subgraph(request)
        self.assertEqual(result, [])

    @patch("msmodelslim.processor.anti_outlier.default.model_adapter.collect_shared_input_modules")
    def test_get_adapter_config_for_subgraph_only_appends_when_producer_is_layernorm(self, mock_collect):
        # If producer is not layernorm, we should not add to subgraphs_list
        input_to_linear = {}
        output_to_layernorm = {}
        mock_collect.return_value = (input_to_linear, output_to_layernorm)
        request = BatchProcessRequest(name="block", module=nn.Linear(4, 4), datas=None, outputs=None)
        result = get_adapter_config_for_subgraph(request)
        self.assertEqual(result, [])
