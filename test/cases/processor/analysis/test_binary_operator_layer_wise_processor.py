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

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.analysis.binary_operator_layer_wise.processor import BinaryOperatorLayerWiseProcessor
from msmodelslim.utils.exception import UnexpectedError


class TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyDecoderLayer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestLayerAnalysisProcessor(unittest.TestCase):
    """测试 LayerAnalysisProcessor（mse_layer_wise 指标）。"""

    def setUp(self):
        self.model = TinyModel()
        self.adapter = MagicMock()
        self.config = SimpleNamespace(
            metrics="mse_layer_wise",
            quant_modules=["layers.0"],
            configs=[MagicMock(name="cfg1")],
        )
        self.request = BatchProcessRequest(
            name="layers.0",
            module=self.model.layers[0],
            datas=[torch.randn(1, 4)],
        )

    def _build_fake_method(self):
        fake_method = MagicMock()
        fake_method.name = "mse_layer_wise"
        return fake_method

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_init_set_empty_state_when_config_valid(self, mock_create_method, mock_from_config):
        fake_method = self._build_fake_method()
        qp = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = qp

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)

        self.assertEqual(processor.config, self.config)
        self.assertIs(processor.adapter, self.adapter)
        self.assertEqual(processor.quant_processors, [qp])
        mock_from_config.assert_has_calls(
            [
                call(self.model, self.config.configs[0], self.adapter),
            ]
        )
        mock_create_method.assert_called_once_with("mse_layer_wise", adapter=self.adapter)
        self.assertIs(processor._analysis_method, fake_method)
        self.assertEqual(processor._layer_scores, [])

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_pre_run_call_quant_processors_when_context_exists(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        fake_method = self._build_fake_method()
        qp = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = qp
        mock_get_current_context.return_value = {"layer_analysis": SimpleNamespace(debug={})}

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)
        processor.pre_run()

        qp.pre_run.assert_called_once()

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_pre_run_raise_unexpected_error_when_context_missing(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        fake_method = self._build_fake_method()
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = MagicMock()
        mock_get_current_context.return_value = None

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)

        with self.assertRaises(UnexpectedError):
            processor.pre_run()

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_process_call_quant_processors_in_order_when_quant_processors_exist(
        self, mock_create_method, mock_from_config
    ):
        fake_method = self._build_fake_method()
        fake_method.compute_score.return_value = 0.0
        mock_create_method.return_value = fake_method
        qp = MagicMock()
        mock_from_config.return_value = qp

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)
        with patch.object(processor, "_run_forward_if_need") as mock_run_forward:

            def _fake_forward(req):
                req.outputs = [torch.randn(1, 4)]

            mock_run_forward.side_effect = _fake_forward
            processor.process(self.request)

        qp.preprocess.assert_called_once_with(self.request)
        qp.process.assert_called_once_with(self.request)
        qp.postprocess.assert_called_once_with(self.request)

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_process_compute_score_and_append_layer_score(self, mock_create_method, mock_from_config):
        fake_method = self._build_fake_method()
        fake_method.compute_score.return_value = 0.5

        mock_create_method.return_value = fake_method
        mock_from_config.return_value = MagicMock()

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)
        self.assertEqual(processor._layer_scores, [])

        quant_out = torch.randn(1, 4)
        float_out = torch.randn(1, 4)

        with (
            patch.object(processor, "_run_forward_if_need") as mock_run_forward,
            patch.object(self.request.module, "load_state_dict", autospec=True) as mock_load_state,
        ):

            def _fake_forward(req):
                # process() does: quant forward first, then float forward.
                if not hasattr(_fake_forward, "called"):
                    _fake_forward.called = 0
                _fake_forward.called += 1
                req.outputs = [quant_out] if _fake_forward.called == 1 else [float_out]

            mock_run_forward.side_effect = _fake_forward

            processor.process(self.request)

            # Float state should be restored between 2 forwards
            mock_load_state.assert_called_once()
            _, kwargs = mock_load_state.call_args
            self.assertIn("strict", kwargs)
            self.assertFalse(kwargs["strict"])

        fake_method.compute_score.assert_called_once_with([float_out], [quant_out])
        self.assertEqual(processor._layer_scores, [{"name": "layers.0", "score": 0.5}])

    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.maybe_barrier_before_linear_quant")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_process_barriers_before_quant_when_distributed(self, mock_create_method, mock_from_config, mock_barrier):
        fake_method = self._build_fake_method()
        fake_method.compute_score.return_value = 0.0
        mock_create_method.return_value = fake_method
        qp = MagicMock()
        mock_from_config.return_value = qp

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)
        with patch.object(processor, "_run_forward_if_need") as mock_run_forward:

            def _fake_forward(req):
                req.outputs = [torch.randn(1, 4)]

            mock_run_forward.side_effect = _fake_forward
            processor.process(self.request)

        mock_barrier.assert_called_once()
        qp.preprocess.assert_called_once_with(self.request)

    @patch("msmodelslim.processor.analysis.distributed_utils.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_layer_wise.processor.LayerWiseMethodFactory.create_method")
    def test_post_run_call_quant_processors_and_set_context_when_scores_ready(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        fake_method = self._build_fake_method()
        fake_method.name = "mse_layer_wise"
        qp = MagicMock()
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = qp

        processor = BinaryOperatorLayerWiseProcessor(self.model, self.config, adapter=self.adapter)
        processor._layer_scores = [{"name": "layers.0", "score": 1.0}]
        fake_ctx = {"layer_analysis": SimpleNamespace(debug={})}
        mock_get_current_context.return_value = fake_ctx

        processor.post_run()

        qp.post_run.assert_called_once()
        self.assertEqual(fake_ctx["layer_analysis"].debug["layer_scores"], processor._layer_scores)
        self.assertEqual(fake_ctx["layer_analysis"].debug["method"], "mse_layer_wise")
        self.assertEqual(fake_ctx["layer_analysis"].debug["quant_modules"], self.config.quant_modules)


if __name__ == "__main__":
    unittest.main()
