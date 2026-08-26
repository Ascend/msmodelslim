"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
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

from msmodelslim.utils.exception import UnexpectedError, UnsupportedError


class TinyIdentity(nn.Module):
    def forward(self, x):
        return x


class TestBinaryOperatorModelWiseProcessor(unittest.TestCase):
    """测试 BinaryOperatorModelWiseProcessor（模型级敏感层分析处理器）。"""

    def setUp(self):
        self.model = TinyIdentity()
        self.adapter = MagicMock()
        self.config = SimpleNamespace(
            metrics="mse_model_wise",
            quant_modules=["*mlp*"],
            configs=[MagicMock(name="cfg1")],
        )

    def _build_fake_method(self):
        fake_method = MagicMock()
        fake_method.name = "mse_model_wise"
        fake_method.compute_score.return_value = 0.1
        return fake_method

    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_init_shouldSetEmptyState_whenConfigValid(self, mock_create_method, mock_from_config):
        """测试 Processor 初始化：config 合法时状态清空，并按 metrics 创建分析方法。"""
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        fake_method = self._build_fake_method()
        mock_create_method.return_value = fake_method
        qp = MagicMock()
        mock_from_config.return_value = qp

        p = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)

        mock_create_method.assert_called_once_with("mse_model_wise", adapter=self.adapter)
        self.assertEqual(p.quant_processors, [qp])
        self.assertEqual(p._base_data_count, 0)
        self.assertEqual(p._block_names, [])
        self.assertEqual(p._float_outputs, [])
        self.assertEqual(p._quant_inputs, [])
        self.assertEqual(p._merged_outputs, [])

    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_preRun_shouldRaiseUnexpectedError_whenContextMissing(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        """测试 Processor.pre_run：上下文缺失时抛 UnexpectedError。"""
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        mock_create_method.return_value = self._build_fake_method()
        mock_from_config.return_value = MagicMock()
        mock_get_current_context.return_value = None

        p = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)
        with self.assertRaises(UnexpectedError):
            p.pre_run()

    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_postRun_shouldRaiseUnexpectedError_whenMergedOutputsLengthInvalid(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        """测试 Processor.post_run：merged_outputs 数量不足时抛 UnexpectedError。"""
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        mock_create_method.return_value = self._build_fake_method()
        mock_from_config.return_value = MagicMock()
        mock_get_current_context.return_value = {"layer_analysis": SimpleNamespace(debug={})}

        p = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)
        p._base_data_count = 2
        p._block_names = ["block0", "block1"]
        p._merged_outputs = [torch.zeros(1)]  # expected 2 * (2+1) = 6

        with self.assertRaises(UnexpectedError):
            p.post_run()

    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_replaceDatas_shouldRaiseUnsupportedError_whenHiddenStatesMismatch(
        self, mock_create_method, mock_from_config
    ):
        """测试 replace_datas：hidden_states 不一致时抛 UnsupportedError。"""
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        mock_create_method.return_value = self._build_fake_method()
        mock_from_config.return_value = MagicMock()

        p = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)
        p._base_data_count = 1

        datas = [((torch.zeros(2, 3),), {})]
        # previous merged output has different tensor => should raise
        p._merged_outputs = [torch.ones(2, 3)]

        with self.assertRaises(UnsupportedError):
            p._replace_request_datas_with_merged_outputs_if_need(datas)

    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_replaceDatas_shouldKeepOriginalRows_whenBoundaryNotChainable(
        self, mock_create_method, mock_from_config
    ):
        """visual 等非链式边界：无法从上一 block 输出注入 hidden 时保留 generator datas。"""
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        mock_create_method.return_value = self._build_fake_method()
        mock_from_config.return_value = MagicMock()

        p = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)
        p._base_data_count = 1

        hidden = torch.zeros(2, 3)
        datas = [((hidden,), {"attention_mask": torch.ones(2, 8)})]
        p._merged_outputs = [{"pooler_output": torch.ones(4, 5)}]

        new_rows = p._replace_request_datas_with_merged_outputs_if_need(datas)
        self.assertIs(new_rows, datas)

    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_replaceDatas_shouldReturnRebuiltRows_whenHiddenStatesMatch(
        self, mock_create_method, mock_from_config
    ):
        """测试 replace_datas：hidden_states 一致时返回按 merged_outputs 重建的 datas。"""
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        mock_create_method.return_value = self._build_fake_method()
        mock_from_config.return_value = MagicMock()

        p = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)
        p._base_data_count = 1

        x = torch.zeros(2, 3)
        datas = [((x,), {"foo": "bar"})]
        p._merged_outputs = [x.clone()]

        new_rows = p._replace_request_datas_with_merged_outputs_if_need(datas)
        self.assertEqual(len(new_rows), 1)
        (args, kwargs) = new_rows[0]
        self.assertTrue(torch.allclose(args[0], x))
        self.assertEqual(kwargs, {"foo": "bar"})

    @patch("msmodelslim.processor.analysis.distributed_utils.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_postRun_shouldWriteContextAndAnnotateNames_whenScoresReady(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        """测试 Processor.post_run：分数可计算时写入 ctx.debug（含 quant_modules）。"""
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        fake_method = self._build_fake_method()
        fake_method.name = "mse_model_wise"
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = MagicMock()

        ctx = {"layer_analysis": SimpleNamespace(debug={})}
        mock_get_current_context.return_value = ctx

        p = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)
        p._base_data_count = 1
        p._block_names = ["model.layers.0"]
        # layout: [ref0, layer0_out0]
        p._merged_outputs = [torch.zeros(1), torch.ones(1)]

        p.post_run()

        self.assertIn("layer_scores", ctx["layer_analysis"].debug)
        self.assertEqual(ctx["layer_analysis"].debug["method"], "mse_model_wise")
        self.assertEqual(ctx["layer_analysis"].debug["quant_modules"], list(self.config.quant_modules))
        self.assertEqual(len(ctx["layer_analysis"].debug["layer_scores"]), 1)
        row = ctx["layer_analysis"].debug["layer_scores"][0]
        self.assertEqual(row["name"], "model.layers.0")
        self.assertEqual(row["score"], 0.1)

    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.maybe_barrier_before_linear_quant")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_process_barriers_before_quant_when_distributed(self, mock_create_method, mock_from_config, mock_barrier):
        from msmodelslim.core.base.protocol import BatchProcessRequest
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        mock_create_method.return_value = self._build_fake_method()
        qp = MagicMock()
        mock_from_config.return_value = qp

        processor = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)
        processor._quant_inputs = [((torch.zeros(1),), {})]
        request = BatchProcessRequest(name="block", module=self.model, datas=[])

        processor.process(request)

        mock_barrier.assert_called_once()
        qp.preprocess.assert_called_once_with(request)


if __name__ == "__main__":
    unittest.main()
