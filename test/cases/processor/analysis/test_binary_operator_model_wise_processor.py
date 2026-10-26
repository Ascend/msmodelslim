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

from msmodelslim.utils.exception import UnexpectedError, UnsupportedError


class TinyIdentity(nn.Module):
    def forward(self, x, **_kwargs):
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
    def test_processor_init_sets_empty_state_when_config_valid(self, mock_create_method, mock_from_config):
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
        self.assertEqual(p._segment_layer_scores, [])

    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_pre_run_raises_unexpected_error_when_context_missing(
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
    def test_processor_pre_run_raises_unsupported_error_when_multi_device_not_supported(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        """多卡环境且指标未声明 supports_distributed 时，pre_run fail-fast。"""
        from msmodelslim.processor.analysis import distributed_utils
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        fake_method = self._build_fake_method()
        fake_method.supports_distributed = False
        mock_create_method.return_value = fake_method
        mock_from_config.return_value = MagicMock()
        mock_get_current_context.return_value = {"layer_analysis": SimpleNamespace(debug={})}

        p = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)
        with (
            patch.object(distributed_utils.dist, "is_initialized", return_value=True),
            patch.object(distributed_utils.dist, "get_world_size", return_value=2),
        ):
            with self.assertRaises(UnsupportedError):
                p.pre_run()

    @patch("msmodelslim.processor.analysis.distributed_utils.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_lifecycle_writes_scores_to_context_when_one_block_chains(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        """公开 lifecycle 跑通后，post_run 把当前链层分数与 quant_modules 写入 ctx。"""
        from msmodelslim.core.base.protocol import BatchProcessRequest
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
        request = BatchProcessRequest(
            name="model.layers.0",
            module=self.model,
            datas=[((torch.zeros(2, 3),), {})],
        )

        p.preprocess(request)
        p.process(request)
        p.postprocess(request)
        p.post_run()

        self.assertEqual(p._base_data_count, 1)
        self.assertEqual(p._block_names, ["model.layers.0"])
        # [ref, layer0_out]
        self.assertEqual(len(p._merged_outputs), 2)
        fake_method.compute_score.assert_called_once()
        self.assertEqual(
            ctx["layer_analysis"].debug["layer_scores"],
            [{"name": "model.layers.0", "score": 0.1}],
        )
        self.assertEqual(ctx["layer_analysis"].debug["method"], "mse_model_wise")
        self.assertEqual(ctx["layer_analysis"].debug["quant_modules"], list(self.config.quant_modules))

    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_preprocess_chains_previous_output_when_hidden_states_match(
        self, mock_create_method, mock_from_config
    ):
        """前一 block 输出与当前输入 hidden_states 一致时，preprocess 重建链式输入并保留非 hidden kwargs。"""
        from msmodelslim.core.base.protocol import BatchProcessRequest
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        mock_create_method.return_value = self._build_fake_method()
        mock_from_config.return_value = MagicMock()

        p = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)
        hidden = torch.zeros(2, 3)
        first = BatchProcessRequest(
            name="model.layers.0",
            module=self.model,
            datas=[((hidden,), {})],
        )
        p.preprocess(first)

        chained = BatchProcessRequest(
            name="model.layers.1",
            module=self.model,
            datas=[((hidden.clone(),), {"attention_mask": True})],
        )
        p.preprocess(chained)

        self.assertEqual(p._block_names, ["model.layers.0", "model.layers.1"])
        self.assertEqual(p._base_data_count, 1)
        args, kwargs = chained.datas[0]
        self.assertIsInstance(args[0], torch.Tensor)
        self.assertTrue(torch.equal(args[0], hidden))
        self.assertEqual(kwargs, {"attention_mask": True})

    @patch("msmodelslim.processor.analysis.distributed_utils.get_current_context")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.AutoSessionProcessor.from_config")
    @patch("msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method")
    def test_processor_post_run_validates_merged_outputs_when_input_chain_broken(
        self, mock_create_method, mock_from_config, mock_get_current_context
    ):
        """链断裂时会话结束再 post_run 应抛 UnexpectedError，避免写出不完整分数。"""
        from msmodelslim.core.base.protocol import BatchProcessRequest
        from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
            BinaryOperatorModelWiseProcessor,
        )

        mock_create_method.return_value = self._build_fake_method()
        mock_from_config.return_value = MagicMock()
        mock_get_current_context.return_value = {"layer_analysis": SimpleNamespace(debug={})}

        p = BinaryOperatorModelWiseProcessor(self.model, self.config, adapter=self.adapter)
        request = BatchProcessRequest(
            name="model.layers.0",
            module=self.model,
            datas=[((torch.zeros(2, 3),), {})],
        )
        p.preprocess(request)

        # 量化链中断：postprocess 未落地 merged outputs，会话结束校验应失败
        with self.assertRaises(UnexpectedError):
            p.post_run()

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
        request = BatchProcessRequest(
            name="block",
            module=self.model,
            datas=[((torch.zeros(1),), {})],
        )
        processor.preprocess(request)
        processor.process(request)

        mock_barrier.assert_called_once()
        qp.preprocess.assert_called_once_with(request)


if __name__ == "__main__":
    unittest.main()
