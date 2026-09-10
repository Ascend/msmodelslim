# -*- coding: UTF-8 -*-

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

msmodelslim/infra/logging_inference_result_displayer.py 的单元测试。
"""

from unittest.mock import patch

from msmodelslim.core.infer_engine.interface import InferenceResult
from msmodelslim.infra.logging_inference_result_displayer import LoggingInferenceResultDisplayer


class TestLoggingInferenceResultDisplayer:
    """对应 LoggingInferenceResultDisplayer。"""

    @staticmethod
    def _result(generated_texts=None, generated_token_ids=None):
        return InferenceResult(
            generated_texts=generated_texts or [],
            generated_token_ids=generated_token_ids or [],
        )

    def test_display_logs_each_sample_when_single_device(self):
        displayer = LoggingInferenceResultDisplayer()
        result = self._result(
            generated_texts=["hello", "world"],
            generated_token_ids=[[1, 2], [3, 4]],
        )
        with patch("msmodelslim.infra.logging_inference_result_displayer.get_logger") as mock_logger:
            displayer.display_result(result, device_indices=None)

        mock_logger.return_value.info.assert_any_call("Fake-quant inference results (%d sample(s)%s):", 2, "")
        mock_logger.return_value.info.assert_any_call(
            "Fake-quant sample[%d] token_ids=%s generated_text=%r", 1, [3, 4], "world"
        )

    def test_display_logs_device_info_when_distributed(self):
        displayer = LoggingInferenceResultDisplayer()
        result = self._result(
            generated_texts=["a", "b", "c", "d"],
            generated_token_ids=[[1], [2], [3], [4]],
        )
        with patch("msmodelslim.infra.logging_inference_result_displayer.get_logger") as mock_logger:
            displayer.display_result(result, device_indices=[2, 3])

        # world_size = 2, header 带设备数
        mock_logger.return_value.info.assert_any_call(
            "Fake-quant inference results (%d sample(s)%s):", 4, ", 2 device(s)"
        )
        # sample 0 -> rank 0 -> device 2; sample 3 -> rank 1 -> device 3
        mock_logger.return_value.info.assert_any_call(
            "Fake-quant sample[%d] (rank=%d, device=%d) token_ids=%s generated_text=%r",
            0,
            0,
            2,
            [1],
            "a",
        )
        mock_logger.return_value.info.assert_any_call(
            "Fake-quant sample[%d] (rank=%d, device=%d) token_ids=%s generated_text=%r",
            3,
            1,
            3,
            [4],
            "d",
        )

    def test_display_logs_header_only_when_no_samples(self):
        displayer = LoggingInferenceResultDisplayer()
        result = self._result()
        with patch("msmodelslim.infra.logging_inference_result_displayer.get_logger") as mock_logger:
            displayer.display_result(result)

        calls = mock_logger.return_value.info.call_args_list
        assert len(calls) == 1
        assert calls[0][0] == ("Fake-quant inference results (%d sample(s)%s):", 0, "")

    def test_display_falls_back_empty_token_ids_when_token_list_shorter(self):
        displayer = LoggingInferenceResultDisplayer()
        result = self._result(
            generated_texts=["only-text"],
            generated_token_ids=[],
        )
        with patch("msmodelslim.infra.logging_inference_result_displayer.get_logger") as mock_logger:
            displayer.display_result(result)

        mock_logger.return_value.info.assert_any_call(
            "Fake-quant sample[%d] token_ids=%s generated_text=%r", 0, [], "only-text"
        )

    def test_display_treats_single_device_as_not_distributed(self):
        displayer = LoggingInferenceResultDisplayer()
        result = self._result(generated_texts=["x"], generated_token_ids=[[7]])
        with patch("msmodelslim.infra.logging_inference_result_displayer.get_logger") as mock_logger:
            displayer.display_result(result, device_indices=[0])

        # device_indices 长度 1 -> world_size=1 -> 非分布式分支
        mock_logger.return_value.info.assert_any_call("Fake-quant inference results (%d sample(s)%s):", 1, "")
        mock_logger.return_value.info.assert_any_call(
            "Fake-quant sample[%d] token_ids=%s generated_text=%r", 0, [7], "x"
        )
