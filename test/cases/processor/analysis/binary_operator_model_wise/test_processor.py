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

Unit tests for BinaryOperatorModelWiseProcessor DP flag and skip mode.
"""

import unittest
from unittest.mock import MagicMock, patch

from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.analysis.binary_operator_model_wise.processor import (
    BinaryOperatorModelWiseProcessor,
    BinaryOperatorModelWiseProcessorConfig,
)


class TestBinaryOperatorModelWiseProcessorExtra(unittest.TestCase):
    @staticmethod
    def _fake_method():
        fake = MagicMock()
        fake.name = "mse_model_wise"
        return fake

    def _build_processor(self, fake_method):
        config = BinaryOperatorModelWiseProcessorConfig(metrics="mse_model_wise", quant_modules=["*"], configs=[])
        with patch(
            "msmodelslim.processor.analysis.binary_operator_model_wise.processor.ModelWiseMethodFactory.create_method",
            return_value=fake_method,
        ):
            return BinaryOperatorModelWiseProcessor(nn.Identity(), config)

    def test_support_distributed_requires_all_quant_processors(self):
        processor = self._build_processor(self._fake_method())
        qp_true = MagicMock()
        qp_true.support_distributed.return_value = True
        qp_false = MagicMock()
        qp_false.support_distributed.return_value = False

        processor.quant_processors = [qp_true, qp_true]
        self.assertTrue(processor.support_distributed())

        processor.quant_processors = [qp_true, qp_false]
        self.assertFalse(processor.support_distributed())

    def test_preprocess_in_skip_mode_records_name_and_forwards_once(self):
        processor = self._build_processor(self._fake_method())
        qp = MagicMock()
        processor.quant_processors = [qp]
        request = BatchProcessRequest(name="block.layers.1", module=nn.Identity(), datas=[])

        with patch.object(processor, "_skip_remaining_blocks", True):
            processor.preprocess(request)

        self.assertEqual(processor._skipped_request_names, ["block.layers.1"])
        self.assertEqual(request.outputs, [])
        qp.process.assert_not_called()

    def test_process_is_noop_in_skip_mode(self):
        processor = self._build_processor(self._fake_method())
        qp = MagicMock()
        processor.quant_processors = [qp]
        request = BatchProcessRequest(name="block.layers.1", module=nn.Identity(), datas=[])

        with patch.object(processor, "_skip_remaining_blocks", True):
            processor.process(request)

        qp.preprocess.assert_not_called()
        qp.process.assert_not_called()
        qp.postprocess.assert_not_called()


if __name__ == "__main__":
    unittest.main()
