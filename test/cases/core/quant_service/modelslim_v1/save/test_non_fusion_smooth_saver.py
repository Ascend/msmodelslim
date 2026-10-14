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

Unit tests for on_non_fusion_smooth_quant_wrapper (commit f2075e1).
- AutoSaverProcessor base raises NotImplementedError.
- AscendV1Saver writes mul_scale and processes wrapped linear.
"""
# pylint: disable=consider-using-from-import

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

import msmodelslim.ir as qir
from msmodelslim.core.quant_service.modelslim_v1.save.saver import AutoSaverProcessor
from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import (
    AscendV1Saver,
    AscendV1Config,
)


class TestAutoSaverProcessorOnNonFusionSmoothQuantWrapper(unittest.TestCase):
    """Base saver raises NotImplementedError for NonFusionSmoothQuantWrapper."""

    def test_base_on_non_fusion_smooth_quant_wrapper_raises(self):
        """AutoSaverProcessor.on_non_fusion_smooth_quant_wrapper raises NotImplementedError."""
        model = MagicMock(spec=nn.Module)
        config = MagicMock()
        adapter = MagicMock()
        saver = AutoSaverProcessor(model, config, adapter)
        linear = nn.Linear(2, 2)
        wrapper = qir.NonFusionSmoothQuantWrapper(torch.tensor([1.0, 1.0]), linear)
        with self.assertRaises(NotImplementedError) as ctx:
            saver.on_non_fusion_smooth_quant_wrapper("model.layer", wrapper)
        self.assertIn("on_non_fusion_smooth_quant_wrapper", str(ctx.exception))
        self.assertIn("AutoSaverProcessor", str(ctx.exception))


class TestAscendV1SaverOnNonFusionSmoothQuantWrapper(unittest.TestCase):
    """AscendV1Saver.on_non_fusion_smooth_quant_wrapper writes scale and processes linear."""

    def setUp(self):
        self.model = nn.Linear(2, 2)
        self.config = AscendV1Config(save_directory=".")
        self.adapter = MagicMock()
        self.saver = AscendV1Saver(self.model, self.config, self.adapter)

    def test_on_non_fusion_smooth_quant_wrapper_calls_write_tensor_and_process_module(self):
        """on_non_fusion_smooth_quant_wrapper writes div.mul_scale and processes .linear."""
        linear = nn.Linear(2, 2)
        scales = torch.tensor([2.0, 2.0])
        wrapper = qir.NonFusionSmoothQuantWrapper(scales, linear)
        prefix = "model.block.fc"

        with patch.object(self.saver, "write_tensor") as mock_write:
            with patch.object(self.saver, "_process_module") as mock_process:
                self.saver.on_non_fusion_smooth_quant_wrapper(prefix, wrapper)
        mock_write.assert_called_once()
        call_args = mock_write.call_args[0]
        self.assertEqual(call_args[0], "model.block.fc.div.mul_scale")
        self.assertEqual(call_args[1], "FLOAT")
        torch.testing.assert_close(call_args[2], wrapper.scales)
        mock_process.assert_called_once_with("model.block.fc.linear", linear)
