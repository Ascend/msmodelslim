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
# pylint: disable=consider-using-from-import

import unittest

import torch
import torch.nn as nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor import LoadProcessor, LoadProcessorConfig


class TestLoadProcessor(unittest.TestCase):
    """LoadProcessor测试类"""

    def setUp(self):
        """测试前的准备工作"""
        self.model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        self.sample_module = nn.Linear(5, 3)

    def test_load_processor_initialization(self):
        config = LoadProcessorConfig(device="cpu")
        processor = LoadProcessor(self.model, config)

        self.assertEqual(processor.device, "cpu")
        self.assertFalse(processor.non_blocking)
        self.assertTrue(processor.is_data_free())

    def test_load_processor_with_gpu_config(self):
        if torch.cuda.is_available():
            config = LoadProcessorConfig(device="cuda", non_blocking=True)
            processor = LoadProcessor(self.model, config)

            self.assertEqual(processor.device, "cuda")
            self.assertTrue(processor.non_blocking)

    def test_preprocess_device_movement(self):
        config = LoadProcessorConfig(device="cpu")
        processor = LoadProcessor(self.model, config)

        self.sample_module.to("cpu")

        request = BatchProcessRequest(name="test_module", module=self.sample_module, datas=None, outputs=None)

        processor.preprocess(request)

        self.assertEqual(next(self.sample_module.parameters()).device, torch.device("cpu"))

    def test_pre_run_model_movement(self):
        config = LoadProcessorConfig(device="cpu")
        processor = LoadProcessor(self.model, config)

        self.model.to("cpu")

        processor.pre_run()

        self.assertEqual(next(self.model.parameters()).device, torch.device("cpu"))

    def test_repr_method(self):
        config = LoadProcessorConfig(device="cpu", non_blocking=True)
        processor = LoadProcessor(self.model, config)

        expected_repr = "LoadProcessor(device=cpu, non_blocking=True)"
        self.assertEqual(repr(processor), expected_repr)

    def test_none_module_handling(self):
        config = LoadProcessorConfig(device="cpu")
        processor = LoadProcessor(self.model, config)

        request = BatchProcessRequest(name="none_module", module=None, datas=None, outputs=None)

        try:
            processor.preprocess(request)
            processor.postprocess(request)
        except Exception as e:
            self.fail(f"处理None模块时不应该抛出异常: {e}")

    def test_none_model_handling(self):
        config = LoadProcessorConfig(device="cpu")
        processor = LoadProcessor(None, config)

        try:
            processor.pre_run()
            processor.post_run()
        except Exception as e:
            self.fail(f"处理None模型时不应该抛出异常: {e}")


if __name__ == "__main__":
    unittest.main()
