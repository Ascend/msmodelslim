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
# pylint: disable=consider-using-from-import,no-member,too-many-function-args

from unittest.mock import patch
import pytest
import torch.nn as nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.calibrator.calibrator_classes.fakequantize_calibrator import (
    FakeQuantizeCalibrator as F,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantModelJsonDescription  # noqa: F401
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim import logger as msmodelslim_logger


class TestF:
    @patch('os.environ', {'sourceBranch': 'test_branch'})
    def __init__(self):
        self.test_model = None
        self.model = nn.Linear(10, 2)
        self.logger = msmodelslim_logger
        self.description = {"model_quant_type": QuantType.W8A16}
        self.json_description = {"key1": "value1", "key2": "value2"}
        self.safetensor_weight = {"key1": "value1", "key2": "value2"}
        self.safetensor_mismatch = {"key3": "value3"}

    def test_cpu_device(self):
        dev_type, dev_id = F.check_device(None, "cpu")
        assert dev_type == "cpu"
        assert dev_id is None

    def test_gpu_device(self):
        dev_type, dev_id = F.check_device(0, "gpu")
        assert dev_type == "gpu"
        assert dev_id == 0

    def test_npu_device(self):
        dev_type, dev_id = F.check_device(0, "npu")
        assert dev_type == "npu"
        assert dev_id == 0

    def test_unsupported_device(self):
        with pytest.raises(ValueError):
            F.check_device(None, "unsupported")

    def test_invalid_dev_id(self):
        with pytest.raises(TypeError):
            F.check_device("invalid_id", "gpu")

    def test_model_type(self):
        model = nn.Linear(10, 2)
        assert isinstance(model, nn.Module)

    def test_invalid_model_type(self):
        model = "invalid"
        with pytest.raises(TypeError):
            F.check_model_type(model)

    def test_model_cpu_device(self):
        dev_type = "cpu"
        model = F.init_model_device(self.model, dev_type, self.logger)
        assert model.device.type == dev_type

    def test_model_gpu_device(self):
        if nn.cuda.is_available():
            dev_type = "cuda"
            model = F.init_model_device(self.model, dev_type, self.logger)
            assert model.device.type == dev_type
        else:
            pytest.skip("CUDA is not available")

    def test_invalid_device(self):
        dev_type = "invalid"
        with pytest.raises(RuntimeError):
            F.init_model_device(self.model, dev_type, self.logger)

    def test_invalid_quant_type(self):
        self.description["model_quant_type"] = "invalid"
        with pytest.raises(ValueError):
            self.test_model.init_quantize_type_model()

    def test_match(self):
        F.QuantModelJsonDescription.check_description_match(self.description, self.safetensor)

    def test_mismatch(self):
        with pytest.raises(ValueError):
            F.QuantModelJsonDescription.check_description_match(self.description, self.safetensor_mismatch)
