#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

# pylint: disable=redefined-outer-name

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.core.quant_service.modelslim_v1.save.interface import AscendV1GlobalModelDtypeInterface
from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config, AscendV1Saver
from msmodelslim.ir.qal import QParam, QScheme, QStorage, QScope, QDType


def _make_w8a8_static_module(out_features=4, in_features=8):
    """Build a minimal W8A8StaticFakeQuantLinear for testing."""
    input_scale = torch.tensor([0.5], dtype=torch.float32)
    input_offset = torch.tensor([0.0], dtype=torch.float32)
    weight_scale = torch.ones(out_features, dtype=torch.float32) * 0.1
    weight = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
    bias = torch.zeros(out_features, dtype=torch.float32)

    x_q_param = QParam(
        scheme=QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=False),
        ext={"scale": input_scale, "offset": input_offset},
    )
    w_q_param = QParam(
        scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True),
        ext={"scale": weight_scale},
    )
    w_q = QStorage(dtype=QDType.INT8, value=weight)
    return qir.W8A8StaticFakeQuantLinear(x_q_param, w_q_param, w_q, bias)


class AdapterBf16(AscendV1GlobalModelDtypeInterface):
    """Adapter that reports bfloat16."""

    def __init__(self, model_path):
        self._model_path = Path(model_path)

    @property
    def model_path(self):
        return self._model_path

    def get_global_model_torch_dtype(self):
        return torch.bfloat16


class AdapterFloat32(AscendV1GlobalModelDtypeInterface):
    """Adapter that reports float32."""

    def __init__(self, model_path):
        self._model_path = Path(model_path)

    @property
    def model_path(self):
        return self._model_path

    def get_global_model_torch_dtype(self):
        return torch.float32


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    if os.path.exists(d):
        shutil.rmtree(d)


class TestResolveIsBf16FromAdapter:
    """Tests for AscendV1Saver._resolve_is_bf16_from_adapter (and thus _global_torch_dtype_is_bf16)."""

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized")
    def test_adapter_bf16_returns_true(self, mock_dist_init, temp_dir):
        mock_dist_init.return_value = False
        config = AscendV1Config(save_directory=temp_dir)
        model = nn.Linear(2, 2)
        adapter = AdapterBf16(temp_dir)
        saver = AscendV1Saver(model, config, adapter)
        assert saver._global_torch_dtype_is_bf16 is True

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized")
    def test_adapter_float32_returns_false(self, mock_dist_init, temp_dir):
        mock_dist_init.return_value = False
        config = AscendV1Config(save_directory=temp_dir)
        model = nn.Linear(2, 2)
        adapter = AdapterFloat32(temp_dir)
        saver = AscendV1Saver(model, config, adapter)
        assert saver._global_torch_dtype_is_bf16 is False

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized")
    def test_adapter_non_interface_returns_false(self, mock_dist_init, temp_dir):
        mock_dist_init.return_value = False
        config = AscendV1Config(save_directory=temp_dir)
        model = nn.Linear(2, 2)
        adapter = MagicMock()
        adapter.model_path = Path(temp_dir)
        saver = AscendV1Saver(model, config, adapter)
        assert saver._global_torch_dtype_is_bf16 is False


class TestW8A8DeqScaleWriteDtype:
    """W8A8 on_w8a8_static writes deq_scale as float32 when bf16, int64 when not bf16."""

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized")
    def test_w8a8_static_deq_scale_int64_when_not_bf16(self, mock_dist_init, temp_dir):
        mock_dist_init.return_value = False
        config = AscendV1Config(save_directory=temp_dir)
        model = nn.Linear(2, 2)
        adapter = AdapterFloat32(temp_dir)
        saver = AscendV1Saver(model, config, adapter)
        w8a8_module = _make_w8a8_static_module()

        with patch.object(saver, "write_tensor") as mock_write:
            saver.on_w8a8_static("layer.linear", w8a8_module)

        deq_calls = [c for c in mock_write.call_args_list if c[0][0].endswith(".deq_scale")]
        assert len(deq_calls) == 1
        _, _, tensor = deq_calls[0][0]
        assert tensor.dtype == torch.int64

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized")
    def test_w8a8_static_deq_scale_float32_when_bf16(self, mock_dist_init, temp_dir):
        mock_dist_init.return_value = False
        config = AscendV1Config(save_directory=temp_dir)
        model = nn.Linear(2, 2)
        adapter = AdapterBf16(temp_dir)
        saver = AscendV1Saver(model, config, adapter)
        w8a8_module = _make_w8a8_static_module()

        with patch.object(saver, "write_tensor") as mock_write:
            saver.on_w8a8_static("layer.linear", w8a8_module)

        deq_calls = [c for c in mock_write.call_args_list if c[0][0].endswith(".deq_scale")]
        assert len(deq_calls) == 1
        _, _, tensor = deq_calls[0][0]
        assert tensor.dtype == torch.float32
