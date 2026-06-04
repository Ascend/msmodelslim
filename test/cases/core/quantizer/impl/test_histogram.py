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

import pytest
import torch

from msmodelslim import ir as qir
from msmodelslim.core.observer.histogram import HistogramObserver, SearchMethod
from msmodelslim.core.quantizer.base import AutoActQuantizer, QConfig
from msmodelslim.core.quantizer.impl.histogram import ActPerTensorHistogram
from msmodelslim.ir.qal.qbase import QDType, QScheme, QScope
from msmodelslim.utils.exception import SpecError


def to_qconfig(q_scheme: QScheme, method: str) -> QConfig:
    q_config = QConfig(
        dtype=q_scheme.dtype.value,
        scope=q_scheme.scope.value,
        symmetric=q_scheme.symmetric,
        method=method,
    )

    if q_scheme.scope == QScope.PER_GROUP:
        q_config.ext["group_size"] = 256

    return q_config


class TestActPerTensorHistogram:
    """Tests for ActPerTensorHistogram."""

    def test_initialization(self):
        """测试初始化"""
        config = QConfig(dtype="int8", scope="per_tensor", method="histogram", symmetric=True)

        quantizer = ActPerTensorHistogram(config)

        assert quantizer.config == config
        assert isinstance(quantizer.histogram_observer, HistogramObserver)
        assert quantizer.q_param is None

    def test_forward_then_can_get_correct_q_param(self):
        """测试前向传播并验证量化参数"""
        config = QConfig(dtype="int8", scope="per_tensor", method="histogram", symmetric=True)

        quantizer = ActPerTensorHistogram(config)

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = quantizer(x)

        q_param = quantizer.get_q_param()
        assert q_param
        assert q_param.scheme == config.to_scheme()
        assert isinstance(q_param.ext, dict)
        assert "scale" in q_param.ext
        assert "offset" in q_param.ext
        assert isinstance(q_param.ext["scale"], torch.Tensor)
        assert isinstance(q_param.ext["offset"], torch.Tensor)
        assert q_param.ext["scale"].shape == (1,)
        assert q_param.ext["offset"].shape == (1,)

        assert result.shape == x.shape

    def test_forward_with_batch_input(self):
        """测试批量输入"""
        config = QConfig(dtype="int8", scope="per_tensor", method="histogram", symmetric=True)

        quantizer = ActPerTensorHistogram(config)

        x = torch.randn(2, 3, 4)
        result = quantizer(x)
        assert result.shape == x.shape

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_tensor_sym, "histogram"),
            to_qconfig(qir.int8_per_tensor_asym, "histogram"),
        ],
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        """测试通过自动量化器创建"""
        quantizer = AutoActQuantizer.from_config(qconfig)
        assert isinstance(quantizer, ActPerTensorHistogram)

    def test_get_q_param_before_forward(self):
        """测试在forward之前获取q_param应该失败"""
        config = QConfig(dtype="int8", scope="per_tensor", method="histogram", symmetric=True)

        quantizer = ActPerTensorHistogram(config)

        with pytest.raises(SpecError, match="No q_param was set"):
            quantizer.get_q_param()

    def test_get_q_param_after_forward(self):
        """测试在forward之后可以正常获取q_param"""
        config = QConfig(dtype="int8", scope="per_tensor", method="histogram", symmetric=True)
        quantizer = ActPerTensorHistogram(config)
        x = torch.randn(8, 8)
        quantizer(x)
        q_param = quantizer.get_q_param()
        assert q_param is not None
        assert isinstance(q_param.ext, dict)
        assert "scale" in q_param.ext
        assert "offset" in q_param.ext
        assert isinstance(q_param.ext["scale"], torch.Tensor)
        assert isinstance(q_param.ext["offset"], torch.Tensor)
        assert q_param.scheme == config.to_scheme()

    def test_forward_with_edge_cases(self):
        """测试边界情况"""
        config = QConfig(dtype="int8", scope="per_tensor", method="histogram", symmetric=True)

        quantizer = ActPerTensorHistogram(config)

        x = torch.tensor([])
        with pytest.raises(SpecError, match="Input tensor is empty"):
            quantizer(x)

        with pytest.raises(SpecError, match="Input must be a valid torch.Tensor"):
            quantizer(None)

        with pytest.raises(SpecError, match="Input must be a valid torch.Tensor"):
            quantizer([1, 2, 3])

        x = torch.tensor([float("inf"), float("-inf")])
        with pytest.raises(SpecError, match="Input tensor is empty"):
            quantizer(x)

        x = torch.tensor([float("nan"), float("nan")])
        with pytest.raises(SpecError, match="Input tensor is empty"):
            quantizer(x)

    def test_forward_with_extreme_values(self):
        """测试极值输入"""
        config = QConfig(dtype="int8", scope="per_tensor", method="histogram", symmetric=True)

        quantizer = ActPerTensorHistogram(config)
        finfo_float32 = torch.finfo(torch.float32)
        x = torch.tensor([finfo_float32.min, finfo_float32.max, -finfo_float32.max]).to(torch.float32)
        result = quantizer(x)
        assert result.shape == x.shape
        assert not torch.isinf(result).any()
        assert not torch.isnan(result).any()

    def test_forward_with_same_values(self):
        """测试全等输入"""
        config = QConfig(dtype="int8", scope="per_tensor", method="histogram", symmetric=True)

        quantizer = ActPerTensorHistogram(config)

        x = torch.ones(10, 10)
        result = quantizer(x)
        assert result.shape == x.shape

    def test_histogram_observer_integration(self):
        """测试直方图观察器集成"""
        config = QConfig(dtype="int8", scope="per_tensor", method="histogram", symmetric=False)

        quantizer = ActPerTensorHistogram(config)

        assert quantizer.histogram_observer.config.search_method == SearchMethod.L2_NORM
        assert quantizer.histogram_observer.config.dtype == QDType.INT8
        assert quantizer.histogram_observer.config.scope == QScope.PER_TENSOR
        assert not quantizer.histogram_observer.config.symmetric

        x = torch.randn(10, 10)
        quantizer(x)

        clip_min, clip_max = quantizer.histogram_observer.get_clip_bounds()
        assert isinstance(clip_min, torch.Tensor)
        assert isinstance(clip_max, torch.Tensor)
        assert not torch.isnan(clip_min)
        assert not torch.isnan(clip_max)
        assert not torch.isinf(clip_min)
        assert not torch.isinf(clip_max)

    def test_quantizer_symmetric_vs_asymmetric(self):
        """测试对称和非对称量化"""
        for symmetric in [True, False]:
            config = QConfig(dtype="int8", scope="per_tensor", method="histogram", symmetric=symmetric)
            quantizer = ActPerTensorHistogram(config)

            x = torch.randn(5, 5)
            result = quantizer(x)
            assert result.shape == x.shape

            q_param = quantizer.get_q_param()
            assert q_param.scheme.symmetric == symmetric
