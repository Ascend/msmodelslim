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

import pytest
import torch
from pydantic import ValidationError

from msmodelslim.ir.qal.qbase import QStorage, QDType, QScheme, QScope
from msmodelslim import ir as qir
from msmodelslim.core.observer import MsMinMaxObserver
from msmodelslim.core.quantizer.base import QConfig, AutoActQuantizer
from msmodelslim.core.quantizer.impl.minmax import ActPerTensorMinmax, ActPerTokenMinmax, WeightPerChannelMinmax


def to_qconfig(q_scheme: QScheme, method: str) -> QConfig:
    q_config = QConfig(
        dtype=q_scheme.dtype.value,
        scope=q_scheme.scope.value,
        symmetric=q_scheme.symmetric,
        method=method,
    )

    if q_scheme.scope == QScope.PER_GROUP:
        q_config.ext['group_size'] = 256

    return q_config


class TestActPerTensorMinmax:
    """测试Per-Tensor激活MinMax量化器"""

    def test_initialization(self):
        """测试初始化"""
        config = QConfig(dtype="int8", scope="per_tensor", method="minmax", symmetric=True)

        quantizer = ActPerTensorMinmax(config)

        assert quantizer.config == config
        assert isinstance(quantizer.minmax_observer, MsMinMaxObserver)
        assert quantizer.q_param is None

    def test_forward_then_can_get_correct_q_param(self):
        """测试前向传播并验证量化参数"""
        config = QConfig(dtype="int8", scope="per_tensor", method="minmax", symmetric=True)

        quantizer = ActPerTensorMinmax(config)

        # 测试输入
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = quantizer(x)

        # 验证q_param被设置
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

        # 验证输出形状
        assert result.shape == x.shape

    def test_forward_with_batch_input(self):
        config = QConfig(dtype="int8", scope="per_token", method="minmax", symmetric=True)

        quantizer = ActPerTokenMinmax(config)

        # 测试不同形状的输入
        x = torch.randn(2, 3, 4)  # (batch, seq, hidden)
        result = quantizer(x)
        assert result.shape == x.shape

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_tensor_sym, "minmax"),
            to_qconfig(qir.int8_per_tensor_asym, "minmax"),
        ],
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        """测试通过自动量化器创建"""
        quantizer = AutoActQuantizer.from_config(qconfig)
        assert isinstance(quantizer, ActPerTensorMinmax)


class TestActPerTokenMinmax:
    """测试Per-Token激活MinMax量化器"""

    def test_initialization(self):
        """测试初始化"""
        config = QConfig(dtype="int8", scope="per_token", method="minmax", symmetric=True)

        quantizer = ActPerTokenMinmax(config)

        assert quantizer.config == config
        assert isinstance(quantizer.minmax_observer, MsMinMaxObserver)
        assert quantizer.q_param is None

    def test_forward_then_can_get_correct_q_param(self):
        """测试前向传播并验证量化参数"""
        config = QConfig(dtype="int8", scope="per_token", method="minmax", symmetric=True)

        quantizer = ActPerTokenMinmax(config)

        # 测试输入 (batch_size, seq_len, hidden_dim)
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        original_shape = x.shape

        result = quantizer(x)

        # 验证q_param被设置
        q_param = quantizer.get_q_param()
        assert q_param
        assert q_param.scheme == config.to_scheme()

        # 验证输出形状保持不变
        assert result.shape == original_shape

    def test_forward_with_batch_input(self):
        """测试不同输入形状的处理"""
        config = QConfig(dtype="int8", scope="per_token", method="minmax", symmetric=True)

        quantizer = ActPerTokenMinmax(config)

        # 测试不同形状的输入
        x1 = torch.randn(2, 3, 4)  # (batch, seq, hidden)

        result1 = quantizer(x1)

        assert result1.shape == x1.shape

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_token_sym, "minmax"),
            to_qconfig(qir.int8_per_token_asym, "minmax"),
        ],
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        """测试通过自动量化器创建"""
        quantizer = AutoActQuantizer.from_config(qconfig)
        assert isinstance(quantizer, ActPerTokenMinmax)


class TestWeightPerChannelMinmax:  # pylint: disable=attribute-defined-outside-init
    """测试Per-Channel权重量化器"""

    def setup_class(self):
        self.config = QConfig(dtype="int8", scope="per_channel", method="minmax", symmetric=True)

    def test_initialization(self):
        """测试初始化"""
        quantizer = WeightPerChannelMinmax(self.config)

        assert quantizer.config == self.config
        assert quantizer.weight is None
        assert quantizer.bias is None

    def test_init_weight_validation(self):
        """测试权重初始化验证"""
        quantizer = WeightPerChannelMinmax(self.config)

        # 测试无效权重类型
        with pytest.raises(ValidationError, match="instance of QStorage"):
            quantizer.init_weight(torch.randn(10, 20))

        # 测试无效bias类型
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        with pytest.raises(ValidationError, match="instance of Tensor"):
            quantizer.init_weight(weight, bias="invalid")

    def test_init_weight_then_forward(self):
        """测试权重初始化并前向传播"""
        quantizer = WeightPerChannelMinmax(self.config)

        # 初始化权重
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        bias = torch.randn(20)

        quantizer.init_weight(weight, bias)

        assert quantizer.weight == weight
        assert quantizer.bias is bias

        # 前向传播
        result = quantizer()

        # 验证q_param被设置
        q_param = quantizer.get_q_param()
        assert q_param
        assert q_param.scheme == self.config.to_scheme()
        assert isinstance(q_param.ext, dict)
        assert "scale" in q_param.ext
        assert "offset" in q_param.ext
        assert isinstance(q_param.ext["scale"], torch.Tensor)
        assert isinstance(q_param.ext["offset"], torch.Tensor)
        # Per-channel的scale和offset应该与输出通道数匹配
        assert q_param.ext["scale"].shape == (weight.value.shape[0],)
        assert q_param.ext["offset"].shape == (weight.value.shape[0],)

        # 验证q_storage被设置
        q_storage = quantizer.get_q_storage()
        assert q_storage is not None

        # 验证输出形状
        assert result.shape == weight.value.shape

    def test_different_weight_shapes(self):
        """测试不同权重形状的处理"""

        # 测试不同形状的权重
        weight_shapes = [(10, 20), (32, 64), (128, 256)]

        for shape in weight_shapes:
            quantizer = WeightPerChannelMinmax(self.config)
            weight = QStorage(QDType.FLOAT, torch.randn(*shape))
            bias = torch.randn(shape[1])

            quantizer.init_weight(weight, bias)
            result = quantizer()
            q_param = quantizer.get_q_param()

            assert result.shape == weight.value.shape
            assert q_param is not None
            assert q_param.scheme == self.config.to_scheme()
            assert q_param.ext["scale"].shape == (shape[0],)
            assert q_param.ext["offset"].shape == (shape[0],)

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_channel_sym, "minmax"),
        ],
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        """测试通过自动量化器创建"""
        from msmodelslim.core.quantizer.base import AutoWeightQuantizer

        quantizer = AutoWeightQuantizer.from_config(qconfig)
        assert isinstance(quantizer, WeightPerChannelMinmax)
