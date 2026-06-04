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

from msmodelslim import ir as qir
from msmodelslim.core.quantizer.base import AutoWeightQuantizer, QConfig
from msmodelslim.core.quantizer.impl.gptq import (
    WeightPerChannelGPTQ,
    WeightPerGroupGPTQ,
    add_batch,
    calculate_hessian_inv,
    get_ext_value,
)
from msmodelslim.ir.qal.qbase import QDType, QScope, QScheme, QStorage
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


class DummyConfig:
    """用于测试 get_ext_value 的简易配置对象"""

    def __init__(self, ext):
        self.ext = ext


class TestGetExtValue:
    """测试 get_ext_value 辅助函数"""

    def test_return_default_when_config_is_none(self):
        default = 0.1
        assert get_ext_value(None, "percdamp", default) == default

    def test_return_default_when_ext_is_none(self):
        cfg = DummyConfig(ext=None)
        default = 0.2
        assert get_ext_value(cfg, "percdamp", default) == default

    def test_return_value_from_ext(self):
        cfg = DummyConfig(ext={"percdamp": 0.3})
        assert get_ext_value(cfg, "percdamp", 0.1) == pytest.approx(0.3)

    def test_ignore_none_in_ext_and_use_default(self):
        cfg = DummyConfig(ext={"percdamp": None})
        assert get_ext_value(cfg, "percdamp", 0.4) == pytest.approx(0.4)


class TestAddBatch:
    """测试 add_batch 辅助函数"""

    def test_initialize_hessian_when_none(self):
        x = torch.randn(4, 8)
        hessian, nsamples = add_batch(None, 0.0, x)

        assert hessian.shape == (8, 8)
        assert nsamples == 4
        assert torch.all(torch.isfinite(hessian))
        assert torch.trace(hessian) > 0

    def test_accumulate_hessian_and_nsamples(self):
        x1 = torch.randn(4, 8)
        hessian, nsamples = add_batch(None, 0.0, x1)

        x2 = torch.randn(6, 8)
        hessian2, nsamples2 = add_batch(hessian.clone(), nsamples, x2)

        assert hessian2.shape == (8, 8)
        assert nsamples2 == nsamples + 6
        assert not torch.allclose(hessian, hessian2)

    def test_support_3d_input(self):
        x = torch.randn(2, 3, 8)  # (batch, seq, hidden)
        hessian, nsamples = add_batch(None, 0.0, x)

        assert hessian.shape == (8, 8)
        # 2 * 3 = 6
        assert nsamples == 6


class TestCalculateHessianInv:
    """测试 calculate_hessian_inv 辅助函数"""

    def test_return_positive_definite_like_inverse(self):
        columns = 6
        rows = 10
        weight = torch.randn(rows, columns, dtype=torch.float32)

        # 构造正定矩阵 hessian
        a = torch.randn(columns, columns)
        hessian = a.t().matmul(a)

        percdamp = 0.01
        h_inv = calculate_hessian_inv(hessian.clone(), percdamp, weight)

        assert h_inv.shape == (columns, columns)
        assert h_inv.dtype == weight.dtype
        # 对角线应为有限且大于0
        diag = torch.diag(h_inv)
        assert torch.all(torch.isfinite(diag))
        assert torch.all(diag > 0)

    def test_handle_dead_columns(self):
        columns = 4
        rows = 6
        weight = torch.randn(rows, columns, dtype=torch.float32)

        # 制造部分对角线为 0 的 hessian
        hessian = torch.eye(columns)
        hessian[1, 1] = 0.0

        percdamp = 0.01
        h_inv = calculate_hessian_inv(hessian, percdamp, weight)

        assert h_inv.shape == (columns, columns)
        # dead 列对应的权重应被置为 0
        assert torch.all(weight[:, 1] == 0)


class TestWeightPerChannelGPTQ:  # pylint: disable=attribute-defined-outside-init
    """测试 Per-Channel GPTQ 量化器"""

    def setup_class(self):
        self.config = QConfig(
            dtype="int8",
            scope="per_channel",
            method="gptq",
            symmetric=True,
        )

    def test_initialization(self):
        quantizer = WeightPerChannelGPTQ(self.config)

        assert quantizer.config == self.config
        assert quantizer.weight is None
        assert quantizer.bias is None
        assert quantizer.w_q_param is None
        assert quantizer.w_q_storage is None
        assert quantizer.hessian is None
        assert quantizer.nsamples == 0
        assert isinstance(quantizer.percdamp, float)
        assert isinstance(quantizer.block_size, int)

    def test_ext_overrides_percdamp_and_block_size(self):
        cfg = QConfig(
            dtype="int8",
            scope="per_channel",
            method="gptq",
            symmetric=True,
        )
        cfg.ext["percdamp"] = 0.2
        cfg.ext["block_size"] = 64

        quantizer = WeightPerChannelGPTQ(cfg)

        assert quantizer.percdamp == pytest.approx(0.2)
        assert quantizer.block_size == 64

    def test_forward_collects_hessian(self):
        quantizer = WeightPerChannelGPTQ(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(4, 8))
        bias = torch.randn(8)
        quantizer.init_weight(weight, bias)

        x = torch.randn(2, 3, 8)
        out = quantizer(x)

        assert quantizer.hessian is not None
        assert quantizer.hessian.shape == (8, 8)
        assert quantizer.nsamples == 6
        # forward 返回原始权重
        assert torch.allclose(out, weight.value)

    def test_get_q_storage_without_hessian_raises(self):
        quantizer = WeightPerChannelGPTQ(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(4, 8))
        quantizer.init_weight(weight)

        with pytest.raises(SpecError, match="No hessian was set"):
            _ = quantizer.get_q_storage()

    def test_get_q_param_without_hessian_raises(self):
        quantizer = WeightPerChannelGPTQ(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(4, 8))
        quantizer.init_weight(weight)

        with pytest.raises(SpecError, match="No hessian was set"):
            _ = quantizer.get_q_param()

    def test_quantization_flow_can_get_q_storage_and_param(self):
        quantizer = WeightPerChannelGPTQ(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(4, 8))
        bias = torch.randn(8)
        quantizer.init_weight(weight, bias)

        # 收集 hessian
        x = torch.randn(3, 8)
        _ = quantizer(x)

        q_storage = quantizer.get_q_storage()
        q_param = quantizer.get_q_param()

        assert q_storage is not None
        assert q_param is not None
        assert q_param.scheme == self.config.to_scheme()
        assert "scale" in q_param.ext
        assert "offset" in q_param.ext

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_channel_sym, "gptq"),
            to_qconfig(qir.int8_per_channel_asym, "gptq"),
            to_qconfig(qir.int4_per_channel_sym, "gptq"),
            to_qconfig(qir.int4_per_channel_asym, "gptq"),
        ],
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        quantizer = AutoWeightQuantizer.from_config(qconfig)
        assert isinstance(quantizer, WeightPerChannelGPTQ)


class TestWeightPerGroupGPTQ:  # pylint: disable=attribute-defined-outside-init
    """测试 Per-Group GPTQ 量化器"""

    def setup_class(self):
        self.config = QConfig(
            dtype="int8",
            scope="per_group",
            method="gptq",
            symmetric=True,
        )
        self.config.ext["group_size"] = 2

    def test_initialization(self):
        quantizer = WeightPerGroupGPTQ(self.config)

        assert quantizer.config == self.config
        assert quantizer.weight is None
        assert quantizer.bias is None
        assert quantizer.w_q_param is None
        assert quantizer.w_q_storage is None
        assert quantizer.hessian is None
        assert quantizer.nsamples == 0
        assert isinstance(quantizer.percdamp, float)
        assert isinstance(quantizer.block_size, int)
        assert isinstance(quantizer.group_size, int)

    def test_forward_collects_hessian(self):
        quantizer = WeightPerGroupGPTQ(self.config)
        # 列数需能被 group_size 整除
        weight = QStorage(QDType.FLOAT, torch.randn(4, 8))
        bias = torch.randn(8)
        quantizer.init_weight(weight, bias)

        x = torch.randn(2, 3, 8)
        out = quantizer(x)

        assert quantizer.hessian is not None
        assert quantizer.hessian.shape == (8, 8)
        assert quantizer.nsamples == 6
        assert torch.allclose(out, weight.value)

    def test_get_q_storage_without_hessian_raises(self):
        quantizer = WeightPerGroupGPTQ(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(4, 8))
        quantizer.init_weight(weight)

        with pytest.raises(SpecError, match="No hessian was set"):
            _ = quantizer.get_q_storage()

    def test_get_q_param_without_hessian_raises(self):
        quantizer = WeightPerGroupGPTQ(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(4, 8))
        quantizer.init_weight(weight)

        with pytest.raises(SpecError, match="No hessian was set"):
            _ = quantizer.get_q_param()

    def test_quantization_flow_can_get_q_storage_and_param_and_group_ext(self):
        quantizer = WeightPerGroupGPTQ(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(4, 8))
        bias = torch.randn(8)
        quantizer.init_weight(weight, bias)

        x = torch.randn(3, 8)
        _ = quantizer(x)

        q_storage = quantizer.get_q_storage()
        q_param = quantizer.get_q_param()

        assert q_storage is not None
        assert q_param is not None
        assert q_param.scheme == self.config.to_scheme()
        assert "scale" in q_param.ext
        assert "offset" in q_param.ext
        assert "group_size" in q_param.ext
        # scale/offset 形状应为 (out_channels, num_groups)
        out_channels = weight.value.shape[0]
        num_groups = weight.value.shape[1] // self.config.ext["group_size"]
        assert q_param.ext["scale"].shape == (out_channels, num_groups)
        assert q_param.ext["offset"].shape == (out_channels, num_groups)

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_group_sym, "gptq"),
            to_qconfig(qir.int8_per_group_asym, "gptq"),
            to_qconfig(qir.int4_per_group_sym, "gptq"),
            to_qconfig(qir.int4_per_group_asym, "gptq"),
        ],
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        quantizer = AutoWeightQuantizer.from_config(qconfig)
        assert isinstance(quantizer, WeightPerGroupGPTQ)
