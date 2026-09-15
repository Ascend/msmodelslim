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
from msmodelslim.ir.api import calculate_qparam, fake_quantize
from msmodelslim.ir.qal import QABCRegistry, QParam, QScheme, QScope, QDType, QStorage
from msmodelslim.core.observer import MsMinMaxObserver
from msmodelslim.core.quantizer.base import QConfig, AutoWeightQuantizer
from msmodelslim.core.quantizer.impl.hqq import (
    WeightPerChannelHQQ,
    hqq_calculate_qparam,
    get_ext_scale,
)
from msmodelslim.utils.exception import SpecError, UnsupportedError


def to_qconfig(symmetric: bool, dtype: str = "int8") -> QConfig:
    """构造 HQQ 权重量化配置（per_channel）。"""
    return QConfig(dtype=dtype, scope="per_channel", symmetric=symmetric, method="hqq")


def random_weight_qstorage(shape=(32, 16), seed: int = 0) -> QStorage:
    torch.manual_seed(seed)
    return QStorage(QDType.FLOAT, torch.randn(*shape))


class TestWeightPerChannelHQQ:  # pylint: disable=attribute-defined-outside-init
    """测试 Per-Channel HQQ 权重量化器（WeightPerChannelHQQ）。"""

    def setup_class(self):
        self.config = to_qconfig(symmetric=False)

    def test_registered_dispatch(self):
        """QABCRegistry 可按 (int8_per_channel_asym, 'hqq') 创建。"""
        quantizer = QABCRegistry.create(AutoWeightQuantizer, (qir.int8_per_channel_asym, "hqq"), self.config)
        assert isinstance(quantizer, WeightPerChannelHQQ)

        # 对称 scheme 未注册给 hqq，应报 UnsupportedError
        with pytest.raises(UnsupportedError):
            QABCRegistry.create(AutoWeightQuantizer, (qir.int8_per_channel_sym, "hqq"), self.config)

    def test_initialization(self):
        """构造后成员变量初值正确。"""
        quantizer = WeightPerChannelHQQ(self.config)

        assert quantizer.config == self.config, "config 不正确"
        assert isinstance(quantizer.minmax_observer, MsMinMaxObserver), "minmax_observer 不正确"
        assert quantizer.weight is None, "weight 初值应为 None"
        assert quantizer.bias is None, "bias 初值应为 None"
        assert quantizer.w_q_param is None, "w_q_param 初值应为 None"
        assert quantizer.w_q_storage is None, "w_q_storage 初值应为 None"
        assert quantizer.is_quantized is False, "is_quantized 初值应为 False"

    def test_validate_symmetric_true_raises(self):
        """symmetric=True 时 validate_ext_config 应直接报错（HQQ 仅支持非对称）。"""
        quantizer = WeightPerChannelHQQ(to_qconfig(symmetric=True))
        with pytest.raises(SpecError, match="asymmetric"):
            quantizer.validate_ext_config()

    def test_validate_asymmetric_ok(self):
        """symmetric=False 时 validate_ext_config 应通过。"""
        quantizer = WeightPerChannelHQQ(self.config)
        quantizer.validate_ext_config()  # 不应抛异常

    def test_forward_before_init_weight(self):
        """未 init_weight 就 forward 应报错。"""
        quantizer = WeightPerChannelHQQ(self.config)
        with pytest.raises(SpecError, match="No weight was set"):
            quantizer.forward(None)

    def test_forward_asymmetric(self):
        """symmetric=False 时正常量化，输出 shape 与权重一致。"""
        quantizer = WeightPerChannelHQQ(self.config)
        weight = random_weight_qstorage(shape=(32, 16))
        quantizer.init_weight(weight)

        out = quantizer.forward(None)

        assert quantizer.is_quantized is True, "量化后 is_quantized 应为 True"
        assert out.shape == (32, 16), f"输出 shape 不正确: {out.shape}"

    def test_get_q_storage_and_q_param_after_forward(self):
        """量化后可获取 q_storage 与 q_param。"""
        quantizer = WeightPerChannelHQQ(self.config)
        quantizer.init_weight(random_weight_qstorage())
        quantizer.forward(None)

        assert quantizer.get_q_storage() is not None, "q_storage 不应为 None"
        assert quantizer.get_q_param() is not None, "q_param 不应为 None"

    def test_get_q_param_offset_negated(self):
        """非对称量化：get_q_param 的 offset 应被取反（匹配 vllm 语义），且 scheme 为 neg_offset。"""
        quantizer = WeightPerChannelHQQ(self.config)
        quantizer.init_weight(random_weight_qstorage())
        quantizer.forward(None)

        q_param = quantizer.get_q_param()
        assert q_param.ext["offset"] is not None
        assert torch.equal(q_param.ext["offset"], -quantizer.w_q_param.ext["offset"]), "非对称 offset 应被取反"
        # 返回 NEG_OFFSET 专用 scheme，使 AutoFakeQuantLinear.create 命中 neg_offset IR
        assert q_param.scheme == qir.int8_per_channel_asym_neg_offset, "get_q_param 应返回 neg_offset 专用 scheme"

    def test_get_q_param_creates_neg_offset_ir(self):
        """get_q_param → AutoFakeQuantLinear.create 应命中 W8A16PerChannelNegOffsetFakeQuantLinear。"""
        quantizer = WeightPerChannelHQQ(self.config)
        quantizer.init_weight(random_weight_qstorage())
        quantizer.forward(None)

        x_q_param = QParam(scheme=QScheme(QScope.PER_TENSOR, QDType.FLOAT, True), ext={})
        ir = qir.AutoFakeQuantLinear.create(
            x_q_param,
            quantizer.get_q_param(),
            quantizer.get_q_storage(),
            None,
        )
        assert isinstance(ir, qir.W8A16PerChannelNegOffsetFakeQuantLinear), (
            f"deploy 应创建 neg_offset IR，实际为 {type(ir).__name__}"
        )

    def test_is_data_free(self):
        """HQQ 是 data-free 方法。"""
        quantizer = WeightPerChannelHQQ(self.config)
        assert quantizer.is_data_free() is True

    def test_support_distributed(self):
        """HQQ 支持分布式。"""
        quantizer = WeightPerChannelHQQ(self.config)
        assert quantizer.support_distributed() is True

    def test_ext_params_fixed(self):
        """ext 参数已固定：不再暴露 n_iter/beta/lp_norm 属性，config.ext 不参与调参。"""
        config = to_qconfig(symmetric=False)
        config.ext["n_iter"] = 50
        config.ext["beta"] = 999
        config.ext["lp_norm"] = 0.9

        quantizer = WeightPerChannelHQQ(config)

        assert not hasattr(quantizer, "n_iter"), "ext 参数已取消，不应有 n_iter 属性"
        assert not hasattr(quantizer, "beta"), "ext 参数已取消，不应有 beta 属性"
        assert not hasattr(quantizer, "lp_norm"), "ext 参数已取消，不应有 lp_norm 属性"

        quantizer.init_weight(random_weight_qstorage())
        quantizer.forward(None)
        assert quantizer.is_quantized is True, "固定参数下仍应能正常量化"


class TestHqqCalculateQparam:
    """测试 HQQ 核心算法函数 hqq_calculate_qparam。"""

    def test_non_2d_raises(self):
        """非 2D 权重应报 SpecError。"""
        weight = QStorage(QDType.FLOAT, torch.randn(4, 8, 16))
        q_param = QParam(
            scheme=QScheme(QScope.PER_CHANNEL, QDType.INT8, False),
            ext={"scale": torch.ones(8), "offset": torch.zeros(8)},
        )
        with pytest.raises(SpecError, match="2D"):
            hqq_calculate_qparam(weight, q_param)

    def test_error_not_worse_than_minmax(self):
        """HQQ 迭代后的重构误差不应劣于 MinMax 初始误差。"""
        torch.manual_seed(0)
        weight_tensor = torch.randn(32, 16)
        weight = QStorage(QDType.FLOAT, weight_tensor)

        # MinMax 初始量化参数
        min_val = weight_tensor.min(dim=0, keepdim=True).values
        max_val = weight_tensor.max(dim=0, keepdim=True).values
        init_q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_CHANNEL, symmetric=False)

        # 初始重构误差
        err_init = torch.mean((weight_tensor - fake_quantize(weight, init_q_param).value) ** 2)
        # HQQ 优化后重构误差
        hqq_q_param = hqq_calculate_qparam(weight, init_q_param)
        err_hqq = torch.mean((weight_tensor - fake_quantize(weight, hqq_q_param).value) ** 2)

        assert err_hqq.item() <= err_init.item() * 1.001, (
            f"HQQ 误差({err_hqq.item()})不应劣于 MinMax({err_init.item()})"
        )

    def test_scale_unchanged(self):
        """hqq_calculate_qparam 固定 scale，只优化 offset。"""
        torch.manual_seed(1)
        weight_tensor = torch.randn(32, 16)
        weight = QStorage(QDType.FLOAT, weight_tensor)

        min_val = weight_tensor.min(dim=0, keepdim=True).values
        max_val = weight_tensor.max(dim=0, keepdim=True).values
        init_q_param = calculate_qparam(min_val, max_val, QDType.INT8, QScope.PER_CHANNEL, symmetric=False)
        scale_before = get_ext_scale(init_q_param)

        hqq_q_param = hqq_calculate_qparam(weight, init_q_param)
        scale_after = get_ext_scale(hqq_q_param)

        assert torch.equal(scale_before, scale_after), "HQQ 迭代后 scale 应保持不变"
