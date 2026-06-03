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

from msmodelslim.ir.qal import QStorage, QDType
from msmodelslim.core.quantizer.base import QConfig

from msmodelslim.core.quantizer.impl.fouroversix import WeightFouroverSixQuantizer


class TestWeightFouroverSixQuantizer:
    """测试 Weight FouroverSix 量化器"""

    @pytest.fixture
    def standard_config(self):
        """标准 mxfp4 per_block fouroversix 量化配置"""
        return QConfig(dtype="mxfp4", scope="per_block", method="fouroversix", symmetric=True, ext={"axes": -1})

    def test_initialization_success_when_valid_config(self, standard_config):
        """测试正常初始化时各项参数挂载符合预期"""
        quantizer = WeightFouroverSixQuantizer(standard_config)
        assert quantizer.axes == -1
        assert quantizer.block_size == 32  # mxfp4 默认 block_size
        assert quantizer.w_q_storage is None
        assert quantizer.w_q_param.ext["block_size"] == 32
        assert quantizer.w_q_param.ext["axes"] == -1
        assert quantizer.w_q_param.ext["scale"] is None

    @pytest.mark.parametrize(
        "axes_value",
        [
            -1,
            0,
            1,
            [-1],
            [0, 1],
        ],
    )
    def test_initialization_success_with_various_axes(self, axes_value):
        """测试不同 axes 配置下的初始化"""
        config = QConfig(
            dtype="mxfp4", scope="per_block", method="fouroversix", symmetric=True, ext={"axes": axes_value}
        )
        quantizer = WeightFouroverSixQuantizer(config)
        assert quantizer.axes == axes_value

    @pytest.mark.parametrize(
        "weight_tensor",
        [
            # 边界 1：全零矩阵 [1, 32]，对应 1 个块
            torch.zeros(1, 32),
            # 边界 2：标准块大小 [1, 64]，对应 2 个块
            torch.randn(1, 64),
            # 边界 3：多个通道 [2, 32]
            torch.randn(2, 32),
        ],
    )
    def test_init_weight_success_with_boundary_values(self, standard_config, weight_tensor):
        """测试不同边界条件下的权重初始化"""
        quantizer = WeightFouroverSixQuantizer(standard_config)
        weight_storage = QStorage(QDType.FLOAT, weight_tensor)
        quantizer.init_weight(weight_storage)

        assert quantizer.w_q_storage is not None
        assert quantizer.w_q_param.ext["scale"] is not None
        assert "scale" in quantizer.w_q_param.ext

    def test_init_weight_selects_best_scale_based_on_mse(self, standard_config):
        """测试 init_weight 能够根据 MSE 选择最优缩放方案"""
        quantizer = WeightFouroverSixQuantizer(standard_config)

        weight_tensor = torch.linspace(-6.0, 6.0, 32).unsqueeze(0)
        weight_storage = QStorage(QDType.FLOAT, weight_tensor)
        quantizer.init_weight(weight_storage)

        assert quantizer.w_q_storage is not None
        assert quantizer.w_q_param.ext["scale"] is not None
        selected_scale = quantizer.w_q_param.ext["scale"]
        assert selected_scale.dim() > 0

    def test_init_weight_with_zero_tensor(self, standard_config):
        """测试全零权重张量的量化处理"""
        quantizer = WeightFouroverSixQuantizer(standard_config)
        weight_tensor = torch.zeros(1, 32)
        weight_storage = QStorage(QDType.FLOAT, weight_tensor)
        quantizer.init_weight(weight_storage)

        assert quantizer.w_q_storage is not None
        assert quantizer.w_q_param.ext["scale"] is not None

    def test_forward_return_dequantized_value(self, standard_config):
        """测试前向传播能够正确反量化权重"""
        quantizer = WeightFouroverSixQuantizer(standard_config)

        weight_tensor = torch.linspace(-6.0, 6.0, 32).unsqueeze(0)
        weight_storage = QStorage(QDType.FLOAT, weight_tensor)
        quantizer.init_weight(weight_storage)

        output = quantizer(x=None)
        assert output is not None
        assert output.shape == weight_tensor.shape

    def test_forward_raise_error_when_not_initialized(self, standard_config):
        """测试异常边界：未初始化权重时调用 forward 会报错"""
        quantizer = WeightFouroverSixQuantizer(standard_config)
        with pytest.raises(Exception):
            quantizer(x=None)

    def test_get_q_storage_return_valid_storage_after_init(self, standard_config):
        """测试 get_q_storage 在初始化后返回有效的 QStorage"""
        quantizer = WeightFouroverSixQuantizer(standard_config)
        weight_tensor = torch.randn(1, 32)
        weight_storage = QStorage(QDType.FLOAT, weight_tensor)
        quantizer.init_weight(weight_storage)

        q_storage = quantizer.get_q_storage()
        assert q_storage is not None
        assert isinstance(q_storage, QStorage)

    def test_get_q_storage_return_none_before_init(self, standard_config):
        """测试 get_q_storage 在初始化前返回 None"""
        quantizer = WeightFouroverSixQuantizer(standard_config)
        q_storage = quantizer.get_q_storage()
        assert q_storage is None

    def test_get_q_param_return_valid_param(self, standard_config):
        """测试 get_q_param 返回有效的 QParam"""
        quantizer = WeightFouroverSixQuantizer(standard_config)
        q_param = quantizer.get_q_param()
        assert q_param is not None
        assert q_param.scheme.dtype.name == "MXFP4"
        assert q_param.scheme.scope.name == "PER_BLOCK"

    @pytest.mark.parametrize(
        "input_scale, expected_exp",
        [
            # 测试 e8m0 舍入：小于 0.5 不进，大于 0.5 进
            (torch.tensor(2.0), torch.tensor(1.0)),  # log2(2)=1.0 -> 1
            (torch.tensor(3.0), torch.tensor(2.0)),  # log2(3)≈1.58 -> 2 (>.5)
            (torch.tensor(1.5), torch.tensor(1.0)),  # log2(1.5)≈0.58 -> 1 (>.5 from 0)
            # 银行家舍入：尾数==0.5时，偶数进1，奇数不进
            (torch.tensor(2.0**0.5), torch.tensor(1.0)),  # log2(√2)=0.5, int=0(偶) -> 1
            (torch.tensor(2.0**1.5), torch.tensor(1.0)),  # log2(2√2)=1.5, int=1(奇) -> 1
            # 边界：<=0 强制为 0
            (torch.tensor(0.0), torch.tensor(0.0)),
            (torch.tensor(-1.0), torch.tensor(0.0)),
        ],
    )
    def test_nearest_neighbor_rounding_to_e8m0(self, standard_config, input_scale, expected_exp):
        """测试 e8m0 舍入函数的正确性"""
        quantizer = WeightFouroverSixQuantizer(standard_config)
        result = quantizer._WeightFouroverSixQuantizer__nearest_neighbor_rounding_to_e8m0(input_scale)
        assert torch.isclose(result, expected_exp, atol=1e-5)

    def test_mse_based_selection_between_two_scales(self, standard_config):
        """测试基于 MSE 的双缩放方案选择机制"""
        quantizer = WeightFouroverSixQuantizer(standard_config)

        weight_tensor = torch.cat([torch.linspace(-6.0, 6.0, 32), torch.linspace(-3.0, 3.0, 32)]).unsqueeze(0)
        weight_storage = QStorage(QDType.FLOAT, weight_tensor)
        quantizer.init_weight(weight_storage)

        assert quantizer.w_q_storage is not None
        selected_scale = quantizer.w_q_param.ext["scale"]
        assert selected_scale.shape[1] == 2  # 两个块

        # 验证两个块选择了不同的缩放方案
        # 第一个块 [-6, 6]：Scale-to-6 (max/6=1.0 -> e8m0=0) 应该更优
        # 第二个块 [-3, 3]：Scale-to-4 (max/4=0.75 -> e8m0=-1) 应该更优
        # 由于 MSE 比较，两个块应该选择不同的 scale
        assert selected_scale[0, 0] != selected_scale[0, 1], "两个块应该根据 MSE 选择不同的缩放方案"
