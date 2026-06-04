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
from unittest.mock import MagicMock

from msmodelslim.ir.qal import QStorage, QDType, QParam
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError

from msmodelslim.core.quantizer.impl.dualscale import MXWeightDualScaleMinmax, MXActDualScaleMinmax


class TestMXWeightDualScaleMinmax:
    """测试 Weight 双级缩放 MinMax 量化器 (512外层大块场景)"""

    @pytest.fixture
    def standard_config(self):
        """标准 512 大块量化配置"""
        return QConfig(
            dtype="mxfp4",
            scope="per_block",
            method="dualscale",
            symmetric=True,
            ext={"axes": -1, "dual_block_size": 512},
        )

    def test_initialization_success_when_valid_config(self, standard_config):
        """测试正常初始化时各项参数挂载符合预期"""
        quantizer = MXWeightDualScaleMinmax(standard_config)
        assert quantizer.axes == -1
        assert quantizer.dual_block_size == 512
        assert quantizer.w_q_storage is None
        assert quantizer.w_q_param.ext["dual_block_size"] == 512

    def test_initialization_raise_schema_validate_error_when_invalid_axes_type(self):
        """测试异常边界：axes 传入非整数或非列表时抛出 SchemaValidateError"""
        invalid_config = QConfig(
            dtype="mxfp4",
            scope="per_block",
            method="dualscale",
            symmetric=True,
            ext={"axes": "invalid_axis_string", "dual_block_size": 512},
        )
        with pytest.raises(SchemaValidateError, match="Invalid value for 'axes'"):
            MXWeightDualScaleMinmax(invalid_config)

    @pytest.mark.parametrize(
        "weight_tensor, expected_dual_scale",
        [
            # 边界 1：全零矩阵边界 [1, 512]，对应 1 个大块
            (torch.zeros(1, 512), torch.zeros(1, 1)),
            # 边界 2：包含 2 个大块的矩阵 [1, 1024]
            # 第一块极大绝对值 12.0 -> dual_scale = 12 / 6 = 2.0
            # 第二块极大绝对值 24.0 -> dual_scale = 24 / 6 = 4.0
            (
                torch.cat(
                    [torch.linspace(-12.0, 12.0, 512).unsqueeze(0), torch.linspace(-24.0, 24.0, 512).unsqueeze(0)],
                    dim=-1,
                ),
                torch.tensor([[[2.0], [4.0]]]),  # 对应 Observer 产出的三维/多维嵌套形状
            ),
        ],
    )
    def test_dual_scale_calculate_correct_scale_when_init_weight_with_boundary_values(
        self, standard_config, weight_tensor, expected_dual_scale
    ):
        """测试不同极值边界下，init_weight 能够通过 Observer 准确计算出分块全局的 dual_scale"""
        quantizer = MXWeightDualScaleMinmax(standard_config)

        # Mock 隔离内层 32 规模小块量化器，拦截其初始化和获取行为
        quantizer.inner_quantizer.init_weight = MagicMock()
        quantizer.inner_quantizer.get_q_param = MagicMock(
            return_value=QParam(scheme=standard_config.to_scheme(), ext={"axes": [1]})
        )
        quantizer.inner_quantizer.get_q_storage = MagicMock(return_value=QStorage(QDType.FLOAT, weight_tensor))

        weight_storage = QStorage(QDType.FLOAT, weight_tensor)
        quantizer.init_weight(weight_storage)

        assert "dual_scale" in quantizer.w_q_param.ext
        calculated_dual_scale = quantizer.w_q_param.ext["dual_scale"]

        assert torch.allclose(calculated_dual_scale.flatten(), expected_dual_scale.flatten(), atol=1e-5)

    def test_forward_raise_unexpected_error_when_dual_scale_is_none(self, standard_config):
        """测试异常边界：若未执行权重初始化直接调用 forward，内部因拿不到 dual_scale 抛出 UnexpectedError"""
        quantizer = MXWeightDualScaleMinmax(standard_config)
        quantizer.w_q_param.ext["axes"] = [1]
        quantizer.inner_quantizer.forward = MagicMock(return_value=torch.randn(1, 512))

        with pytest.raises(UnexpectedError, match="The parameter 'dual_scale' cannot be None"):
            quantizer(x=None)

    def test_forward_return_correct_dequant_value_when_valid_inner_dequant_provided(self, standard_config):
        """测试前向传播：完成初始化后，forward 能够正确执行轴变换并将内层去量化值乘以外部全局大尺度"""
        quantizer = MXWeightDualScaleMinmax(standard_config)

        # 构造输入：[1, 512]，最大绝对值为 24.0 -> dual_scale 应当为 24.0 / 6.0 = 4.0
        weight_tensor = torch.linspace(-24.0, 24.0, 512).unsqueeze(0)
        weight_storage = QStorage(QDType.FLOAT, weight_tensor)

        # 设定内层量化器返回的模拟反量化值（全 1 矩阵）
        mock_inner_dequant = torch.ones(1, 512)
        quantizer.inner_quantizer.forward = MagicMock(return_value=mock_inner_dequant)
        quantizer.inner_quantizer.get_q_param = MagicMock(
            return_value=QParam(scheme=standard_config.to_scheme(), ext={"axes": [1]})
        )
        quantizer.inner_quantizer.get_q_storage = MagicMock(return_value=weight_storage)

        # 填充 axes 并计算出真正的 dual_scale (4.0)
        quantizer.init_weight(weight_storage)
        output = quantizer(x=None)

        # 预期输出 = 内层数据 (1.0) * 外层大块尺度 (4.0) = 4.0
        expected_output = mock_inner_dequant * 4.0
        assert torch.allclose(output, expected_output, atol=1e-5)

    def test_get_q_storage_raise_unexpected_error_when_storage_is_not_initialized(self, standard_config):
        """测试仓储获取异常：未做初始化时调取 w_q_storage 抛出异常"""
        quantizer = MXWeightDualScaleMinmax(standard_config)
        with pytest.raises(UnexpectedError, match="self.w_q_storage' cannot be None"):
            quantizer.get_q_storage()

    def test_get_q_param_raise_unexpected_error_when_param_is_none(self, standard_config):
        """测试参数获取异常：当 w_q_param 为空时引发 UnexpectedError"""
        quantizer = MXWeightDualScaleMinmax(standard_config)
        quantizer.w_q_param = None
        with pytest.raises(UnexpectedError, match="self.w_q_param' cannot be None"):
            quantizer.get_q_param()


class TestMXActDualScaleMinmax:
    """测试 Activation 双级缩放 MinMax 量化器"""

    @pytest.fixture
    def act_config(self):
        """标准激活层配置"""
        return QConfig(
            dtype="mxfp4",
            scope="per_block",
            method="dualscale",
            symmetric=True,
            ext={"axes": -1, "dual_block_size": 512},
        )

    def test_initialization_success_when_valid_config(self, act_config):
        """测试激活量化器初始化成功后，各核心内部状态、配置树及 data_free 属性的正确性"""
        quantizer = MXActDualScaleMinmax(act_config)
        assert quantizer.axes == -1
        assert quantizer.dual_block_size == 512
        assert quantizer.is_data_free() is True

    def test_initialization_raise_schema_validate_error_when_invalid_axes_type(self):
        """测试异常边界：初始化激活量化器时，传入非法非法的 axes 类型应抛出异常"""
        invalid_config = QConfig(
            dtype="mxfp4",
            scope="per_block",
            method="dualscale",
            symmetric=True,
            ext={"axes": {}, "dual_block_size": 512},  # 传入字典非法类型
        )
        with pytest.raises(SchemaValidateError, match="Invalid value for 'axes'"):
            MXActDualScaleMinmax(invalid_config)

    @pytest.mark.parametrize("boundary_tensor", [torch.randn(1, 1024), torch.randn(2, 512), torch.tensor([])])
    def test_forward_return_original_tensor_when_any_boundary_values_passed(self, act_config, boundary_tensor):
        """测试前向传播：根据 Data-Free 设计，激活层 forward 为透传逻辑，任何张量均原样返回"""
        quantizer = MXActDualScaleMinmax(act_config)
        output = quantizer(boundary_tensor)
        assert output is boundary_tensor

    def test_get_q_param_return_fallback_scheme_when_q_param_is_none(self, act_config):
        """测试参数安全读取：若 q_param 被意外置空，应通过 config 降级返回基础的量化 Scheme 结构"""
        quantizer = MXActDualScaleMinmax(act_config)
        quantizer.q_param = None

        fallback_param = quantizer.get_q_param()
        assert fallback_param is not None
        assert fallback_param.scheme == act_config.to_scheme()
