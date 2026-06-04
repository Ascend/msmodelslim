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

from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig, LinearQuantizer
from msmodelslim.ir.qal.qbase import QDType, QScope
from msmodelslim.utils.exception import SchemaValidateError, SpecError


def _int8_config():
    """Weight-only INT8 per-channel symmetric (registered minmax scheme)."""
    weight = QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, method="minmax", symmetric=True)
    return LinearQConfig(weight=weight)


class TestLinearQConfig:
    """Tests for LinearQConfig."""

    def test_defaults_include_float_act_when_weight_only_given(self):
        """场景：仅配置 weight。预期：act 为默认 float none。"""
        weight = QConfig(dtype=QDType.INT8, scope=QScope.PER_TENSOR, method="minmax", symmetric=True)
        cfg = LinearQConfig(weight=weight)
        assert cfg.act.dtype == QDType.FLOAT

    def test_raises_schema_error_when_required_fields_missing(self):
        """场景：缺少必需字段。预期：SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            LinearQConfig()

    def test_stores_mxfp4_act_and_weight_when_given(self):
        """场景：传入 mxfp4 act/weight 配置。预期：字段保存正确。"""
        act_config = QConfig(dtype="mxfp4", scope="per_block", method="minmax", symmetric=True)
        weight_config = QConfig(dtype="mxfp4", scope="per_block", method="minmax", symmetric=True)
        config = LinearQConfig(act=act_config, weight=weight_config)
        assert config.act == act_config
        assert config.weight == weight_config


class TestLinearQuantizer:
    """Tests for LinearQuantizer external API."""

    def test_enable_sync_propagates_when_called(self):
        """场景：调用 enable_sync。预期：子量化器 enable_sync 被调用。"""
        quantizer = LinearQuantizer(_int8_config())
        quantizer.input_quantizer.enable_sync = MagicMock()
        quantizer.weight_quantizer.enable_sync = MagicMock()
        quantizer.enable_sync()
        quantizer.input_quantizer.enable_sync.assert_called_once()
        quantizer.weight_quantizer.enable_sync.assert_called_once()

    def test_support_distributed_returns_true_when_both_support(self):
        """场景：子量化器均支持分布式。预期：返回 True。"""
        quantizer = LinearQuantizer(_int8_config())
        quantizer.input_quantizer.support_distributed = MagicMock(return_value=True)
        quantizer.weight_quantizer.support_distributed = MagicMock(return_value=True)
        assert quantizer.support_distributed() is True

    def test_is_data_free_returns_true_when_both_data_free(self):
        """场景：子量化器均为 data_free。预期：返回 True。"""
        quantizer = LinearQuantizer(_int8_config())
        quantizer.input_quantizer.is_data_free = MagicMock(return_value=True)
        quantizer.weight_quantizer.is_data_free = MagicMock(return_value=True)
        assert quantizer.is_data_free() is True

    def test_validate_config_calls_sub_validators_when_invoked(self):
        """场景：调用 validate_config。预期：子量化器 validate_ext_config 被调用。"""
        quantizer = LinearQuantizer(_int8_config())
        quantizer.input_quantizer.validate_ext_config = MagicMock()
        quantizer.weight_quantizer.validate_ext_config = MagicMock()
        quantizer.validate_config()
        quantizer.input_quantizer.validate_ext_config.assert_called_once()
        quantizer.weight_quantizer.validate_ext_config.assert_called_once()

    def test_forward_runs_linear_when_setup_done(self):
        """场景：setup 后 forward。预期：输出 shape 正确。"""
        quantizer = LinearQuantizer(_int8_config())
        linear = nn.Linear(4, 3, bias=False)
        quantizer.setup(linear)
        out = quantizer(torch.randn(2, 4))
        assert out.shape == (2, 3)

    def test_deploy_returns_fake_quant_module_when_configured(self):
        """场景：正常 setup 后 deploy。预期：返回 fake quant 模块。"""
        quantizer = LinearQuantizer(_int8_config())
        linear = nn.Linear(4, 3, bias=False)
        quantizer.setup(linear)
        quantizer(torch.randn(2, 4))
        with patch("msmodelslim.core.quantizer.linear.qir.AutoFakeQuantLinear.create") as mock_create:
            mock_create.return_value = nn.Linear(4, 3)
            deployed = quantizer.deploy()
            assert deployed is not None
            mock_create.assert_called_once()

    def test_setup_raises_validation_error_when_module_none(self):
        """场景：setup 传入 None。预期：ValidationError。"""
        act_config = QConfig(dtype="mxfp4", scope="per_block", method="minmax", symmetric=True)
        weight_config = QConfig(dtype="mxfp4", scope="per_block", method="minmax", symmetric=True)
        quantizer = LinearQuantizer(LinearQConfig(act=act_config, weight=weight_config))
        with pytest.raises(ValidationError):
            quantizer.setup(None)

    def test_forward_raises_spec_error_when_not_setup_with_int8_config(self):
        """场景：int8 配置未 setup 直接 forward。预期：SpecError。"""
        act_config = QConfig(dtype="int8", scope="per_tensor", method="minmax", symmetric=True)
        weight_config = QConfig(dtype="int8", scope="per_channel", method="minmax", symmetric=True)
        quantizer = LinearQuantizer(LinearQConfig(act=act_config, weight=weight_config))
        with pytest.raises(SpecError):
            quantizer(torch.randn(10, 10))

    def test_deploy_returns_module_when_int8_setup_done(self):
        """场景：int8 配置 setup 后 deploy。预期：返回非 None 模块。"""
        act_config = QConfig(dtype="int8", scope="per_tensor", method="minmax", symmetric=True)
        weight_config = QConfig(dtype="int8", scope="per_channel", method="minmax", symmetric=True)
        config = LinearQConfig(act=act_config, weight=weight_config)
        quantizer = LinearQuantizer(config)
        quantizer.setup(nn.Linear(10, 10))
        quantizer(torch.randn((10,)))
        deployed_quantizer = quantizer.deploy()
        assert deployed_quantizer is not None
