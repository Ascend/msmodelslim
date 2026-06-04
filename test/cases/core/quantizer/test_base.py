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
from pydantic import ValidationError

from msmodelslim.core.quantizer.base import AutoActQuantizer, AutoWeightQuantizer, QConfig
from msmodelslim.ir.qal.qbase import QDType, QScheme, QScope
from msmodelslim.utils.exception import SchemaValidateError


class TestQConfig:
    """Tests for QConfig."""

    def test_qconfig_creation(self):
        """测试QConfig创建"""
        config = QConfig(dtype="int8", scope="per_tensor", method="minmax", symmetric=True, ext={"group_size": 128})

        assert config.dtype == QDType.INT8
        assert config.scope == QScope.PER_TENSOR
        assert config.method == "minmax"
        assert config.symmetric is True
        assert config.ext == {"group_size": 128}

    def test_qconfig_default_ext(self):
        """测试QConfig默认ext参数"""
        config = QConfig(dtype="int8", scope="per_tensor", method="minmax", symmetric=True)

        assert config.ext == {}

    def test_qconfig_to_scheme(self):
        """测试QConfig转换为QScheme"""
        config = QConfig(dtype="int8", scope="per_tensor", method="minmax", symmetric=True)

        scheme = config.to_scheme()
        assert isinstance(scheme, QScheme)
        assert scheme.scope == QScope.PER_TENSOR
        assert scheme.dtype == QDType.INT8
        assert scheme.symmetric is True

    def test_qconfig_to_scheme_asymmetric(self):
        """测试非对称量化配置转换"""
        config = QConfig(dtype="int8", scope="per_channel", method="minmax", symmetric=False)

        scheme = config.to_scheme()
        assert scheme.scope == QScope.PER_CHANNEL
        assert scheme.dtype == QDType.INT8
        assert scheme.symmetric is False

    def test_qconfig_validation_invalid_dtype(self):
        """测试无效dtype参数验证"""
        with pytest.raises(SchemaValidateError):
            QConfig.model_validate(dict(dtype="invalid_dtype", scope="per_tensor", method="minmax", symmetric=True))

    def test_qconfig_validation_invalid_scope(self):
        """测试无效scope参数验证"""
        with pytest.raises(SchemaValidateError):
            QConfig(dtype="int8", scope="invalid_scope", method="minmax", symmetric=True)

    def test_qconfig_validation_missing_required_fields(self):
        """测试缺失必需字段验证"""
        with pytest.raises(SchemaValidateError):
            QConfig(
                dtype="int8",
                method="minmax",
                symmetric=True,
            )

    def test_qconfig_with_ext_parameters(self):
        """测试带扩展参数的配置"""
        config = QConfig(
            dtype="int8",
            scope="per_group",
            method="minmax",
            symmetric=True,
            ext={"group_size": 64, "custom_param": "value"},
        )

        assert config.ext["group_size"] == 64
        assert config.ext["custom_param"] == "value"


class TestAutoActQuantizer:
    """Tests for AutoActQuantizer factory."""

    def test_from_config_with_invalid_config(self):
        """测试from_config工厂方法使用无效配置"""
        with pytest.raises(ValidationError):
            AutoActQuantizer.from_config(None)

        with pytest.raises(SchemaValidateError):
            AutoActQuantizer.from_config({"dtype": "int8"})

    def test_register_new_subclass_and_can_create_instance_using_from_config(self):
        from msmodelslim.ir.qal.qregistry import QABCRegistry

        test_scheme = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True).to_scheme()

        @QABCRegistry.register(dispatch_key=(test_scheme, "test"), abc_class=AutoActQuantizer)
        class MyActQuantizer(AutoActQuantizer):
            def __init__(self, config: QConfig):
                super().__init__()

        config = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True)
        quantizer = AutoActQuantizer.from_config(config)
        assert isinstance(quantizer, MyActQuantizer)


class TestAutoWeightQuantizer:
    """Tests for AutoWeightQuantizer factory."""

    def test_from_config_with_invalid_config(self):
        """测试from_config工厂方法使用无效配置"""
        with pytest.raises(ValidationError):
            AutoWeightQuantizer.from_config(None)

        with pytest.raises(SchemaValidateError):
            AutoWeightQuantizer.from_config({"dtype": "int8"})

    def test_register_new_subclass_and_can_create_instance_using_from_config(self):
        from msmodelslim.ir.qal.qregistry import QABCRegistry

        test_scheme = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True).to_scheme()

        @QABCRegistry.register(dispatch_key=(test_scheme, "test"), abc_class=AutoWeightQuantizer)
        class MyWeightQuantizer(AutoWeightQuantizer):
            def __init__(self, config: QConfig):
                super().__init__()

            def init_weight(self, weight, bias=None):
                pass

            def forward(self, x=None):
                return x

            def get_q_storage(self):
                from msmodelslim.core.quantizer.base import QStorage

                return QStorage()

            def get_q_param(self):
                from msmodelslim.core.quantizer.base import QParam

                return QParam()

        config = QConfig(dtype="int8", scope="per_channel", method="test", symmetric=True)
        quantizer = AutoWeightQuantizer.from_config(config)
        assert isinstance(quantizer, MyWeightQuantizer)
