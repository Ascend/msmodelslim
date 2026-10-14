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

Unit tests for msmodelslim.core.tune_strategy.common.config_builder.base_builder
"""

from msmodelslim.core.practice import Metadata
from msmodelslim.core.quant_service.modelslim_v1.save import AscendV1Config
from msmodelslim.core.tune_strategy.common.config_builder.base_builder import QuantizationConfigBuilder
from msmodelslim.core.tune_strategy.common.config_builder.quantization_config import (
    QuantizationConfig,
    TuningSearchSpace,
)
from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig


def _default_quant_save_list():
    return [AscendV1QuantFormatConfig(part_file_size=4)]


class ConcreteBuilder(QuantizationConfigBuilder):
    """Concrete builder for testing abstract base."""

    def build_spec_process(self, **kwargs):
        return []


class TestQuantizationConfigBuilder:
    """QuantizationConfigBuilder 基类单元测试"""

    def test_build_returns_full_QuantizationConfig_with_defaults_when_no_components_passed(self):
        """
        场景：build() 不传 process/metadata 等，但显式传入符合 save schema 的 spec_save（QuantFormatConfig）。
        预期：返回 QuantizationConfig，apiversion、metadata、spec.runner、spec.dataset、spec.save 符合预期。
        """
        builder = ConcreteBuilder()
        config = builder.build(spec_save=_default_quant_save_list())
        assert isinstance(config, QuantizationConfig)
        assert config.apiversion == QuantizationConfigBuilder.DEFAULT_APIVERSION
        assert config.metadata.config_id == "unknown"
        assert config.spec.runner == "auto"
        assert config.spec.dataset == QuantizationConfigBuilder.DEFAULT_DATASET
        assert config.spec.process == []
        assert len(config.spec.save) == 1
        assert isinstance(config.spec.save[0], AscendV1QuantFormatConfig)
        assert config.spec.save[0].type == "ascendv1_saver"

    def test_build_spec_save_returns_AscendV1Config_when_called_without_override(self):
        """
        场景：直接调用 build_spec_save 默认实现。
        预期：返回 AscendV1Config（运行时由 quant_service 路由为旧版 Saver processor）。
        """
        save = ConcreteBuilder().build_spec_save()
        assert len(save) == 1
        assert isinstance(save[0], AscendV1Config)
        assert save[0].type == "ascendv1_saver"

    def test_build_uses_passed_metadata_when_metadata_provided(self):
        """
        场景：build() 显式传入 metadata。
        预期：返回的 config.metadata 与传入一致，不再使用 build_metadata 默认值。
        """
        builder = ConcreteBuilder()
        meta = Metadata(config_id="custom", label={"k": "v"})
        config = builder.build(metadata=meta, spec_save=_default_quant_save_list())
        assert config.metadata.config_id == "custom"
        assert config.metadata.label == {"k": "v"}

    def test_get_tuning_search_space_returns_empty_TuningSearchSpace_when_called_without_args(self):
        """
        场景：子类未重写 get_tuning_search_space，直接调用默认实现。
        预期：返回 TuningSearchSpace 且 anti_outlier_strategies 为 None。
        """
        builder = ConcreteBuilder()
        space = builder.get_tuning_search_space()
        assert isinstance(space, TuningSearchSpace)
        assert space.anti_outlier_strategies is None
