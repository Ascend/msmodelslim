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

Unit tests for msmodelslim.core.tune_strategy.common.config_builder.quantization_config
"""

from msmodelslim.core.practice import Metadata
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1ServiceConfig
from msmodelslim.core.tune_strategy.common.config_builder.quantization_config import (
    QuantizationConfig,
    TuningSearchSpace,
)
from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig
from msmodelslim.processor.base import AutoProcessorConfigList


def _default_save():
    """ModelslimV1ServiceConfig.save 为 QuantFormatConfig 列表（discriminator 为 format type）。"""
    return [AscendV1QuantFormatConfig(part_file_size=4)]


class TestQuantizationConfig:
    """QuantizationConfig 单元测试。"""

    def test_QuantizationConfig_field_match_when_apiversion_metadata_spec_provided(self):
        """
        场景：构造 QuantizationConfig 时传入 apiversion、metadata、spec。
        预期：各字段与入参一致。
        """
        metadata = Metadata(config_id="test", label={})
        spec = ModelslimV1ServiceConfig(
            runner="auto",
            process=[],
            dataset="mix_calib.jsonl",
            save=_default_save(),
        )
        config = QuantizationConfig(apiversion="modelslim_v1", metadata=metadata, spec=spec)
        assert config.apiversion == "modelslim_v1"
        assert config.metadata.config_id == "test"
        assert config.spec.runner == "auto"
        assert config.spec.dataset == "mix_calib.jsonl"

    def test_QuantizationConfig_uses_default_metadata_when_metadata_omitted(self):
        """
        场景：构造时未传 metadata，仅传 apiversion 与 spec。
        预期：metadata 为非 None 的 Metadata 实例。
        """
        spec = ModelslimV1ServiceConfig(runner="auto", process=[], dataset="d", save=_default_save())
        config = QuantizationConfig(apiversion="v1", spec=spec)
        assert isinstance(config.metadata, Metadata)


class TestTuningSearchSpace:
    """TuningSearchSpace 单元测试。"""

    def test_TuningSearchSpace_optional_fields_are_none_when_created_without_args(self):
        """
        场景：无参构造 TuningSearchSpace。
        预期：anti_outlier_strategies、allowed_datasets、max_rollback_layers 均为 None（看护默认值）。
        """
        space = TuningSearchSpace()
        assert space.anti_outlier_strategies is None
        assert space.allowed_datasets is None
        assert space.max_rollback_layers is None

    def test_TuningSearchSpace_anti_outlier_strategies_match_when_valid_list_provided(self):
        """
        场景：构造时传入合法的 anti_outlier_strategies（processor 配置列表的列表）。
        预期：字段被校验并解析，类型与内容与入参一致。
        """
        strategies: AutoProcessorConfigList = [[{"type": "flex_smooth_quant"}]]
        space = TuningSearchSpace(anti_outlier_strategies=strategies)
        assert len(space.anti_outlier_strategies) == 1
        assert len(space.anti_outlier_strategies[0]) == 1
        assert space.anti_outlier_strategies[0][0].type == "flex_smooth_quant"

    def test_TuningSearchSpace_field_match_when_allowed_datasets_and_max_rollback_layers_provided(self):
        """
        场景：构造时传入 allowed_datasets、max_rollback_layers。
        预期：对应字段与入参一致。
        """
        space = TuningSearchSpace(
            allowed_datasets=["d1.jsonl"],
            max_rollback_layers=10,
        )
        assert space.allowed_datasets == ["d1.jsonl"]
        assert space.max_rollback_layers == 10

    def test_TuningSearchSpace_accepts_extra_fields_when_extra_allow(self):
        """
        场景：构造时传入未在模型中定义的额外字段（ConfigDict extra=allow）。
        预期：额外字段可被访问。
        """
        space = TuningSearchSpace(custom_field="value")
        assert getattr(space, "custom_field", None) == "value"
