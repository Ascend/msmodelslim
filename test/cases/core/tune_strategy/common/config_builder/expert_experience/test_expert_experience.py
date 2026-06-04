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

from msmodelslim.core.tune_strategy.common.config_builder.expert_experience.expert_experience import (
    StructureConfig,
    ExpertExperienceLoader,
    ExpertExperienceConfigBuilder,
)
from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig
from msmodelslim.utils.exception import UnsupportedError, SchemaValidateError


class TestStructureConfig:
    """StructureConfig 单元测试"""

    def test_StructureConfig_field_match_when_all_fields_provided(self):
        """
        场景：构造 StructureConfig 时显式传入 type、include、exclude。
        预期：各字段与入参一致，用于看护完整输入下的构造行为。
        """
        cfg = StructureConfig(
            type="GQA",
            include=["*self_attn*"],
            exclude=["*kv_b_proj"],
        )
        assert cfg.type == "GQA"
        assert cfg.include == ["*self_attn*"]
        assert cfg.exclude == ["*kv_b_proj"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"type": "MHA"},
            {"type": "MHA", "include": []},
            {"type": "MHA", "include": ["*self_attn*", "", "*mlp*"]},
        ],
        ids=["include_omitted", "include_empty_list", "include_contains_empty_string"],
    )
    def test_StructureConfig_raises_SchemaValidateError_when_include_invalid(self, kwargs):
        """
        场景：include 未传、为空列表或含空字符串时构造 StructureConfig。
        预期：抛出 SchemaValidateError 且消息含 include。
        """
        with pytest.raises(SchemaValidateError) as exc_info:
            StructureConfig(**kwargs)
        assert "include" in str(exc_info.value).lower()

    def test_StructureConfig_exclude_optional_defaults_to_none_when_omitted(self):
        """传入 type、include 时不传 exclude，预期 exclude 为 None。"""
        cfg = StructureConfig(type="MHA", include=["*"])
        assert cfg.type == "MHA"
        assert cfg.include == ["*"]
        assert cfg.exclude is None


class TestExpertExperienceLoader:
    """ExpertExperienceLoader 单元测试"""

    def test_get_supported_quant_types_returns_list_when_config_loaded(self):
        """
        场景：加载 expert_experience.yaml 后调用 get_supported_quant_types。
        预期：返回列表且包含 w8a8、w4a8。
        """
        types = ExpertExperienceLoader.get_supported_quant_types()
        assert isinstance(types, list)
        assert "w8a8" in types
        assert "w4a8" in types

    def test_get_qconfig_returns_dict_when_quant_type_and_structure_supported(self):
        """
        场景：loader 使用支持的 quant_type=w8a8、structure_type=MHA 调用 get_qconfig。
        预期：返回非空字典且包含 act、weight 键。
        """
        loader = ExpertExperienceLoader()
        qconfig = loader.get_qconfig("w8a8", "MHA")
        assert qconfig is not None
        assert "act" in qconfig
        assert "weight" in qconfig

    def test_get_qconfig_returns_none_when_structure_maps_to_bf16(self):
        """
        场景：structure_type 在 yaml 中映射为 bf16（如 DSA）时调用 get_qconfig。
        预期：返回 None。
        """
        loader = ExpertExperienceLoader()
        qconfig = loader.get_qconfig("w8a8", "DSA")
        assert qconfig is None

    def test_get_qconfig_raises_UnsupportedError_when_quant_type_unsupported(self):
        """
        场景：quant_type 不在 supported_quant_types 内（如 w4a4）时调用 get_qconfig。
        预期：抛出 UnsupportedError 且消息含 quant_type 或 Unsupported。
        """
        loader = ExpertExperienceLoader()
        with pytest.raises(UnsupportedError) as exc_info:
            loader.get_qconfig("w4a4", "MHA")
        assert "quant_type" in str(exc_info.value).lower() or "Unsupported" in str(exc_info.value)

    def test_get_qconfig_raises_UnsupportedError_when_structure_type_unsupported(self):
        """
        场景：structure_type 在 mapping 中不存在（如 UnknownStructure）时调用 get_qconfig。
        预期：抛出 UnsupportedError。
        """
        loader = ExpertExperienceLoader()
        with pytest.raises(UnsupportedError):
            loader.get_qconfig("w8a8", "UnknownStructure")

    def test_get_anti_outlier_strategy_templates_returns_list_when_quant_type_supported(self):
        """
        场景：使用支持的 quant_type 调用 get_anti_outlier_strategy_templates。
        预期：返回非空列表，且每项为含 type 键的字典。
        """
        loader = ExpertExperienceLoader()
        templates = loader.get_anti_outlier_strategy_templates("w8a8")
        assert isinstance(templates, list)
        assert len(templates) >= 1
        for t in templates:
            assert isinstance(t, dict)
            assert "type" in t

    def test_get_anti_outlier_strategy_templates_raises_UnsupportedError_when_quant_type_unsupported(self):
        """
        场景：quant_type 不支持（如 w4a4）时调用 get_anti_outlier_strategy_templates。
        预期：抛出 UnsupportedError。
        """
        loader = ExpertExperienceLoader()
        with pytest.raises(UnsupportedError):
            loader.get_anti_outlier_strategy_templates("w4a4")


class TestExpertExperienceConfigBuilder:
    """ExpertExperienceConfigBuilder 单元测试"""

    def test_build_metadata_label_match_when_quant_type_w8a8_or_w4a8(self):
        """
        场景：build_metadata 传入支持的 quant_type（w8a8 / w4a8）。
        预期：config_id 为 standing_high_with_experience，label 中 w_bit、a_bit、is_sparse、kv_cache 与量化类型一致。
        """
        builder = ExpertExperienceConfigBuilder()
        meta_w8 = builder.build_metadata(quant_type="w8a8")
        assert meta_w8.config_id == "standing_high_with_experience"
        assert meta_w8.label["w_bit"] == 8
        assert meta_w8.label["a_bit"] == 8
        assert meta_w8.label["is_sparse"] is False
        assert meta_w8.label["kv_cache"] is False
        meta_w4 = builder.build_metadata(quant_type="w4a8")
        assert meta_w4.label["w_bit"] == 4
        assert meta_w4.label["a_bit"] == 8

    def test_build_metadata_fallback_to_8_8_when_quant_type_invalid(self):
        """
        场景：build_metadata 传入无法解析的 quant_type。
        预期：label 中 w_bit、a_bit 回退为 8、8。
        """
        builder = ExpertExperienceConfigBuilder()
        meta = builder.build_metadata(quant_type="invalid")
        assert meta.label["w_bit"] == 8
        assert meta.label["a_bit"] == 8

    def test_build_spec_process_raises_UnsupportedError_when_structure_configs_missing(self):
        """
        场景：build_spec_process 未传入 structure_configs（必填）。
        预期：抛出 UnsupportedError 且消息含 structure_configs。
        """
        builder = ExpertExperienceConfigBuilder()
        with pytest.raises(UnsupportedError) as exc_info:
            builder.build_spec_process()
        assert "structure_configs" in str(exc_info.value)

    def test_build_spec_process_returns_linear_quant_list_when_valid_structure_configs_and_quant_type(self):
        """
        场景：build_spec_process 传入合法 structure_configs 与 quant_type。
        预期：返回非空 process 列表，首项为 linear_quant 且 include/exclude 与配置一致。
        """
        builder = ExpertExperienceConfigBuilder()
        structure_configs = [StructureConfig(type="MHA", include=["*"], exclude=[])]
        process = builder.build_spec_process(
            quant_type="w8a8",
            structure_configs=structure_configs,
        )
        assert len(process) >= 1
        assert process[0].type == "linear_quant"
        assert process[0].include == ["*"]
        assert process[0].exclude == []

    def test_build_spec_process_adds_skipped_include_to_exclude_when_bf16_structure_present(self):
        """
        场景：structure_configs 中含映射为 bf16 的结构（如 DSA）与可量化结构（如 MHA）。
        预期：被跳过的结构的 include 会加入其他 linear_quant 的 exclude。
        """
        builder = ExpertExperienceConfigBuilder()
        structure_configs = [
            StructureConfig(type="DSA", include=["*dsa*"]),  # bf16, skipped
            StructureConfig(type="MHA", include=["*self_attn*"], exclude=[]),
        ]
        process = builder.build_spec_process(
            quant_type="w8a8",
            structure_configs=structure_configs,
        )
        assert len(process) >= 1
        assert "*dsa*" in process[0].exclude

    def test_build_spec_process_raises_UnsupportedError_when_all_structures_skipped(self):
        """
        场景：structure_configs 中全部结构均被跳过（如均为 bf16）。
        预期：抛出 UnsupportedError 且消息含 No valid structure 或 skipped。
        """
        builder = ExpertExperienceConfigBuilder()
        structure_configs = [
            StructureConfig(type="DSA", include=["*"]),
            StructureConfig(type="SWA", include=["*"]),
        ]
        with pytest.raises(UnsupportedError) as exc_info:
            builder.build_spec_process(
                quant_type="w8a8",
                structure_configs=structure_configs,
            )
        assert "No valid structure" in str(exc_info.value) or "skipped" in str(exc_info.value).lower()

    def test_get_tuning_search_space_returns_non_empty_anti_outlier_strategies_when_quant_type_supported(self):
        """
        场景：使用支持的 quant_type 调用 get_tuning_search_space。
        预期：返回的 TuningSearchSpace 中 anti_outlier_strategies 非空。
        """
        builder = ExpertExperienceConfigBuilder()
        space = builder.get_tuning_search_space(quant_type="w8a8")
        assert space.anti_outlier_strategies is not None
        assert len(space.anti_outlier_strategies) >= 1

    def test_build_returns_full_QuantizationConfig_when_valid_quant_type_and_structure_configs(self):
        """
        场景：build 传入合法 quant_type 与 structure_configs。
        预期：返回的配置含 apiversion、metadata.config_id、spec.process 及默认 dataset。
        """
        builder = ExpertExperienceConfigBuilder()
        structure_configs = [StructureConfig(type="MHA", include=["*"], exclude=[])]
        config = builder.build(
            quant_type="w8a8",
            structure_configs=structure_configs,
            spec_save=[AscendV1QuantFormatConfig(part_file_size=4)],
        )
        assert config.apiversion == "modelslim_v1"
        assert config.metadata.config_id == "standing_high_with_experience"
        assert len(config.spec.process) >= 1
        assert config.spec.dataset == "mix_calib.jsonl"
