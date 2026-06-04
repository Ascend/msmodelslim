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

from msmodelslim.core.practice.interface import (
    Metadata,
    PracticeConfig,
    ScenarioTagMatch,
)
from msmodelslim.utils.exception import SchemaValidateError


def _make_practice_config(verified_tags=None, verified_model_types=None):
    return PracticeConfig(
        apiversion="modelslim_v1",
        metadata=Metadata(
            config_id="test_config",
            score=90,
            label={"w_bit": 8, "a_bit": 8},
            verified_model_types=verified_model_types or [],
            verified_tags=verified_tags or {},
        ),
    )


class TestScenarioTagMatch:
    """Tests for ScenarioTagMatch enum."""

    def test_enum_values_match_expected_strings(self):
        """场景：读取枚举。预期：值为 no_match / match / standby。"""
        assert ScenarioTagMatch.NO_MATCH.value == "no_match"
        assert ScenarioTagMatch.MATCH.value == "match"
        assert ScenarioTagMatch.STANDBY.value == "standby"


class TestMetadata:
    """Tests for Metadata."""

    def test_metadata_accepts_valid_score_when_in_range(self):
        """场景：score 在合法范围。预期：构造成功。"""
        meta = Metadata(config_id="cfg1", score=50.0)
        assert meta.score == 50.0

    def test_metadata_raises_schema_validate_error_when_score_negative(self):
        """场景：score 为负。预期：抛出 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            Metadata(config_id="cfg1", score=-1.0)


class TestPracticeConfig:
    """Tests for PracticeConfig."""

    def test_extract_quant_config_omits_metadata_when_called(self):
        """场景：含 metadata 的 PracticeConfig。预期：extract 后无 metadata 字段。"""
        config = _make_practice_config()
        extracted = config.extract_quant_config()
        assert extracted.apiversion == "modelslim_v1"
        assert not hasattr(extracted, "metadata") or "metadata" not in extracted.model_dump()

    def test_matches_scenario_tags_returns_no_match_when_no_verified_scenarios(self):
        """场景：model_type 无 verified_tags。预期：NO_MATCH。"""
        config = _make_practice_config()
        assert config.matches_scenario_tags("Qwen2.5-7B", ["mindie"]) == ScenarioTagMatch.NO_MATCH

    def test_matches_scenario_tags_returns_match_when_tags_subset_of_scenario(self):
        """场景：scenario_tags 为 verified 场景子集。预期：MATCH。"""
        config = _make_practice_config(verified_tags={"Qwen2.5-7B": [["mindie", "npu"], ["vllm", "cpu"]]})
        result = config.matches_scenario_tags("Qwen2.5-7B", ["mindie", "npu"])
        assert result == ScenarioTagMatch.MATCH

    def test_matches_scenario_tags_returns_standby_when_tags_not_in_any_scenario(self):
        """场景：有 verified 场景但 tags 不匹配。预期：STANDBY。"""
        config = _make_practice_config(verified_tags={"Qwen2.5-7B": [["mindie", "npu"]]})
        result = config.matches_scenario_tags("Qwen2.5-7B", ["vllm"])
        assert result == ScenarioTagMatch.STANDBY

    def test_matches_scenario_tags_returns_match_when_scenario_tags_none(self):
        """场景：无 scenario_tags。预期：MATCH（存在 verified 场景即可）。"""
        config = _make_practice_config(verified_tags={"Qwen2.5-7B": [["mindie", "npu"]]})
        result = config.matches_scenario_tags("Qwen2.5-7B", None)
        assert result == ScenarioTagMatch.MATCH

    def test_matches_scenario_tags_is_case_insensitive_when_comparing_tags(self):
        """场景：大小写不同的 tag。预期：仍 MATCH。"""
        config = _make_practice_config(verified_tags={"Qwen2.5-7B": [["MindIE", "NPU"]]})
        result = config.matches_scenario_tags("Qwen2.5-7B", ["mindie", "npu"])
        assert result == ScenarioTagMatch.MATCH
