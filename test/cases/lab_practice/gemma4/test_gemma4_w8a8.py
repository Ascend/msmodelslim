#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from pathlib import Path

from msmodelslim.core.practice.interface import PracticeConfig
from msmodelslim.utils.security import yaml_safe_load


class TestGemma4W8A8Practice:
    def test_metadata_contains_verified_model_and_tags_when_config_is_loaded(self):
        repo_root = Path(__file__).resolve().parents[4]
        config_path = repo_root / "lab_practice" / "gemma4" / "gemma4_w8a8.yaml"

        config = PracticeConfig.model_validate(yaml_safe_load(str(config_path)))

        assert config.metadata.config_id == config_path.stem
        assert config.metadata.label == {
            "w_bit": 8,
            "a_bit": 8,
            "is_sparse": False,
            "kv_cache": False,
        }
        assert config.metadata.verified_tags == {
            "gemma-4-31B-it": [
                ["vLLM_Ascend", "Atlas_A2_Inference"],
                ["vLLM_Ascend", "Atlas_A3_Inference"],
                ["vLLM_Ascend", "Ascend_950"],
            ]
        }
