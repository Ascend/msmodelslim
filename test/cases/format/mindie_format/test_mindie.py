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

Unit tests for msmodelslim.format.mindie_format.mindie.
"""

from __future__ import annotations

from msmodelslim.format.mindie_format.mindie import MindIEQuantFormatConfig


class TestMindIEQuantFormatConfig:
    """Tests for MindIEQuantFormatConfig."""

    def test_config_defaults_when_created(self):
        config = MindIEQuantFormatConfig()

        assert config.type == "mindie_format_saver"
        assert config.part_file_size == 4
        assert config.ext == {}

    def test_set_save_directory_update_path_when_called(self):
        config = MindIEQuantFormatConfig()

        config.set_save_directory("/tmp/save")

        assert config.save_directory == "/tmp/save"

    def test_model_dump_exclude_ext_when_ext_empty(self):
        config = MindIEQuantFormatConfig()

        dumped = config.model_dump()

        assert "ext" not in dumped

    def test_model_dump_include_ext_when_ext_nonempty(self):
        config = MindIEQuantFormatConfig(ext={"key": "value"})

        dumped = config.model_dump()

        assert dumped["ext"] == {"key": "value"}
