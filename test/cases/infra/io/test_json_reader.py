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

msmodelslim.infra.io.json_reader 模块的单元测试
"""

from unittest.mock import patch

from msmodelslim.infra.io.json_reader import JsonReader


class TestJsonReader:
    """测试 JsonReader 类的路径校验与 JSON 解析行为"""

    @patch("msmodelslim.infra.io.json_reader.json_safe_load")
    @patch("msmodelslim.infra.io.json_reader.get_valid_read_path")
    def test_load_return_parsed_json_when_path_valid(self, mock_get_valid, mock_load):
        """正常路径：load 先做安全校验再返回解析后的 JSON 对象"""
        mock_get_valid.return_value = "/safe/config.json"
        mock_load.return_value = {"quantization_config": {"bits": 8}}

        result = JsonReader("config.json").load()

        assert result == {"quantization_config": {"bits": 8}}
        mock_get_valid.assert_called_once_with("config.json")
        mock_load.assert_called_once_with("/safe/config.json")

    @patch("msmodelslim.infra.io.json_reader.json_safe_load")
    @patch("msmodelslim.infra.io.json_reader.get_valid_read_path")
    def test_load_pass_resolved_path_to_safe_load(self, mock_get_valid, mock_load):
        """行为：json_safe_load 收到的是 get_valid_read_path 解析后的绝对路径"""
        mock_get_valid.return_value = "/abs/tenant/config.json"
        mock_load.return_value = {"weight_map": {"a": "shard.safetensors"}}

        result = JsonReader("relative/config.json").load()

        assert result == {"weight_map": {"a": "shard.safetensors"}}
        mock_load.assert_called_once_with("/abs/tenant/config.json")
