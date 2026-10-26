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

msmodelslim.infra.io.json_writer 模块的单元测试
"""

import os
from unittest.mock import patch

from msmodelslim.infra.io.json_writer import JsonWriter


class TestJsonWriter:
    """测试 JsonWriter 的聚合写入与落盘行为"""

    def test_write_aggregate_desc_by_prefix(self):
        """正常路径：write 按 prefix 聚合描述到 value_map"""
        writer = JsonWriter(save_directory="/tmp", file_name="out.json")

        writer.write("weight", {"a": 1})
        writer.write("config", ["x", "y"])

        assert writer.value_map == {"weight": {"a": 1}, "config": ["x", "y"]}

    @patch("msmodelslim.infra.io.json_writer.json_safe_dump")
    def test_close_dump_aggregated_map_with_indent_four(self, mock_dump):
        """行为：close 使用 indent=4 将聚合结果写回目标文件"""
        writer = JsonWriter(save_directory="/tmp", file_name="out.json")
        writer.write("weight", {"a": 1})

        writer.close()

        mock_dump.assert_called_once_with({"weight": {"a": 1}}, os.path.join("/tmp", "out.json"), indent=4)

    @patch("msmodelslim.infra.io.json_writer.json_safe_dump")
    def test_dump_use_passed_indent(self, mock_dump):
        """行为：dump 使用调用方传入的 indent"""
        writer = JsonWriter(save_directory="/tmp", file_name="out.json")

        writer.dump({"k": "v"}, indent=8)

        mock_dump.assert_called_once_with({"k": "v"}, os.path.join("/tmp", "out.json"), indent=8)
