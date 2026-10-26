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

msmodelslim.infra.io.default_json_reader_factory 模块的单元测试
"""

from unittest.mock import patch

from msmodelslim.infra.io.default_json_reader_factory import DefaultJsonReaderFactory
from msmodelslim.infra.io.json_reader import JsonReader


class TestDefaultJsonReaderFactory:
    """测试 DefaultJsonReaderFactory 的公开创建入口"""

    def test_create_json_reader_return_json_reader(self):
        """正常路径：工厂返回 JsonReader 实例"""
        reader = DefaultJsonReaderFactory().create_json_reader("/data/config.json")

        assert isinstance(reader, JsonReader)

    @patch("msmodelslim.infra.io.json_reader.json_safe_load", return_value={"ok": True})
    @patch("msmodelslim.infra.io.json_reader.get_valid_read_path", return_value="/resolved/config.json")
    def test_created_reader_reads_from_given_path(self, mock_get_valid, mock_load):
        """行为：工厂创建的 reader 读取调用方传入的路径"""
        reader = DefaultJsonReaderFactory().create_json_reader("config.json")

        assert reader.load() == {"ok": True}
        mock_get_valid.assert_called_once_with("config.json")
