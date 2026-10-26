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

msmodelslim.infra.io.default_json_writer_factory 模块的单元测试
"""

import os
from unittest.mock import patch

from msmodelslim.infra.io.default_json_writer_factory import DefaultJsonWriterFactory
from msmodelslim.infra.io.json_writer import JsonWriter


class TestDefaultJsonWriterFactory:
    """测试 DefaultJsonWriterFactory 的公开创建入口"""

    def test_create_json_writer_return_json_writer(self):
        """正常路径：工厂返回 JsonWriter 实例"""
        writer = DefaultJsonWriterFactory().create_json_writer("/out", "w.json")

        assert isinstance(writer, JsonWriter)

    @patch("msmodelslim.infra.io.json_writer.json_safe_dump")
    def test_created_writer_writes_to_given_location(self, mock_dump):
        """行为：工厂创建的 writer 落盘到调用方指定的目录与文件名"""
        writer = DefaultJsonWriterFactory().create_json_writer("/out", "w.json")
        writer.write("a", 1)

        writer.close()

        mock_dump.assert_called_once_with({"a": 1}, os.path.join("/out", "w.json"), indent=4)
