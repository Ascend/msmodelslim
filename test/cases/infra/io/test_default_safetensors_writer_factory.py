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

msmodelslim.infra.io.default_safetensors_writer_factory 模块的单元测试
"""

from unittest.mock import patch

from msmodelslim.infra.io.buffered_safetensors_writer import ONE_GB_FILE_BYTES, BufferedSafetensorsWriter
from msmodelslim.infra.io.default_safetensors_writer_factory import DefaultSafetensorsWriterFactory
from msmodelslim.infra.io.safetensors_writer import SafetensorsWriter


def _passthrough(path, *args, **kwargs):
    return path


class TestDefaultSafetensorsWriterFactory:
    """测试 DefaultSafetensorsWriterFactory 依据 part_file_size 选择写入器"""

    @patch("msmodelslim.infra.io.buffered_safetensors_writer.get_write_directory", side_effect=_passthrough)
    @patch("msmodelslim.infra.io.safetensors_writer.get_valid_write_path", side_effect=_passthrough)
    def test_create_buffered_writer_when_part_size_positive(self, mock_get_valid, mock_get_write_dir):
        """正数 part_file_size：返回分片写入器并按其换算最大分片字节数"""
        writer = DefaultSafetensorsWriterFactory().create_safetensors_writer(
            part_file_size=2, save_directory="/out", save_prefix="model"
        )

        assert isinstance(writer, BufferedSafetensorsWriter)
        assert writer.max_size == 2 * ONE_GB_FILE_BYTES
        assert writer.save_prefix == "model"

    @patch("msmodelslim.infra.io.safetensors_writer.get_valid_write_path", side_effect=_passthrough)
    def test_create_single_writer_when_part_size_zero(self, mock_get_valid):
        """part_file_size 为 0：返回单文件写入器而非分片写入器"""
        writer = DefaultSafetensorsWriterFactory().create_safetensors_writer(
            part_file_size=0, save_directory="/out", save_prefix="model"
        )

        assert isinstance(writer, SafetensorsWriter)
        assert not isinstance(writer, BufferedSafetensorsWriter)

    @patch("msmodelslim.infra.io.safetensors_writer.get_valid_write_path", side_effect=_passthrough)
    def test_create_single_writer_when_part_size_none(self, mock_get_valid):
        """part_file_size 为 None：返回单文件写入器"""
        writer = DefaultSafetensorsWriterFactory().create_safetensors_writer(
            part_file_size=None, save_directory="/out", save_prefix="model"
        )

        assert isinstance(writer, SafetensorsWriter)
        assert not isinstance(writer, BufferedSafetensorsWriter)
