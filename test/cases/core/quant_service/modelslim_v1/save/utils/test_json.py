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

import os
import tempfile
from unittest.mock import patch

from msmodelslim.core.quant_service.modelslim_v1.save.utils.json import JsonWriter


class TestJsonWriter:
    """Tests for JsonWriter."""

    def test_write_accumulates_values_when_called(self):
        """场景：多次 write。预期：value_map 累积键值。"""
        writer = JsonWriter(save_directory="/tmp", file_name="desc.json")
        writer.write("layer0", {"dtype": "W8A8"})
        writer.write("layer1", {"dtype": "W4A8"})
        assert writer.value_map == {
            "layer0": {"dtype": "W8A8"},
            "layer1": {"dtype": "W4A8"},
        }

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.utils.json.json_safe_dump")
    def test_close_writes_json_file_when_called(self, mock_json_safe_dump):
        """场景：write 后 close。预期：json_safe_dump 写入合并后的 value_map。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JsonWriter(save_directory=tmpdir, file_name="out.json")
            writer.write("key", {"v": 1})
            writer.close()

            expected_path = os.path.join(tmpdir, "out.json")
            mock_json_safe_dump.assert_called_once_with({"key": {"v": 1}}, expected_path, indent=4)
            assert os.path.isdir(tmpdir)
