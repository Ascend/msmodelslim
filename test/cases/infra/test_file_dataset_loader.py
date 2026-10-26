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

msmodelslim.infra.file_dataset_loader 模块的单元测试
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from msmodelslim.infra.file_dataset_loader import FileDatasetLoader
from msmodelslim.utils.exception import InvalidDatasetError, SchemaValidateError


def _passthrough(path, *args, **kwargs):
    return path


class TestFileDatasetLoader:
    """测试 FileDatasetLoader 的 JSON/JSONL 加载与校验"""

    @patch("msmodelslim.infra.file_dataset_loader.json_safe_load", return_value=["q1", "q2"])
    @patch("msmodelslim.infra.file_dataset_loader.check_read_permission")
    @patch("msmodelslim.infra.file_dataset_loader.get_valid_read_path", side_effect=_passthrough)
    def test_get_json_dataset_return_list(self, mock_get_valid, mock_perm, mock_load):
        """正常路径：.json 文件经 json_safe_load 解析为字符串列表"""
        result = FileDatasetLoader(Path("/data")).get_dataset_by_name("questions.json")

        assert result == ["q1", "q2"]
        mock_load.assert_called_once()
        mock_get_valid.assert_called_once()

    @patch("msmodelslim.infra.file_dataset_loader.SafeGenerator.load_jsonl", return_value=["l1", "l2"])
    @patch("msmodelslim.infra.file_dataset_loader.check_read_permission")
    @patch("msmodelslim.infra.file_dataset_loader.get_valid_read_path", side_effect=_passthrough)
    def test_get_jsonl_dataset_return_list(self, mock_get_valid, mock_perm, mock_jsonl):
        """正常路径：非 .json 文件经 SafeGenerator.load_jsonl 解析为列表"""
        result = FileDatasetLoader(Path("/data")).get_dataset_by_name("questions.jsonl")

        assert result == ["l1", "l2"]
        mock_jsonl.assert_called_once()

    def test_get_dataset_raise_schema_error_when_id_not_str(self):
        """异常：dataset_id 非字符串时立即抛 SchemaValidateError"""
        with pytest.raises(SchemaValidateError):
            FileDatasetLoader(Path("/data")).get_dataset_by_name(123)

    @patch("msmodelslim.infra.file_dataset_loader.json_safe_load", return_value={"not": "a list"})
    @patch("msmodelslim.infra.file_dataset_loader.check_read_permission")
    @patch("msmodelslim.infra.file_dataset_loader.get_valid_read_path", side_effect=_passthrough)
    def test_get_dataset_raise_invalid_error_when_json_not_list(self, mock_get_valid, mock_perm, mock_load):
        """异常：JSON 根节点不是列表时抛 InvalidDatasetError"""
        with pytest.raises(InvalidDatasetError):
            FileDatasetLoader(Path("/data")).get_dataset_by_name("data.json")
