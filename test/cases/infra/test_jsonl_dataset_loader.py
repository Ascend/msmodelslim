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

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from msmodelslim.infra.dataset_loader.jsonl_dataset_loader import JsonlDatasetLoader
from msmodelslim.utils.exception import InvalidDatasetError


@pytest.fixture(autouse=True)
def mock_security_check():
    with patch(
        "msmodelslim.infra.dataset_loader.jsonl_dataset_loader.get_valid_read_path",
        side_effect=lambda p, *args, **kwargs: p,
    ):
        yield


def _write_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class TestJsonlDatasetLoader:
    def test_is_support_return_true_when_with_index_json(self, tmp_path: Path):
        p = tmp_path / "index.json"
        p.write_text("{}", encoding="utf-8")
        assert JsonlDatasetLoader().is_support(p) is True

    def test_is_support_return_true_when_with_index_jsonl(self, tmp_path: Path):
        p = tmp_path / "index.jsonl"
        p.write_text("", encoding="utf-8")
        assert JsonlDatasetLoader().is_support(p) is True

    def test_is_support_return_false_when_with_non_index_file(self, tmp_path: Path):
        p = tmp_path / "data.json"
        p.write_text("{}", encoding="utf-8")
        assert JsonlDatasetLoader().is_support(p) is False

    def test_load_data_return_samples_when_with_valid_jsonl(self, tmp_path: Path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"a")
        index = tmp_path / "index.jsonl"
        _write_jsonl(index, [{"text": "hello", "audio": "a.wav"}])
        data = JsonlDatasetLoader(default_text="d").load_data(index)
        assert len(data) == 1
        assert data[0].text == "hello"
        assert data[0].audio is not None

    def test_load_data_return_default_text_when_text_missing(self, tmp_path: Path):
        index = tmp_path / "index.jsonl"
        _write_jsonl(index, [{"image": None}])
        data = JsonlDatasetLoader(default_text="from-default").load_data(index)
        assert len(data) == 1
        assert data[0].text == "from-default"

    def test_load_data_raise_error_when_with_invalid_jsonl_line(self, tmp_path: Path):
        index = tmp_path / "index.jsonl"
        index.write_text("{bad\n", encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)

    def test_load_data_raise_error_when_with_empty_entries(self, tmp_path: Path):
        index = tmp_path / "index.jsonl"
        index.write_text("\n\n", encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)

    def test_load_data_raise_error_when_with_unsupported_fixed_extension(self, tmp_path: Path):
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"a")
        index = tmp_path / "index.jsonl"
        _write_jsonl(index, [{"text": "hello", "audio": "a.ogg"}])
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)

    def test_load_data_raise_error_when_text_whitespace_only(self, tmp_path: Path):
        index = tmp_path / "index.jsonl"
        _write_jsonl(index, [{"text": "   "}])
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)

    def test_load_data_raise_error_when_image_field_not_non_empty_string(self, tmp_path: Path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        index = tmp_path / "index.jsonl"
        _write_jsonl(index, [{"text": "hi", "image": ""}])
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)

    def test_load_data_raise_error_when_image_suffix_not_supported(self, tmp_path: Path):
        (tmp_path / "a.bmp").write_bytes(b"x")
        index = tmp_path / "index.jsonl"
        _write_jsonl(index, [{"text": "hi", "image": "a.bmp"}])
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)

    def test_load_data_raise_error_when_jsonl_line_not_object(self, tmp_path: Path):
        index = tmp_path / "index.jsonl"
        index.write_text('["not object"]\n', encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)

    def test_load_data_return_one_entry_when_index_json_is_single_object(self, tmp_path: Path):
        index = tmp_path / "index.json"
        index.write_text('{"text": "single"}', encoding="utf-8")
        data = JsonlDatasetLoader().load_data(index)
        assert len(data) == 1
        assert data[0].text == "single"

    def test_load_data_raise_error_when_index_json_parse_fails(self, tmp_path: Path):
        index = tmp_path / "index.json"
        index.write_text("{ invalid", encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)

    def test_load_data_raise_error_when_index_json_list_has_non_object(self, tmp_path: Path):
        index = tmp_path / "index.json"
        index.write_text('[{"text": "a"}, "not object"]', encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)

    def test_load_data_raise_error_when_index_json_root_not_object_or_list(self, tmp_path: Path):
        index = tmp_path / "index.json"
        index.write_text('"string root"', encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)

    def test_load_data_raise_error_when_image_path_not_exist(self, tmp_path: Path):
        index = tmp_path / "index.jsonl"
        _write_jsonl(index, [{"text": "hi", "image": "nonexist.jpg"}])
        with pytest.raises(InvalidDatasetError):
            JsonlDatasetLoader().load_data(index)
