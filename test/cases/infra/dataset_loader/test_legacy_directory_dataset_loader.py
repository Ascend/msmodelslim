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

from msmodelslim.infra.dataset_loader.legacy_directory_dataset_loader import LegacyDirectoryDatasetLoader
from msmodelslim.utils.exception import InvalidDatasetError


@pytest.fixture(autouse=True)
def mock_security_check():
    with patch(
        "msmodelslim.infra.dataset_loader.legacy_directory_dataset_loader.get_valid_read_path",
        side_effect=lambda p, *args, **kwargs: p,
    ):
        yield


def _write_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class TestLegacyDirectoryDatasetLoader:
    def test_is_support_return_true_when_with_directory(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        assert LegacyDirectoryDatasetLoader().is_support(d) is True

    def test_is_support_return_true_when_with_json_file(self, tmp_path: Path):
        p = tmp_path / "a.json"
        p.write_text("{}", encoding="utf-8")
        assert LegacyDirectoryDatasetLoader().is_support(p) is True

    def test_load_data_return_samples_when_with_image_only_directory(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 1
        assert data[0].text == "D"
        assert Path(data[0].image).name == "a.jpg"

    def test_load_data_return_mixed_samples_when_with_jsonl_and_images(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        (d / "b.jpg").write_bytes(b"b")
        _write_jsonl(d / "data.jsonl", [{"image": "a.jpg", "text": "custom"}])
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 2
        texts = sorted([x.text for x in data])
        assert texts == ["D", "custom"]

    def test_load_data_return_text_samples_when_with_text_jsonl_file(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, ["hello", {"text": "world"}])
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(f)
        assert len(data) == 2
        assert sorted([x.text for x in data]) == ["hello", "world"]

    def test_load_data_raise_error_when_with_unsupported_file(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            LegacyDirectoryDatasetLoader().load_data(f)

    def test_load_data_raise_error_when_with_no_default_supported_images(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.bmp").write_bytes(b"x")
        with pytest.raises(InvalidDatasetError):
            LegacyDirectoryDatasetLoader().load_data(d)

    def test_is_support_return_true_when_with_jsonl_file(self, tmp_path: Path):
        p = tmp_path / "a.jsonl"
        p.write_text("", encoding="utf-8")
        assert LegacyDirectoryDatasetLoader().is_support(p) is True

    def test_load_data_return_text_samples_when_with_pure_text_json_file(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('[{"text": "a"}, "b"]', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader().load_data(f)
        assert len(data) == 2
        assert sorted([x.text for x in data]) == ["a", "b"]

    def test_load_data_raise_error_when_with_multiple_json_files_in_directory(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        (d / "f1.json").write_text("[]", encoding="utf-8")
        (d / "f2.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            LegacyDirectoryDatasetLoader().load_data(d)

    def test_load_data_skip_invalid_jsonl_line_and_continue(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "ok"}\n{ bad }\n{"text": "two"}\n', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader().load_data(f)
        assert len(data) == 2
        assert data[0].text == "ok"
        assert data[1].text == "two"

    def test_load_data_return_one_when_json_root_is_single_object(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('{"text": "only"}', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader().load_data(f)
        assert len(data) == 1
        assert data[0].text == "only"

    def test_load_data_raise_error_when_json_file_invalid(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text("{ invalid", encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            LegacyDirectoryDatasetLoader().load_data(f)

    def test_load_data_skip_invalid_formats_in_jsonl_and_keep_valid(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"text": "valid"}\n{"no_text": 1}\n[]\n"good"\n', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader().load_data(f)
        assert len(data) == 2
        assert "valid" in [x.text for x in data]
        assert "good" in [x.text for x in data]

    def test_load_data_raise_error_when_no_valid_text_in_file(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"x": 1}\n[]\n', encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            LegacyDirectoryDatasetLoader().load_data(f)

    def test_load_data_return_mixed_when_directory_has_json_file_not_jsonl(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        (d / "b.jpg").write_bytes(b"b")
        (d / "meta.json").write_text('[{"image": "a.jpg", "text": "cap"}]', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 2
        texts = sorted([x.text for x in data])
        assert "cap" in texts
        assert "D" in texts

    def test_load_data_skip_image_entry_with_invalid_image_ref(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        _write_jsonl(d / "data.jsonl", [{"image": "", "text": "x"}, {"image": "a.jpg", "text": "ok"}])
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 1
        assert data[0].text == "ok"

    def test_load_data_skip_image_entry_when_image_not_in_directory(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        _write_jsonl(d / "data.jsonl", [{"image": "missing.jpg", "text": "x"}, {"image": "a.jpg", "text": "ok"}])
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 1
        assert data[0].text == "ok"

    def test_load_data_use_default_text_when_image_entry_has_empty_text(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        _write_jsonl(d / "data.jsonl", [{"image": "a.jpg", "text": ""}])
        data = LegacyDirectoryDatasetLoader(default_text="Default").load_data(d)
        assert len(data) == 1
        assert data[0].text == "Default"

    def test_load_data_skip_text_only_entry_when_text_empty(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"text": "ok"}, {"text": ""}, {"text": "  "}])
        data = LegacyDirectoryDatasetLoader().load_data(f)
        assert len(data) == 1
        assert data[0].text == "ok"

    def test_load_data_skip_plain_string_when_whitespace_only(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        f.write_text('"  "\n"ok"\n', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader().load_data(f)
        assert len(data) == 1
        assert data[0].text == "ok"

    def test_load_data_skip_jsonl_line_with_both_image_and_text_missing(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        _write_jsonl(d / "data.jsonl", [{"other": 1}, {"image": "a.jpg", "text": "ok"}])
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 1
        assert data[0].text == "ok"

    def test_load_data_skip_image_with_unsupported_suffix_in_mixed(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        _write_jsonl(d / "data.jsonl", [{"image": "a.bmp", "text": "x"}, {"image": "a.jpg", "text": "ok"}])
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 1
        assert data[0].text == "ok"

    def test_load_data_use_default_text_when_image_entry_has_no_text_field(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        _write_jsonl(d / "data.jsonl", [{"image": "a.jpg"}])
        data = LegacyDirectoryDatasetLoader(default_text="DefaultCaption").load_data(d)
        assert len(data) == 1
        assert data[0].text == "DefaultCaption"

    def test_load_data_skip_text_only_entry_when_text_is_none(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        _write_jsonl(f, [{"text": "ok"}, {"text": None}, "last"])
        data = LegacyDirectoryDatasetLoader().load_data(f)
        assert len(data) == 2
        assert data[0].text == "ok"
        assert data[1].text == "last"

    def test_load_data_skip_plain_string_when_empty(self, tmp_path: Path):
        f = tmp_path / "data.jsonl"
        f.write_text('""\n"ok"\n', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader().load_data(f)
        assert len(data) == 1
        assert data[0].text == "ok"

    def test_load_data_mixed_from_json_root_single_object(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        (d / "meta.json").write_text('{"image": "a.jpg", "text": "cap"}', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 1
        assert data[0].text == "cap"

    def test_load_data_mixed_from_json_list(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        (d / "b.jpg").write_bytes(b"b")
        (d / "meta.json").write_text(
            '[{"image": "a.jpg", "text": "first"}, {"image": "b.jpg"}]',
            encoding="utf-8",
        )
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 2
        assert data[0].text == "first"
        assert data[1].text == "D"

    def test_load_data_skip_jsonl_line_with_invalid_format(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        f = d / "data.jsonl"
        f.write_text('[1,2]\n{"image": "a.jpg", "text": "ok"}\n', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 1
        assert data[0].text == "ok"

    def test_load_data_raise_error_when_mixed_json_file_invalid(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        (d / "meta.json").write_text("{ invalid", encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            LegacyDirectoryDatasetLoader(default_text="D").load_data(d)

    def test_load_data_image_count_mismatch_when_json_references_same_image_twice(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        (d / "b.jpg").write_bytes(b"b")
        _write_jsonl(d / "data.jsonl", [{"image": "a.jpg", "text": "x"}, {"image": "a.jpg", "text": "y"}])
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 3
        texts = sorted([x.text for x in data])
        assert "D" in texts
        assert "x" in texts
        assert "y" in texts

    def test_load_data_skip_jsonl_line_with_non_dict_non_string(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        (d / "data.jsonl").write_text('null\n{"image": "a.jpg", "text": "ok"}\n', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 1
        assert data[0].text == "ok"

    def test_load_data_skip_json_root_with_no_valid_entry(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('{"x": 1}', encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            LegacyDirectoryDatasetLoader().load_data(f)

    def test_load_data_skip_json_list_item_with_invalid_format(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('[{"text": "ok"}, 123, null]', encoding="utf-8")
        data = LegacyDirectoryDatasetLoader().load_data(f)
        assert len(data) == 1
        assert data[0].text == "ok"

    def test_load_data_mixed_json_skips_text_only_invalid_entry(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        (d / "meta.json").write_text(
            '[{"text": ""}, {"image": "a.jpg", "text": "cap"}]',
            encoding="utf-8",
        )
        data = LegacyDirectoryDatasetLoader(default_text="D").load_data(d)
        assert len(data) == 1
        assert data[0].text == "cap"

    def test_load_data_raise_error_when_pure_text_json_file_unreadable(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('{"text": "x"}', encoding="utf-8")
        with patch(
            "msmodelslim.infra.dataset_loader.legacy_directory_dataset_loader.open",
            side_effect=OSError("read error"),
        ):
            with pytest.raises(InvalidDatasetError):
                LegacyDirectoryDatasetLoader().load_data(f)
