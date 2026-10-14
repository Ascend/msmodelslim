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

from msmodelslim.infra.dataset_loader.indexed_directory_dataset_loader import IndexedDirectoryDatasetLoader
from msmodelslim.utils.exception import InvalidDatasetError


@pytest.fixture(autouse=True)
def mock_security_check():
    with (
        patch(
            "msmodelslim.infra.dataset_loader.indexed_directory_dataset_loader.get_valid_read_path",
            side_effect=lambda p, *args, **kwargs: p,
        ),
        patch(
            "msmodelslim.infra.dataset_loader.jsonl_dataset_loader.get_valid_read_path",
            side_effect=lambda p, *args, **kwargs: p,
        ),
    ):
        yield


def _write_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class TestIndexedDirectoryDatasetLoader:
    def test_is_support_return_true_when_with_index_json(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "index.json").write_text("[]", encoding="utf-8")
        assert IndexedDirectoryDatasetLoader().is_support(d) is True

    def test_is_support_return_true_when_with_index_jsonl(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "index.jsonl").write_text("", encoding="utf-8")
        assert IndexedDirectoryDatasetLoader().is_support(d) is True

    def test_is_support_return_false_when_without_index_files(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        assert IndexedDirectoryDatasetLoader().is_support(d) is False

    def test_load_data_return_samples_when_with_index_jsonl(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        wav = d / "a.wav"
        wav.write_bytes(b"a")
        _write_jsonl(d / "index.jsonl", [{"text": "hello", "audio": "a.wav"}])
        data = IndexedDirectoryDatasetLoader(default_text="d").load_data(d)
        assert len(data) == 1
        assert data[0].text == "hello"
        assert data[0].audio is not None

    def test_load_data_raise_error_when_with_both_index_files(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        (d / "index.json").write_text("[]", encoding="utf-8")
        (d / "index.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            IndexedDirectoryDatasetLoader().load_data(d)

    def test_load_data_raise_error_when_without_index_files(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        with pytest.raises(InvalidDatasetError):
            IndexedDirectoryDatasetLoader().load_data(d)
