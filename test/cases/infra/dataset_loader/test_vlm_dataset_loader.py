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

from msmodelslim.infra.dataset_loader.vlm_dataset_loader import (
    VLMDatasetLoader,
    VlmCalibSample,
    _resolve_dataset_path,
)
from msmodelslim.utils.exception import InvalidDatasetError


@pytest.fixture(autouse=True)
def mock_security_check():
    def fake(p, *args, **kwargs):
        return p

    with (
        patch(
            "msmodelslim.infra.dataset_loader.vlm_dataset_loader.get_valid_read_path",
            side_effect=fake,
        ),
        patch(
            "msmodelslim.infra.dataset_loader.jsonl_dataset_loader.get_valid_read_path",
            side_effect=fake,
        ),
        patch(
            "msmodelslim.infra.dataset_loader.indexed_directory_dataset_loader.get_valid_read_path",
            side_effect=fake,
        ),
        patch(
            "msmodelslim.infra.dataset_loader.legacy_directory_dataset_loader.get_valid_read_path",
            side_effect=fake,
        ),
    ):
        yield


def _write_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _write_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class TestVlmDatasetLoader:
    # --- _resolve_dataset_path / 路径解析 ---
    def test_resolve_dataset_path_return_path_when_with_absolute_path(self, tmp_path: Path):
        p = tmp_path / "a.json"
        p.write_text("{}", encoding="utf-8")
        assert _resolve_dataset_path(str(p), None) == p

    def test_resolve_dataset_path_return_joined_path_when_with_dataset_dir(self, tmp_path: Path):
        base = tmp_path / "base"
        base.mkdir()
        d = base / "short"
        d.mkdir()
        assert _resolve_dataset_path("short", base) == d

    def test_resolve_dataset_path_return_resolved_path_when_relative_path_exists_in_cwd(
        self, tmp_path: Path, monkeypatch
    ):
        rel_file = tmp_path / "rel.json"
        rel_file.write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        result = _resolve_dataset_path("rel.json", None)
        assert result.exists()
        assert result.name == "rel.json"

    def test_resolve_dataset_path_return_joined_path_when_short_name_under_dataset_dir(self, tmp_path: Path):
        dataset_dir = tmp_path / "lab_calib"
        dataset_dir.mkdir()
        data_dir = dataset_dir / "calibData"
        data_dir.mkdir()
        result = _resolve_dataset_path("calibData", dataset_dir)
        assert result == data_dir

    def test_resolve_dataset_path_return_fallback_path_when_dataset_dir_missing_but_cwd_has_file(
        self, tmp_path: Path, monkeypatch
    ):
        data_file = tmp_path / "fallback.json"
        data_file.write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        fake_dataset_dir = tmp_path / "nonexist"
        result = _resolve_dataset_path("fallback.json", fake_dataset_dir)
        assert result.exists()

    def test_resolve_dataset_path_return_fallback_when_short_name_missing_under_dataset_dir_but_resolved_exists(
        self, tmp_path: Path
    ):
        (tmp_path / "rel_name").mkdir()
        dataset_dir = tmp_path / "other"
        dataset_dir.mkdir()
        from msmodelslim.infra.dataset_loader import vlm_dataset_loader as vlm_mod

        with patch.object(vlm_mod.Path, "exists", side_effect=[False, False, True]):
            result = _resolve_dataset_path("rel_name", dataset_dir)
        assert result.name == "rel_name"

    def test_resolve_dataset_path_raise_error_when_path_not_exists(self):
        with pytest.raises(InvalidDatasetError) as exc_info:
            _resolve_dataset_path("nonexist.json", None)
        assert "does not exist" in str(exc_info.value)

    def test_resolve_dataset_path_raise_error_when_all_candidates_fail(self, tmp_path: Path):
        dataset_dir = tmp_path / "lab_calib"
        dataset_dir.mkdir()
        with pytest.raises(InvalidDatasetError) as exc_info:
            _resolve_dataset_path("missing.json", dataset_dir)
        error_msg = str(exc_info.value)
        assert "none exists" in error_msg
        assert "dataset_dir" in error_msg

    # --- get_dataset_by_name：Jsonl / Indexed / Legacy 路由与端到端 ---
    def test_get_dataset_by_name_return_samples_when_with_index_jsonl_file(self, tmp_path: Path):
        index = tmp_path / "index.jsonl"
        _write_jsonl(index, [{"text": "hello"}])
        data = VLMDatasetLoader().get_dataset_by_name(str(index))
        assert len(data) == 1
        assert data[0].text == "hello"

    def test_get_dataset_by_name_return_samples_when_with_index_json_file(self, tmp_path: Path):
        index = tmp_path / "index.json"
        _write_json(index, [{"text": "json test"}])
        data = VLMDatasetLoader().get_dataset_by_name(str(index))
        assert len(data) == 1
        assert data[0].text == "json test"

    def test_get_dataset_by_name_return_samples_when_with_indexed_directory(self, tmp_path: Path):
        d = tmp_path / "d"
        d.mkdir()
        _write_json(d / "index.json", [{"text": "hello-dir"}])
        data = VLMDatasetLoader().get_dataset_by_name(str(d))
        assert len(data) == 1
        assert data[0].text == "hello-dir"

    def test_get_dataset_by_name_return_samples_when_with_indexed_dir_and_index_jsonl(self, tmp_path: Path):
        dir_path = tmp_path / "data"
        dir_path.mkdir()
        _write_jsonl(dir_path / "index.jsonl", [{"text": "indexed dir"}])
        data = VLMDatasetLoader().get_dataset_by_name(str(dir_path))
        assert len(data) == 1
        assert data[0].text == "indexed dir"

    def test_get_dataset_by_name_return_legacy_default_text_when_with_image_directory(self, tmp_path: Path):
        d = tmp_path / "img"
        d.mkdir()
        (d / "a.jpg").write_bytes(b"a")
        loader = VLMDatasetLoader()
        loader.default_text = "from-yaml"
        data = loader.get_dataset_by_name(str(d))
        assert len(data) == 1
        assert data[0].text == "from-yaml"

    def test_get_dataset_by_name_return_samples_when_with_legacy_json_file(self, tmp_path: Path):
        json_file = tmp_path / "data.json"
        _write_json(json_file, [{"text": "legacy json"}])
        data = VLMDatasetLoader().get_dataset_by_name(str(json_file))
        assert len(data) == 1
        assert data[0].text == "legacy json"

    def test_get_dataset_by_name_return_samples_when_with_legacy_jsonl_file(self, tmp_path: Path):
        jsonl_file = tmp_path / "calib.jsonl"
        _write_jsonl(jsonl_file, [{"text": "legacy jsonl"}])
        data = VLMDatasetLoader().get_dataset_by_name(str(jsonl_file))
        assert len(data) == 1
        assert data[0].text == "legacy jsonl"

    def test_get_dataset_by_name_return_samples_when_short_name_under_dataset_dir(self, tmp_path: Path):
        dataset_dir = tmp_path / "lab_calib"
        dataset_dir.mkdir()
        calib_dir = dataset_dir / "calibData"
        calib_dir.mkdir()
        (calib_dir / "img.jpg").write_bytes(b"img")
        loader = VLMDatasetLoader(dataset_dir=dataset_dir)
        data = loader.get_dataset_by_name("calibData")
        assert len(data) == 1
        assert data[0].text == "Describe this image in detail."

    def test_get_dataset_by_name_use_indexed_loader_when_dir_has_index_and_images(self, tmp_path: Path):
        dir_path = tmp_path / "data"
        dir_path.mkdir()
        (dir_path / "img.jpg").write_bytes(b"img")
        _write_jsonl(dir_path / "index.jsonl", [{"text": "indexed"}])
        data = VLMDatasetLoader().get_dataset_by_name(str(dir_path))
        assert len(data) == 1
        assert data[0].text == "indexed"

    def test_get_dataset_by_name_return_samples_with_audio_when_index_jsonl_has_audio(self, tmp_path: Path):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"audio")
        index_jsonl = tmp_path / "index.jsonl"
        _write_jsonl(index_jsonl, [{"text": "audio sample", "audio": "audio.wav"}])
        data = VLMDatasetLoader().get_dataset_by_name(str(index_jsonl))
        assert len(data) == 1
        assert Path(data[0].audio).name == "audio.wav"

    def test_get_dataset_by_name_return_samples_with_video_when_indexed_dir_has_video(self, tmp_path: Path):
        dir_path = tmp_path / "videos"
        dir_path.mkdir()
        (dir_path / "video.mp4").write_bytes(b"video")
        _write_json(dir_path / "index.json", [{"text": "video sample", "video": "video.mp4"}])
        data = VLMDatasetLoader().get_dataset_by_name(str(dir_path))
        assert len(data) == 1
        assert Path(data[0].video).name == "video.mp4"

    def test_get_dataset_by_name_return_mixed_modalities_when_indexed_dir_has_mixed(self, tmp_path: Path):
        dir_path = tmp_path / "mixed"
        dir_path.mkdir()
        (dir_path / "img.jpg").write_bytes(b"img")
        (dir_path / "audio.wav").write_bytes(b"audio")
        (dir_path / "video.mp4").write_bytes(b"video")
        _write_jsonl(
            dir_path / "index.jsonl",
            [
                {"text": "text only"},
                {"text": "with image", "image": "img.jpg"},
                {"text": "with audio", "audio": "audio.wav"},
                {"text": "with video", "video": "video.mp4"},
            ],
        )
        data = VLMDatasetLoader().get_dataset_by_name(str(dir_path))
        assert len(data) == 4
        assert data[0].image is None
        assert Path(data[1].image).name == "img.jpg"
        assert Path(data[2].audio).name == "audio.wav"
        assert Path(data[3].video).name == "video.mp4"

    def test_get_dataset_by_name_return_legacy_mixed_when_dir_has_images_and_jsonl(self, tmp_path: Path):
        dir_path = tmp_path / "legacy"
        dir_path.mkdir()
        (dir_path / "img1.jpg").write_bytes(b"1")
        (dir_path / "img2.jpg").write_bytes(b"2")
        _write_jsonl(
            dir_path / "data.jsonl",
            [
                {"text": "text only"},
                {"image": "img1.jpg", "text": "custom text"},
            ],
        )
        data = VLMDatasetLoader().get_dataset_by_name(str(dir_path))
        assert len(data) == 3
        text_only = [s for s in data if s.image is None]
        assert len(text_only) == 1
        img2_samples = [s for s in data if s.image and "img2" in s.image]
        assert len(img2_samples) == 1
        assert img2_samples[0].text == "Describe this image in detail."

    def test_get_dataset_by_name_raise_error_when_with_unsupported_file(self, tmp_path: Path):
        p = tmp_path / "a.txt"
        p.write_text("x", encoding="utf-8")
        with pytest.raises(InvalidDatasetError):
            VLMDatasetLoader().get_dataset_by_name(str(p))

    def test_get_dataset_by_name_raise_error_when_with_missing_path(self, tmp_path: Path):
        with pytest.raises(InvalidDatasetError):
            VLMDatasetLoader().get_dataset_by_name(str(tmp_path / "missing"))

    def test_get_dataset_by_name_raise_error_when_path_not_exists(self, tmp_path: Path):
        with pytest.raises(InvalidDatasetError) as exc_info:
            VLMDatasetLoader().get_dataset_by_name(str(tmp_path / "nonexist"))
        assert "does not exist" in str(exc_info.value)

    def test_get_dataset_by_name_raise_error_when_unsupported_type(self, tmp_path: Path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("unsupported", encoding="utf-8")
        with pytest.raises(InvalidDatasetError) as exc_info:
            VLMDatasetLoader().get_dataset_by_name(str(txt_file))
        assert "Unsupported" in str(exc_info.value)

    # --- VlmCalibSample（同模块数据结构，一并测试）---
    def test_vlm_calib_sample_has_text_only_when_only_text_given(self):
        sample = VlmCalibSample(text="test")
        assert sample.text == "test"
        assert sample.image is None
        assert sample.audio is None
        assert sample.video is None

    def test_vlm_calib_sample_has_all_fields_when_all_modalities_given(self):
        sample = VlmCalibSample(
            text="test",
            image="/path/to/img.jpg",
            audio="/path/to/audio.wav",
            video="/path/to/video.mp4",
        )
        assert sample.text == "test"
        assert sample.image == "/path/to/img.jpg"
        assert sample.audio == "/path/to/audio.wav"
        assert sample.video == "/path/to/video.mp4"
