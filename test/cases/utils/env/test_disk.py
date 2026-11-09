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

Tests for ``msmodelslim.utils.env.disk``: model size and partition free space.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from msmodelslim.utils.env import get_dir_size, get_free_space
from msmodelslim.utils.env import disk as env_disk


def _create_symlink(link, target, *, target_is_directory=False):
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip("symlink creation is unavailable: %s" % error)


def test_dir_size_returns_zero_for_empty_directory(tmp_path):
    model = tmp_path / "model"
    model.mkdir()

    assert get_dir_size(model) == 0


def test_dir_size_returns_minus_one_for_empty_string():
    result = get_dir_size("")

    assert result == -1
    assert isinstance(result, int)


def test_dir_size_sums_files_in_nested_directories(tmp_path):
    model = tmp_path / "model"
    nested = model / "nested"
    nested.mkdir(parents=True)
    (model / "a.bin").write_bytes(b"x" * 3)
    (nested / "b.bin").write_bytes(b"x" * 7)

    assert get_dir_size(model) == 10


def test_dir_size_returns_minus_one_for_invalid_root(tmp_path):
    regular_file = tmp_path / "model.bin"
    regular_file.write_bytes(b"x")

    assert get_dir_size(tmp_path / "missing") == -1
    assert get_dir_size(regular_file) == -1


@pytest.mark.parametrize("invalid_root", [None, "", " ", "\t\n", "\0", "bad\0path", 0, 1.5, b""])
def test_dir_size_returns_minus_one_for_invalid_path_types_and_values(invalid_root):
    result = get_dir_size(invalid_root)

    assert result == -1
    assert isinstance(result, int)


def test_dir_size_returns_minus_one_when_root_stat_fails(monkeypatch, tmp_path):
    def fail_stat(_path):
        raise OSError("stat failed")

    monkeypatch.setattr(env_disk.os, "stat", fail_stat)

    assert get_dir_size(tmp_path) == -1


def test_dir_size_returns_minus_one_when_child_stat_fails(monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    readable = model / "readable.bin"
    unreadable = model / "unreadable.bin"
    readable.write_bytes(b"x")
    unreadable.write_bytes(b"x")
    real_stat = env_disk.os.stat

    def fail_selected_stat(path, *args, **kwargs):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(unreadable)):
            raise OSError("child stat failed")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(env_disk.os, "stat", fail_selected_stat)

    assert get_dir_size(model) == -1


def test_dir_size_returns_minus_one_when_directory_stat_fails(monkeypatch, tmp_path):
    model = tmp_path / "model"
    nested = model / "nested"
    nested.mkdir(parents=True)
    real_stat = env_disk.os.stat

    def fail_selected_stat(path, *args, **kwargs):
        if os.path.normcase(os.fspath(path)) == os.path.normcase(str(nested)):
            raise OSError("directory stat failed")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(env_disk.os, "stat", fail_selected_stat)

    assert get_dir_size(model) == -1


def test_dir_size_returns_minus_one_when_walk_reports_error(monkeypatch, tmp_path):
    def walk_with_error(_root, onerror=None, followlinks=False):
        del followlinks
        if onerror is not None:
            onerror(PermissionError("directory cannot be read"))
        return iter(())

    monkeypatch.setattr(env_disk.os, "walk", walk_with_error)

    assert get_dir_size(tmp_path) == -1


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_dir_size_returns_minus_one_when_walk_raises(monkeypatch, tmp_path, error_type):
    def walk_raises(_root, onerror=None, followlinks=False):
        del onerror, followlinks
        raise error_type("walk failed")

    monkeypatch.setattr(env_disk.os, "walk", walk_raises)

    result = get_dir_size(tmp_path)

    assert result == -1
    assert isinstance(result, int)


def test_dir_size_counts_symlinked_dir_and_prunes_cycles(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "w.bin").write_bytes(b"x" * 100)
    model = tmp_path / "model"
    model.mkdir()
    _create_symlink(model / "linked", weights, target_is_directory=True)
    assert get_dir_size(model) == 100

    _create_symlink(model / "loop", model, target_is_directory=True)
    assert get_dir_size(model) == 100


def test_dir_size_follows_symlinked_file(tmp_path):
    target = tmp_path / "weights.bin"
    target.write_bytes(b"x" * 25)
    model = tmp_path / "model"
    model.mkdir()
    _create_symlink(model / "linked.bin", target)

    assert get_dir_size(model) == 25


def test_dir_size_follows_symlink_as_root(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weight.bin").write_bytes(b"x" * 25)
    link = tmp_path / "model-link"
    _create_symlink(link, model, target_is_directory=True)

    assert get_dir_size(link) == 25


def test_dir_size_returns_minus_one_for_broken_symlink_root(tmp_path):
    link = tmp_path / "model-link"
    _create_symlink(link, tmp_path / "missing", target_is_directory=True)

    result = get_dir_size(link)

    assert result == -1
    assert isinstance(result, int)


def test_dir_size_returns_minus_one_for_broken_symlink(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    _create_symlink(model / "broken.bin", tmp_path / "missing.bin")

    result = get_dir_size(model)

    assert result == -1
    assert isinstance(result, int)


def test_dir_size_dedups_hardlinked_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    target = model / "a.bin"
    target.write_bytes(b"x" * 50)
    try:
        os.link(target, model / "b.bin")
    except OSError as error:
        pytest.skip("hard links are not supported: %s" % error)

    assert get_dir_size(model) == 50


def test_dir_size_returns_minus_one_when_subdir_unreadable(tmp_path):
    if sys.platform.startswith("win") or getattr(os, "geteuid", lambda: 0)() == 0:
        pytest.skip("permission checks are bypassed on Windows or as root")

    model = tmp_path / "model"
    model.mkdir()
    (model / "a.bin").write_bytes(b"x")
    locked = model / "locked"
    locked.mkdir()
    (locked / "b.bin").write_bytes(b"x")
    os.chmod(locked, 0)
    try:
        assert get_dir_size(model) == -1
    finally:
        os.chmod(locked, 0o755)


def test_free_space_uses_nearest_existing_ancestor(monkeypatch, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    target = existing / "missing" / "deeper"
    queried_paths = []

    def disk_usage(path):
        queried_paths.append(Path(path))
        return SimpleNamespace(free=1234)

    monkeypatch.setattr(env_disk.shutil, "disk_usage", disk_usage)

    assert get_free_space(target) == 1234
    assert queried_paths == [existing]


def test_free_space_accepts_existing_regular_file(monkeypatch, tmp_path):
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"x")
    queried_paths = []

    def disk_usage(path):
        queried_paths.append(Path(path))
        return SimpleNamespace(free=1234)

    monkeypatch.setattr(env_disk.shutil, "disk_usage", disk_usage)

    result = get_free_space(model_file)

    assert result == 1234
    assert isinstance(result, int)
    assert queried_paths == [model_file]


def test_free_space_accepts_zero_free_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(env_disk.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))

    result = get_free_space(tmp_path)

    assert result == 0
    assert isinstance(result, int)


def test_free_space_returns_minus_one_for_empty_string():
    result = get_free_space("")

    assert result == -1
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "invalid_path",
    [None, "", " ", "\t\n", "\0", "bad\0path", 0, 1.5, b""],
)
def test_free_space_returns_minus_one_for_invalid_path(invalid_path):
    result = get_free_space(invalid_path)

    assert result == -1
    assert isinstance(result, int)


def test_free_space_returns_minus_one_when_disk_usage_fails(monkeypatch, tmp_path):
    def disk_usage(_path):
        raise OSError("disk query failed")

    monkeypatch.setattr(env_disk.shutil, "disk_usage", disk_usage)

    result = get_free_space(tmp_path)

    assert result == -1
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "usage",
    [
        SimpleNamespace(free=-1),
        SimpleNamespace(free=True),
        SimpleNamespace(free=False),
        SimpleNamespace(free=float("nan")),
        object(),
        SimpleNamespace(free="invalid"),
        SimpleNamespace(free=float("inf")),
        SimpleNamespace(free=1.5),
    ],
)
def test_free_space_returns_minus_one_for_invalid_disk_usage(monkeypatch, tmp_path, usage):
    monkeypatch.setattr(env_disk.shutil, "disk_usage", lambda _path: usage)

    result = get_free_space(tmp_path)

    assert result == -1
    assert isinstance(result, int)
