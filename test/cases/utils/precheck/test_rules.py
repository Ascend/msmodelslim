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

Tests for ``msmodelslim.utils.precheck.rules``: string fields, path permissions,
disk space and NPU state.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from msmodelslim.core.const import DeviceType
from msmodelslim.utils import precheck
from msmodelslim.utils.exception import EnvError, SchemaValidateError, SecurityError
from msmodelslim.utils.precheck import (
    DISK_SPACE_GROUP,
    NPU_INDEX_GROUP,
    NPU_OCCUPANCY_GROUP,
    PrecheckManager,
    PrecheckStatus,
    check_disk_space,
    npu_index_check,
    npu_occupancy_check,
    readable_path_check,
    string_field_check,
    writable_path_check,
)
from msmodelslim.utils.precheck import rules as precheck_rules


def _rules():
    return precheck_rules


def _device_module():
    from msmodelslim.utils.validation import device

    return device


def test_string_field_check_accepts_string():
    assert string_field_check("model") is True


def test_string_field_check_rejects_wrong_type():
    with pytest.raises(SchemaValidateError, match="must be a string"):
        string_field_check(123)


def test_string_field_check_reports_too_long_value():
    with pytest.raises(SecurityError, match="should be less than"):
        string_field_check("x" * 201)


def test_string_field_skip_when_none():
    with pytest.raises(SchemaValidateError, match="must be a string"):
        string_field_check(None)
    assert string_field_check("x") is True
    assert string_field_check(None, "config_path", skip_when_none=True) is False
    assert string_field_check("x", "config_path", skip_when_none=True) is True


def test_string_field_check_reports_parameter_name():
    with pytest.raises(SecurityError) as error_info:
        string_field_check("x" * 201, "model_path")
    assert "model_path" in str(error_info.value)


def test_readable_path_accepts_directory_and_file(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    target = tmp_path / "cfg.yaml"
    target.write_text("x", encoding="utf-8")

    assert readable_path_check(str(model_path)) is True
    assert readable_path_check(str(target), file=True) is True


def test_readable_path_skip_when_none():
    assert readable_path_check(None, file=True, skip_when_none=True) is False
    with pytest.raises(SchemaValidateError, match="must be a string or Path"):
        readable_path_check(None)


def test_readable_path_rejects_wrong_kind_and_missing_path(monkeypatch, tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    def _convert(expected_is_dir):
        def _converter(value):
            path = Path(value)
            if not path.exists() or path.is_dir() != expected_is_dir:
                raise SecurityError("path kind mismatch")
            return path

        return _converter

    monkeypatch.setattr(precheck_rules, "convert_to_readable_dir", _convert(expected_is_dir=True))
    monkeypatch.setattr(precheck_rules, "convert_to_readable_file", _convert(expected_is_dir=False))

    with pytest.raises(SecurityError):
        readable_path_check(str(target))
    with pytest.raises(SecurityError):
        readable_path_check(str(tmp_path), file=True)
    with pytest.raises(SecurityError):
        readable_path_check(str(tmp_path / "missing"))


def test_readable_path_checks_read_permission(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    target = tmp_path / "cfg.yaml"
    target.write_text("x", encoding="utf-8")

    def _deny(_path):
        raise OSError("read permission denied")

    monkeypatch.setattr(precheck_rules, "check_read_permission", _deny)
    with pytest.raises(OSError):
        readable_path_check(str(model_path))
    with pytest.raises(OSError):
        readable_path_check(str(target), file=True)


def test_writable_path_does_not_create_directory(monkeypatch, tmp_path):
    save_path = tmp_path / "new" / "save"
    monkeypatch.setattr(precheck_rules, "get_valid_path", lambda value: value)

    assert writable_path_check(str(save_path)) is True
    assert not save_path.exists()


def test_writable_path_rejects_file_parent(monkeypatch, tmp_path):
    parent_file = tmp_path / "parent"
    parent_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(precheck_rules, "get_valid_path", lambda value: value)

    with pytest.raises(SecurityError):
        writable_path_check(str(parent_file / "save"))


def test_writable_path_rejects_existing_file_target(tmp_path):
    target = tmp_path / "save"
    target.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SecurityError):
        writable_path_check(str(target))


def test_writable_path_rejects_wrong_type():
    with pytest.raises(SchemaValidateError):
        writable_path_check(123)


def test_writable_path_accepts_writable_directory(tmp_path):
    save_path = tmp_path / "save"
    save_path.mkdir()

    assert writable_path_check(str(save_path)) is True


def test_rule_groups_are_stable():
    assert precheck.STRING_FIELD_GROUP == "string.fields"
    assert precheck.PATH_PERMISSION_GROUP == "path.permission"
    assert precheck.DISK_SPACE_GROUP == "disk"
    assert precheck.NPU_INDEX_GROUP == "npu.index"
    assert precheck.NPU_OCCUPANCY_GROUP == "npu"
    assert string_field_check.group == precheck.STRING_FIELD_GROUP
    assert readable_path_check.group == precheck.PATH_PERMISSION_GROUP
    assert writable_path_check.group == precheck.PATH_PERMISSION_GROUP
    assert check_disk_space.group == DISK_SPACE_GROUP
    assert npu_index_check.group == NPU_INDEX_GROUP
    assert npu_occupancy_check.group == NPU_OCCUPANCY_GROUP


def test_disk_check_supports_ratio_and_empty_model(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(rules, "get_dir_size", lambda _: 100)
    monkeypatch.setattr(rules, "get_free_space", lambda _: 149)

    with pytest.raises(EnvError):
        rules.check_disk_space("model", "save", size_ratio=1.5)

    monkeypatch.setattr(rules, "get_free_space", lambda _: 150)
    assert "model directory size" in rules.check_disk_space("model", "save", size_ratio=1.5)

    monkeypatch.setattr(rules, "get_dir_size", lambda _: 0)
    with pytest.raises(EnvError):
        rules.check_disk_space("model", "save")


def test_disk_check_uses_default_ratio(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(rules, "get_dir_size", lambda _: 100)
    monkeypatch.setattr(rules, "get_free_space", lambda _: 100)

    assert "model directory size" in rules.check_disk_space("model", "save")


def test_disk_check_passes_when_model_size_unmeasurable(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(rules, "get_dir_size", lambda _: -1)

    assert rules.check_disk_space("model", "save") is True


def test_disk_check_passes_when_free_space_unavailable(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(rules, "get_dir_size", lambda _: 100)
    monkeypatch.setattr(rules, "get_free_space", lambda _: -1)

    assert rules.check_disk_space("model", "save") is True


@pytest.mark.parametrize("model_path", ["", " ", "\t\n"])
def test_disk_check_passes_blank_model_path_without_path_normalization(monkeypatch, model_path):
    rules = _rules()
    probed_paths = []

    def get_dir_size(path):
        probed_paths.append(path)
        return -1

    monkeypatch.setattr(rules, "get_dir_size", get_dir_size)

    assert rules.check_disk_space(model_path, "save") is True
    assert probed_paths == [model_path]


@pytest.mark.parametrize("save_path", ["", " ", "\t\n"])
def test_disk_check_passes_blank_save_path_without_path_normalization(monkeypatch, save_path):
    rules = _rules()
    probed_paths = []

    def get_free_space(path):
        probed_paths.append(path)
        return -1

    monkeypatch.setattr(rules, "get_dir_size", lambda _: 100)
    monkeypatch.setattr(rules, "get_free_space", get_free_space)

    assert rules.check_disk_space("model", save_path) is True
    assert probed_paths == [save_path]


@pytest.mark.parametrize("probe_name", ["get_dir_size", "get_free_space"])
def test_disk_check_passes_when_probe_raises(monkeypatch, probe_name):
    rules = _rules()
    monkeypatch.setattr(rules, "get_dir_size", lambda _: 100)
    monkeypatch.setattr(rules, "get_free_space", lambda _: 100)

    def fail_probe(_):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(rules, probe_name, fail_probe)

    assert rules.check_disk_space("model", "save") is True


@pytest.mark.parametrize("probe_name", ["get_dir_size", "get_free_space"])
@pytest.mark.parametrize("invalid_result", [True, False, -1, 1.5, float("nan"), float("inf"), None])
def test_disk_check_passes_malformed_probe_result(monkeypatch, probe_name, invalid_result):
    rules = _rules()
    monkeypatch.setattr(rules, "get_dir_size", lambda _: 100)
    monkeypatch.setattr(rules, "get_free_space", lambda _: 100)
    monkeypatch.setattr(rules, probe_name, lambda _: invalid_result)

    assert rules.check_disk_space("model", "save") is True


def test_disk_probe_failure_is_a_silent_pass_by_manager(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(rules, "get_dir_size", lambda _: -1)
    manager = PrecheckManager()
    manager.register(rules.check_disk_space.bind("model", "save"))

    report = manager.run({rules.DISK_SPACE_GROUP})
    result = report.results[0]

    assert not report.failed
    assert result.status == PrecheckStatus.PASS
    assert result.reason == ""


def test_npu_index_check_checks_cpu_and_npu_indices(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(
        _device_module().torch,
        "npu",
        SimpleNamespace(device_count=lambda: 2),
        raising=False,
    )

    assert rules.npu_index_check(DeviceType.CPU, None) is True
    assert rules.npu_index_check(DeviceType.CPU, []) is True
    assert rules.npu_index_check(DeviceType.CPU, [0]) is True
    with pytest.raises(SchemaValidateError):
        rules.npu_index_check(DeviceType.CPU, [1])
    with pytest.raises(SchemaValidateError):
        rules.npu_index_check(DeviceType.CPU, [False])
    assert rules.npu_index_check(DeviceType.NPU, None) is True
    assert rules.npu_index_check(DeviceType.NPU, []) is True
    assert rules.npu_index_check(DeviceType.NPU, [1]) is True
    with pytest.raises(SchemaValidateError):
        rules.npu_index_check(DeviceType.NPU, [2])
    assert rules.npu_index_check("invalid", None) is False


def test_npu_index_check_rejects_default_card_when_no_npu_is_available(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(
        _device_module().torch,
        "npu",
        SimpleNamespace(device_count=lambda: 0),
        raising=False,
    )

    with pytest.raises(SchemaValidateError, match="exceed maximum"):
        rules.npu_index_check(DeviceType.NPU, None)


def test_npu_index_probe_failure_is_a_silent_pass_by_manager(monkeypatch):
    rules = _rules()

    def fail_device_count():
        raise RuntimeError("NPU driver unavailable")

    monkeypatch.setattr(
        _device_module().torch,
        "npu",
        SimpleNamespace(device_count=fail_device_count),
        raising=False,
    )
    manager = PrecheckManager()
    manager.register(rules.npu_index_check.bind(DeviceType.NPU, [0]))

    report = manager.run()
    result = report.results[0]

    assert not report.failed
    assert result.status == PrecheckStatus.PASS
    assert result.reason == ""
    assert not rules.npu_index_check.skippable


@pytest.mark.parametrize("device_count", [True, False, -1, 1.5, float("nan"), float("inf"), "2", None])
def test_npu_index_check_passes_invalid_device_count_probe(monkeypatch, device_count):
    rules = _rules()
    monkeypatch.setattr(
        _device_module().torch,
        "npu",
        SimpleNamespace(device_count=lambda: device_count),
        raising=False,
    )

    assert rules.npu_index_check(DeviceType.NPU, [0]) is True


@pytest.mark.parametrize("device_indices", ["0", 0, (0,), {0}, {"0": 0}])
def test_npu_index_check_rejects_non_list(monkeypatch, device_indices):
    rules = _rules()
    monkeypatch.setattr(
        _device_module().torch,
        "npu",
        SimpleNamespace(device_count=lambda: 2),
        raising=False,
    )

    with pytest.raises(SchemaValidateError, match="must be a list or None"):
        rules.npu_index_check(DeviceType.NPU, device_indices)


@pytest.mark.parametrize("device_indices", [["0"], [1.5], [None], [True], [0, False]])
def test_npu_index_check_rejects_non_integer_values(device_indices):
    rules = _rules()

    with pytest.raises(SchemaValidateError, match="must be integers"):
        rules.npu_index_check(DeviceType.NPU, device_indices)


def test_npu_index_check_rejects_negative_values():
    rules = _rules()

    with pytest.raises(SchemaValidateError, match="non-negative"):
        rules.npu_index_check(DeviceType.NPU, [-1])


def test_npu_index_check_rejects_duplicate_indices(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(
        _device_module().torch,
        "npu",
        SimpleNamespace(device_count=lambda: 2),
        raising=False,
    )

    with pytest.raises(SchemaValidateError, match="must be unique"):
        rules.npu_index_check(DeviceType.NPU, [0, 0])


def test_npu_index_check_rejects_multiple_cpu_indices(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(
        _device_module().torch,
        "npu",
        SimpleNamespace(device_count=lambda: 2),
        raising=False,
    )

    with pytest.raises(SchemaValidateError, match="does not support multi-device"):
        rules.npu_index_check(DeviceType.CPU, [0, 1])


def test_npu_occupancy_check_uses_free_ratio_min(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(rules, "get_npu_memory", lambda _: (1900, 2000))
    assert "free memory" in rules.npu_occupancy_check(DeviceType.NPU, 0, 0.9)

    monkeypatch.setattr(rules, "get_npu_memory", lambda _: (1000, 2000))
    with pytest.raises(EnvError):
        rules.npu_occupancy_check(DeviceType.NPU, 0, 0.9)

    monkeypatch.setattr(rules, "get_npu_memory", lambda _: (-1, -1))
    assert rules.npu_occupancy_check(DeviceType.NPU, 0, 0.9) is True


def test_npu_occupancy_check_supports_custom_free_ratio_min(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(rules, "get_npu_memory", lambda _: (1500, 2000))

    assert "free memory" in rules.npu_occupancy_check(DeviceType.NPU, 0, free_ratio_min=0.7)
    with pytest.raises(EnvError):
        rules.npu_occupancy_check(DeviceType.NPU, 0, free_ratio_min=0.8)


def test_npu_occupancy_check_passes_zero_total_memory(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(rules, "get_npu_memory", lambda _: (0, 0))

    assert rules.npu_occupancy_check(DeviceType.NPU, 0, 0.9) is True


def test_npu_occupancy_check_passes_when_probe_raises(monkeypatch):
    rules = _rules()

    def fail_memory_query(_):
        raise RuntimeError("memory query failed")

    monkeypatch.setattr(rules, "get_npu_memory", fail_memory_query)

    assert rules.npu_occupancy_check(DeviceType.NPU, 0, 0.9) is True


@pytest.mark.parametrize(
    "memory_info",
    [
        None,
        100,
        (100,),
        (100, 200, 300),
        [],
        (True, 200),
        (100, True),
        (float("nan"), 200),
        (100, float("inf")),
        (100.0, 200.0),
        (None, 200),
        (100, None),
    ],
)
def test_npu_occupancy_check_passes_malformed_probe_result(monkeypatch, memory_info):
    rules = _rules()
    monkeypatch.setattr(rules, "get_npu_memory", lambda _: memory_info)

    assert rules.npu_occupancy_check(DeviceType.NPU, 0, 0.9) is True


def test_npu_probe_failure_is_a_silent_pass_by_manager(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(rules, "get_npu_memory", lambda _: (-1, -1))
    manager = PrecheckManager()
    manager.register(
        rules.npu_occupancy_check.bind(
            DeviceType.NPU,
            0,
            free_ratio_min=0.9,
            rule_id="npu.NPU:0",
        )
    )

    report = manager.run({rules.NPU_OCCUPANCY_GROUP})
    result = report.results[0]

    assert not report.failed
    assert result.status == PrecheckStatus.PASS
    assert result.reason == ""


def test_npu_occupancy_check_evaluates_each_card_index(monkeypatch):
    rules = _rules()
    monkeypatch.setattr(
        rules,
        "get_npu_memory",
        lambda index: (1900 if index == 0 else 1000, 2000),
    )

    assert "free memory" in rules.npu_occupancy_check(DeviceType.NPU, 0, 0.9)
    with pytest.raises(EnvError):
        rules.npu_occupancy_check(DeviceType.NPU, 1, 0.9)


@pytest.mark.parametrize("device_index", [-1, True])
def test_npu_occupancy_check_skips_invalid_index(monkeypatch, device_index):
    rules = _rules()
    monkeypatch.setattr(rules, "get_npu_memory", lambda _: pytest.fail("memory query should not run"))

    assert rules.npu_occupancy_check(DeviceType.NPU, device_index, 0.9) is False
