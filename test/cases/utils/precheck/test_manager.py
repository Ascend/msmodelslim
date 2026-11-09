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

Tests for ``msmodelslim.utils.precheck.manager``: rule declaration, binding,
registration and reporting.
"""

import inspect
from functools import partial
from typing import Any
from unittest.mock import MagicMock

import pytest

from msmodelslim.utils import precheck
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.precheck import PrecheckManager, PrecheckReport, PrecheckResult, PrecheckStatus
from msmodelslim.utils.precheck import manager as precheck_manager


def _pass_result(_rule_id):
    return True


def test_manager_runs_rules_in_registration_order():
    calls = []

    @precheck.precheck_rule(rule_id="first")
    def first():
        calls.append("first")
        return _pass_result("first")

    @precheck.precheck_rule(rule_id="second")
    def second():
        calls.append("second")
        return _pass_result("second")

    manager = PrecheckManager()
    manager.register(second)
    manager.register(first)

    report = manager.run()

    assert calls == ["second", "first"]
    assert [result.check_id for result in report.results] == ["second", "first"]


def test_manager_binds_function_args_and_kwargs():
    calls = []

    @precheck.precheck_rule(rule_id="bound")
    def bound_check(value, suffix=""):
        calls.append((value, suffix))
        return _pass_result("bound")

    manager = PrecheckManager()
    manager.register(bound_check.bind(3, suffix="ok"))

    report = manager.run()

    assert calls == [(3, "ok")]
    assert not report.failed


def test_manager_converts_required_rule_exception_to_fail():
    @precheck.precheck_rule(rule_id="required")
    def broken():
        raise RuntimeError("boom")

    manager = PrecheckManager()
    manager.register(broken)

    report = manager.run()

    assert report.failed
    assert report.results[0].reason == "boom"


def test_manager_converts_optional_rule_exception_to_fail():
    @precheck.precheck_rule(rule_id="npu", skippable=True)
    def unavailable():
        raise RuntimeError("probe unavailable")

    manager = PrecheckManager()
    manager.register(unavailable)

    report = manager.run({"npu"})

    assert report.failed
    assert report.results[0].status == PrecheckStatus.FAIL


def test_manager_converts_false_to_skip():
    @precheck.precheck_rule(rule_id="probe")
    def unavailable():
        return False

    manager = PrecheckManager()
    manager.register(unavailable)

    result = manager.run().results[0]

    assert result.status == PrecheckStatus.SKIP
    assert not result.config_skipped


def test_fail_without_action_leaves_suggestion_empty():
    @precheck.precheck_rule(rule_id="quant.arguments")
    def broken():
        raise SchemaValidateError("quant_type and config_path only one can be provided")

    manager = PrecheckManager()
    manager.register(broken)

    report = manager.run()

    result = report.results[0]
    assert result.status == PrecheckStatus.FAIL
    assert result.reason == "quant_type and config_path only one can be provided"
    assert result.suggestion == ""


def test_run_marks_unconfigured_optional_rules_as_skip():
    calls = []

    @precheck.precheck_rule(rule_id="required")
    def required():
        calls.append("required")
        return _pass_result("required")

    @precheck.precheck_rule(rule_id="disk", skippable=True)
    def optional():
        calls.append("disk")
        return _pass_result("disk")

    manager = PrecheckManager()
    manager.register(required)
    manager.register(optional)

    report = manager.run()

    assert calls == ["required"]  # 未启用的可选项不执行
    assert [result.check_id for result in report.results] == ["required", "disk"]
    skipped = report.results[1]
    assert skipped.status == PrecheckStatus.SKIP
    assert "quant_precheck_item" in skipped.reason
    assert skipped.config_skipped is True
    assert skipped.skippable

    report = manager.run({"disk"})

    assert calls == ["required", "required", "disk"]
    assert [(result.check_id, result.status) for result in report.results] == [
        ("required", PrecheckStatus.PASS),
        ("disk", PrecheckStatus.PASS),
    ]


def test_run_executes_mandatory_rules_regardless_of_config():
    manager = PrecheckManager()
    manager.register(precheck.PrecheckRule(check=lambda: _pass_result("required"), rule_id="required"))

    report = manager.run({"required", "unknown"})

    assert [result.check_id for result in report.results] == ["required"]
    assert not report.failed


def test_bind_rejects_argument_count_mismatch():
    @precheck.precheck_rule(rule_id="bound")
    def no_args():
        return _pass_result("bound")

    with pytest.raises(TypeError):
        no_args.bind("unexpected")


def test_bind_rejects_argument_type_mismatch():
    @precheck.precheck_rule(rule_id="typed")
    def typed_check(count: int, ratio: float = 1.0):
        return _pass_result("typed")

    with pytest.raises(TypeError):
        typed_check.bind("1")
    with pytest.raises(TypeError):
        typed_check.bind(1, ratio="high")

    manager = PrecheckManager()
    manager.register(typed_check.bind(1, ratio=2.0, rule_id="typed_ok"))
    assert not manager.run().failed


def test_bind_checks_container_element_types():
    @precheck.precheck_rule(rule_id="list")
    def list_check(indices: list[int]):
        return _pass_result("list")

    with pytest.raises(TypeError):
        list_check.bind(["0"])

    manager = PrecheckManager()
    manager.register(list_check.bind([0, 1], rule_id="list_ok"))
    assert not manager.run().failed


def test_bind_accepts_any_annotated_values():
    @precheck.precheck_rule(rule_id="flexible")
    def flexible_check(value: Any, payload: Any = None):
        return _pass_result("flexible")

    manager = PrecheckManager()
    manager.register(flexible_check.bind(object(), payload=123))

    assert not manager.run().failed


def test_manager_register_skips_duplicate(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(precheck_manager, "get_logger", lambda: logger)

    manager = PrecheckManager()
    manager.register(precheck.PrecheckRule(check=lambda: _pass_result("required"), rule_id="required"))
    manager.register(precheck.PrecheckRule(check=lambda: _pass_result("required"), rule_id="required"))

    assert len(manager._rules) == 1
    assert any("Duplicate precheck rule id" in call.args[0] for call in logger.warning.call_args_list)


def test_report_to_dict_uses_stable_fields():
    report = PrecheckReport(
        (PrecheckResult("path.permission", "model_path", PrecheckStatus.FAIL, "不可读", "检查权限"),)
    )

    assert report.to_dict() == {
        "results": [
            {
                "check_id": "model_path",
                "group": "path.permission",
                "status": "FAIL",
                "reason": "不可读",
                "suggestion": "检查权限",
            }
        ]
    }


def test_precheck_rule_decorator_wraps_function_into_rule():
    @precheck.precheck_rule(group="disk", rule_id="disk", skippable=True)
    def check_disk(value):
        return "size %d" % value

    assert isinstance(check_disk, precheck.PrecheckRule)
    assert check_disk.group == "disk"
    assert check_disk.skippable
    assert check_disk.needs_binding

    manager = PrecheckManager()
    manager.register(check_disk.bind(7))

    report = manager.run({"disk"})
    result = report.results[0]
    assert result.check_id == "disk"
    assert result.status == PrecheckStatus.PASS
    assert result.reason == "size 7"


def test_precheck_rule_decorator_bare_form_uses_function_name():
    @precheck.precheck_rule
    def zero_arg_check():
        return "ok"

    assert isinstance(zero_arg_check, precheck.PrecheckRule)
    assert zero_arg_check.id == "zero_arg_check"
    assert zero_arg_check.group == "zero_arg_check"
    assert not zero_arg_check.needs_binding

    manager = PrecheckManager()
    manager.register(zero_arg_check)
    assert manager.run().results[0].reason == "ok"


def test_decorated_rule_stays_callable_like_wrapped_function():
    @precheck.precheck_rule(rule_id="string")
    def check_value(value):
        return "got %s" % value

    assert check_value(5) == "got 5"


def test_bind_overrides_rule_id_and_keeps_metadata():
    @precheck.precheck_rule(group="npu", skippable=True)
    def check_card(device_index):
        return "card %s" % device_index

    bound = check_card.bind(3, rule_id="npu.NPU:3")

    assert bound.id == "npu.NPU:3"
    assert bound.group == "npu"
    assert bound.skippable
    assert not bound.needs_binding
    assert bound.check() == "card 3"


def test_register_rejects_plain_function():
    manager = PrecheckManager()

    with pytest.raises(TypeError):
        manager.register(lambda: _pass_result("fn"))


def test_register_rejects_unbound_rule():
    @precheck.precheck_rule(rule_id="needs_arg")
    def needs_arg(value):
        return value

    manager = PrecheckManager()

    with pytest.raises(TypeError):
        manager.register(needs_arg)


def test_register_accepts_custom_rule_subclass():
    class AlwaysFail(precheck.PrecheckRule):
        def __init__(self):
            super().__init__(group="custom", rule_id="custom")

        def check(self):
            raise RuntimeError("custom failure")

    manager = PrecheckManager()
    rule = AlwaysFail()
    manager.register(rule)

    report = manager.run()
    assert report.failed
    assert report.results[0].group == "custom"
    assert report.results[0].reason == "custom failure"


def test_custom_rule_subclass_with_constructor_args():
    class BoundRule(precheck.PrecheckRule):
        def __init__(self, value):
            super().__init__(group="bound", rule_id="bound")
            self.value = value

        def check(self):
            return "value=%s" % self.value

    manager = PrecheckManager()
    manager.register(BoundRule(42))

    result = manager.run().results[0]
    assert result.status == PrecheckStatus.PASS
    assert result.reason == "value=42"


def test_decorator_rejects_reserved_parameter_name():
    with pytest.raises(TypeError):

        @precheck.precheck_rule(rule_id="reserved")
        def uses_reserved(rule_id):
            return rule_id


def test_default_rule_id_unwraps_partial():
    def check_value(value):
        return value

    assert precheck.PrecheckRule(func=partial(check_value, 1)).id == "check_value"


def test_default_rule_id_requires_explicit_id_without_name():
    class CallableWithoutName:
        def __call__(self):
            return True

    with pytest.raises(TypeError):
        precheck.PrecheckRule(func=CallableWithoutName())


def test_rule_exposes_original_signature_and_func():
    @precheck.precheck_rule(group="disk", rule_id="disk")
    def check_disk(model_path: str, ratio: float = 1.0) -> bool:
        return True

    assert list(inspect.signature(check_disk).parameters) == ["model_path", "ratio"]
    assert check_disk.func is not None

    bound = check_disk.bind("/data")
    assert inspect.signature(bound).parameters == {}
    assert bound.func is None


def test_bound_rule_call_forwards_to_check():
    @precheck.precheck_rule(rule_id="value")
    def check_value(value):
        return "got %s" % value

    bound = check_value.bind(7)

    assert bound() == "got 7"
    assert bound() == bound.check()
    with pytest.raises(TypeError):
        bound(8)


def test_register_warns_when_unbound_rule_uses_default_args(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(precheck_manager, "get_logger", lambda: logger)

    @precheck.precheck_rule(rule_id="defaults")
    def with_defaults(value="x"):
        return value

    manager = PrecheckManager()
    manager.register(with_defaults)

    assert manager.run().results[0].status == PrecheckStatus.PASS
    assert any("without bind()" in call.args[0] for call in logger.warning.call_args_list)
