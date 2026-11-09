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

from unittest.mock import MagicMock

import pytest

from msmodelslim.app.naive_quantization import application as app_module
from msmodelslim.utils.config import CONFIG_FILE_PATH
from msmodelslim.utils.exception import EnvError, SchemaValidateError, SecurityError
from msmodelslim.utils.precheck import PrecheckReport, PrecheckResult, PrecheckStatus


def test_log_skipped_checks_logs_config_skipped_results_by_group(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(app_module, "get_logger", lambda: logger)
    report = PrecheckReport(
        (
            PrecheckResult(
                "disk",
                "disk",
                PrecheckStatus.SKIP,
                "removed from quant_precheck_item",
                skippable=True,
                config_skipped=True,
            ),
            PrecheckResult("npu", "npu.NPU:0", PrecheckStatus.SKIP, skippable=True, config_skipped=True),
            PrecheckResult("npu", "npu.NPU:1", PrecheckStatus.SKIP, skippable=True, config_skipped=True),
            PrecheckResult("path.permission", "model_path", PrecheckStatus.PASS),
        )
    )

    app_module.log_skipped_checks(report)

    info_messages = [call.args[0] % call.args[1:] for call in logger.info.call_args_list]
    assert info_messages == [
        "disk skipped; if you want this check, add disk to %s" % CONFIG_FILE_PATH,
        "npu skipped; if you want this check, add npu to %s" % CONFIG_FILE_PATH,
    ]
    logger.warning.assert_not_called()


def test_raise_for_report_returns_when_no_failure():
    report = PrecheckReport((PrecheckResult("path.permission", "model_path", PrecheckStatus.PASS),))

    app_module.raise_for_report(report)


def test_raise_for_report_reraises_single_parameter_error():
    error = SchemaValidateError("quant_type must be a QuantType", action="use QuantType")
    report = PrecheckReport(
        (
            PrecheckResult(
                "quant.arguments",
                "quant.arguments",
                PrecheckStatus.FAIL,
                "quant_type must be a QuantType",
                "use QuantType",
                error=error,
            ),
        )
    )

    with pytest.raises(SchemaValidateError) as exc_info:
        app_module.raise_for_report(report)

    assert exc_info.value is error


def test_raise_for_report_reraises_single_environment_error():
    error = EnvError("no space left", action="free space")
    report = PrecheckReport(
        (
            PrecheckResult(
                "disk", "disk", PrecheckStatus.FAIL, "no space left", "free space", skippable=True, error=error
            ),
        )
    )

    with pytest.raises(EnvError) as exc_info:
        app_module.raise_for_report(report)

    assert exc_info.value is error


def test_raise_for_report_aggregates_mixed_failures_as_env_error():
    report = PrecheckReport(
        (
            PrecheckResult(
                "disk",
                "disk",
                PrecheckStatus.FAIL,
                "no space left",
                "free space",
                skippable=True,
                error=EnvError("no space left", action="free space"),
            ),
            PrecheckResult(
                "quant.arguments",
                "quant.arguments",
                PrecheckStatus.FAIL,
                "bad argument",
                "fix argument",
                error=SchemaValidateError("bad argument", action="fix argument"),
            ),
        )
    )

    with pytest.raises(EnvError) as exc_info:
        app_module.raise_for_report(report)

    assert str(exc_info.value) == (
        "Code: 100, Message: Precheck failed\n"
        "disk: [EnvError] Code: 100, Message: no space left, TIP: free space\n"
        "quant.arguments: [SchemaValidateError] Code: 203, Message: bad argument, TIP: fix argument\n"
        "TIP: remove disk from quant_precheck_item in %s to skip" % CONFIG_FILE_PATH
    )


def test_raise_for_report_lists_each_failure_with_its_own_exception():
    report = PrecheckReport(
        (
            PrecheckResult(
                "path.permission",
                "path.permission.model_path",
                PrecheckStatus.FAIL,
                "The path /home/mnt doesn't exist or not a directory.",
                "Please make sure the path exists and is a directory.",
                error=SecurityError(
                    "The path /home/mnt doesn't exist or not a directory.",
                    action="Please make sure the path exists and is a directory.",
                ),
            ),
            PrecheckResult(
                "npu",
                "npu.NPU:3",
                PrecheckStatus.FAIL,
                "NPU:3 free memory 0.73/29.49 GB, free ratio 0.02 below threshold 0.90",
                "Please switch to another card or wait for the running task to finish",
                skippable=True,
                error=EnvError(
                    "NPU:3 free memory 0.73/29.49 GB, free ratio 0.02 below threshold 0.90",
                    action="Please switch to another card or wait for the running task to finish",
                ),
            ),
        )
    )

    with pytest.raises(EnvError) as exc_info:
        app_module.raise_for_report(report)

    # 外层统一为 EnvError "Precheck failed"，内部逐项给出该失败项的完整异常（含类型/错误码/TIP）。
    assert str(exc_info.value) == (
        "Code: 100, Message: Precheck failed\n"
        "model_path: [SecurityError] Code: 204, Message: The path /home/mnt doesn't exist or not a directory., "
        "TIP: Please make sure the path exists and is a directory.\n"
        "NPU:3: [EnvError] Code: 100, Message: NPU:3 free memory 0.73/29.49 GB, "
        "free ratio 0.02 below threshold 0.90, "
        "TIP: Please switch to another card or wait for the running task to finish\n"
        "TIP: remove npu from quant_precheck_item in %s to skip" % CONFIG_FILE_PATH
    )


def test_raise_for_report_uses_env_error_for_multiple_environment_failures():
    report = PrecheckReport(
        (
            PrecheckResult(
                "disk",
                "disk",
                PrecheckStatus.FAIL,
                "no space left",
                "free space",
                skippable=True,
                error=EnvError("no space left"),
            ),
            PrecheckResult(
                "npu",
                "npu.NPU:0",
                PrecheckStatus.FAIL,
                "card occupied",
                "switch card",
                skippable=True,
                error=EnvError("card occupied"),
            ),
        )
    )

    with pytest.raises(EnvError) as exc_info:
        app_module.raise_for_report(report)

    message = str(exc_info.value)
    assert "NPU:0: [EnvError] Code: 100, Message: card occupied" in message
    assert "remove disk, npu from quant_precheck_item" in message


def test_raise_for_report_wraps_unexpected_exception_as_env_error():
    report = PrecheckReport(
        (PrecheckResult("custom", "custom", PrecheckStatus.FAIL, "boom", error=RuntimeError("boom")),)
    )

    with pytest.raises(EnvError) as exc_info:
        app_module.raise_for_report(report)

    assert str(exc_info.value) == "Code: 100, Message: Precheck failed\ncustom: boom"


def test_raise_for_report_attaches_all_failures_when_aggregating():
    report = PrecheckReport(
        (
            PrecheckResult(
                "disk",
                "disk",
                PrecheckStatus.FAIL,
                "no space left",
                "free space",
                skippable=True,
                error=EnvError("no space left"),
            ),
            PrecheckResult(
                "npu",
                "npu.NPU:0",
                PrecheckStatus.FAIL,
                "card occupied",
                "switch card",
                skippable=True,
                error=EnvError("card occupied"),
            ),
        )
    )

    with pytest.raises(EnvError) as exc_info:
        app_module.raise_for_report(report)

    failures = exc_info.value.precheck_failures
    assert [item.check_id for item in failures] == ["disk", "npu.NPU:0"]
    assert [type(item.error).__name__ for item in failures] == ["EnvError", "EnvError"]
    assert [item.suggestion for item in failures] == ["free space", "switch card"]


def test_raise_for_report_attaches_failures_when_reraising_single_error():
    error = SchemaValidateError("bad argument")
    report = PrecheckReport(
        (PrecheckResult("quant.arguments", "quant.arguments", PrecheckStatus.FAIL, "bad argument", error=error),)
    )

    with pytest.raises(SchemaValidateError) as exc_info:
        app_module.raise_for_report(report)

    assert exc_info.value is error
    assert [item.check_id for item in exc_info.value.precheck_failures] == ["quant.arguments"]
