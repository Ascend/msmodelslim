#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append accuracy cache entry (replaces MCP accuracy_append)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Bootstrap: add shared library to sys.path before any cross-skill imports
# ---------------------------------------------------------------------------
_common_dir = Path(__file__).resolve().parents[2] / "tuning-loop-lib" / "scripts"
if str(_common_dir) not in sys.path:
    sys.path.insert(0, str(_common_dir))

from script_utils import emit_result, ensure_msmodelslim, parse_optional_json, with_plugin_timeout_retry  # noqa: E402


def _load_practice(practice_path: str):
    from msmodelslim.core.practice import PracticeConfig
    from msmodelslim.utils.security import yaml_safe_load

    content = yaml_safe_load(practice_path)
    return PracticeConfig.model_validate(content)


def _load_evaluate_config(evaluate_config_path: str):
    from msmodelslim.infra.service_oriented_evaluate_service import ServiceOrientedEvaluateServiceConfig
    from msmodelslim.utils.security import yaml_safe_load

    config_dict = yaml_safe_load(evaluate_config_path)
    return ServiceOrientedEvaluateServiceConfig.model_validate(config_dict)


def accuracy_append(
    save_path: str,
    evaluate_config_path: str,
    practice_path: str,
    evaluate_result: Dict[str, Any],
) -> Dict[str, Any]:
    from msmodelslim.core.tune_strategy import EvaluateResult
    from msmodelslim.infra.yaml_practice_accuracy_manager import YamlTuningAccuracyManager

    # F0-9 修复（2026-09-03 实测根因）：v1 practice 的 model_validate 经插件加载触发
    # processor/__init__.py 全量执行（实测 ~4s+，quant.attention 段 import torch 有 ~2.9s gap），
    # 落在框架插件 5s 超时窗口（SIGALRM 原地中断）内会被截断，残留「前 13 类已注册、后段缺失」
    # 的半截 registry：首轮报 303；G3 重试后 registry 只增不减，稳定报 203（linear_quant 不在
    # 期望 tags）。修复 = 先全量预热注册，把耗时导入挪出插件超时窗口；此后插件加载仅注册自身，
    # 秒级完成，retry 退化为纯兜底。
    import msmodelslim.processor  # noqa: F401

    def _run() -> Dict[str, Any]:
        history_path = str(Path(save_path) / "history")
        accuracy = YamlTuningAccuracyManager().load_accuracy(history_path)
        evaluate_config = _load_evaluate_config(evaluate_config_path)
        practice_obj = _load_practice(practice_path)
        evaluate_obj = EvaluateResult.model_validate(evaluate_result)
        accuracy.append_accuracy(practice_obj, evaluate_config, evaluate_obj)
        return {"ok": True, "message": "accuracy appended"}

    # G3：插件冷启动 5s 超时(Code 303)属环境慢而非逻辑错，同进程重试可自愈
    # （F0-9 修复后 processor 已预热，303 由源头消除，此处仅兜底异常场景）
    try:
        return with_plugin_timeout_retry(_run)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Append accuracy cache entry.")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--evaluate-config-path", required=True)
    parser.add_argument("--practice-path", required=True)
    parser.add_argument("--evaluate-result", required=True, help="JSON object of EvaluateResult")
    args = parser.parse_args()

    ensure_msmodelslim()
    evaluate_result = parse_optional_json(args.evaluate_result, default={})
    if not isinstance(evaluate_result, dict):
        return emit_result({"ok": False, "error": "evaluate-result must be a JSON object"})
    return emit_result(
        accuracy_append(
            args.save_path,
            args.evaluate_config_path,
            args.practice_path,
            evaluate_result,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
