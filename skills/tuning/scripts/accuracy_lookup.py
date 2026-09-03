#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lookup accuracy cache (replaces MCP accuracy_lookup)."""

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

from script_utils import emit_result, ensure_msmodelslim, with_plugin_timeout_retry  # noqa: E402


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


def accuracy_lookup(save_path: str, evaluate_config_path: str, practice_path: str) -> Dict[str, Any]:
    from msmodelslim.infra.yaml_practice_accuracy_manager import YamlTuningAccuracyManager

    # F0-9 修复（2026-09-03 实测根因）：同 accuracy_append——先全量预热注册 processor 子类，
    # 避免 v1 practice 的 model_validate 在插件 5s 超时窗口内执行 processor/__init__.py 全量
    # 导入被 SIGALRM 截断成半截 registry（首轮 303、retry 后 203）。
    import msmodelslim.processor  # noqa: F401

    def _run() -> Dict[str, Any]:
        history_path = str(Path(save_path) / "history")
        accuracy = YamlTuningAccuracyManager().load_accuracy(history_path)
        evaluate_config = _load_evaluate_config(evaluate_config_path)
        practice_obj = _load_practice(practice_path)
        evaluate_result = accuracy.get_accuracy(practice_obj, evaluate_config)
        if evaluate_result is None:
            return {"ok": True, "cache_hit": False}
        return {
            "ok": True,
            "cache_hit": True,
            "evaluate_result": evaluate_result.model_dump(),
        }

    # G3：插件冷启动 5s 超时(Code 303)属环境慢而非逻辑错，同进程重试可自愈
    # （F0-9 修复后 processor 已预热，303 由源头消除，此处仅兜底异常场景）
    try:
        return with_plugin_timeout_retry(_run)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Lookup accuracy cache for practice + evaluate config.")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--evaluate-config-path", required=True)
    parser.add_argument("--practice-path", required=True)
    args = parser.parse_args()

    ensure_msmodelslim()
    return emit_result(accuracy_lookup(args.save_path, args.evaluate_config_path, args.practice_path))


if __name__ == "__main__":
    sys.exit(main())
