#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run model evaluation from Evaluation YAML (replaces MCP evaluation_run)."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Bootstrap: add shared library to sys.path before any cross-skill imports
# ---------------------------------------------------------------------------
_common_dir = Path(__file__).resolve().parents[3] / "tuning-loop-lib" / "scripts"
if str(_common_dir) not in sys.path:
    sys.path.insert(0, str(_common_dir))

from script_utils import emit_result, ensure_msmodelslim, parse_int_list  # noqa: E402
from shared import to_device_type  # noqa: E402


def _serialize_evaluate_result(evaluate_result: Any) -> Dict[str, Any]:
    """Return a JSON-compatible representation without losing Decimal precision."""
    return evaluate_result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# G1 长度配置门禁：evaluation.aisbench.max_out_len 必须 < max-model-len
# 依据实测(F0-5)：二者相等时 vLLM 以 VLLMValidationError 拒绝全部请求 -> accuracy=0.0，
# 假基线曾被当作达标向上传播。此预检在拉起推理服务前拦截，属第二道硬闸
# （第一道在 evaluation/evaluation-cfg 生成配置时）。
# ---------------------------------------------------------------------------
def _length_config_violation(evaluate_config_dict: Dict[str, Any]) -> Optional[str]:
    try:
        max_out_len = evaluate_config_dict["evaluation"]["aisbench"]["max_out_len"]
        max_model_len = evaluate_config_dict["inference_engine"]["args"]["max-model-len"]
    except (KeyError, TypeError):
        return None  # 字段缺失时不拦截，交由下游 schema 校验
    if max_out_len is None or max_model_len is None:
        return None
    if max_out_len >= max_model_len:
        return (
            f"长度配置自洽性校验失败: evaluation.aisbench.max_out_len={max_out_len} "
            f"必须小于 inference_engine.args.max-model-len={max_model_len} "
            f"(vLLM 约束 prompt+output<=context，相等/更大将拒绝全部请求导致 accuracy=0.0)。"
            f"请修正 max_out_len 后重新评测。"
        )
    return None


# ---------------------------------------------------------------------------
# G2 accuracy=0.0 诊断与上报：禁止 0 分假达标向上传播
# 判定规则（依据用户裁定「accuracy 为 0 应确认为什么是 0，无法确认则上报」）：
#   - vllm 服务日志含 VLLMValidationError / maximum context length -> 服务/配置全拒
#   - aisbench summary 含可解析 accuracy 且 == 0.0 且无全拒 -> 请求成功但真 0 分（真实 0，如实回传）
#   - 其它/无法确认 -> 上报 EVALUATION_INVALID 附证据，由执行者按 evaluate SKILL 自修复后重测，
#     仍失败再向编排层升级。
# ---------------------------------------------------------------------------
def _diagnose_zero_accuracy(save_path: str) -> Dict[str, Any]:
    save = Path(save_path)
    evidence: List[Dict[str, Any]] = []
    vllm_reject = False

    log_candidates = sorted(
        (p for p in save.rglob("vllm_server.log") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if log_candidates:
        log = log_candidates[0]
        try:
            tail = log.read_text(encoding="utf-8", errors="ignore")[-20000:]
        except Exception:  # noqa: BLE001
            tail = ""
        vllm_reject = ("VLLMValidationError" in tail) or ("maximum context length" in tail)
        evidence.append({"type": "vllm_server_log", "path": str(log), "rejected_by_vllm": vllm_reject})

    summary_files = sorted(
        glob.glob(str(save / "aisbench_output" / "*" / "summary")),
        key=os.path.getmtime,
        reverse=True,
    )
    parsed_acc: Optional[float] = None
    if summary_files:
        s = summary_files[0]
        try:
            text = Path(s).read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            text = ""
        m = re.search(r"accuracy[\"':=\s]*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if m:
            parsed_acc = float(m.group(1))
        evidence.append(
            {
                "type": "aisbench_summary",
                "path": s,
                "parsed_accuracy": parsed_acc,
                "preview": text[:400],
            }
        )
    return {"evidence": evidence, "vllm_reject": vllm_reject, "aisbench_accuracy": parsed_acc}


def _is_zero(value: Any) -> bool:
    try:
        return float(str(value)) == 0.0
    except (TypeError, ValueError):
        return False


def run_evaluation(
    quant_model_path: str,
    evaluate_id: str,
    evaluate_config_path: str,
    save_path: str,
    device: str = "npu",
    device_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateContext
    from msmodelslim.infra.service_oriented_evaluate_service import (
        ServiceOrientedEvaluateService,
        ServiceOrientedEvaluateServiceConfig,
    )
    from msmodelslim.utils.security import yaml_safe_load

    try:
        evaluate_config_dict = yaml_safe_load(evaluate_config_path)

        violation = _length_config_violation(evaluate_config_dict)
        if violation:
            return {"ok": False, "error_code": "VALIDATION_ERROR", "error": violation}

        evaluate_config = ServiceOrientedEvaluateServiceConfig.model_validate(evaluate_config_dict)
        evaluate_service = ServiceOrientedEvaluateService()
        evaluate_result = evaluate_service.evaluate(
            context=EvaluateContext(
                evaluate_id=evaluate_id,
                device=to_device_type(device),
                device_indices=device_indices,
                working_dir=Path(save_path),
            ),
            evaluate_config=evaluate_config,
            model_path=Path(quant_model_path),
        )
        serialized = _serialize_evaluate_result(evaluate_result)

        zero_datasets = [a for a in serialized.get("accuracies", []) if _is_zero(a.get("accuracy"))]
        if zero_datasets:
            diag = _diagnose_zero_accuracy(save_path)
            if diag["aisbench_accuracy"] is not None and diag["aisbench_accuracy"] == 0.0 and not diag["vllm_reject"]:
                # 请求成功但确实答对 0 题：真实 0 分，如实回传，由编排层判定不达标
                return {
                    "ok": True,
                    "evaluate_result": serialized,
                    "warning": (
                        "accuracy=0.0 已确认：aisbench summary 显示请求成功且得分为 0，"
                        "属真实 0 分（非服务/配置问题），由编排层按不达标处理。"
                    ),
                }
            # 无法确认根因 / 确认是服务或配置问题 -> 上报，由执行者自修复后重测
            return {
                "ok": False,
                "error_code": "EVALUATION_INVALID",
                "error": (
                    f"存在 accuracy=0.0 的数据集 {zero_datasets} 且未能确认为真实 0 分"
                    f"(vllm_reject={diag['vllm_reject']}, aisbench_accuracy={diag['aisbench_accuracy']})。"
                    f"请按 evaluate SKILL「0 分处置」自查根因：若为 max_out_len/max-model-len 等配置问题，"
                    f"修正配置后重测一次；仍为 0 或无法确认则向编排层上报本错误与证据。"
                ),
                "evidence": diag["evidence"],
                "evaluate_result": serialized,
            }
        return {
            "ok": True,
            "evaluate_result": serialized,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation from Evaluation YAML.")
    parser.add_argument("--quant-model-path", required=True)
    parser.add_argument("--evaluate-id", required=True)
    parser.add_argument("--evaluate-config-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--device-indices",
        default=None,
        help="Comma-separated indices or JSON array, e.g. 0,1 or [0,1]",
    )
    args = parser.parse_args()

    ensure_msmodelslim()
    result = run_evaluation(
        quant_model_path=args.quant_model_path,
        evaluate_id=args.evaluate_id,
        evaluate_config_path=args.evaluate_config_path,
        save_path=args.save_path,
        device=args.device,
        device_indices=parse_int_list(args.device_indices),
    )
    return emit_result(result)


if __name__ == "__main__":
    sys.exit(main())
