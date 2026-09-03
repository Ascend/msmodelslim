#!/usr/bin/env python3
"""步骤1：生成结构覆盖完备的减层随机权重测试模型。

验证语义（非全量）：
  模型由若干结构类型堆叠而成；减层只需覆盖每种结构至少一次。
  默认取「覆盖完备的最小前缀层数」L，再 from_config 建随机权重模型。
  禁止盲目前 N 层（可能漏 MoE / full_attn 等）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoConfig
import transformers

# 配置中按层对齐的 list 字段（减层时一并截断）
_PER_LAYER_LIST_KEYS = (
    "layer_types",
    "swiglu_limits",
    "swiglu_limits_shared",
)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _copy_non_weight_files(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            continue
        if name.endswith(".safetensors"):
            continue
        if name.endswith(".index.json"):
            continue
        shutil.copy2(src, dst)


def _text_cfg_view(cfg: dict) -> Tuple[dict, bool]:
    """返回 (text_cfg_dict, is_nested_under_text_config)。"""
    if isinstance(cfg.get("text_config"), dict):
        return cfg["text_config"], True
    return cfg, False


def _parse_moe_layer_indices(text_cfg: dict, n_layers: int) -> Set[int]:
    """从 config 解析 MoE 层下标集合；无 MoE 信号则返回空集。"""
    enum = text_cfg.get("moe_layers_enum")
    if enum is not None:
        if isinstance(enum, str):
            return {int(x) for x in enum.split(",") if str(x).strip() != ""}
        if isinstance(enum, (list, tuple)):
            return {int(x) for x in enum}
    if not text_cfg.get("use_moe") and not text_cfg.get("num_experts") and not text_cfg.get("moe_num_experts"):
        return set()
    # 常见回退：除第 0 层外均为 MoE，或按 moe_every_n_layer / moe_layer_offset
    every = text_cfg.get("moe_every_n_layer")
    offset = int(text_cfg.get("moe_layer_offset") or 0)
    if every:
        every = int(every)
        return {i for i in range(n_layers) if i >= offset and (i - offset) % every == 0}
    # DeepSeek 类：首层 dense，其余 MoE
    return set(range(1, n_layers))


def _structure_label(layer_idx: int, layer_types: Optional[List[Any]], moe_indices: Set[int]) -> str:
    """单层结构标签：注意力类型 × FFN 类型（堆叠重复则标签相同）。"""
    if layer_types and layer_idx < len(layer_types):
        attn = f"attn:{layer_types[layer_idx]}"
    else:
        attn = "attn:default"
    if moe_indices:
        ffn = "ffn:moe" if layer_idx in moe_indices else "ffn:dense"
    else:
        ffn = "ffn:dense"
    return f"{attn}+{ffn}"


def plan_structure_cover(
    cfg: dict,
    min_layers: int = 1,
    max_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """计算结构覆盖完备的最小前缀层数。

    Returns:
        dict: cover_layers, first_occurrence, all_labels, incomplete (若 max 截断导致漏盖)
    """
    text_cfg, _ = _text_cfg_view(cfg)
    n_layers = int(text_cfg.get("num_hidden_layers") or 1)
    layer_types = text_cfg.get("layer_types")
    if not isinstance(layer_types, list):
        layer_types = None
    moe_indices = _parse_moe_layer_indices(text_cfg, n_layers)

    first_occ: Dict[str, int] = {}
    for i in range(n_layers):
        label = _structure_label(i, layer_types, moe_indices)
        if label not in first_occ:
            first_occ[label] = i

    if not first_occ:
        cover = max(min_layers, 1)
        return {
            "cover_layers": cover,
            "first_occurrence": {},
            "all_labels": [],
            "incomplete": False,
            "original_layers": n_layers,
        }

    needed = max(first_occ.values()) + 1
    cover = max(needed, min_layers)
    incomplete = False
    missing: List[str] = []
    if max_layers is not None and cover > max_layers:
        # 在 max 前缀内仍未出现的标签 → 不完备
        covered_in_max = {lab for lab, idx in first_occ.items() if idx < max_layers}
        missing = sorted(set(first_occ) - covered_in_max)
        cover = max_layers
        incomplete = bool(missing)

    return {
        "cover_layers": cover,
        "first_occurrence": first_occ,
        "all_labels": sorted(first_occ.keys()),
        "incomplete": incomplete,
        "missing_labels": missing,
        "original_layers": n_layers,
        "moe_layer_count": len(moe_indices),
    }


def _truncate_per_layer_lists(text_cfg: dict, num_layers: int) -> dict:
    out = dict(text_cfg)
    out["num_hidden_layers"] = num_layers
    for key in _PER_LAYER_LIST_KEYS:
        val = out.get(key)
        if isinstance(val, list):
            out[key] = val[:num_layers]
    # moe_layers_enum：过滤到减层范围内，保持原类型（str / list）
    enum = out.get("moe_layers_enum")
    if isinstance(enum, str):
        kept = [x.strip() for x in enum.split(",") if x.strip() and int(x) < num_layers]
        out["moe_layers_enum"] = ",".join(kept)
    elif isinstance(enum, list):
        out["moe_layers_enum"] = [int(x) for x in enum if int(x) < num_layers]
    return out


def _shrink_config(cfg: dict, num_layers: int) -> dict:
    """按结构覆盖层数减层（前缀截断 + 同步 per-layer / moe 枚举）。"""
    out = dict(cfg)
    text_cfg, nested = _text_cfg_view(out)
    shrunk = _truncate_per_layer_lists(text_cfg, num_layers)
    if nested:
        out["text_config"] = shrunk
    else:
        out.update(shrunk)
    return out


def _build_random_model_from_config(config):
    candidate_auto_model_names = [
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModel",
    ]
    errors = []
    for cls_name in candidate_auto_model_names:
        auto_cls = getattr(transformers, cls_name, None)
        if auto_cls is None:
            continue
        try:
            model = auto_cls.from_config(config, trust_remote_code=True, torch_dtype=torch.float32)
            return model, cls_name
        except Exception as e:  # pragma: no cover - best-effort fallback chain
            errors.append(f"{cls_name}: {repr(e)}")

    raise RuntimeError(
        "无法根据配置构建模型。已尝试: " + ", ".join(candidate_auto_model_names) + "\n错误详情:\n" + "\n".join(errors)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Step1: 结构覆盖完备的减层随机权重测试模型（非全量验证）")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="可选：减层上限。默认不设上限，取结构覆盖完备的最小前缀。"
        "若上限导致结构覆盖不完备则失败（除非 --allow-incomplete-cover）。",
    )
    parser.add_argument(
        "--min-layers",
        type=int,
        default=1,
        help="减层下限（默认 1）。结构种类少于此时仍至少保留该层数。",
    )
    parser.add_argument(
        "--allow-incomplete-cover",
        action="store_true",
        help="允许在 --num-layers 截断后结构覆盖不完备（不推荐；仅调试）。",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只打印结构覆盖计划，不生成模型。",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    src_cfg = os.path.join(args.model_path, "config.json")
    if not os.path.exists(src_cfg):
        print(f"[ERROR] 缺少配置文件: {src_cfg}")
        return 1

    cfg = _read_json(src_cfg)
    plan = plan_structure_cover(cfg, min_layers=args.min_layers, max_layers=args.num_layers)

    print("[INFO] 验证语义: 减层结构覆盖（非全量）")
    print(f"[INFO] 原层数: {plan['original_layers']} → 覆盖层数: {plan['cover_layers']}")
    print(f"[INFO] 结构标签全集 ({len(plan['all_labels'])}): {plan['all_labels']}")
    for lab, idx in sorted(plan["first_occurrence"].items(), key=lambda x: x[1]):
        print(f"[INFO]   首次覆盖 layer[{idx}] = {lab}")
    if plan.get("moe_layer_count"):
        print(f"[INFO] MoE 层计数(原配置): {plan['moe_layer_count']}")

    if plan["incomplete"]:
        msg = (
            f"结构覆盖不完备: --num-layers={args.num_layers} 漏掉 {plan.get('missing_labels')}. "
            "请去掉上限或增大 --num-layers，使前缀覆盖全部结构标签。"
        )
        if args.allow_incomplete_cover:
            print(f"[WARN] {msg} （已 --allow-incomplete-cover，继续）")
        else:
            print(f"[ERROR] {msg}")
            return 2

    if args.plan_only:
        print("[OK] plan-only 完成")
        return 0

    _copy_non_weight_files(args.model_path, args.output_path)
    shrunk = _shrink_config(cfg, plan["cover_layers"])
    _write_json(os.path.join(args.output_path, "config.json"), shrunk)
    # 落盘覆盖计划，供 step2~4 / 评审引用
    _write_json(
        os.path.join(args.output_path, "structure_cover_plan.json"),
        {
            "semantics": "reduced_layer_structure_cover",
            "not_full_model": True,
            **plan,
        },
    )

    config = AutoConfig.from_pretrained(args.output_path, trust_remote_code=True)
    model, used_cls_name = _build_random_model_from_config(config)
    print(f"[INFO] 使用模型类: {used_cls_name}")
    model = model.to(args.device).eval()

    # remote-code 兼容：部分旧式 modeling 的 `_tied_weights_keys` 是 list，
    # transformers>=5.5 期望 dict（见 modeling_utils._get_tied_weight_keys 内 .keys()）。
    # 测试模型无 tie 精度诉求，统一清空避免 save_pretrained 崩溃。
    for _, submodule in model.named_modules():
        tied = getattr(submodule, "_tied_weights_keys", None)
        if isinstance(tied, list):
            submodule._tied_weights_keys = {}

    model.save_pretrained(args.output_path)
    stale_index = os.path.join(args.output_path, "model.safetensors.index.json")
    if os.path.exists(stale_index) and os.path.exists(os.path.join(args.output_path, "model.safetensors")):
        os.remove(stale_index)
    print(f"[OK] step1完成: {args.output_path} (cover_layers={plan['cover_layers']}, labels={len(plan['all_labels'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
