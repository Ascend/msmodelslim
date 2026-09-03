#!/usr/bin/env python3
"""步骤3：验证权重一致性（精简版，支持 tie-weight 等价克隆豁免）。"""

import argparse
import glob
import os
import sys

import torch


def _load_weights(model_path):
    try:
        from safetensors.torch import load_file

        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if files:
            merged = {}
            for file in files:
                merged.update(load_file(file))
            return merged
    except Exception:  # nosec B110 - safetensors 缺失时回退到 pytorch_model.bin
        pass

    pt_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(pt_path):
        return torch.load(pt_path, map_location="cpu", weights_only=True)
    return {}


def _find_equivalent(key, source_map, target_map, tolerance):
    """在 target_map 中寻找与 source_map[key] 同 shape 且数值全等的克隆键。

    tie-word-embeddings 模型（如 Qwen3）：msmodelslim saver 有意把 embed_tokens.weight
    克隆为 lm_head.weight 以兼容推理，而 transformers 原模型只保存一份 —— 严格键集
    比对会对这类模型误报 FAIL。数值全等（max diff <= tolerance）即视为等价克隆，属可接受差异。
    """
    target_tensor = source_map[key]
    for cand_key in target_map:
        cand = target_map[cand_key]
        if cand.shape != target_tensor.shape:
            continue
        if torch.abs(cand.float() - target_tensor.float()).max().item() <= tolerance:
            return cand_key
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-path", required=True)
    parser.add_argument("--quantized-path", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    left = _load_weights(args.original_path)
    right = _load_weights(args.quantized_path)
    if not left or not right:
        print("[ERROR] step3失败: 权重加载失败")
        return 1

    left_keys, right_keys = set(left.keys()), set(right.keys())

    # 1) 公共键直接纳入数值比较；单侧键先尝试等价克隆豁免，无克隆对应才算真不一致
    pairs = [(k, k) for k in sorted(left_keys & right_keys)]
    accepted_clones = []  # (单侧键, 对应克隆键, 方向说明)
    missing = []  # 仅左侧且无克隆对应（真缺失）
    unexpected = []  # 仅右侧且无克隆对应（真多余）

    for key in sorted(left_keys - right_keys):
        eq = _find_equivalent(key, left, right, args.tolerance)
        if eq is not None:
            accepted_clones.append((key, eq, "quantized 侧含数值全等克隆"))
            pairs.append((key, eq))
        else:
            missing.append(key)
    for key in sorted(right_keys - left_keys):
        eq = _find_equivalent(key, right, left, args.tolerance)
        if eq is not None:
            accepted_clones.append((key, eq, "original 侧含数值全等克隆"))
            pairs.append((eq, key))
        else:
            unexpected.append(key)

    if missing or unexpected:
        print("[ERROR] step3失败: 权重键不一致（已排除等价克隆）")
        print(f"[INFO] 仅左侧数量(真缺失): {len(missing)}")
        for k in missing[:10]:
            print(f"  - {k}")
        print(f"[INFO] 仅右侧数量(真多余): {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  - {k}")
        return 1

    # 2) 数值一致性：公共键 + 被豁免的克隆配对
    max_diff = 0.0
    for left_key, right_key in pairs:
        left_t = left[left_key]
        right_t = right[right_key]
        if left_t.shape != right_t.shape:
            print(f"[ERROR] step3失败: 形状不一致 {left_key} vs {right_key}")
            return 1
        diff = torch.abs(left_t.float() - right_t.float()).max().item()
        max_diff = max(max_diff, diff)
        if diff > args.tolerance:
            print(f"[ERROR] step3失败: 权重差异超阈值 {left_key} diff={diff:.2e}")
            return 1

    print(f"[OK] step3完成: max_diff={max_diff:.2e}, 等价克隆豁免 {len(accepted_clones)} 项")
    for key, eq, why in accepted_clones:
        print(f"[INFO] 豁免: {key} == {eq} ({why})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
