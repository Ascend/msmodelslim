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

Post-process AscendV1 quantized weights: convert deq_scale from float32/bf16 to int64
(bit-pattern of float32 stored as int64), so that checkpoints saved with bf16 default
can be used where int64 deq_scale is expected. Supports single-file and sharded layouts.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from typing import Optional, Set

import numpy as np
import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _deqscale2int64(scale: torch.Tensor) -> torch.Tensor:
    """
    Interpret float32 deq_scale as int32 bit pattern and store as int64.
    Same semantics as AscendV1 saver (non-bf16 path). No msmodelslim dependency.
    """
    scale = scale.cpu().numpy()
    scale = np.frombuffer(scale.tobytes(), dtype=np.int32).astype(np.int64)
    return torch.tensor(scale)


ASCENDV1_DESC_JSON_NAME = "quant_model_description.json"
ASCENDV1_SAFETENSORS_NAME = "quant_model_weights.safetensors"
ASCENDV1_SAFETENSORS_INDEX_NAME = "quant_model_weights.safetensors.index.json"
DEQ_SCALE_QUANT_TYPES = ("W8A8", "W8A8_MIX")
SUPPORTED_CONFIG_EXTENSIONS = (".json", ".py")
MAX_CONFIG_FILES = 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert deq_scale in AscendV1 quant weights from float32/bf16 to int64."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Directory containing quant_model_description.json and safetensors weight(s).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If not set, overwrites in-place (with temp + rename).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print which keys would be converted, do not write files.",
    )
    return parser.parse_args()


def get_deq_scale_keys_from_description(model_path: str) -> Optional[Set[str]]:
    """Read quant_model_description.json and return set of deq_scale keys (W8A8/W8A8_MIX)."""
    desc_path = os.path.join(model_path, ASCENDV1_DESC_JSON_NAME)
    if not os.path.isfile(desc_path):
        return None
    with open(desc_path, "r", encoding="utf-8") as f:
        desc = json.load(f)
    if not isinstance(desc, dict):
        return None
    keys = set()
    for key, value in desc.items():
        if isinstance(value, str) and value in DEQ_SCALE_QUANT_TYPES and key.endswith(".deq_scale"):
            keys.add(key)
    return keys if keys else None


def is_deq_scale_key_candidate(key: str) -> bool:
    """Fallback: key looks like a deq_scale (e.g. ends with .deq_scale)."""
    return key.endswith(".deq_scale") or ".deq_scale" in key


def convert_tensor_if_needed(tensor: torch.Tensor, key: str, dry_run: bool, converted: list, skipped: list):
    """Convert float32/bf16 deq_scale to int64; return converted tensor or original."""
    if tensor.dtype == torch.int64:
        skipped.append(key)
        return tensor
    if tensor.dtype == torch.float32:
        out = _deqscale2int64(tensor)
        converted.append(key)
        return out
    if tensor.dtype == torch.bfloat16:
        out = _deqscale2int64(tensor.to(torch.float32))
        converted.append(key)
        return out
    skipped.append(key)
    return tensor


def process_single_file(
    model_path: str,
    output_dir: str,
    deq_scale_keys: Optional[Set[str]],
    dry_run: bool,
    converted: list,
    skipped: list,
):
    """Process single quant_model_weights.safetensors file."""
    src_file = os.path.join(model_path, ASCENDV1_SAFETENSORS_NAME)
    if not os.path.isfile(src_file):
        return False
    tensors = load_file(src_file, device="cpu")
    modified = False
    for key in list(tensors.keys()):
        if deq_scale_keys is not None and key not in deq_scale_keys:
            continue
        if not is_deq_scale_key_candidate(key):
            continue
        t = tensors[key]
        if t.dtype not in (torch.float32, torch.bfloat16, torch.int64):
            continue
        new_t = convert_tensor_if_needed(t, key, dry_run, converted, skipped)
        if new_t is not t:
            tensors[key] = new_t
            modified = True
    if not modified or dry_run:
        if dry_run and converted:
            logger.info("Dry run: would convert keys in single file: %s", src_file)
        return True
    out_file = os.path.join(output_dir, ASCENDV1_SAFETENSORS_NAME)
    if output_dir == model_path:
        fd, tmp_path = tempfile.mkstemp(suffix=".safetensors", dir=output_dir)
        os.close(fd)
        try:
            save_file(tensors, tmp_path)
            os.replace(tmp_path, src_file)
        except Exception:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
            raise
    else:
        save_file(tensors, out_file)
    logger.info("Processed single file: %s -> %s", src_file, out_file)
    return True


def process_sharded(
    model_path: str,
    output_dir: str,
    deq_scale_keys: Optional[Set[str]],
    dry_run: bool,
    converted: list,
    skipped: list,
):
    """Process sharded layout using quant_model_weights.safetensors.index.json."""
    index_path = os.path.join(model_path, ASCENDV1_SAFETENSORS_INDEX_NAME)
    if not os.path.isfile(index_path):
        return False
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    weight_map = index_data.get("weight_map")
    if not weight_map:
        logger.warning("Index file has no weight_map: %s", index_path)
        return True
    # key -> filename (basename)
    file_to_keys = defaultdict(list)
    for key, filename in weight_map.items():
        file_to_keys[filename].append(key)
    for filename, keys_in_file in file_to_keys.items():
        src_file = os.path.join(model_path, filename)
        if not os.path.isfile(src_file):
            logger.warning("Shard file not found: %s", src_file)
            continue
        tensors = load_file(src_file, device="cpu")
        modified = False
        for key in keys_in_file:
            if deq_scale_keys is not None and key not in deq_scale_keys:
                continue
            if not is_deq_scale_key_candidate(key):
                continue
            if key not in tensors:
                continue
            t = tensors[key]
            if t.dtype not in (torch.float32, torch.bfloat16, torch.int64):
                continue
            new_t = convert_tensor_if_needed(t, key, dry_run, converted, skipped)
            if new_t is not t:
                tensors[key] = new_t
                modified = True
        if not modified or dry_run:
            continue
        out_file = os.path.join(output_dir, filename)
        if output_dir == model_path:
            fd, tmp_path = tempfile.mkstemp(suffix=".safetensors", dir=output_dir)
            os.close(fd)
            try:
                save_file(tensors, tmp_path)
                os.replace(tmp_path, src_file)
            except Exception:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)
                raise
        else:
            save_file(tensors, out_file)
        logger.info("Processed shard: %s -> %s", src_file, out_file)
    if dry_run and converted:
        logger.info("Dry run: would convert keys in sharded files under: %s", model_path)
    return True


def copy_all_to_output(src_dir: str, dst_dir: str, weight_map: Optional[dict]):
    """Copy config, description, index, and all weight files to output dir."""
    if src_dir == dst_dir:
        return
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir, 0o750)
    names = os.listdir(src_dir)
    if len(names) > MAX_CONFIG_FILES:
        raise ValueError(f"Too many files in directory ({len(names)}), limit {MAX_CONFIG_FILES}.")
    for name in names:
        src_path = os.path.join(src_dir, name)
        if not os.path.isfile(src_path):
            continue
        _, ext = os.path.splitext(name)
        is_safetensors = name.endswith(".safetensors")
        is_config = ext in SUPPORTED_CONFIG_EXTENSIONS
        if is_safetensors and weight_map is not None and name not in set(weight_map.values()):
            continue
        if not is_config and not is_safetensors:
            continue
        dst_path = os.path.join(dst_dir, name)
        shutil.copy2(src_path, dst_path)
        os.chmod(dst_path, 0o600)
    logger.info("Copied config and weight files to: %s", dst_dir)


def main():
    args = parse_args()
    model_path = os.path.abspath(os.path.expanduser(args.model_path))
    if not os.path.isdir(model_path):
        logger.error("model_path is not a directory: %s", model_path)
        sys.exit(1)
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or model_path))
    if output_dir != model_path and not args.dry_run:
        os.makedirs(output_dir, 0o750)

    # When writing to a different dir, copy full layout first then convert in place under output_dir
    work_dir = model_path
    if output_dir != model_path and not args.dry_run:
        weight_map = None
        index_path = os.path.join(model_path, ASCENDV1_SAFETENSORS_INDEX_NAME)
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map") or {}
        copy_all_to_output(model_path, output_dir, weight_map)
        work_dir = output_dir
    elif output_dir != model_path and args.dry_run:
        work_dir = model_path

    deq_scale_keys = get_deq_scale_keys_from_description(work_dir)
    if deq_scale_keys is not None:
        logger.info("Using deq_scale keys from description: %d keys", len(deq_scale_keys))
    else:
        logger.info("No description or no W8A8/W8A8_MIX deq_scale entries; will infer from key name and dtype.")

    converted = []
    skipped = []

    index_path = os.path.join(work_dir, ASCENDV1_SAFETENSORS_INDEX_NAME)
    single_path = os.path.join(work_dir, ASCENDV1_SAFETENSORS_NAME)
    ret = False
    if os.path.isfile(index_path):
        ret = process_sharded(work_dir, work_dir, deq_scale_keys, args.dry_run, converted, skipped)
    elif os.path.isfile(single_path):
        ret = process_single_file(work_dir, work_dir, deq_scale_keys, args.dry_run, converted, skipped)
    else:
        logger.error(
            "Neither %s nor %s found in %s",
            ASCENDV1_SAFETENSORS_NAME,
            ASCENDV1_SAFETENSORS_INDEX_NAME,
            work_dir,
        )
        sys.exit(1)
    if not ret:
        logger.error("Processing failed (file missing or invalid).")
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run: would convert %d keys, skip %d.", len(converted), len(skipped))
        for k in converted[:20]:
            logger.info("  convert: %s", k)
        if len(converted) > 20:
            logger.info("  ... and %d more", len(converted) - 20)
        return

    logger.info("Converted %d deq_scale keys to int64, skipped %d.", len(converted), len(skipped))


if __name__ == "__main__":
    main()
