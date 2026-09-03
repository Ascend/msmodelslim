#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Worker 直接写盘（npu_multi + AscendV1）。

背景：convert 把结果经进程队列回传主进程串行落盘，大量小任务的串行开销成为瓶颈。
本模块让每个 worker 用 ``SaveProcessorAdapter`` 同一套 factory 建 saver，
写入 staging 分片后只回传元数据，主进程收尾 merge。

写盘本身仍走 ``AscendV1Saver.postprocess`` / ``on_w8a8_*``，不另写格式。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from torch import nn

from msmodelslim.utils.logging import get_logger

logger = get_logger()

_STAGING_PREFIX = ".direct"
# 描述文件头字段，推断 model_quant_type 时排除，避免把 "version" 等当成量化类型。
_DESC_HEADER_KEYS = frozenset({"version", "model_quant_type", "group_size", "metadata", "optional"})
# 与 AscendV1Saver.on_w8a8_mx_dynamic_per_block 等 MX 路径写入的 group_size 一致。
_MX_GROUP_SIZE = 32
_MX_QUANT_TYPES = frozenset(
    {
        "W8A8_MXFP8",
        "W4A8_MXFP",
        "W4A4_MXFP4",
        "W4A4_MXFP4_DUALSCALE",
        "W4A4_MXFP4_SVD",
    }
)


def _infer_desc_header(desc_map: dict[str, Any]) -> tuple[str, int]:
    """从逐 tensor 描述推断文件级 ``model_quant_type`` / ``group_size``。"""
    from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Saver

    priority = AscendV1Saver.QUANT_TYPE_PRIORITY
    found = [
        value
        for key, value in desc_map.items()
        if key not in _DESC_HEADER_KEYS and isinstance(value, str) and value in priority and value != "FLOAT"
    ]
    if not found:
        return "Unknown", 0
    quant_type = max(found, key=priority.index)
    return quant_type, _MX_GROUP_SIZE if quant_type in _MX_QUANT_TYPES else 0


def worker_staging_dir(save_path: str, rank: int) -> str:
    """worker rank 的 staging 子目录名。"""
    return f"{_STAGING_PREFIX}_{rank}"


def main_staging_dir(save_path: str) -> str:
    """主进程 passthrough 的 staging 子目录名。"""
    return f"{_STAGING_PREFIX}_main"


def open_staging_saver(context, staging_subdir: str):
    """在 ``context.save_path/<staging_subdir>`` 下用 convert 同一套 factory 打开 saver。"""
    from msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter import _create_saver

    staging = Path(context.save_path) / staging_subdir
    staging.mkdir(parents=True, exist_ok=True)
    bundle = _create_saver(context, nn.Module(), save_dir=str(staging))
    bundle.saver.pre_run()
    return bundle.saver


def release_processed_modules(saver) -> None:
    """丢掉 saver 对已写模块的强引用，避免 host/NPU 内存随任务数线性上涨。"""
    processed = getattr(saver, "processed_modules", None)
    if processed:
        processed.clear()


def write_result(saver, module_path: str, module: nn.Module) -> None:
    """把 worker 本地转换结果通过 saver 直接落盘，写完立即释放模块引用。"""
    from msmodelslim.core.base.protocol import BatchProcessRequest

    try:
        module.to("cpu")
        saver.postprocess(BatchProcessRequest(name=module_path, module=module, datas=None, outputs=None))
    finally:
        release_processed_modules(saver)


def collect_saver_meta(saver) -> tuple[dict[str, str], dict[str, Any]]:
    """读取 saver 已写入的 (键→分片, 键→描述)。须在 writer close / post_run 之后调用。"""
    return (
        dict(saver.safetensors_writer.saved_keys_map),
        dict(saver.json_writer.value_map),
    )


def finalize_saver(saver) -> tuple[dict[str, str], dict[str, Any]]:
    """关闭 worker 侧 writer（不走 post_run，避免每人写一份最终 index/拷配置）。"""
    # saved_keys_map 在 close()（写分片 + 生成索引）时才填充，须先 close 再读取
    saver.safetensors_writer.close()
    return collect_saver_meta(saver)


def merge_staged_output(
    save_path: str,
    model_path: str,
    worker_metas: list[tuple[str, dict[str, str], dict[str, Any]]],
    main_meta: tuple[dict[str, str], dict[str, Any]] | None,
) -> None:
    """
    合并主进程 staging 与各 worker staging 的分片，生成最终 AscendV1 输出。

    ``worker_metas`` 元素为 ``(staging_subdir, weight_map, desc_map)``；
    将所有 staging 分片重命名为全局序号 ``quant_model_weights-{i}-of-{N}.safetensors``，
    写入 ``quant_model_weights.safetensors.index.json`` 与 ``quant_model_description.json``，
    拷贝模型配置文件并清理 staging。
    """
    from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import (
        ASCENDV1_DESC_JSON_NAME,
        ASCENDV1_SAFETENSORS_NAME,
        copy_files,
        remove_quantization_config,
    )
    from msmodelslim.core.quant_service.modelslim_v1.save.utils.safetensors import get_index_json
    from msmodelslim.utils.security import json_safe_dump

    shard_prefix = ASCENDV1_SAFETENSORS_NAME.removesuffix(".safetensors")
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    desc_map: dict[str, Any] = {}
    # 待重命名：临时键 → (源文件路径, 该分片内包含的 tensor 键)
    shard_plan: dict[str, tuple[Path, list[str]]] = {}

    def _collect(subdir: str, wm: dict[str, str], dm: dict[str, Any]) -> None:
        staging = save_dir / subdir
        for key, fname in wm.items():
            tmp = f"{subdir}|{fname}"
            weight_map[key] = tmp
            desc_map[key] = dm.get(key, "FLOAT")
            entry = shard_plan.setdefault(tmp, (staging / os.path.basename(fname), []))
            entry[1].append(key)

    for subdir, wm, dm in worker_metas:
        _collect(subdir, wm, dm)
    if main_meta is not None:
        _collect(main_staging_dir(save_path), main_meta[0], main_meta[1])

    total = len(shard_plan)
    final_name: dict[str, str] = {}
    total_size = 0
    for i, (tmp, (src, keys)) in enumerate(shard_plan.items(), 1):
        if not src.exists():
            raise FileNotFoundError(f"direct-write staged shard missing: {src}")
        total_size += src.stat().st_size
        dst = f"{shard_prefix}-{i:05d}-of-{total:05d}.safetensors"
        shutil.move(str(src), str(save_dir / dst))
        final_name[tmp] = dst

    weight_map = {key: final_name[tmp] for key, tmp in weight_map.items()}
    index_name = f"{shard_prefix}.safetensors.index.json"
    json_safe_dump(get_index_json(weight_map, total_size), str(save_dir / index_name), indent=2)

    # worker 跳过 post_run，主进程只写 passthrough，saver.model_quant_type 仍是 Unknown。
    # 文件级类型从已合并的逐 tensor 描述推断，与 AscendV1Saver.update_quant_type 同一套优先级。
    quant_type, group_size = _infer_desc_header(desc_map)
    desc_map.setdefault("version", "1.0.0")
    desc_map.setdefault("model_quant_type", quant_type)
    desc_map.setdefault("group_size", group_size)
    desc_map.setdefault("metadata", {})
    desc_map.setdefault("optional", {})
    json_safe_dump(desc_map, str(save_dir / ASCENDV1_DESC_JSON_NAME), indent=4)

    try:
        copy_files(model_path, str(save_dir))
        remove_quantization_config(str(save_dir))
    except Exception as exc:  # noqa: BLE001 - 配置拷贝失败不阻断权重输出
        logger.warning("copy model config files failed: %s", exc)

    for subdir, _, _ in worker_metas:
        shutil.rmtree(save_dir / subdir, ignore_errors=True)
    if main_meta is not None:
        shutil.rmtree(save_dir / main_staging_dir(save_path), ignore_errors=True)
    logger.info("Merged direct-write output: %d shards, %d tensors", total, len(weight_map))


def cleanup_staging(save_path: str) -> None:
    """转换中断时清理残留 staging 目录。"""
    save_dir = Path(save_path)
    if not save_dir.exists():
        return
    for sub in save_dir.iterdir():
        if sub.is_dir() and sub.name.startswith(_STAGING_PREFIX):
            shutil.rmtree(sub, ignore_errors=True)
