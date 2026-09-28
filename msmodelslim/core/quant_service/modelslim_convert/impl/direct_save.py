#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Worker 直接写盘（npu_multi + AscendV1 / HuggingFace）。

背景：convert 把结果经进程队列回传主进程串行落盘，大量小任务的串行开销成为瓶颈。
本模块让每个 worker 用 ``SaveProcessorAdapter`` 同一套 factory 建 saver，
写入 staging 分片后只回传元数据，主进程收尾 merge。

写盘仍走既有 saver：``AscendV1Saver.postprocess`` / HuggingFace ``QuantSaveProcessor``。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch import nn

from msmodelslim.core.quant_service.modelslim_v1.save.utils.safetensors import get_index_json
from msmodelslim.format.compressed_tensors_format.compressed_tensors import CompressedTensorsQuantFormat
from msmodelslim.processor.save.processor import QuantSaveProcessor
from msmodelslim.utils.exception import UnexpectedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import json_safe_dump

if TYPE_CHECKING:
    from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Saver

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
    """丢掉 saver / format 对已写模块的强引用，避免 host 内存随任务数线性上涨。"""
    processed = getattr(saver, "processed_modules", None)
    if processed:
        processed.clear()
    if isinstance(saver, QuantSaveProcessor) and isinstance(saver._format, CompressedTensorsQuantFormat):
        saver._format.release_module_refs()


def write_result(saver, module_path: str, module: nn.Module) -> None:
    """把 worker 本地转换结果通过 saver 直接落盘，写完立即释放模块引用。"""
    from msmodelslim.core.base.protocol import BatchProcessRequest

    try:
        module.to("cpu")
        saver.postprocess(BatchProcessRequest(name=module_path, module=module, datas=None, outputs=None))
    finally:
        release_processed_modules(saver)


def _safetensors_writer(saver: AscendV1Saver | QuantSaveProcessor):
    """AscendV1Saver 把 writer 挂在自身；HF QuantSaveProcessor 挂在 compressed_tensors format 上。

    HF format 在 ``finalize_export`` close 后会置空 live writer，merge 须改读已关闭的 writer。
    """
    if not isinstance(saver, QuantSaveProcessor):
        return saver.safetensors_writer
    fmt = saver._format
    if isinstance(fmt, CompressedTensorsQuantFormat):
        return fmt.written_safetensors_writer()
    return None


def collect_saver_meta(saver) -> tuple[dict[str, str], dict[str, Any]]:
    """读取 saver 已写入的 (键→分片, 键→描述)。须在 writer close / post_run 之后调用。"""
    writer = _safetensors_writer(saver)
    if writer is None:
        raise UnexpectedError(
            "direct-write saver has no safetensors writer after close; refusing to merge an empty weight_map",
            action="Keep QuantSaveProcessor._closed_writer after finalize_export, "
            "or leave AscendV1Saver.safetensors_writer set after post_run.",
        )
    weight_map = dict(writer.saved_keys_map)
    # HF 没有描述文件，只有 AscendV1Saver 带 json_writer。
    desc_map = {} if isinstance(saver, QuantSaveProcessor) else dict(saver.json_writer.value_map)
    return weight_map, desc_map


def finalize_saver(saver) -> tuple[dict[str, str], dict[str, Any]]:
    """关闭 worker 侧 writer（不走 post_run，避免每人写一份最终 index/拷配置）。"""
    # saved_keys_map 在 close()（写分片 + 生成索引）时才填充，须先 close 再读取
    writer = _safetensors_writer(saver)
    if writer is None:
        raise UnexpectedError(
            "direct-write saver has no safetensors writer to close",
            action="Open the staging saver with open_staging_saver before write_result.",
        )
    writer.close()
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


def _hf_index_total_size(save_dir: Path, subdirs: list[str], file_size_sum: int) -> int:
    """Prefer staging index ``metadata.total_size`` (tensor bytes), matching non-direct HF."""
    total = 0
    for subdir in subdirs:
        idx = save_dir / subdir / "model.safetensors.index.json"
        if not idx.is_file():
            return file_size_sum
        data = json.loads(idx.read_text(encoding="utf-8"))
        size = (data.get("metadata") or {}).get("total_size")
        if size is None:
            return file_size_sum
        total += int(size)
    return total


def merge_hf_staged_output(
    save_path: str,
    model_path: str,
    worker_metas: list[tuple[str, dict[str, str], dict[str, Any]]],
    main_meta: tuple[dict[str, str], dict[str, Any]] | None,
) -> None:
    """合并 HF/compressed_tensors 的 staging 分片，主进程只写 ``model.safetensors.index.json``。"""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    shard_plan: dict[str, tuple[Path, list[str]]] = {}

    def _weight_map_or_index(subdir: str, wm: dict[str, str]) -> dict[str, str]:
        if wm:
            return wm
        # 仅当内存 map 为空、但该 staging 已写出 index 时才读盘。
        # 生产路径靠 _closed_writer / live writer；本分支是最后兜底，避免 rmtree 静默丢分片。
        idx = save_dir / subdir / "model.safetensors.index.json"
        if not idx.is_file():
            return wm
        data = json.loads(idx.read_text(encoding="utf-8"))
        return dict(data.get("weight_map") or {})

    def _collect(subdir: str, wm: dict[str, str]) -> None:
        staging = save_dir / subdir
        for key, fname in wm.items():
            tmp = f"{subdir}|{fname}"
            weight_map[key] = tmp
            entry = shard_plan.setdefault(tmp, (staging / os.path.basename(fname), []))
            entry[1].append(key)

    for subdir, wm, _desc in worker_metas:
        _collect(subdir, _weight_map_or_index(subdir, wm))
    if main_meta is not None:
        main_sub = main_staging_dir(save_path)
        _collect(main_sub, _weight_map_or_index(main_sub, main_meta[0]))

    total = len(shard_plan)
    final_name: dict[str, str] = {}
    file_size_sum = 0
    for i, (tmp, (src, _keys)) in enumerate(shard_plan.items(), 1):
        if not src.exists():
            raise FileNotFoundError(f"direct-write staged shard missing: {src}")
        file_size_sum += src.stat().st_size
        dst = f"model-{i:05d}-of-{total:05d}.safetensors"
        shutil.move(str(src), str(save_dir / dst))
        final_name[tmp] = dst

    weight_map = {key: final_name[tmp] for key, tmp in weight_map.items()}
    staging_subdirs = [subdir for subdir, _, _ in worker_metas]
    if main_meta is not None:
        staging_subdirs.append(main_staging_dir(save_path))
    # 与非直写 HF 一致：index.metadata.total_size 是张量字节和，不是分片文件体积。
    total_size = _hf_index_total_size(save_dir, staging_subdirs, file_size_sum)
    json_safe_dump(get_index_json(weight_map, total_size), str(save_dir / "model.safetensors.index.json"), indent=2)

    main_dir = save_dir / main_staging_dir(save_path)
    if main_dir.is_dir():
        for item in main_dir.iterdir():
            if item.suffix == ".safetensors" or item.name.endswith(".index.json"):
                continue
            target = save_dir / item.name
            if item.is_file() and not target.exists():
                shutil.copy2(item, target)

    for subdir, _, _ in worker_metas:
        shutil.rmtree(save_dir / subdir, ignore_errors=True)
    if main_meta is not None:
        shutil.rmtree(main_dir, ignore_errors=True)
    logger.info(
        "Merged HF direct-write output: %d shards, %d tensors (model=%s)",
        total,
        len(weight_map),
        model_path,
    )


def cleanup_staging(save_path: str) -> None:
    """转换中断时清理残留 staging 目录。"""
    save_dir = Path(save_path)
    if not save_dir.exists():
        return
    for sub in save_dir.iterdir():
        if sub.is_dir() and sub.name.startswith(_STAGING_PREFIX):
            shutil.rmtree(sub, ignore_errors=True)
