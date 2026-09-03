#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
SaveProcessorAdapter（convert_design.md §11）。

将转换后的虚拟树交给既有保存栈，不重复实现写盘逻辑：

  - **``dst_format=ascendv1``**（MXFP8 产品路径）：``AscendV1Saver`` — W8A8_MXFP8 权重仅在昇腾 NPU 运行，须用此格式。
  - **``dst_format=huggingface|compressed_tensors``**：``QuantSaveProcessor`` + compressed_tensors —
    用于 **FLOAT / bf16** 等 HF 侧导出（如 fp8_block → bf16），**不**作为 MXFP8 生产落盘格式。

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.format.registry import parse_format_config
from msmodelslim.model.interface import IModel
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.processor.save.processor import (
    QuantSaveProcessor,
    QuantSaveProcessorConfig,
)
from msmodelslim.utils.logging import get_logger

logger = get_logger()

_HF_DST = ("huggingface", "hf", "compressed_tensors")
_ASCEND_DST = ("ascendv1", "ascendv1_saver")


def _lazy_init_unsaved_modules(context: ConvertContext, tree: nn.Module) -> None:
    """保存前加载未参与 IR 转换的模块（PassthroughModule、未 quant 的 ModelFreeLinear）。"""
    from msmodelslim.core.quant_service.modelslim_convert.virtual_module import (
        ModelFreeModule,
    )
    from msmodelslim.infra.io.shard_handle_cache import ShardHandleCache

    reader = context.reader
    if reader is None:
        return
    n = 0
    had_shard_attr = hasattr(reader, "shard_handle_cache")
    old_shard_cache = getattr(reader, "shard_handle_cache", None)
    shard_cache = ShardHandleCache(max_shards=context.config.parallel.shard_cache_size)
    reader.shard_handle_cache = shard_cache
    try:
        for mod in tree.modules():
            if isinstance(mod, ModelFreeModule) and not mod.lazy_initialized:
                mod.lazy_init(reader, device="cpu")
                n += 1
    finally:
        shard_cache.clear()
        if had_shard_attr:
            reader.shard_handle_cache = old_shard_cache
        else:
            try:
                delattr(reader, "shard_handle_cache")
            except AttributeError:
                pass
    if n:
        logger.info("Lazy-loaded %d module(s) before save (passthrough / FLOAT linear)", n)


def _build_adapter(context: ConvertContext) -> IModel:
    model_type = context.config.model_family or "convert"
    return BaseModelAdapter(
        model_type=model_type,
        model_path=Path(context.model_path),
    )


@dataclass
class _SaverBundle:
    """一次保存会话用的 saver 与元数据（format 分派的唯一出口）。"""

    saver: Any
    label: str
    part_file_size: int
    iterate_named_modules: bool


def is_npu_direct_write(parallel_mode: str, dst_format: str) -> bool:
    """npu_multi + AscendV1：worker 直接写盘，主进程只写 passthrough 到 staging。"""
    return parallel_mode == "npu_multi" and dst_format.lower() in _ASCEND_DST


def _create_saver(context: ConvertContext, tree: nn.Module, save_dir: str | None = None) -> _SaverBundle:
    """按 ``dst_format`` 构造既有 saver；流式 ``begin`` 与整树 ``save`` 共用。"""
    dst = context.config.dst_format.lower()
    save_dir = str(context.save_path) if save_dir is None else save_dir
    adapter = _build_adapter(context)
    part_file_size = context.config.part_file_size

    if dst in _HF_DST:
        format_cfg = parse_format_config({"type": "compressed_tensors", "part_file_size": part_file_size})
        cfg = QuantSaveProcessorConfig(type="saver", format=format_cfg)
        cfg.set_save_directory(save_dir)
        return _SaverBundle(
            saver=QuantSaveProcessor(tree, cfg, adapter),
            label="HF/compressed_tensors",
            part_file_size=part_file_size,
            iterate_named_modules=True,
        )
    if dst in _ASCEND_DST:
        from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import (
            AscendV1Config,
            AscendV1Saver,
        )

        cfg = AscendV1Config(save_directory=save_dir, part_file_size=part_file_size)
        return _SaverBundle(
            saver=AscendV1Saver(model=tree, config=cfg, adapter=adapter),
            label="AscendV1",
            part_file_size=part_file_size,
            iterate_named_modules=False,
        )
    raise ValueError(f"Unsupported dst_format for convert save: {dst}")


@dataclass
class _SaveSession:
    context: ConvertContext
    tree: nn.Module
    bundle: _SaverBundle
    accepted: int = 0
    closed: bool = False
    direct_write: bool = False


class SaveProcessorAdapter:
    """convert 保存阶段：流式生命周期与整树 ``save`` 共用 ``_create_saver``。

    复用既有 ``AscendV1Saver`` / ``QuantSaveProcessor`` 的
    ``pre_run`` / ``postprocess`` / ``post_run``，不另立保存协议。
    """

    def __init__(self) -> None:
        self._session: _SaveSession | None = None
        self.main_meta: tuple[dict[str, str], dict[str, Any]] | None = None
        self._direct_worker_metas: list = []

    def set_direct_worker_metas(self, metas: list) -> None:
        """注入 npu_multi worker 回传的 staging 元数据，供 finalize 时 merge。"""
        self._direct_worker_metas = list(metas)

    def begin(self, context: ConvertContext, tree: nn.Module) -> None:
        if self._session is not None and not self._session.closed:
            raise RuntimeError("SaveProcessorAdapter.begin() called while a session is already open")
        direct_write = is_npu_direct_write(context.parallel_mode, context.config.dst_format)
        save_dir = None
        if direct_write:
            from msmodelslim.core.quant_service.modelslim_convert.impl.direct_save import (
                main_staging_dir,
            )

            save_dir = str(Path(context.save_path) / main_staging_dir(str(context.save_path)))
        bundle = _create_saver(context, tree, save_dir=save_dir)
        bundle.saver.pre_run()
        self.main_meta = None
        self._direct_worker_metas = []
        self._session = _SaveSession(
            context=context,
            tree=tree,
            bundle=bundle,
            direct_write=direct_write,
        )

    def accept(self, module_path: str, module: nn.Module) -> None:
        session = self._require_open_session()
        session.bundle.saver.postprocess(
            BatchProcessRequest(name=module_path, module=module, datas=None, outputs=None),
        )
        # convert 随后把树节点换成空占位符；必须丢掉 memo，否则已写权重只增不减。
        from msmodelslim.core.quant_service.modelslim_convert.impl.direct_save import (
            release_processed_modules,
        )

        release_processed_modules(session.bundle.saver)
        session.accepted += 1

    def finalize(self) -> None:
        session = self._require_open_session()
        try:
            logger.info("Streaming save finalize: lazy init passthrough / unconverted modules")
            _lazy_init_unsaved_modules(session.context, session.tree)
            logger.info("Streaming save finalize: write metadata and close saver")
            session.bundle.saver.post_run()
            if session.direct_write:
                from msmodelslim.core.quant_service.modelslim_convert.impl.direct_save import (
                    collect_saver_meta,
                    merge_staged_output,
                )

                self.main_meta = collect_saver_meta(session.bundle.saver)
                if self._direct_worker_metas:
                    merge_staged_output(
                        str(session.context.save_path),
                        str(session.context.model_path),
                        self._direct_worker_metas,
                        self.main_meta,
                    )
            logger.info(
                "Streaming saved %s checkpoint (modules=%d, part_file_size=%s)",
                session.bundle.label,
                session.accepted,
                session.bundle.part_file_size,
            )
        finally:
            session.closed = True

    def abort(self) -> None:
        session = self._session
        if session is None or session.closed:
            return
        session.closed = True
        if session.direct_write:
            from msmodelslim.core.quant_service.modelslim_convert.impl.direct_save import (
                cleanup_staging,
            )

            cleanup_staging(str(session.context.save_path))
        logger.warning(
            "Streaming save aborted before finalize; partial files under %s may be incomplete.",
            session.context.save_path,
        )

    def save(self, context: ConvertContext, tree: nn.Module) -> None:
        bundle = _create_saver(context, tree)
        _lazy_init_unsaved_modules(context, tree)
        bundle.saver.pre_run()
        if bundle.iterate_named_modules:
            for name, module in tree.named_modules():
                if name:
                    bundle.saver.postprocess(
                        BatchProcessRequest(name=name, module=module, datas=None, outputs=None),
                    )
        bundle.saver.post_run()
        logger.info(
            "Saved %s checkpoint to %s (part_file_size=%s)",
            bundle.label,
            context.save_path,
            bundle.part_file_size,
        )

    def _require_open_session(self) -> _SaveSession:
        session = self._session
        if session is None:
            raise RuntimeError("SaveProcessorAdapter session is not open; call begin() first")
        if session.closed:
            raise RuntimeError("SaveProcessorAdapter session is already closed")
        return session
