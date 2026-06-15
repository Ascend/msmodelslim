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

from pathlib import Path

from torch import nn

from msmodelslim.core.convert.protocol import ConvertContext, ISaveProcessorAdapter
from msmodelslim.format.registry import parse_format_config
from msmodelslim.model.interface import IModel
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.processor.save.processor import QuantSaveProcessor, QuantSaveProcessorConfig
from msmodelslim.utils.logging import get_logger

logger = get_logger()


def _lazy_init_unsaved_modules(context: ConvertContext, tree: nn.Module) -> None:
    """保存前加载未参与 IR 转换的模块（PassthroughModule、未 quant 的 ModelFreeLinear）。"""
    from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeModule
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


class SaveProcessorAdapter(ISaveProcessorAdapter):
    def save(self, context: ConvertContext, tree: nn.Module) -> None:
        dst = context.config.dst_format.lower()
        save_dir = str(context.save_path)
        model_type = context.config.model_family or "convert"
        adapter = BaseModelAdapter(
            model_type=model_type,
            model_path=Path(context.model_path),
        )

        if dst in ("huggingface", "hf", "compressed_tensors"):
            self._save_compressed_tensors(context, tree, save_dir, adapter)
        elif dst in ("ascendv1", "ascendv1_saver"):
            self._save_ascendv1(context, tree, save_dir, adapter)
        else:
            raise ValueError(f"Unsupported dst_format for convert save: {dst}")

    @staticmethod
    def _save_compressed_tensors(
        context: ConvertContext,
        tree: nn.Module,
        save_dir: str,
        adapter: IModel,
    ) -> None:
        format_cfg = parse_format_config({"type": "compressed_tensors", "part_file_size": 4})
        cfg = QuantSaveProcessorConfig(type="saver", format=format_cfg)
        cfg.set_save_directory(save_dir)
        _lazy_init_unsaved_modules(context, tree)
        saver = QuantSaveProcessor(tree, cfg, adapter)
        saver.pre_run()
        from msmodelslim.core.base.protocol import BatchProcessRequest

        for name, module in tree.named_modules():
            if name:
                saver.postprocess(BatchProcessRequest(name=name, module=module, datas=None, outputs=None))
        saver.post_run()
        logger.info("Saved HF/compressed_tensors checkpoint to %s", save_dir)

    @staticmethod
    def _save_ascendv1(
        context: ConvertContext,
        tree: nn.Module,
        save_dir: str,
        adapter: IModel,
    ) -> None:
        from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config, AscendV1Saver

        _lazy_init_unsaved_modules(context, tree)
        cfg = AscendV1Config(save_directory=save_dir, part_file_size=4)
        saver = AscendV1Saver(model=tree, config=cfg, adapter=adapter)
        saver.pre_run()
        saver.post_run()
        logger.info("Saved AscendV1 checkpoint to %s", save_dir)
