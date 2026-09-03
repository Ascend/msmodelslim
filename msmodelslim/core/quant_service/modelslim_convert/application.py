#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
ConvertApplication：离线权重转换编排入口（convert_design.md §7）。

流水线（固定顺序，不经过 quant_service / runner 旁路）：
  1. CheckpointReader.read_catalog — 仅从 index 建 catalog
  2. PreprocessExecutor — preprocess_rules 结构变换
  3. VirtualModelTreeBuilder — module_rules 建虚拟 nn.Module 树
  4. DefaultIRTaskBuilder — convert_rules 枚举 IRTask
  5. IRRouter.resolve — 为每个任务解析 IR 边链
  6. ConvertExecutor — lazy_init + processor/convert 链
  7. SaveProcessorAdapter — processor/save → format 落盘
"""

from __future__ import annotations

import time

from tqdm import tqdm
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.device import (
    resolve_multi_worker_devices,
    resolve_worker_device,
)
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.router import IRRouter
from msmodelslim.core.convert.tasks import RoutedTask
from msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter import SaveProcessorAdapter
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import (
    set_submodule_by_path,
)
from msmodelslim.utils.logging import get_logger, logger_setter

logger = get_logger()


@logger_setter(prefix="msmodelslim.core.quant_service.modelslim_convert")
class ConvertApplication:
    """
    依赖注入式编排器：各阶段实现由 factory 组装，本类只负责阶段顺序与上下文传递。

    Attributes:
        _reader_factory: ``(model_path) -> ICheckpointReader``
        _preprocess / _tree_builder / _task_builder / _executor / _save_adapter: 各阶段执行器
        _router: 路由解析（executor 内 transform 共用同一 IRRouter 实例）
    """

    def __init__(
        self,
        checkpoint_reader_factory,
        preprocess_executor,
        tree_builder,
        task_builder,
        executor,
        save_adapter: SaveProcessorAdapter,
        router: IRRouter | None = None,
    ) -> None:
        self._reader_factory = checkpoint_reader_factory
        self._preprocess = preprocess_executor
        self._tree_builder = tree_builder
        self._task_builder = task_builder
        self._executor = executor
        self._save_adapter = save_adapter
        self._router = router or IRRouter.default()

    def run(self, config: ConvertConfig) -> None:
        """执行一次完整 convert 任务。"""
        pipeline_t0 = time.perf_counter()
        context = ConvertContext(config=config)
        # NPU 路径：CLI 给出卡号且设备可用 → npu_multi（进程数=卡数）。
        # CPU 路径：否则按 YAML worker_backend 走 process / thread。
        devices = config.parallel.device_indices
        resolved_multi = resolve_multi_worker_devices(devices) if devices else []
        if resolved_multi:
            context.parallel_mode = "npu_multi"
            context.resolved_worker_devices = resolved_multi
            # 主进程仍 CPU：streaming 落盘与队列回传均在 CPU 侧完成。
            context.resolved_worker_device = "cpu"
        elif config.parallel.worker_backend == "process":
            context.parallel_mode = "process"
            context.resolved_worker_device = "cpu"
        else:
            context.parallel_mode = "thread"
            context.resolved_worker_device = resolve_worker_device(config.parallel.worker_device)
        reader = self._reader_factory(config.model_path)
        context.reader = reader
        logger.info(
            "Convert parallel_mode=%s, backend=%s, device=%s, devices=%s",
            context.parallel_mode,
            config.parallel.worker_backend,
            context.resolved_worker_device,
            context.resolved_worker_devices or None,
        )

        phase_t0 = time.perf_counter()
        with tqdm(total=1, desc="read checkpoint index") as pbar:
            raw_catalog = reader.read_catalog()
            pbar.update(1)
        logger.info("Convert phase timing: read_index=%.2fs", time.perf_counter() - phase_t0)

        phase_t0 = time.perf_counter()
        catalog_result = self._preprocess.run(context, raw_catalog, config.preprocess_rules)
        context.preprocess_result = catalog_result
        context.catalog = catalog_result.catalog
        logger.info("Convert phase timing: preprocess=%.2fs", time.perf_counter() - phase_t0)

        phase_t0 = time.perf_counter()
        with tqdm(total=1, desc="build virtual module tree") as pbar:
            tree = self._tree_builder.build(context, catalog_result.catalog)
            pbar.update(1)
        context.virtual_tree = tree
        logger.info("Convert phase timing: build_tree=%.2fs", time.perf_counter() - phase_t0)

        phase_t0 = time.perf_counter()
        with tqdm(total=1, desc="build IR tasks") as pbar:
            ir_tasks = self._task_builder.build(context, tree, catalog_result.catalog)
            pbar.update(1)
        logger.info(
            "Convert phase timing: build_tasks=%.2fs (ir_tasks=%d)",
            time.perf_counter() - phase_t0,
            len(ir_tasks),
        )

        phase_t0 = time.perf_counter()
        routed: list[RoutedTask] = []
        for task in tqdm(ir_tasks, desc="route IR tasks", leave=False):
            edges = self._router.resolve(
                task.source_ir.kind,
                task.target_ir,
                task.route_constraints,
            )
            route_names = [task.source_ir.kind] + [e.dst_ir for e in edges]
            routed.append(RoutedTask(task=task, route=edges, route_ir_names=route_names))
        logger.info(
            "Convert phase timing: route_tasks=%.2fs (routed=%d)",
            time.perf_counter() - phase_t0,
            len(routed),
        )

        self._run_convert_with_streaming_save(context, tree, routed)
        logger.info("Convert finished in %.2fs", time.perf_counter() - pipeline_t0)

    def _run_convert_with_streaming_save(self, context: ConvertContext, tree, routed: list[RoutedTask]) -> None:
        phase_t0 = time.perf_counter()
        self._save_adapter.begin(context, tree)
        accepted = 0
        completed = False
        try:
            for result in self._executor.run(context, routed):
                if result.already_saved:
                    set_submodule_by_path(tree, result.module_path, nn.Module())
                else:
                    module = result.resolve_module()
                    self._save_adapter.accept(result.module_path, module)
                    set_submodule_by_path(tree, result.module_path, nn.Module())
                    del module
                accepted += 1
            completed = True
        finally:
            if completed:
                metas = getattr(self._executor, "direct_worker_metas", None)
                if isinstance(metas, list) and metas:
                    self._save_adapter.set_direct_worker_metas(metas)
                self._save_adapter.finalize()
            else:
                self._save_adapter.abort()
        logger.info(
            "Convert phase timing: convert_ir_stream_save=%.2fs (converted_modules=%d)",
            time.perf_counter() - phase_t0,
            accepted,
        )
