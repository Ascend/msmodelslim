#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
NPU 多卡并行转换：每张 NPU 卡一个子进程，组间动态负载均衡。

架构（对齐 convert_design.md 的 executor 扩展）：
  - 主进程 ``MultiNpuConvertScheduler`` 用 ``torch.multiprocessing.spawn`` 派生
    ``world_size`` 个子进程，每进程 ``torch.npu.set_device(f"npu:card")`` 绑卡；
  - 动态任务队列（spawn ``ctx.Queue``，无界）：主进程把所有 dependency group
    依序入队 + ``world_size`` 个 None 哨兵，空转进程自动多取，缓解 MoE 大组拖尾；
  - 子进程组内串行（``max_workers=1``，等价现有 ``npu_max_workers=1`` 语义，防同卡 OOM）；
  - 结果默认经 ``result_queue`` 逐条流式回传；npu_multi 且 AscendV1 时 worker 直接写盘，
    队列只回传轻量 ``saved`` 进度与 ``worker_meta``，主进程收尾 merge staging；
  - 出错经 ``error_queue`` 回传并中止。

注意：
  - ``_npu_worker_fn`` 必须是模块顶层函数（spawn 需可 pickle）；
  - 不引入 torch.distributed/HCCL，convert 的 IR 任务彼此独立、无需跨进程通信。
"""

from __future__ import annotations

import queue
import time
import traceback
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.multiprocessing import spawn

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.tasks import IRResult, RoutedTask
from msmodelslim.core.convert.types import IRKind
from msmodelslim.core.quant_service.modelslim_convert.impl.worker import (
    GroupWorkSummary,
    bootstrap_convert_worker,
)
from msmodelslim.utils.logging import get_logger

logger = get_logger()

_QUEUE_GET_TIMEOUT_S = 0.2
# mp.spawn join 超时：worker 正常消费 None 哨兵后应秒级退出，给足缓冲避免误判。
_SPAWN_JOIN_TIMEOUT_S = 60.0
# 结果队列有界长度：单结果约数 MB~数十 MB，限制主进程无界积压导致的 OOM；
# 队列满时 worker 的 put 阻塞形成背压。task/error 队列保持无界，避免多队列互等死锁。
# 与 executor 中 CPU ProcessPool 的无界 Manager.Queue 不是同一个常量。
_NPU_RESULT_QUEUE_MAXSIZE = 256
# 无进展兜底超时（秒）：worker 被外部 kill（未写 error_queue）且其余 worker 也卡死时，
# 主进程长期收不到任何结果会永久挂起；此处兜底中断。正常转换结果会持续到达，不会触发。
_RUN_TIMEOUT_S = 3600.0


@dataclass(frozen=True)
class _NpuWorkPayload:
    """跨进程传递的静态配置（须可 pickle，不含 Queue）。"""

    model_path: str
    config: ConvertConfig
    device_indices: list[int]
    save_path: str | None = None
    budget: int | None = None
    return_mode: str = "state_dict"


def _npu_worker_fn(
    rank: int,
    world_size: int,
    task_queue,
    result_queue,
    error_queue,
    payload: _NpuWorkPayload,
) -> None:
    """
    ``mp.spawn`` worker 目标：单卡子进程，循环消费 task_queue 直到 None 哨兵。

    必须在模块顶层定义以便 spawn 上下文 pickle。
    """
    card = payload.device_indices[rank]
    torch.npu.set_device(f"npu:{card}")
    # 单算子限制单线程：8 进程并发时避免 OpenMP 线程超订（并行度由卡间提供）。
    torch.set_num_threads(1)

    saver = None

    def _put_result(result: IRResult) -> None:
        if payload.save_path:
            from msmodelslim.core.quant_service.modelslim_convert.impl.direct_save import (
                write_result,
            )

            write_result(saver, result.module_path, result.module)
            result.module = None
            result.state_dict = None
            result_queue.put(("saved", result.module_path))
        else:
            result_queue.put(("result", result))

    try:
        context, runner = bootstrap_convert_worker(payload.model_path, payload.config, f"npu:{card}")
        if payload.save_path:
            from msmodelslim.core.quant_service.modelslim_convert.impl.direct_save import (
                open_staging_saver,
                worker_staging_dir,
            )

            saver = open_staging_saver(context, worker_staging_dir(payload.save_path, rank))

        while True:
            group = task_queue.get()
            if group is None:
                break
            runner.run_group(
                context=context,
                group=group,
                max_workers=1,
                budget=payload.budget,
                catalog=None,
                return_mode=payload.return_mode,
                result_sink=_put_result,
            )
            result_queue.put(("summary", GroupWorkSummary.from_runner(runner)))
        if payload.save_path and saver is not None:
            from msmodelslim.core.quant_service.modelslim_convert.impl.direct_save import (
                finalize_saver,
                worker_staging_dir,
            )

            weight_map, desc_map = finalize_saver(saver)
            result_queue.put(
                (
                    "worker_meta",
                    (worker_staging_dir(payload.save_path, rank), weight_map, desc_map),
                ),
            )
    except BaseException as exc:  # noqa: BLE001 - 子进程异常必须回传主进程
        error_queue.put((rank, traceback.format_exc()))
        logger.error("NPU worker rank=%d card=npu:%d failed: %s", rank, card, exc)
        raise


class MultiNpuConvertScheduler:
    """
    npu_multi 调度器：spawn 每卡一个子进程，动态队列分发组，主进程流式聚合结果。
    """

    def __init__(
        self,
        model_path: str,
        config: ConvertConfig,
        device_indices: list[int],
        groups: list[list[RoutedTask]],
        save_path: str | None = None,
        budget: int | None = None,
        return_mode: str = "state_dict",
    ) -> None:
        self._model_path = model_path
        self._config = config
        self._device_indices = list(device_indices)
        self._groups = groups
        self._save_path = save_path
        self._budget = budget
        self._return_mode = return_mode
        self._summaries: list[GroupWorkSummary] = []
        self._worker_metas: list[tuple[str, dict, dict]] = []

    @property
    def summaries(self) -> list[GroupWorkSummary]:
        """每完成一个组回传一条摘要（与 groups 一一对应，顺序不定）。"""
        return list(self._summaries)

    @property
    def worker_metas(self) -> list[tuple[str, dict, dict]]:
        """worker 直接写盘后回传的 (staging_subdir, weight_map, desc_map)。"""
        return list(self._worker_metas)

    def run(self) -> Iterator[IRResult]:
        world_size = len(self._device_indices)
        if world_size < 1:
            raise ValueError("MultiNpuConvertScheduler requires at least one device index")

        payload = _NpuWorkPayload(
            model_path=self._model_path,
            config=self._config,
            device_indices=self._device_indices,
            save_path=self._save_path,
            budget=self._budget,
            return_mode=self._return_mode,
        )
        task_queue = torch.multiprocessing.get_context("spawn").Queue()
        result_queue = torch.multiprocessing.get_context("spawn").Queue(maxsize=_NPU_RESULT_QUEUE_MAXSIZE)
        error_queue = torch.multiprocessing.get_context("spawn").Queue()

        procs = spawn(
            _npu_worker_fn,
            args=(world_size, task_queue, result_queue, error_queue, payload),
            nprocs=world_size,
            join=False,
        )
        logger.info("Convert npu_multi: spawned %d worker processes", world_size)
        try:
            # 全部组依序入队，随后 world_size 个 None 哨兵终止各进程。
            for group in self._groups:
                task_queue.put(group)
            for _ in range(world_size):
                task_queue.put(None)

            summaries = 0
            last_progress = time.monotonic()
            while summaries < len(self._groups):
                self._raise_if_worker_error(error_queue)
                if time.monotonic() - last_progress > _RUN_TIMEOUT_S:
                    raise RuntimeError(
                        f"Multi-NPU convert stalled: no results for {_RUN_TIMEOUT_S:.0f}s "
                        f"({summaries}/{len(self._groups)} group summaries)"
                    )
                try:
                    tag, item = result_queue.get(timeout=_QUEUE_GET_TIMEOUT_S)
                except queue.Empty:
                    # 队列此刻为空且所有 worker 已退出：不会再有结果/摘要到达。
                    # 若仍未收满 summary，说明有 worker 异常退出（如被 kill 且未写 error_queue）。
                    if all(not p.is_alive() for p in procs.processes):
                        raise RuntimeError(
                            f"All NPU workers exited before completing all groups "
                            f"({summaries}/{len(self._groups)} summaries)"
                        )
                    continue
                last_progress = time.monotonic()
                if tag == "result":
                    yield item
                elif tag == "saved":
                    yield IRResult(module_path=item, final_ir=IRKind.FLOAT, already_saved=True)
                elif tag == "worker_meta":
                    self._worker_metas.append(item)
                elif tag == "summary":
                    self._summaries.append(item)
                    summaries += 1
                else:
                    raise RuntimeError(f"Unknown npu_multi result tag: {tag!r}")
            # worker_meta 在 worker 退出前才发，收满 summary 后继续排空。
            if self._save_path:
                while len(self._worker_metas) < world_size:
                    self._raise_if_worker_error(error_queue)
                    try:
                        tag, item = result_queue.get(timeout=_QUEUE_GET_TIMEOUT_S)
                    except queue.Empty:
                        if all(not p.is_alive() for p in procs.processes):
                            break
                        continue
                    if tag == "worker_meta":
                        self._worker_metas.append(item)
                    elif tag == "saved":
                        yield IRResult(module_path=item, final_ir=IRKind.FLOAT, already_saved=True)
            # 收尾：确认无 worker 报错（错误已在上方 error_queue 检查中抛出）
            self._raise_if_worker_error(error_queue)
        finally:
            self._join_procs(procs)

    @staticmethod
    def _raise_if_worker_error(error_queue) -> None:
        """非阻塞检查 error_queue；有 worker 报错即抛异常（唯一失败判定来源）。

        成功与否由主循环按 ``summary`` 计数收敛判断，不以 worker 是否退出为准，
        避免「worker 快于主进程落盘、全部退出而队列尚有未读结果」时误判失败。
        """
        try:
            rank, tb = error_queue.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(f"NPU worker rank={rank} failed:\n{tb}")

    @staticmethod
    def _join_procs(procs) -> None:
        """join 子进程；超时未退出的 terminate 兜底。"""
        for p in procs.processes:
            p.join(timeout=_SPAWN_JOIN_TIMEOUT_S)
        for p in procs.processes:
            if p.is_alive():
                logger.warning("Terminating stuck NPU worker pid=%s", p.pid)
                p.terminate()
        for p in procs.processes:
            p.join(timeout=_SPAWN_JOIN_TIMEOUT_S)
