#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

"""
DistributedTaskScheduler（分波调度器）的单元测试

测试范围：
    - 多进程 worker 侧 DTS 行为与同步

多 rank worker（``_run_*_worker``）由 ``dts_distributed_spawn.run_distributed_spawn`` 以
``torch.multiprocessing.spawn`` 拉起；子进程内统一 ``gloo`` + ``file://`` 初始化与销毁。
"""

import os
import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

# 多进程 spawn 会在子进程重新 import 本模块；部分环境装了 torch_npu 但缺少运行时库时，
# torch 的 device backend auto-load 会导致 import torch 直接失败。这里统一关闭自动加载。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch.nn as nn

from msmodelslim.utils.distributed import (
    DistributedTaskScheduler,
    TaskExecutionRecord,
    TaskSyncContext,
)
from test.cases.utils.distributed.dts_test_internals import _TaskSpec



def _run_sync_worker_fn(
    rank: int,
    world_size: int,
    module_cls,
    module_kwargs: dict,
    owner_rank: int,
    results_queue=None,
):
    """分布式同步测试：由 ``run_distributed_spawn`` 在子进程内 init gloo 后调用。"""
    import traceback

    import torch

    from test.cases.utils.distributed.dts_test_internals import default_module_state_sync

    try:
        # 创建模块并设置不同的初始值
        module = module_cls(**module_kwargs)

        # 根据 rank 设置不同的初始值
        with torch.no_grad():
            for param in module.parameters(recurse=False):
                if rank == owner_rank:
                    param.fill_(1.0)  # owner rank 设置为 1.0
                else:
                    param.fill_(0.0)  # 其他 rank 设置为 0.0

            for buffer in module.buffers(recurse=False):
                if rank == owner_rank:
                    buffer.fill_(2.0)  # owner rank 设置为 2.0
                else:
                    buffer.fill_(0.0)  # 其他 rank 设置为 0.0

        # 创建 task spec 和 record
        task = _TaskSpec(task_id="task1")
        record = TaskExecutionRecord(task_id=task.task_id, executor_rank=owner_rank)
        sync_ctx = TaskSyncContext(
            model=module,
            rank=rank,
            world_size=world_size,
        )

        # 执行同步
        default_module_state_sync(record, sync_ctx, module)

        # 回传完整 post_sync 副本，用于验证大对象传输场景。
        post_sync_values = {}
        for name, param in module.named_parameters(recurse=False):
            post_sync_values[f"param_{name}"] = param.data.clone()
        for name, buffer in module.named_buffers(recurse=False):
            post_sync_values[f"buffer_{name}"] = buffer.clone()

        def _tensor_dict_to_numpy_copy(td: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v.detach().cpu().numpy().copy() for k, v in td.items()}

        result = {
            "ok": True,
            "rank": rank,
            "post_sync": _tensor_dict_to_numpy_copy(post_sync_values),
            "module_type": module_cls.__name__,
        }
        if results_queue is not None:
            results_queue.put(result)
    except Exception:
        err = {
            "ok": False,
            "rank": rank,
            "error": traceback.format_exc(),
        }
        try:
            if results_queue is not None:
                results_queue.put(err)
        except Exception:
            pass



def _run_disable_parallel_worker(rank: int, world_size: int, results_queue) -> None:
    """验证 disable_parallel 控制的 shared 执行与同步跳过语义（由 ``run_distributed_spawn`` 拉起）。"""
    import torch
    import torch.distributed as dist
    from torch import nn

    from msmodelslim.utils.distributed import (
        DistributedTaskScheduler,
        TaskExecutionRecord,
        clear_distributed_task_work_queue,
    )
    from test.cases.utils.distributed.dts_test_internals import (
        _DtsMultiRankParallelWaveScheduler,
        _DtsSequentialWaveScheduler,
    )

    # 防止跨用例残留队列全局状态影响结果
    clear_distributed_task_work_queue()

    class _ScalarMod(nn.Module):
        def __init__(self, init_val: float):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_mod1 = _ScalarMod(0.0)
            self.shared_mod2 = _ScalarMod(0.0)

    r = rank

    def _fn(payload: Any):
        slot = int((payload or {}).get("slot", -1))
        if slot == 1:
            model.shared_mod1.p.data.fill_(100.0 + float(r))
        elif slot == 2:
            model.shared_mod2.p.data.fill_(200.0 + float(r))
        else:
            raise RuntimeError(f"unexpected task payload slot={slot}")
        return None

    def _make_shared_sync_default(slot: int):
        def _shared_sync_default(record: TaskExecutionRecord, sync_ctx):
            # broadcast executor_rank 的 shared_mod 值
            src = record.executor_rank
            obj = [None]
            if sync_ctx.rank == src:
                if slot == 1:
                    obj[0] = float(sync_ctx.model.shared_mod1.p.detach().cpu().item())
                elif slot == 2:
                    obj[0] = float(sync_ctx.model.shared_mod2.p.detach().cpu().item())
                else:
                    raise RuntimeError(f"unexpected task payload slot={slot}")
            dist.broadcast_object_list(obj, src=src)
            if slot == 1:
                sync_ctx.model.shared_mod1.p.data.fill_(obj[0])
            elif slot == 2:
                sync_ctx.model.shared_mod2.p.data.fill_(obj[0])
            else:
                raise RuntimeError(f"unexpected task payload slot={slot}")

        return _shared_sync_default

    def _shared_sync_should_not_run(record: TaskExecutionRecord, sync_ctx):
        raise AssertionError("shared sync should be skipped when disable_parallel=True")

    results: Dict[str, Any] = {"rank": rank}

    # ---------------------------
    # 单波并行（多 rank）：_DtsMultiRankParallelWaveScheduler
    # ---------------------------
    model = _Root()
    scheduler = _DtsMultiRankParallelWaveScheduler(model=model)
    with scheduler:
        scheduler.submit(_fn, args=({"slot": 1},), dependencies=[],
            sync_fn=_make_shared_sync_default(1),
        )
        scheduler.submit(_fn, args=({"slot": 2},), dependencies=[],
            sync_fn=_make_shared_sync_default(2),
        )
        _ = scheduler.run()
    results["impl_default"] = {
        "shared1": float(model.shared_mod1.p.detach().cpu().item()),
        "shared2": float(model.shared_mod2.p.detach().cpu().item()),
    }

    # ---------------------------
    # 单波无并行（多 rank）：_DtsSequtentialWaveScheduler
    # ---------------------------
    model = _Root()
    scheduler = _DtsSequentialWaveScheduler(model=model)
    with scheduler:
        scheduler.submit(_fn, args=({"slot": 1},), dependencies=[],
            sync_fn=_shared_sync_should_not_run,
        )
        scheduler.submit(_fn, args=({"slot": 2},), dependencies=[],
            sync_fn=_shared_sync_should_not_run,
        )
        _ = scheduler.run()
    results["impl_disable"] = {
        "shared1": float(model.shared_mod1.p.detach().cpu().item()),
        "shared2": float(model.shared_mod2.p.detach().cpu().item()),
    }

    # ---------------------------
    # DistributedTaskScheduler(wave): default
    # ---------------------------
    model = _Root()
    wave_scheduler = DistributedTaskScheduler(
        model=model,
        disable_parallel=False,
    )
    with wave_scheduler:
        wave_scheduler.submit(_fn, args=({"slot": 1},), dependencies=[],
            sync_fn=_make_shared_sync_default(1),
        )
        wave_scheduler.submit(_fn, args=({"slot": 2},), dependencies=[],
            sync_fn=_make_shared_sync_default(2),
        )
        _ = wave_scheduler.run()
    results["wave_default"] = {
        "shared1": float(model.shared_mod1.p.detach().cpu().item()),
        "shared2": float(model.shared_mod2.p.detach().cpu().item()),
    }

    # ---------------------------
    # DistributedTaskScheduler(wave): disable_parallel
    # ---------------------------
    model = _Root()
    wave_scheduler = DistributedTaskScheduler(
        model=model,
        disable_parallel=True,
    )
    with wave_scheduler:
        wave_scheduler.submit(_fn, args=({"slot": 1},), dependencies=[],
            sync_fn=_shared_sync_should_not_run,
        )
        wave_scheduler.submit(_fn, args=({"slot": 2},), dependencies=[],
            sync_fn=_shared_sync_should_not_run,
        )
        _ = wave_scheduler.run()
    results["wave_disable"] = {
        "shared1": float(model.shared_mod1.p.detach().cpu().item()),
        "shared2": float(model.shared_mod2.p.detach().cpu().item()),
    }

    results_queue.put(results)


def _run_dts_multirank_perf_worker(
    rank: int,
    world_size: int,
    work_queue,
    results_queue,
    num_tasks: int,
    exec_sleep_s: float,
    sync_sleep_s: float,
    use_work_queue: bool,
) -> None:
    """多进程性能单测：``run_distributed_spawn`` 已 init gloo，此处走 ``world_size>1`` 的 ``run()``。

    Args:
        work_queue: 父进程构造的 ``multiprocessing.Queue``；``use_work_queue=False`` 时传 ``None``，
            子进程不注入共享队列（静态 ``idx % world_size`` owner）。
    """
    import time
    import traceback

    from torch import nn

    from msmodelslim.utils.distributed import (
        TaskExecutionRecord,
        TaskSyncContext,
        clear_distributed_task_work_queue,
        set_distributed_task_work_queue,
    )
    from test.cases.utils.distributed.dts_test_internals import _DtsMultiRankParallelWaveScheduler

    clear_distributed_task_work_queue()
    if use_work_queue and work_queue is not None:
        set_distributed_task_work_queue(work_queue)

    try:
        model = nn.Linear(2, 2)

        def _sync_slow(_record: TaskExecutionRecord, _sync_ctx: TaskSyncContext) -> None:
            if sync_sleep_s > 0:
                time.sleep(float(sync_sleep_s))

        sync_fn = _sync_slow if sync_sleep_s > 0 else None

        from functools import partial

        sch = _DtsMultiRankParallelWaveScheduler(model=model)
        with sch:
            for _ in range(int(num_tasks)):
                sch.submit(
                    partial(time.sleep, float(exec_sleep_s)),
                    dependencies=[],
                    sync_fn=sync_fn,
                )
            t0 = time.perf_counter()
            records = sch.run()
            t_run = time.perf_counter() - t0

        sum_exec = sum(float(r.exec_time_s) for r in records)
        sum_sync = sum(float(r.sync_time_s) for r in records)
        ratio = (t_run / sum_exec) if sum_exec > 0 else -1.0

        results_queue.put(
            {
                "ok": True,
                "rank": rank,
                "t_run": t_run,
                "sum_exec": sum_exec,
                "sum_sync": sum_sync,
                "ratio": ratio,
                "executors": [int(r.executor_rank) for r in records],
            }
        )
    except Exception:
        try:
            results_queue.put(
                {
                    "ok": False,
                    "rank": rank,
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        clear_distributed_task_work_queue()


def _run_dts_dp_serial_oracle_equivalence_worker(
    rank: int,
    world_size: int,
    results_queue,
    task_seed: int,
    init_seed: int,
) -> None:
    """多卡：串行 oracle 与 ``DistributedTaskScheduler``（并行波次 + 默认同步）终态 ``state_dict`` 应对齐。

各 rank 使用相同随机种子生成任务序列并执行相同 ``submit`` 顺序；oracle 在本地独立重放以得到 ref。
    """
    import random
    import traceback

    import torch
    from torch import nn

    from msmodelslim.utils.distributed import (
        DistributedTaskScheduler,
        clear_distributed_task_work_queue,
    )

    from test.cases.utils.distributed.dts_test_utils import build_dts_dependency_mock_model

    clear_distributed_task_work_queue()
    try:
        n_tasks = 40

        dep_pool = [
            "module1",
            "module2",
            "m1",
            "m2",
            "m3",
            "layer.0.q_proj",
            "layer.0.k_proj",
            "layer.0.v_proj",
            "layer.0.o_proj",
            "layer.1.q_proj",
            "layer.1.k_proj",
            "layer.2.q_proj",
            "layer.3.o_proj",
            "model.layers.0.self_attn.q_proj",
            "shared",
            "shared_module",
        ]

        rng = random.Random(task_seed)
        specs: List[Dict[str, Any]] = []
        for _ in range(n_tasks):
            specs.append(
                {
                    "path": rng.choice(dep_pool),
                    "mult": rng.uniform(0.98, 1.02),
                    "add": rng.uniform(-0.02, 0.02),
                }
            )

        def _apply_linear_inplace(model: nn.Module, path: str, mult: float, add: float) -> None:
            m = model.get_submodule(path)
            if not isinstance(m, nn.Linear):
                raise RuntimeError(f"expected nn.Linear at {path!r}, got {type(m)}")
            with torch.no_grad():
                m.weight.data.mul_(mult).add_(add)
                if m.bias is not None:
                    m.bias.data.mul_(mult).add_(add)

        def _init_params_bounded(m: nn.Module) -> None:
            with torch.no_grad():
                for p in m.parameters():
                    p.uniform_(0.7, 1.3)

        def _assert_finite_sd(sd: Dict[str, torch.Tensor], where: str) -> None:
            for k, t in sd.items():
                if not torch.isfinite(t).all():
                    raise RuntimeError(f"non-finite tensor in {where}: {k}")

        # --- oracle（严格按提交序）---
        torch.manual_seed(init_seed)
        model_o = build_dts_dependency_mock_model()
        _init_params_bounded(model_o)
        for spec in specs:
            _apply_linear_inplace(model_o, spec["path"], spec["mult"], spec["add"])
        ref_sd = {k: v.detach().cpu().clone() for k, v in model_o.state_dict().items()}
        del model_o
        _assert_finite_sd(ref_sd, "oracle")

        # --- DTS 并行路径 ---
        DistributedTaskScheduler.set_global_disable_parallel(False)
        torch.manual_seed(init_seed)
        model_d = build_dts_dependency_mock_model()
        _init_params_bounded(model_d)

        with DistributedTaskScheduler(model_d) as sched:
            for spec in specs:
                p = spec["path"]
                mu = float(spec["mult"])
                ad = float(spec["add"])

                def _fn(p=p, mu=mu, ad=ad) -> None:
                    _apply_linear_inplace(model_d, p, mu, ad)

                sched.submit(_fn, dependencies=[p], parallel=True)
            sched.run()

        dts_sd = {k: v.detach().cpu() for k, v in model_d.state_dict().items()}
        _assert_finite_sd(dts_sd, "dts")

        ok = True
        worst_key = None
        worst = 0.0
        atol, rtol = 1e-5, 1e-5
        for k in ref_sd:
            if not torch.allclose(ref_sd[k], dts_sd[k], atol=atol, rtol=rtol):
                ok = False
                diff = (ref_sd[k] - dts_sd[k]).abs().max().item()
                if diff > worst:
                    worst = diff
                    worst_key = k

        results_queue.put(
            {
                "ok": ok,
                "rank": int(rank),
                "task_seed": int(task_seed),
                "init_seed": int(init_seed),
                "worst_key": worst_key,
                "worst": float(worst),
            }
        )
    except Exception:
        try:
            results_queue.put(
                {
                    "ok": False,
                    "rank": int(rank),
                    "task_seed": int(task_seed),
                    "init_seed": int(init_seed),
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        clear_distributed_task_work_queue()


def _run_dts_submit_hash_mismatch_worker(rank: int, world_size: int, results_queue) -> None:
    """多卡：rank 间 submit 语义不同（args）应在 run 前报错。"""
    import traceback
    from torch import nn
    from msmodelslim.utils.distributed import DistributedTaskScheduler

    try:
        model = nn.Module()
        model.m1 = nn.Linear(1, 1)
        scheduler = DistributedTaskScheduler(model=model)
        with scheduler:
            payload = "rank0" if rank == 0 else "rank1"
            scheduler.submit(lambda x=payload: x, args=(payload,), dependencies=["m1"], parallel=True)
            scheduler.run()
        results_queue.put({"ok": False, "rank": rank, "error": "expected mismatch error but run succeeded"})
    except Exception as e:
        results_queue.put({"ok": True, "rank": rank, "error": str(e)})
    except BaseException:
        results_queue.put({"ok": False, "rank": rank, "error": traceback.format_exc()})


def _run_dts_submit_hash_tensor_meta_worker(rank: int, world_size: int, results_queue) -> None:
    """多卡：tensor 值不同但元信息相同时，语义哈希应一致并可正常运行。"""
    import traceback
    import torch
    from torch import nn
    from msmodelslim.utils.distributed import DistributedTaskScheduler

    try:
        model = nn.Module()
        model.m1 = nn.Linear(1, 1)
        scheduler = DistributedTaskScheduler(model=model)
        with scheduler:
            t = torch.ones(2, 3, dtype=torch.float32) if rank == 0 else torch.zeros(2, 3, dtype=torch.float32)
            scheduler.submit(lambda x=t: None, args=(t,), dependencies=["m1"], parallel=True)
            records = scheduler.run()
        results_queue.put({"ok": True, "rank": rank, "record_count": len(records)})
    except Exception:
        results_queue.put({"ok": False, "rank": rank, "error": traceback.format_exc()})


def _run_dts_heterogeneous_submit_supported_worker(rank: int, world_size: int, results_queue) -> None:
    """多卡：异构提交顺序（2 local + 8 shared vs 8 shared + 2 local）应可执行且不报错。"""
    import traceback
    from torch import nn
    from msmodelslim.utils.distributed import DistributedTaskScheduler
    from test.cases.utils.distributed.dts_test_internals import dts_waves

    try:
        model = nn.Module()
        model.m1 = nn.Linear(1, 1)
        scheduler = DistributedTaskScheduler(model=model, disable_parallel=False)
        with scheduler:
            # rank0: 2(local) + 8(share); rank1: 8(share) + 2(local)
            if rank == 0:
                plan = [(False, "local")] * 2 + [(True, "share")] * 8
            else:
                plan = [(True, "share")] * 8 + [(False, "local")] * 2

            for idx, (parallel, _tag) in enumerate(plan):
                scheduler.submit(lambda _i=idx: _i, dependencies=[], parallel=parallel)
            records = scheduler.run()

        waves = dts_waves(scheduler)
        results_queue.put(
            {
                "ok": True,
                "rank": rank,
                "record_count": len(records),
                "wave_count": len(waves),
                "wave_task_counts": [len(w._tasks) for w in waves],
                "executor_ranks": [int(r.executor_rank) for r in records],
            }
        )
    except Exception:
        results_queue.put({"ok": False, "rank": rank, "error": traceback.format_exc()})


