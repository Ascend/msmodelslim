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

模块级分布式同步：DTSMixin、默认参数/buffer 同步与张量 broadcast 原语。
"""
# pylint: disable=unsubscriptable-object

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import distributed as dist
from torch import nn

from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext


def broadcast_tensor_process_group_safe(tensor: torch.Tensor, src: int) -> None:
    """对 ``tensor`` 执行 ``dist.broadcast``；HCCL/NCCL 进程组不支持 CPU 张量时，经本 rank 当前加速器设备做一次 staging。"""
    if not dist.is_initialized():
        return
    backend = str(dist.get_backend()).lower()
    if tensor.device.type != "cpu" or backend not in ("hccl", "nccl"):
        dist.broadcast(tensor, src=src)
        return
    rank = dist.get_rank()
    if backend == "hccl":
        dev = torch.device(f"npu:{torch.npu.current_device()}")
    else:
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    buf = tensor.detach().to(dev).contiguous()
    dist.broadcast(buf, src=src)
    if rank != src:
        tensor.copy_(buf.cpu())


class DTSMixin(ABC):
    """依赖模块自定义分布式同步：实现 ``distributed_sync``。"""

    @abstractmethod
    def distributed_sync(self, record: TaskExecutionRecord, sync_ctx: TaskSyncContext) -> None:
        """各 rank 调用；集合通信顺序须一致。"""

    @staticmethod
    def default_state_sync(
        record: TaskExecutionRecord,
        sync_ctx: TaskSyncContext,
        module: Optional[nn.Module],
    ) -> None:
        """调用 DTS 默认的模块参数/缓冲区同步实现。"""
        default_module_state_sync(record, sync_ctx, module)

    @staticmethod
    def broadcast_tensor(tensor: torch.Tensor, src: int) -> None:
        """跨 rank broadcast 张量（对 CPU 张量做后端兼容处理）。"""
        broadcast_tensor_process_group_safe(tensor, src)


def default_module_state_sync(
    record: TaskExecutionRecord,
    sync_ctx: TaskSyncContext,
    module: Optional[nn.Module] = None,
) -> None:
    if module is None or not dist.is_initialized() or sync_ctx.world_size <= 1:
        return

    owner_rank = record.executor_rank

    def _infer_module_device(m: nn.Module) -> torch.device:
        for p in m.parameters(recurse=False):
            return p.device
        for b in m.buffers(recurse=False):
            return b.device
        return torch.device("cpu")

    def _sync_named_tensor(
        is_param: bool, name: str, tensor_like: Optional[torch.Tensor], requires_grad: bool = False
    ) -> None:
        meta = [None]
        if sync_ctx.rank == owner_rank:
            if tensor_like is None:
                raise RuntimeError(
                    f"Owner rank={owner_rank} missing {'parameter' if is_param else 'buffer'} {name!r} during default sync."
                )
            meta[0] = {
                "shape": tuple(tensor_like.shape),
                "dtype": tensor_like.dtype,
                "requires_grad": bool(requires_grad),
            }
        dist.broadcast_object_list(meta, src=owner_rank)
        owner_meta = meta[0]

        current = tensor_like
        if current is None or tuple(current.shape) != owner_meta["shape"] or current.dtype != owner_meta["dtype"]:
            new_tensor = torch.empty(
                owner_meta["shape"], dtype=owner_meta["dtype"], device=_infer_module_device(module)
            )
            if is_param:
                new_param = nn.Parameter(new_tensor, requires_grad=owner_meta["requires_grad"])
                setattr(module, name, new_param)
                current = new_param
            else:
                module.register_buffer(name, new_tensor)
                current = module._buffers[name]
        broadcast_tensor_process_group_safe(current.data if is_param else current, owner_rank)

    owner_param_names = [None]
    owner_buffer_names = [None]
    if sync_ctx.rank == owner_rank:
        owner_param_names[0] = [name for name, _ in module.named_parameters(recurse=False)]
        owner_buffer_names[0] = [name for name, _ in module.named_buffers(recurse=False)]
    dist.broadcast_object_list(owner_param_names, src=owner_rank)
    dist.broadcast_object_list(owner_buffer_names, src=owner_rank)

    for pname in owner_param_names[0] or []:
        p = module._parameters.get(pname)
        _sync_named_tensor(
            is_param=True, name=pname, tensor_like=p, requires_grad=p.requires_grad if p is not None else False
        )
    for bname in owner_buffer_names[0] or []:
        b = module._buffers.get(bname)
        _sync_named_tensor(is_param=False, name=bname, tensor_like=b)
