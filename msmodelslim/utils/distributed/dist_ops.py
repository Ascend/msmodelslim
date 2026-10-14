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
"""

from enum import Enum
from typing import Union, List

import torch
import torch.distributed as dist


from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import UnsupportedError, SchemaValidateError


class ReduceOperation(str, Enum):
    """
    Supported reduce operations for distributed synchronization.

    Attributes:
        MIN: Find minimum value across all processes
        MAX: Find maximum value across all processes
        SUM: Sum values across all processes
        MEAN: Calculate mean value across all processes
        PROD: Calculate product across all processes
    """

    MIN = "min"
    MAX = "max"
    SUM = "sum"
    MEAN = "mean"
    PROD = "prod"


def sync_base_operation(tensor: torch.Tensor, op: Union[ReduceOperation, str], group=None) -> torch.Tensor:
    """
    执行不增加显存开销的分布式原地操作

    支持的操作为: min、max、sum、mean、prod

    Args:
        tensor: 要操作的张量，操作结果会原地更新到此张量中
        op: 操作类型，可以使用 ReduceOperation 枚举或字符串
            - ReduceOperation.MIN / 'min': 最小值
            - ReduceOperation.MAX / 'max': 最大值
            - ReduceOperation.SUM / 'sum': 求和
            - ReduceOperation.MEAN / 'mean': 平均值
            - ReduceOperation.PROD / 'prod': 乘积
        group: 进程组，默认为 None（使用默认进程组）

    Returns:
        原地更新后的张量（与输入tensor是同一个对象）

    """

    # Convert string to enum if necessary
    if isinstance(op, str):
        try:
            op = ReduceOperation(op.lower())
        except ValueError as e:
            raise UnsupportedError(
                f"Unsupported operation: {op}. "
                f"Supported operations are: {', '.join([operation.value for operation in ReduceOperation])}",
            ) from e

    # Perform the operation based on enum value
    if op == ReduceOperation.MIN:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
    elif op == ReduceOperation.MAX:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    elif op == ReduceOperation.SUM:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    elif op == ReduceOperation.PROD:
        dist.all_reduce(tensor, op=dist.ReduceOp.PRODUCT, group=group)
    elif op == ReduceOperation.MEAN:
        # 当前hccl后端不支持，通过 mean = sum / world_size 计算结果
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        world_size = dist.get_world_size(group)
        tensor.div_(world_size)

    return tensor


def sync_gather_tensors(
    tensor: torch.Tensor, variable_shapes: bool = False, on_cpu: bool = False, group: dist.ProcessGroup = None
) -> list:
    """
    在所有进程间收集张量，如果在npu上进行会增加显存开销

    Args:
        tensor: 要收集的本地张量
        variable_shapes: 是否支持不同形状的张量聚合（仅在 on_cpu=False 时有效）
                        - False: 所有进程的张量形状必须相同（更快，默认）
                        - True: 支持不同形状（需要额外通信开销）
        on_cpu: 聚合操作在哪里进行
                - False: 在 NPU 上进行聚合（使用 HCCL，默认）
                - True: 在 CPU 上进行聚合（避免 NPU 显存溢出）
        group: 进程组，默认为 None（使用默认进程组）

    Returns:
        收集到的张量列表，列表长度为 world_size
        - 如果 on_cpu=False: 张量在 NPU 上
        - 如果 on_cpu=True: 张量在 CPU 上
    """

    world_size = dist.get_world_size(group)

    if on_cpu:
        # 场景：在 CPU 上进行聚合（针对大张量避免 NPU 显存溢出）
        tensor_cpu = tensor.cpu()
        tensor_list = [None] * world_size
        dist.all_gather_object(tensor_list, tensor_cpu, group=group)  # all_gather_object操作速度很慢
        get_logger().debug("Gathered %s tensors on CPU", world_size)
        return tensor_list

    else:
        # 场景：在 NPU 上进行聚合（使用 HCCL 高效通信）
        if variable_shapes:
            with torch.device(tensor.device):
                # 同步张量形状
                local_shape = torch.tensor(tensor.shape, dtype=torch.long)
                shape_list = [torch.zeros_like(local_shape) for _ in range(dist.get_world_size())]
                dist.all_gather(shape_list, local_shape)

                # 初始化存储
                tensor_list = [torch.zeros(*s.tolist(), dtype=tensor.dtype) for s in shape_list]

                # 收集数据
                dist.all_gather(tensor_list, tensor)
                get_logger().debug("Gathered %s tensors with variable shapes on NPU using HCCL", world_size)
                return tensor_list

        else:
            # NPU 上聚合 - 相同形状（最快路径）
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(tensor_list, tensor, group=group)

            get_logger().debug("Gathered %s tensors with same shape on NPU using HCCL", world_size)

        return tensor_list


def sync_gather_tensor_lists(
    tensor_list: List[torch.Tensor], on_cpu: bool = False, group: dist.ProcessGroup = None
) -> List[torch.Tensor]:
    """
    在所有进程间收集张量列表，并展平成一个列表

    例如：4 卡，每张卡有 12 个 tensor，收集后返回 48 个 tensor 的列表。

    Args:
        tensor_list: 各进程的张量列表
        on_cpu: 聚合操作在哪里进行
                - False: 在 NPU 上进行聚合（使用 HCCL，默认，更快）
                - True: 在 CPU 上进行聚合（避免 NPU 显存溢出，但速度较慢）
        group: 进程组，默认为 None（使用默认进程组）

    Returns:
        所有进程的张量展平后的列表，设备与输入tensor_list的设备一致
    """
    if not tensor_list:
        raise SchemaValidateError(
            "tensor_list is empty, please check the input",
            action="Please provide a non-empty tensor_list with at least one tensor",
        )

    world_size = dist.get_world_size(group)

    # 记录输入设备
    input_device = tensor_list[0].device

    execution_device = torch.device("cpu") if on_cpu else torch.device(f"npu:{torch.npu.current_device()}")

    if input_device != execution_device:
        tensor_list = [t.to(execution_device) for t in tensor_list]

    # CPU路径：使用all_gather_object，速度较慢
    if on_cpu:
        gathered_lists = [None] * world_size
        dist.all_gather_object(gathered_lists, tensor_list, group=group)

        # 展平结果
        flattened = []
        for rank_list in gathered_lists:
            flattened.extend(rank_list)

        # 移回原设备
        if input_device != execution_device:
            flattened = [t.to(input_device) for t in flattened]

        get_logger().debug(
            "Gathered and flattened %d tensors from %d ranks, input device: %s, execution device: cpu",
            len(flattened),
            world_size,
            input_device,
        )

        return flattened

    # NPU路径，使用all_gather，速度更快
    # 1. 堆叠张量和收集形状信息
    flat_tensors = [t.flatten() for t in tensor_list]
    stacked = torch.cat(flat_tensors)

    shapes = torch.cat([torch.tensor(t.shape, dtype=torch.int64, device=execution_device) for t in tensor_list])
    num_tensors = torch.tensor(len(tensor_list), dtype=torch.int64, device=execution_device)

    # 2. 收集所有进程的信息，数量、形状、张量
    num_tensors_list = [torch.zeros_like(num_tensors) for _ in range(world_size)]
    dist.all_gather(num_tensors_list, num_tensors, group=group)

    shape_size = torch.tensor(shapes.numel(), dtype=torch.int64, device=execution_device)
    shape_sizes = [torch.zeros_like(shape_size) for _ in range(world_size)]
    dist.all_gather(shape_sizes, shape_size, group=group)

    data_size = torch.tensor(stacked.numel(), dtype=torch.int64, device=execution_device)
    data_sizes = [torch.zeros_like(data_size) for _ in range(world_size)]
    dist.all_gather(data_sizes, data_size, group=group)

    # 3. 收集最大值并创建缓冲区
    max_shape_size = max(s.item() for s in shape_sizes)
    max_data_size = max(s.item() for s in data_sizes)

    gathered_shapes = torch.zeros(max_shape_size * world_size, dtype=torch.int64, device=execution_device)
    gathered_data = torch.zeros(max_data_size * world_size, dtype=stacked.dtype, device=execution_device)

    # 4. 填充并发送数据
    padded_shapes = torch.zeros(max_shape_size, dtype=torch.int64, device=execution_device)
    padded_shapes[: shapes.numel()] = shapes

    padded_data = torch.zeros(max_data_size, dtype=stacked.dtype, device=execution_device)
    padded_data[: stacked.numel()] = stacked

    # 5. 执行收集
    dist.all_gather(
        tensor_list=[gathered_shapes[i * max_shape_size : (i + 1) * max_shape_size] for i in range(world_size)],
        tensor=padded_shapes,
        group=group,
    )

    dist.all_gather(
        tensor_list=[gathered_data[i * max_data_size : (i + 1) * max_data_size] for i in range(world_size)],
        tensor=padded_data,
        group=group,
    )

    del padded_shapes, padded_data, stacked, flat_tensors, shapes

    # 6. 重建张量列表
    flattened_tensors = []

    for rank in range(world_size):
        n = num_tensors_list[rank].item()
        rank_shape_size = shape_sizes[rank].item()
        rank_data_size = data_sizes[rank].item()

        # 提取形状
        start_idx = rank * max_shape_size
        rank_shapes = gathered_shapes[start_idx : start_idx + rank_shape_size]
        rank_shapes = rank_shapes.view(n, -1)

        # 提取数据
        data_start = rank * max_data_size
        rank_data = gathered_data[data_start : data_start + rank_data_size]

        # 重建张量
        offset = 0
        for i in range(n):
            shape = tuple(rank_shapes[i].tolist())
            num_elements = torch.prod(torch.tensor(shape)).item()
            tensor_data = rank_data[offset : offset + num_elements]
            tensor = tensor_data.view(shape).contiguous()
            flattened_tensors.append(tensor)
            offset += num_elements

    del gathered_shapes, gathered_data

    # 移回原设备
    if input_device != execution_device:
        flattened_tensors = [t.to(input_device) for t in flattened_tensors]

    get_logger().debug(
        "Gathered and flattened %d tensors from %d ranks, input device: %s, execution device: %s",
        len(flattened_tensors),
        world_size,
        input_device,
        execution_device,
    )

    return flattened_tensors
