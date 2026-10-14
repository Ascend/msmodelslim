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
# pylint: disable=logging-fstring-interpolation

import os
import socket

import torch
import torch.distributed as dist

from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception import SchemaValidateError, EnvError


def find_free_port(start_port=29500, max_attempts=100):
    """
    查找可用的端口

    Args:
        start_port: 起始端口号（必须 >= 1024，避免特权端口）
        max_attempts: 最大尝试次数

    Returns:
        可用的端口号(整数)

    Raises:
        SchemaValidateError: 如果 start_port 小于 1024 或大于 65535
        EnvError: 如果无法找到可用端口
    """
    # 安全检查：限制端口范围
    if start_port < 1024:
        raise SchemaValidateError(f"start_port must be >= 1024 (got {start_port})")
    if start_port > 65535:
        raise SchemaValidateError(f"start_port must be <= 65535 (got {start_port})")

    # 确保不超过最大端口号
    end_port = min(start_port + max_attempts, 65536)

    for port in range(start_port, end_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # 安全：SO_REUSEADDR 应在 bind 之前设置
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # 安全：只绑定到 localhost，不暴露在所有网络接口
                s.bind(('127.0.0.1', port))
                return port
        except OSError as e:
            # Port is already in use or cannot be bound, try next port
            get_logger().debug(f"Port {port} is not available: {e}, trying next port")
            continue

    # 安全：错误消息不暴露具体端口范围细节
    raise EnvError(f"Cannot find a free port (searched {max_attempts} ports starting from {start_port})")


def setup_distributed(rank, world_size, backend, master_port=None, device_index=None):
    """
    设置分布式环境

    Args:
        rank: 进程组中的rank（用于进程组通信）
        world_size: 进程组大小
        backend: 分布式后端 ('hccl' 或 'nccl')
        master_port: 主节点端口。如果为None，会自动查找可用端口（推荐用法）
                     如果指定端口，会尝试使用该端口，失败则自动查找可用端口
                     端口必须 >= 1024（避免特权端口）且 <= 65535
        device_index: 实际设备索引。如果为None，则使用rank作为设备索引（向后兼容）
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # 确定实际使用的设备索引
    actual_device_idx = device_index if device_index is not None else rank

    # 初始化设备
    if backend == 'hccl':
        # when reboot, use torch.npu.init() to init hccl
        torch.npu.set_device(f"npu:{actual_device_idx}")
    else:
        torch.cuda.set_device(actual_device_idx)

    # 初始化进程组
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
