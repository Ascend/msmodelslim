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

from typing import Tuple

import torch.distributed as dist

from msmodelslim.utils.exception import SchemaValidateError


def resolve_expert_ep_range(num_experts: int) -> Tuple[int, int, int, int]:
    """Return ``(ep_size, ep_rank, start, end)`` for contiguous expert sharding.

    Single-process / uninitialized dist → full range ``[0, num_experts)``.
    ``num_experts <= 0`` → ``(1, 0, 0, 0)``.
    """
    if num_experts <= 0:
        return 1, 0, 0, 0
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return 1, 0, 0, num_experts

    world_size = dist.get_world_size()
    if num_experts % world_size != 0:
        raise SchemaValidateError(
            f"The total number of experts ({num_experts}) must be divisible by the world size ({world_size})."
        )
    n_local = num_experts // world_size
    rank = dist.get_rank()
    start = rank * n_local
    end = start + n_local
    return world_size, rank, start, end


def _resolve_expert_num(config) -> int:
    if hasattr(config, "num_experts") and isinstance(
        config.num_experts, int
    ):  # Mock时hasattr一直为True，返回Mock类型无法遍历
        return config.num_experts
    if (
        hasattr(config, "n_routed_experts")
        and hasattr(config, "n_shared_experts")
        and isinstance(config.n_routed_experts, int)
    ):
        return config.n_routed_experts
    return 0


def _get_expert_range(config):
    """Return ``(start, end)`` local expert half-open range for this rank."""
    _, _, start, end = resolve_expert_ep_range(_resolve_expert_num(config))
    return start, end
