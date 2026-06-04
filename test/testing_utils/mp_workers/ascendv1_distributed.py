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

Spawn worker helpers for distributed AscendV1 saver tests.

Kept under ``testing_utils.mp_workers`` (on pytest pythonpath) so multiprocessing
spawn can unpickle the target; avoid defining workers under ``test/.../save/``
which pytest may expose as a top-level ``save`` package.
"""

import os
from unittest.mock import MagicMock, patch

import torch.distributed as dist
from torch import nn

from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1_distributed import (
    DistributedAscendV1Config,
    DistributedAscendV1Saver,
)


class SimpleModel(nn.Module):
    """Simple model for distributed saver tests."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


def iter_tasks_mp_worker(rank, world_size, temp_dir, queue, results):
    """Worker for multi-process _iter_tasks test."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500)

    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
    try:
        adapter = MagicMock()
        adapter.model_path = temp_dir

        config = DistributedAscendV1Config(save_directory=temp_dir, part_file_size=4)
        model = SimpleModel()

        with patch(
            "msmodelslim.core.quant_service.modelslim_v1.save.ascendv1_distributed.get_distributed_task_work_queue",
            return_value=queue,
        ):
            saver = DistributedAscendV1Saver(model, config, adapter)
            saver.safetensors_writer = MagicMock()
            saver.json_writer = MagicMock()
            results[rank] = [n for n, _ in saver._iter_tasks("", model)]
    finally:
        dist.destroy_process_group()
