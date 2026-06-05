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

import os
import unittest
from unittest.mock import patch, MagicMock

import torch

from msmodelslim.utils.distributed.dist_setup import setup_distributed


class TestSetupDistributedCudaBackend(unittest.TestCase):
    """测试setup_distributed使用cuda后端的行为"""

    def test_setup_distributed_sets_env_vars_when_cuda_backend(self):
        """正常：cuda后端应正确设置环境变量"""
        mock_cuda = MagicMock()
        mock_init_process_group = MagicMock()

        with (
            patch.object(torch, "cuda", mock_cuda, create=True),
            patch("msmodelslim.utils.distributed.dist_setup.dist.init_process_group", mock_init_process_group),
        ):
            setup_distributed(rank=1, world_size=4, backend="nccl", master_port=29500, device_index=2)

            self.assertEqual(os.environ["MASTER_ADDR"], "127.0.0.1")
            self.assertEqual(os.environ["MASTER_PORT"], "29500")
            self.assertEqual(os.environ["RANK"], "1")
            self.assertEqual(os.environ["WORLD_SIZE"], "4")
            mock_cuda.set_device.assert_called_once_with(2)
            mock_init_process_group.assert_called_once_with(backend="nccl", world_size=4, rank=1)

    def test_setup_distributed_uses_rank_as_device_when_device_index_none_and_cuda(self):
        """边界：device_index为None时cuda应使用rank作为设备索引"""
        mock_cuda = MagicMock()
        mock_init_process_group = MagicMock()

        with (
            patch.object(torch, "cuda", mock_cuda, create=True),
            patch("msmodelslim.utils.distributed.dist_setup.dist.init_process_group", mock_init_process_group),
        ):
            setup_distributed(rank=3, world_size=8, backend="nccl", master_port=29500)

            mock_cuda.set_device.assert_called_once_with(3)


class TestSetupDistributedMasterPortNone(unittest.TestCase):
    """测试setup_distributed在master_port为None时的行为"""

    def test_setup_distributed_sets_master_port_to_none_str_when_port_is_none(self):
        """边界：master_port为None时应将环境变量设为'None'"""
        mock_npu = MagicMock()
        mock_init_process_group = MagicMock()

        with (
            patch.object(torch, "npu", mock_npu, create=True),
            patch("msmodelslim.utils.distributed.dist_setup.dist.init_process_group", mock_init_process_group),
        ):
            setup_distributed(rank=0, world_size=2, backend="hccl", master_port=None)

            self.assertEqual(os.environ["MASTER_PORT"], "None")
