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

import unittest
from unittest.mock import patch, MagicMock

import torch

# 在没有torch.npu的环境中，导入dist_ops会触发对torch.npu的访问
# 临时mock torch.npu以完成导入，导入后立即清理
_torch_npu_existed = hasattr(torch, 'npu')
if not _torch_npu_existed:
    torch.npu = MagicMock()

from msmodelslim.utils.distributed.dist_ops import sync_gather_tensor_lists  # noqa: E402
from msmodelslim.utils.exception import SchemaValidateError  # noqa: E402

if not _torch_npu_existed:
    del torch.npu
    del _torch_npu_existed
else:
    del _torch_npu_existed


class TestSyncGatherTensorListsWhenEmpty(unittest.TestCase):
    """测试sync_gather_tensor_lists在空输入时的行为"""

    def test_sync_gather_tensor_lists_raises_schema_error_when_empty_list(self):
        """异常：空列表应抛出SchemaValidateError"""
        with self.assertRaises(SchemaValidateError) as ctx:
            sync_gather_tensor_lists([])
        self.assertIn("empty", str(ctx.exception))


class TestSyncGatherTensorListsOnCpu(unittest.TestCase):
    """测试sync_gather_tensor_lists在CPU路径下的行为"""

    @patch("msmodelslim.utils.distributed.dist_ops.dist.get_world_size", return_value=2)
    @patch("msmodelslim.utils.distributed.dist_ops.dist.all_gather_object")
    def test_sync_gather_tensor_lists_returns_flattened_when_on_cpu(self, mock_gather, _):
        """正常：CPU路径应返回展平后的列表"""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0])

        def gather_side_effect(gathered, local_list, group=None):
            gathered[0] = local_list
            gathered[1] = [torch.tensor([4.0, 5.0]), torch.tensor([6.0])]

        mock_gather.side_effect = gather_side_effect

        result = sync_gather_tensor_lists([t1, t2], on_cpu=True)

        self.assertEqual(len(result), 4)

    @patch("msmodelslim.utils.distributed.dist_ops.dist.get_world_size", return_value=2)
    @patch("msmodelslim.utils.distributed.dist_ops.dist.all_gather_object")
    def test_sync_gather_tensor_lists_preserves_data_when_on_cpu(self, mock_gather, _):
        """正常：CPU路径应保留原始数据"""
        t1 = torch.tensor([1.0, 2.0])

        def gather_side_effect(gathered, local_list, group=None):
            gathered[0] = local_list
            gathered[1] = [torch.tensor([3.0, 4.0])]

        mock_gather.side_effect = gather_side_effect

        result = sync_gather_tensor_lists([t1], on_cpu=True)

        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(result[1], torch.tensor([3.0, 4.0])))


class TestSyncGatherTensorListsOnNpu(unittest.TestCase):
    """测试sync_gather_tensor_lists在NPU路径下的行为"""

    @patch("msmodelslim.utils.distributed.dist_ops.torch.npu", create=True)
    @patch("msmodelslim.utils.distributed.dist_ops.dist.get_world_size", return_value=2)
    @patch("msmodelslim.utils.distributed.dist_ops.dist.all_gather")
    def test_sync_gather_tensor_lists_returns_flattened_when_on_npu(self, mock_all_gather, _, mock_npu):
        """正常：NPU路径应返回展平后的列表"""
        mock_npu.current_device.return_value = 0
        # 在无NPU环境中，torch.device("npu:0")会报错，需要mock torch.device
        _real_device = torch.device

        def _safe_device(dev_str):
            if isinstance(dev_str, str) and dev_str.startswith("npu"):
                return _real_device("cpu")
            return _real_device(dev_str)

        with patch("msmodelslim.utils.distributed.dist_ops.torch.device", side_effect=_safe_device):

            def all_gather_side_effect(tensor_list, tensor, group=None):
                for t in tensor_list:
                    t.copy_(tensor)

            mock_all_gather.side_effect = all_gather_side_effect

            t1 = torch.tensor([1.0, 2.0])
            t2 = torch.tensor([3.0])

            result = sync_gather_tensor_lists([t1, t2], on_cpu=False)

            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)

    @patch("msmodelslim.utils.distributed.dist_ops.torch.npu", create=True)
    @patch("msmodelslim.utils.distributed.dist_ops.dist.get_world_size", return_value=1)
    @patch("msmodelslim.utils.distributed.dist_ops.dist.all_gather")
    def test_sync_gather_tensor_lists_single_rank_when_on_npu(self, mock_all_gather, _, mock_npu):
        """边界：单rank时应返回原始列表"""
        mock_npu.current_device.return_value = 0
        _real_device = torch.device

        def _safe_device(dev_str):
            if isinstance(dev_str, str) and dev_str.startswith("npu"):
                return _real_device("cpu")
            return _real_device(dev_str)

        with patch("msmodelslim.utils.distributed.dist_ops.torch.device", side_effect=_safe_device):

            def all_gather_side_effect(tensor_list, tensor, group=None):
                for t in tensor_list:
                    t.copy_(tensor)

            mock_all_gather.side_effect = all_gather_side_effect

            t1 = torch.tensor([1.0, 2.0])
            result = sync_gather_tensor_lists([t1], on_cpu=False)

            self.assertEqual(len(result), 1)
