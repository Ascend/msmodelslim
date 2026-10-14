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

from unittest.mock import patch

import pytest
import numpy as np

from msmodelslim.pytorch.weight_compression.compress_utils import (
    HIGH_SPARSE_MODE,
    LOW_SPARSE_MODE,
    compress_weight_fun,
    pseudo_sparse,
    round_up,
    transform_nd2nz,
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType


class TestCompressUtils:
    def test_pseudo_sparse(self):
        np.random.seed(2024)  # 不能删除，因为调用的pseudo_sparse函数中有numpy的随机数设置
        arr = np.array([1, 1, 1, 1])
        ratio = 0.5
        res = pseudo_sparse(arr, ratio)
        expected = np.array([1, 1, 0, 0])
        assert np.array_equal(res, expected)

    def test_round_up(self):
        val, align = 10, 1
        expected = 10
        assert round_up(val, align) == expected

        val, align = 5, 0
        expected = 0
        assert round_up(val, align) == expected

    def test_transform_nd2nz(self):
        nd_arr = np.random.rand(1024, 1024)
        nz_arr = transform_nd2nz(nd_arr)
        assert nz_arr.shape == (32, 64, 16, 32)

    @pytest.mark.parametrize(
        "sparse_type, expected_mode",
        [
            (QuantType.W16A16S, str(HIGH_SPARSE_MODE)),
            (QuantType.W8A8S, str(LOW_SPARSE_MODE)),
        ],
    )
    def test_should_invoke_popen_with_list_command_when_compressing_weights_given_supported_sparse_type(
        self,
        tmp_path,
        sparse_type,
        expected_mode,
    ):
        # given
        weights = np.array([[1, 2], [3, 4]], dtype=np.int8)

        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.command = command
                self.kwargs = kwargs
                self.returncode = 0
                np.array([11, 12], dtype=np.int8).tofile(command[9])
                np.array([1, 0], dtype=np.int8).tofile(command[10])
                np.array([2, 2, 1], dtype=np.uint32).tofile(command[11])

            def communicate(self, timeout=None):
                return b"", b""

        # when
        with patch(
            "msmodelslim.pytorch.weight_compression.compress_utils.subprocess.Popen",
            side_effect=FakeProcess,
        ) as mock_popen:
            info, output, index = compress_weight_fun(
                weights,
                record_detail_root=str(tmp_path),
                sparse_type=sparse_type,
            )

        # then
        assert mock_popen.call_count == 1
        command = mock_popen.call_args.args[0]
        assert isinstance(command, list)
        assert command[1:3] == ["2", "2"]
        assert command[6] == expected_mode
        assert mock_popen.call_args.kwargs["shell"] is False
        assert np.array_equal(output, np.array([11, 12], dtype=np.int8))
        assert np.array_equal(index, np.array([1, 0], dtype=np.int8))
        assert info == [2, 2, 1]
