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

Unit tests for msmodelslim.format.common.deqscale.
"""

from __future__ import annotations

import numpy as np
import torch

from msmodelslim.format.common.deqscale import deqscale2int64, deqscale2int64_by_dtype


class TestDeqscale2Int64:
    """Tests for deqscale2int64."""

    def test_deqscale2int64_convert_bit_pattern_when_float32(self):
        scale = torch.tensor([1.0, 2.0], dtype=torch.float32)

        result = deqscale2int64(scale)

        expected = np.frombuffer(scale.cpu().numpy().tobytes(), dtype=np.int32).astype(np.int64)
        assert result.dtype == torch.int64
        assert torch.equal(result, torch.tensor(expected))


class TestDeqscale2Int64ByDtype:
    """Tests for deqscale2int64_by_dtype."""

    def test_deqscale2int64_by_dtype_return_unchanged_when_bf16(self):
        scale = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

        result = deqscale2int64_by_dtype(scale, is_bf16=True)

        assert result is scale

    def test_deqscale2int64_by_dtype_convert_when_not_bf16(self):
        scale = torch.tensor([1.0], dtype=torch.float32)

        result = deqscale2int64_by_dtype(scale, is_bf16=False)

        assert result.dtype == torch.int64
