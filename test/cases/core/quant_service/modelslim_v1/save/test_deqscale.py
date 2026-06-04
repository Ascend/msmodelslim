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

import numpy as np
import torch

from msmodelslim.core.quant_service.modelslim_v1.save.utils.deqscale import (
    deqscale2int64,
    deqscale2int64_by_dtype,
)


class TestDeqscale2Int64:
    """Tests for deqscale2int64."""

    def test_output_dtype_is_int64(self):
        """Output tensor should be int64."""
        scale = torch.randn(4, dtype=torch.float32)
        out = deqscale2int64(scale)
        assert out.dtype == torch.int64

    def test_shape_preserved_1d(self):
        """Shape is preserved for 1D input."""
        scale = torch.randn(8, dtype=torch.float32)
        out = deqscale2int64(scale)
        assert out.shape == scale.shape

    def test_roundtrip_1d(self):
        """Roundtrip: float32 -> int64 -> float32 recovers original values."""
        scale = torch.tensor([1.0, 2.0, -1.5, 0.0], dtype=torch.float32)
        out = deqscale2int64(scale)
        assert out.dtype == torch.int64
        # int64 stores int32 bit pattern; recover float32 via same bit pattern
        back = out.numpy().astype(np.int32).view(np.float32)
        back_t = torch.from_numpy(back.copy())
        torch.testing.assert_close(scale, back_t)


class TestDeqscale2Int64ByDtype:
    """Tests for deqscale2int64_by_dtype."""

    def test_is_bf16_true_returns_unchanged(self):
        """When is_bf16=True, return scale as-is (float32)."""
        scale = torch.randn(4).float()
        out = deqscale2int64_by_dtype(scale, is_bf16=True)
        assert out is scale
        assert out.dtype == torch.float32

    def test_is_bf16_false_returns_int64(self):
        """When is_bf16=False, return deqscale2int64(scale)."""
        scale = torch.randn(4).float()
        out = deqscale2int64_by_dtype(scale, is_bf16=False)
        expected = deqscale2int64(scale)
        assert out.dtype == torch.int64
        torch.testing.assert_close(out, expected)
