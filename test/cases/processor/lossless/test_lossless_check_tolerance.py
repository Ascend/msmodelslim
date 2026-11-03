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

import pytest
import torch

from msmodelslim.processor.lossless.metrics.builtins import (
    _infer_tolerance,
    _max_relative_diff,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_tolerance_accepts_one_representable_rounding_step(dtype):
    reference = torch.tensor([[-2.0, 0.5, 1.0]], dtype=dtype)
    eps = torch.finfo(dtype).eps
    processed = (reference.float() + eps).to(dtype)
    atol, rtol = _infer_tolerance(reference, processed)

    assert torch.allclose(reference.float(), processed.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_tolerance_rejects_deleted_rotation_scale_mismatch(dtype):
    reference = torch.tensor([[1.0, 0.0]], dtype=dtype)
    processed = torch.tensor([[0.0, 1.0]], dtype=dtype)
    atol, rtol = _infer_tolerance(reference, processed)

    assert not torch.allclose(reference.float(), processed.float(), atol=atol, rtol=rtol)


def test_zero_relative_metric_is_finite_and_stable():
    reference = torch.zeros(4, dtype=torch.float32)
    processed = torch.zeros(4, dtype=torch.float32)

    metric = _max_relative_diff(reference, processed)

    assert metric == 0.0
    assert torch.isfinite(torch.tensor(metric))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e3])
def test_tolerance_accepts_one_representable_step_across_scales(dtype, scale):
    reference = torch.tensor([scale], dtype=dtype)
    storage_dtype = torch.int32 if dtype == torch.float32 else torch.int16
    processed = (reference.view(storage_dtype) + 1).view(dtype)
    atol, rtol = _infer_tolerance(reference, processed)

    assert torch.allclose(reference.float(), processed.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e3])
def test_tolerance_rejects_material_error_across_scales(dtype, scale):
    reference = torch.tensor([scale], dtype=dtype)
    delta = max(0.1, abs(scale) * 0.1)
    processed = torch.tensor([scale + delta], dtype=dtype)
    atol, rtol = _infer_tolerance(reference, processed)

    assert not torch.allclose(reference.float(), processed.float(), atol=atol, rtol=rtol)
