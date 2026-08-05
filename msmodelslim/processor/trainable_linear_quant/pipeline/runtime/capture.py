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

# Unified block forward capture for trainable linear quant.

from __future__ import annotations

import gc
from typing import List

import torch

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.utils.logging import get_logger

from msmodelslim.processor.trainable_linear_quant.data import BlockOutput


def _output_to_cpu(output: BlockOutput) -> BlockOutput:
    if isinstance(output, (tuple, list)):
        return type(output)(o.cpu() if isinstance(o, torch.Tensor) else o for o in output)
    if isinstance(output, torch.Tensor):
        return output.cpu()
    return output


def log_npu_mem(tag: str) -> None:
    if hasattr(torch, "npu"):
        allocated = torch.npu.memory_allocated() / 1024**3
        reserved = torch.npu.memory_reserved() / 1024**3
        get_logger().info(
            "[MEM %s] allocated=%.2fGB, reserved=%.2fGB",
            tag,
            allocated,
            reserved,
        )


def _capture_block_outputs(
    request: BatchProcessRequest,
    to_cpu: bool,
    gc_per_sample: bool,
    log_device_memory: bool,
) -> List[BlockOutput]:
    if log_device_memory:
        log_npu_mem("BEFORE_FORWARD")

    outputs: List[BlockOutput] = []
    for data in request.datas:
        args, kwargs = data
        if not args and not kwargs:
            continue
        output = request.module(*args, **kwargs)
        if to_cpu:
            output = _output_to_cpu(output)
        outputs.append(output)
        if gc_per_sample:
            gc.collect()
            if hasattr(torch, "npu"):
                torch.npu.empty_cache()

    if log_device_memory:
        log_npu_mem("AFTER_FORWARD")

    return outputs


def capture_float_teacher(request: BatchProcessRequest) -> List[BlockOutput]:
    """Run float teacher forward, move outputs to CPU, and release device memory."""
    outputs = _capture_block_outputs(
        request,
        to_cpu=True,
        gc_per_sample=True,
        log_device_memory=True,
    )
    request.outputs = None
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
    log_npu_mem("AFTER_CLEANUP")
    return outputs


def capture_quant_propagation(request: BatchProcessRequest) -> List[BlockOutput]:
    """Run forward with quantized weights and keep outputs on device for propagation."""
    outputs = _capture_block_outputs(
        request,
        to_cpu=False,
        gc_per_sample=False,
        log_device_memory=False,
    )
    request.outputs = outputs
    return outputs


__all__ = ["capture_float_teacher", "capture_quant_propagation", "log_npu_mem"]
