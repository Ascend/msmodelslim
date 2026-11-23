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

from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from pydantic import BaseModel, model_validator

import msmodelslim.core.quantization_schema as qir
from msmodelslim.core.quantization_schema.api import calculate_qparam, quantize, dequantize
from msmodelslim.core.quantization_schema.qal import QABCRegistry, QDType, QParam, QScheme, QScope, QStorage
from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
from msmodelslim.utils.exception import SpecError, SchemaValidateError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.distributed.task_scheduler import (
    TaskExecutionRecord,
    TaskSyncContext,
)
from msmodelslim.utils.distributed.task_scheduler.sync import broadcast_tensor_process_group_safe
from msmodelslim.core.quantization_schema.utils import pad_and_reshape, undo_pad_and_reshape
from ..base import AutoWeightQuantizer, QConfig


# hif4_sweep: HiF4 W4A4 per-block E6M2 offset sweep search 权重量化方法。
#
# 算法原理：
#
# 标准的 HiF4 minmax 方法通过 max_lv1 来计算 scale_factor：
#   scale_factor = max_lv1 / 7
# 然后依次做 BF16 截断 → 7-bit 尾数 round → 2-bit 尾数 round（E6M2 编码）。
# 由于一个hif4 block有64个数，仅通过其中AbsMax值来确定 E6M2 的量化路径，其量化结果未必是最优的。
#
# hif4_sweep 的核心思想：
# 根据推导，理论最优的 lv1-scale 必定处于 [scale_factor * 7/10, scale_factor * 7/4] 区间内,
# 对应到 scale_factor 的 E6M2 编码空间上的 offset 为 [-3, 4] 。
# 对每个 block，以 minmax 确定的 E6M2 编码为中心，在
# [sweep_lower, sweep_upper] 范围内遍历所有 offset：
#   e6m2_candidate = clamp(base_e6m2 + offset, 1, 254)
# 对每个 candidate 逐一量化 lv-2,lv-3 scale 和 S1P2 ，
# 并比较量化 MSE，选择 MSE 最小的量化结果。
#
# 设计特点：
# - Weight 量化：惰性量化，量化延迟到 forward() 首次调用时执行（once-only）。
#   支持多卡 DTS 加速：owner rank 执行 sweep 搜索后通过 distributed_sync
#   广播量化结果到非 owner rank，避免重复搜索。
# - E6M2 编解码函数（float_to_e6m2 / e6m2_to_float）定义在本文件中。
# - 默认搜索范围：W [-3, 4]。
# ============================================================================


# ============================================================================
# E6M2 Encoding/Decoding
# ============================================================================


def float_to_e6m2(scale_factor: torch.Tensor) -> torch.Tensor:
    """Convert E6M2-rounded float32 tensor to uint8 E6M2 encoding.

    E6M2 format: 6-bit exponent (bias=48) + 2-bit mantissa = 8 bits total.
    Value = (1 + M/4) * 2^(E - 48), where E = exp_e6m2, M = mant_2bit.

    The input tensor should already be E6M2-representable (i.e., rounded
    through the bf16→fp32→7bit→2bit pipeline in calculate_hifx_qparam).

    Args:
        scale_factor: float32 tensor with E6M2-representable values.

    Returns:
        uint8 tensor with E6M2 encoding, same shape as input.
    """
    val = scale_factor.contiguous()
    bits = val.view(torch.int32)
    exp_f32 = ((bits >> 23) & 0xFF).to(torch.int32)
    mant_2bit = ((bits >> 21) & 0x03).to(torch.int32)
    exp_e6m2 = (exp_f32 - 79).clamp(0, 63)
    return ((exp_e6m2 << 2) | mant_2bit).to(torch.uint8)


def e6m2_to_float(e6m2_val: torch.Tensor) -> torch.Tensor:
    """Convert uint8 E6M2 encoding to float32 tensor.

    Args:
        e6m2_val: uint8 tensor with E6M2 encoding.

    Returns:
        float32 tensor, same shape as input.
    """
    e6m2_int = e6m2_val.to(torch.int32)
    exp_e6m2 = (e6m2_int >> 2) & 0x3F
    mant_2bit = e6m2_int & 0x03
    exp_f32 = exp_e6m2 + 79
    valid = (e6m2_int != 0) & (e6m2_int != 0xFF) & (exp_f32 > 0) & (exp_f32 < 255)
    f32_bits = torch.where(
        valid,
        (exp_f32 << 23) | (mant_2bit << 21),
        torch.zeros_like(exp_f32),
    )
    return f32_bits.view(torch.float32)


# ============================================================================
# Scale Factor Computation (matching calculate_hifx_qparam)
# ============================================================================


def _compute_base_scale_factor(max_lv1: torch.Tensor) -> torch.Tensor:
    """Compute E6M2-encoded scale_factor from max_lv1, matching calculate_hifx_qparam.

    This replicates the scale_factor computation in calculate_hifx_qparam
    so we can obtain the base E6M2 encoding before sweep search.
    scale_ratio is fixed to 1.0 (hif4_sweep does not tune this parameter;
    the search space is already covered by the E6M2 offset sweep).

    BF16 truncation (.to(bfloat16).to(float32)) is applied to align with
    hardware behavior: HiF4 hardware computes scale_factor in BF16 precision,
    so the software path must match to avoid deployment divergence.
    """
    div7 = torch.ones_like(max_lv1) / 7.0
    div7 = div7.to(torch.bfloat16).to(torch.float32)
    scale_factor = max_lv1 * div7
    scale_factor = scale_factor.to(torch.bfloat16).to(torch.float32)
    scale_factor = torch.clip(scale_factor, min=2 ** (-48), max=49152)
    # 7-bit mantissa rounding
    e_sf = torch.floor(torch.log2(scale_factor))
    mant_sf = scale_factor / (2**e_sf) * (2**7)
    scale_factor = torch.round(mant_sf) / (2**7) * (2**e_sf)
    # 2-bit mantissa rounding (E6M2 encoding)
    e_sf = torch.floor(torch.log2(scale_factor))
    scale_factor = torch.round(scale_factor * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)
    return scale_factor


# ============================================================================
# Sweep Search Core
# ============================================================================


def _compute_block_mse(
    abs_data: torch.Tensor,
    candidate_scale_factor: torch.Tensor,
    max_lv2: torch.Tensor,
    max_lv3: torch.Tensor,
) -> torch.Tensor:
    """Compute per-block MSE for a given scale_factor candidate.

    Args:
        abs_data: [N, 8, 2, 4] absolute values in HiF4 block layout.
        candidate_scale_factor: [N] candidate scale_factor per block.
        max_lv2: [N, 8] per-sub-block max for lv2 scaling.
        max_lv3: [N, 8, 2] per-sub-sub-block max for lv3 scaling.

    Returns:
        [N] per-block MSE.
    """
    clip2 = 4.0
    clip3 = 2.0

    # Step 1: rec_sf with BF16 truncation (matches calculate_hifx_qparam)
    rec_sf = (1.0 / candidate_scale_factor).to(torch.bfloat16).to(torch.float32)  # [N]

    # Step 2: scale_lv2 — shape [N, 8, 1, 1] (for broadcast with abs_data [N, 8, 2, 4])
    scale_lv2 = max_lv2 * rec_sf.unsqueeze(-1)  # [N, 8]
    scale_lv2 = torch.exp2((scale_lv2.clip(0, clip2) / clip2).floor())  # [N, 8]
    scale_lv2_4d = scale_lv2.unsqueeze(-1).unsqueeze(-1)  # [N, 8, 1, 1]

    # Step 3: scale_lv3 — shape [N, 8, 2, 1]
    scale_lv3 = max_lv3 * rec_sf.unsqueeze(-1).unsqueeze(-1) / scale_lv2.unsqueeze(-1)  # [N, 8, 2]
    scale_lv3 = torch.exp2((scale_lv3.clip(0, clip3) / clip3).floor())  # [N, 8, 2]
    scale_lv3_4d = scale_lv3.unsqueeze(-1)  # [N, 8, 2, 1]

    # Step 4: scale_factor_4d — shape [N, 1, 1, 1]
    sf_4d = candidate_scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1, 1]

    # Step 5: x_norm = |x| / total_scale (matches hif4_per_block_quantize order:
    #   inp / scale_lv2 / scale_lv3 * rec_sf, which equals inp / total_scale)
    x_norm = abs_data / scale_lv2_4d / scale_lv3_4d * rec_sf.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Step 6: 3-bit mantissa quantization (mbits=3)
    # Matches hif4_per_block_quantize: floor(x * 2^(mbits-1) + 0.5) / 2^(mbits-1)
    mantissa = torch.floor(x_norm * 4 + 0.5) / 4
    mantissa = torch.clamp(mantissa, max=2 - 2 ** (-2))

    # Step 7: reconstructed (matches hif4_per_block_dequantize: quantized * scale_lv3 * scale_lv2 * scale_factor)
    reconstructed = mantissa * scale_lv3_4d * scale_lv2_4d * sf_4d

    # Step 8: MSE per block (sum over all 64 elements)
    return torch.sum((abs_data - reconstructed) ** 2, dim=(-1, -2, -3))


def sweep_search_best_offset(
    block_data: torch.Tensor,
    base_e6m2: torch.Tensor,
    sweep_lower: int,
    sweep_upper: int,
    max_lv2: torch.Tensor,
    max_lv3: torch.Tensor,
) -> torch.Tensor:
    """Search for the best E6M2 offset per block that minimizes MSE.

    Args:
        block_data: [N, 8, 2, 4] block data in HiF4 layout.
        base_e6m2: [N] uint8 base E6M2 encoding of scale_factor.
        sweep_lower: Lower bound of offset search range.
        sweep_upper: Upper bound of offset search range.
        max_lv2: [N, 8] per-sub-block max for lv2 scaling.
        max_lv3: [N, 8, 2] per-sub-sub-block max for lv3 scaling.

    Returns:
        [N] int8 tensor of best offsets per block.
    """
    abs_data = block_data.abs()
    N = abs_data.shape[0]
    best_offsets = torch.zeros(N, dtype=torch.int8, device=block_data.device)
    best_mse = torch.full((N,), float('inf'), dtype=torch.float32, device=block_data.device)

    for offset in range(sweep_lower, sweep_upper + 1):
        candidate_e6m2 = (base_e6m2.to(torch.int32) + offset).clamp(1, 254).to(torch.uint8)
        candidate_scale = e6m2_to_float(candidate_e6m2)
        # Zero scale means invalid encoding
        valid = candidate_scale > 0
        mse = torch.where(
            valid,
            _compute_block_mse(abs_data, candidate_scale, max_lv2, max_lv3),
            torch.full_like(best_mse, float('inf')),
        )
        improved = mse < best_mse
        best_mse = torch.where(improved, mse, best_mse)
        best_offsets = torch.where(
            improved, torch.tensor(offset, dtype=torch.int8, device=block_data.device), best_offsets
        )

    return best_offsets


# ============================================================================
# Config
# ============================================================================


class Hif4SweepExtConfig(BaseModel):
    """Structured config for hif4_sweep quantization method."""

    model_config = {"extra": "forbid"}

    axes: Union[int, List[int]] = -1
    w_sweep_lower: int = -3
    w_sweep_upper: int = 4

    @model_validator(mode='after')
    def validate_sweep_range(self):
        if self.w_sweep_lower < -3 or self.w_sweep_upper > 4:
            raise ValueError(f"w_sweep range [{self.w_sweep_lower}, {self.w_sweep_upper}] exceeds valid range [-3, 4].")
        if self.w_sweep_lower > self.w_sweep_upper:
            raise ValueError(f"w_sweep_lower ({self.w_sweep_lower}) must be <= w_sweep_upper ({self.w_sweep_upper}).")
        return self


# ============================================================================
# Weight Quantizer
# ============================================================================


@QABCRegistry.multi_register(
    dispatch_key=[
        (qir.hif4_per_block_sym, "hif4_sweep"),
    ],
    abc_type=AutoWeightQuantizer,
)
@logger_setter()
class HiFxWeightPerBlockSweep(AutoWeightQuantizer):
    """HiF4 W4A4 weight quantizer with E6M2 offset sweep search.

    For each block of 64 elements, searches for the E6M2 offset that
    minimizes quantization MSE within [w_sweep_lower, w_sweep_upper].

    量化延迟到 forward() 首次调用时执行（惰性量化）。
    基于 DTS 的 Data-Free Weight-Only 量化通过 invoke forward 触发
    量化参数计算；将量化统一放在 _quantize() 中可支持多卡加速场景下
    由 forward 统一触发量化入口，避免多卡间逻辑分支重复。
    """

    def __init__(self, config: QConfig):
        super().__init__()
        self.config = config
        self.sweep_cfg = Hif4SweepExtConfig(**config.ext)
        if not isinstance(self.sweep_cfg.axes, (int, list)):
            raise SchemaValidateError(f"Invalid value for 'axes': {self.sweep_cfg.axes}. Expected int or list[int].")
        self.block_size = config.dtype.hifx_finfo.block_size
        self.weight: Optional[QStorage] = None
        self.bias: Optional[torch.Tensor] = None
        self.w_q_param: Optional[QParam] = None
        self.w_q_storage: Optional[QStorage] = None
        self.is_quantized = False
        self.orig_shape = None
        self.orig_dtype = None

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.is_quantized:
            if self.weight is None:
                raise SpecError("No weight was set", action="please call init_weight first")
            self._quantize()
            del self.weight
            self.weight = None
            self.is_quantized = True

        weight_q, axes, padc, dim0 = pad_and_reshape(
            self.w_q_storage.value, self.orig_shape, self.sweep_cfg.axes, self.config.dtype.hifx_finfo
        )
        weight_q_dq = dequantize(QStorage(QDType.FLOAT, weight_q), self.w_q_param)
        kwargs = {"padc": padc, "padded_shape": weight_q_dq.value.shape, "orig_shape": self.orig_shape, "dim0": dim0}
        weight_q_dq.value = undo_pad_and_reshape(weight_q_dq.value.to(self.orig_dtype), axes, **kwargs)
        return weight_q_dq.value

    def init_weight(self, weight: QStorage, bias: Optional[torch.Tensor] = None) -> None:
        self.weight = weight
        self.bias = bias

    def _quantize(self) -> None:
        axes = self.sweep_cfg.axes
        minmax_config = MinMaxBlockObserverConfig(axes=axes)
        minmax_block_observer = MsMinMaxBlockObserver(minmax_config)
        weight_value = self.weight.value.detach()
        self.orig_shape, self.orig_dtype = weight_value.shape, weight_value.dtype
        weight_value = weight_value.to(torch.float32)

        weight_value, axes, padc, dim0 = pad_and_reshape(
            weight_value, self.orig_shape, axes, self.config.dtype.hifx_finfo
        )

        # Step 1: Compute 3-level hierarchy max (same as minmax)
        weight_level = torch.abs(weight_value)
        max_levels = {}
        quant_level = 3
        for ax in range(axes, axes - 3, -1):
            minmax_block_observer.update(weight_level, sync=False, shared_exp_axes=[ax])
            _, max_lv = minmax_block_observer.get_min_max()
            max_levels[quant_level] = max_lv
            quant_level -= 1
            weight_level = max_lv

        # Step 2: Sweep search for best offset
        base_scale_factor = _compute_base_scale_factor(max_levels[1])
        base_e6m2 = float_to_e6m2(base_scale_factor)

        # Flatten data to [N, 8, 2, 4] for search (HiF4 block layout)
        flat_data = weight_value.reshape(-1, 8, 2, 4)
        flat_base_e6m2 = base_e6m2.reshape(-1)
        flat_max_lv2 = max_levels[2].reshape(-1, 8)
        flat_max_lv3 = max_levels[3].reshape(-1, 8, 2)

        best_offsets = sweep_search_best_offset(
            flat_data,
            flat_base_e6m2,
            sweep_lower=self.sweep_cfg.w_sweep_lower,
            sweep_upper=self.sweep_cfg.w_sweep_upper,
            max_lv2=flat_max_lv2,
            max_lv3=flat_max_lv3,
        )
        # Reshape best_offsets back to match scale_factor shape
        best_offsets = best_offsets.reshape(base_e6m2.shape)

        # Step 3: Compute QParam (standard minmax, no best_offsets)
        self.w_q_param = calculate_qparam(
            min_val=None,
            max_val=None,
            q_dtype=QDType(self.config.dtype),
            q_scope=QScope(self.config.scope),
            symmetric=self.config.symmetric,
            max_levels=max_levels,
            axes=axes,
            scale_ratio=1.0,
        )

        # Step 4: Apply sweep offsets to scale_factor and recompute lv2/lv3
        scale_factor = self.w_q_param.ext["weight_scale_factor"]
        e6m2_encoded = float_to_e6m2(scale_factor)
        e6m2_shifted = (e6m2_encoded.to(torch.int32) + best_offsets.to(torch.int32)).clamp(1, 254).to(torch.uint8)
        scale_factor = e6m2_to_float(e6m2_shifted)
        self.w_q_param.ext["weight_scale_factor"] = scale_factor

        rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(torch.float32)
        clip2, clip3 = 4.0, 2.0
        scale_lv2 = max_levels[2] * rec_sf
        scale_lv2 = torch.exp2((scale_lv2.clip(0, clip2) / clip2).floor())
        scale_lv3 = torch.exp2(((max_levels[3] * rec_sf / scale_lv2).clip(0, clip3) / clip3).floor())
        self.w_q_param.ext["weight_scale_lv2"] = scale_lv2
        self.w_q_param.ext["weight_scale_lv3"] = scale_lv3

        self.w_q_param.ext['w_sweep_offsets'] = best_offsets

        self.w_q_storage = quantize(QStorage(QDType.FLOAT, weight_value), self.w_q_param)
        kwargs = {"padc": padc, "padded_shape": weight_value.shape, "orig_shape": self.orig_shape, "dim0": dim0}
        self.w_q_storage.value = undo_pad_and_reshape(self.w_q_storage.value.to(self.orig_dtype), axes, **kwargs)

        get_logger().info(
            "hif4_sweep weight: search range=[%d, %d], offset distribution: %s",
            self.sweep_cfg.w_sweep_lower,
            self.sweep_cfg.w_sweep_upper,
            _offset_stats(best_offsets),
        )

    def get_q_storage(self) -> QStorage:
        if self.w_q_storage is None:
            _ = self.forward(None)
        return self.w_q_storage

    def get_q_param(self) -> QParam:
        if self.w_q_param is None:
            _ = self.forward(None)
        return self.w_q_param

    def is_data_free(self) -> bool:
        return True

    def support_distributed(self) -> bool:
        return True

    # ── DTSMixin: distributed_sync ──────────────────────────────────────

    def distributed_sync(self, record: TaskExecutionRecord, sync_ctx: TaskSyncContext) -> None:
        """分布式同步：owner rank broadcast 量化状态到所有 rank。"""
        if not dist.is_initialized() or sync_ctx.world_size <= 1:
            return
        _broadcast_sweep_state(self, record.executor_rank)

    def load_quantized_from_broadcast_tensors(
        self,
        scale_factor: torch.Tensor,
        scale_lv2: torch.Tensor,
        scale_lv3: torch.Tensor,
        w_sweep_offsets: torch.Tensor,
        w_q_storage_value: torch.Tensor,
        orig_shape: torch.Size,
        orig_dtype: torch.dtype,
    ) -> None:
        """在非 owner rank 上重建量化状态（与 forward 首次量化后一致）。"""
        scheme = QScheme(
            scope=QScope(self.config.scope),
            dtype=QDType(self.config.dtype),
            symmetric=self.config.symmetric,
        )
        self.w_q_param = QParam(
            scheme=scheme,
            ext={
                "weight_scale_factor": scale_factor,
                "weight_scale_lv2": scale_lv2,
                "weight_scale_lv3": scale_lv3,
                "w_sweep_offsets": w_sweep_offsets,
            },
        )
        self.w_q_storage = QStorage(dtype=QDType.HiF4, value=w_q_storage_value)
        self.is_quantized = True
        self.orig_shape = orig_shape
        self.orig_dtype = orig_dtype
        if self.weight is not None:
            del self.weight
            self.weight = None


# ============================================================================
# DTS Broadcast Helper
# ============================================================================


def _broadcast_sweep_state(wq: "HiFxWeightPerBlockSweep", owner_rank: int) -> None:
    """Broadcast 权重量化器状态从 owner rank 到所有 rank。

    广播 5 个张量：weight_scale_factor, weight_scale_lv2, weight_scale_lv3,
    w_sweep_offsets, w_q_storage.value，以及元数据（shape/dtype/orig_shape/orig_dtype）。
    """
    rank = dist.get_rank()

    meta: List[Dict[str, Any]] = [{}]
    if rank == owner_rank:
        if not wq.is_quantized:
            raise SpecError(
                "Cannot broadcast quantizer state: weight quantizer has not been initialized. "
                "Call forward() first to trigger quantization."
            )
        sf = wq.w_q_param.ext["weight_scale_factor"]
        lv2 = wq.w_q_param.ext["weight_scale_lv2"]
        lv3 = wq.w_q_param.ext["weight_scale_lv3"]
        offsets = wq.w_q_param.ext["w_sweep_offsets"]
        storage = wq.w_q_storage.value
        meta[0] = {
            "sf_shape": list(sf.shape),
            "sf_dtype": str(sf.dtype),
            "lv2_shape": list(lv2.shape),
            "lv2_dtype": str(lv2.dtype),
            "lv3_shape": list(lv3.shape),
            "lv3_dtype": str(lv3.dtype),
            "offsets_shape": list(offsets.shape),
            "offsets_dtype": str(offsets.dtype),
            "storage_shape": list(storage.shape),
            "storage_dtype": str(storage.dtype),
            "orig_shape": list(wq.orig_shape),
            "orig_dtype": str(wq.orig_dtype),
        }
    dist.broadcast_object_list(meta, src=owner_rank)

    if rank == owner_rank:
        sf = wq.w_q_param.ext["weight_scale_factor"].contiguous()
        lv2 = wq.w_q_param.ext["weight_scale_lv2"].contiguous()
        lv3 = wq.w_q_param.ext["weight_scale_lv3"].contiguous()
        offsets = wq.w_q_param.ext["w_sweep_offsets"].contiguous()
        storage = wq.w_q_storage.value.contiguous()
    else:
        m = meta[0]
        sf = torch.empty(m["sf_shape"], dtype=getattr(torch, m["sf_dtype"].split(".")[-1]))
        lv2 = torch.empty(m["lv2_shape"], dtype=getattr(torch, m["lv2_dtype"].split(".")[-1]))
        lv3 = torch.empty(m["lv3_shape"], dtype=getattr(torch, m["lv3_dtype"].split(".")[-1]))
        offsets = torch.empty(m["offsets_shape"], dtype=getattr(torch, m["offsets_dtype"].split(".")[-1]))
        storage = torch.empty(m["storage_shape"], dtype=getattr(torch, m["storage_dtype"].split(".")[-1]))

    broadcast_tensor_process_group_safe(sf, src=owner_rank)
    broadcast_tensor_process_group_safe(lv2, src=owner_rank)
    broadcast_tensor_process_group_safe(lv3, src=owner_rank)
    broadcast_tensor_process_group_safe(offsets, src=owner_rank)
    broadcast_tensor_process_group_safe(storage, src=owner_rank)

    if rank != owner_rank:
        m = meta[0]
        orig_shape = torch.Size(m["orig_shape"])
        orig_dtype = getattr(torch, m["orig_dtype"].split(".")[-1])
        wq.load_quantized_from_broadcast_tensors(sf, lv2, lv3, offsets, storage, orig_shape, orig_dtype)


# ============================================================================
# Helpers
# ============================================================================


def _offset_stats(offsets: torch.Tensor) -> str:
    """Format offset distribution for logging."""
    unique, counts = torch.unique(offsets, return_counts=True)
    total = offsets.numel()
    parts = []
    for u, c in zip(unique.tolist(), counts.tolist()):
        parts.append(f"{int(u)}:{c / total * 100:.1f}%")
    return " ".join(parts)
