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

# pylint: disable=attribute-defined-outside-init

import pytest
import torch
from unittest import mock

from msmodelslim.core.quantizer.base import QConfig, AutoWeightQuantizer
from msmodelslim.core.quantization_schema.qal import QDType, QScope, QStorage
from msmodelslim.core.quantizer.impl.hif4_sweep import (
    Hif4SweepExtConfig,
    HiFxWeightPerBlockSweep,
    _compute_base_scale_factor,
    _compute_block_mse,
    sweep_search_best_offset,
    _broadcast_sweep_state,
)
from msmodelslim.utils.exception import SpecError, SchemaValidateError


def _make_hif4_sweep_qconfig(**ext_kwargs) -> QConfig:
    return QConfig(
        dtype=QDType.HiF4,
        scope=QScope.PER_BLOCK,
        symmetric=True,
        method="hif4_sweep",
        ext=ext_kwargs,
    )


# ============================================================================
# TestHif4SweepExtConfig
# ============================================================================


class TestHif4SweepExtConfig:
    """Test suite for Hif4SweepExtConfig — config model for hif4_sweep parameters."""

    # ---------- 正常情形 ----------

    def test_parse_returns_defaults_when_empty_input(self):
        """正常：空输入应返回全部默认值。"""
        cfg = Hif4SweepExtConfig.model_validate({})
        assert cfg.axes == -1
        assert cfg.w_sweep_lower == -3
        assert cfg.w_sweep_upper == 4

    def test_parse_sets_custom_values_when_valid_input(self):
        """正常：合法自定义输入应正确覆盖默认值。"""
        cfg = Hif4SweepExtConfig.model_validate({"axes": -1, "w_sweep_lower": -2, "w_sweep_upper": 3})
        assert cfg.w_sweep_lower == -2
        assert cfg.w_sweep_upper == 3

    def test_parse_sets_axes_when_list_provided(self):
        """正常：axes 接受 list 类型。"""
        cfg = Hif4SweepExtConfig.model_validate({"axes": [-1, -2]})
        assert cfg.axes == [-1, -2]

    # ---------- 边界情形 ----------

    def test_parse_accepts_when_w_sweep_lower_equals_negative3(self):
        """边界：w_sweep_lower = -3 应被接受。"""
        cfg = Hif4SweepExtConfig.model_validate({"w_sweep_lower": -3})
        assert cfg.w_sweep_lower == -3

    def test_parse_accepts_when_w_sweep_upper_equals_4(self):
        """边界：w_sweep_upper = 4 应被接受。"""
        cfg = Hif4SweepExtConfig.model_validate({"w_sweep_upper": 4})
        assert cfg.w_sweep_upper == 4

    def test_parse_accepts_when_lower_equals_upper(self):
        """边界：w_sweep_lower == w_sweep_upper 应被接受（单候选等价 minmax）。"""
        cfg = Hif4SweepExtConfig.model_validate({"w_sweep_lower": 0, "w_sweep_upper": 0})
        assert cfg.w_sweep_lower == 0
        assert cfg.w_sweep_upper == 0

    # ---------- 异常情形 ----------

    def test_parse_raises_when_w_sweep_lower_below_negative3(self):
        """异常：w_sweep_lower < -3 应抛出 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            Hif4SweepExtConfig.model_validate({"w_sweep_lower": -4})

    def test_parse_raises_when_w_sweep_upper_above_4(self):
        """异常：w_sweep_upper > 4 应抛出 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            Hif4SweepExtConfig.model_validate({"w_sweep_upper": 5})

    def test_parse_raises_when_lower_greater_than_upper(self):
        """异常：w_sweep_lower > w_sweep_upper 应抛出 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            Hif4SweepExtConfig.model_validate({"w_sweep_lower": 2, "w_sweep_upper": 1})

    def test_parse_raises_when_extra_field_provided(self):
        """异常：传入 a_sweep_lower 等不允许的字段应报错（extra=forbid）。"""
        with pytest.raises(SchemaValidateError):
            Hif4SweepExtConfig.model_validate({"a_sweep_lower": 0})

    def test_parse_raises_when_clip2_provided(self):
        """异常：传入 clip2 应报错（已从 config 中移除）。"""
        with pytest.raises(SchemaValidateError):
            Hif4SweepExtConfig.model_validate({"clip2": 4})


# ============================================================================
# TestComputeBaseScaleFactor
# ============================================================================


class TestComputeBaseScaleFactor:
    """Test suite for _compute_base_scale_factor."""

    # ---------- 正常情形 ----------

    def test_compute_returns_finite_when_positive_input(self):
        """正常：正数输入应返回有限的 scale_factor。"""
        max_lv1 = torch.tensor([1.0, 10.0, 100.0])
        result = _compute_base_scale_factor(max_lv1)
        assert result.shape == max_lv1.shape
        assert torch.all(torch.isfinite(result))
        assert torch.all(result > 0)

    def test_compute_returns_e6m2_representable_when_typical_input(self):
        """正常：结果应可被 float_to_e6m2 正确编码。"""
        from msmodelslim.core.quantizer.impl.hif4_sweep import float_to_e6m2, e6m2_to_float

        max_lv1 = torch.tensor([7.0, 14.0, 21.0])
        result = _compute_base_scale_factor(max_lv1)
        encoded = float_to_e6m2(result)
        decoded = e6m2_to_float(encoded)
        assert torch.allclose(result, decoded, rtol=1e-6)

    # ---------- 边界情形 ----------

    def test_compute_returns_valid_when_very_small_input(self):
        """边界：极小输入应返回合法的 scale_factor。"""
        max_lv1 = torch.tensor([1e-10])
        result = _compute_base_scale_factor(max_lv1)
        assert torch.all(result > 0)
        assert torch.all(torch.isfinite(result))


# ============================================================================
# TestComputeBlockMse
# ============================================================================


class TestComputeBlockMse:
    """Test suite for _compute_block_mse."""

    # ---------- 正常情形 ----------

    def test_compute_returns_non_negative_when_valid_input(self):
        """正常：MSE 应为非负值。"""
        abs_data = torch.rand(4, 8, 2, 4)
        candidate_sf = torch.tensor([0.1, 0.2, 0.3, 0.4])
        max_lv2 = torch.rand(4, 8)
        max_lv3 = torch.rand(4, 8, 2)
        mse = _compute_block_mse(abs_data, candidate_sf, max_lv2, max_lv3)
        assert mse.shape == (4,)
        assert torch.all(mse >= 0)

    def test_compute_returns_zero_when_perfect_reconstruction(self):
        """正常：当数据恰好为量化格点时 MSE 应接近 0。"""
        # 构造恰好落在量化格点上的数据
        candidate_sf = torch.tensor([1.0])
        max_lv2 = torch.zeros(1, 8)
        max_lv3 = torch.zeros(1, 8, 2)
        # 当 scale_lv2=1, scale_lv3=1, sf=1 时，量化格点为 0, 0.25, 0.5, 0.75, 1.0
        abs_data = torch.zeros(1, 8, 2, 4)
        abs_data[0, 0, 0, 0] = 0.5
        mse = _compute_block_mse(abs_data, candidate_sf, max_lv2, max_lv3)
        assert mse[0] < 1e-10


# ============================================================================
# TestSweepSearchBestOffset
# ============================================================================


class TestSweepSearchBestOffset:
    """Test suite for sweep_search_best_offset."""

    # ---------- 正常情形 ----------

    def test_search_returns_int8_when_valid_input(self):
        """正常：返回 int8 tensor，shape 与 block 数一致。"""
        from msmodelslim.core.quantizer.impl.hif4_sweep import float_to_e6m2

        block_data = torch.randn(8, 8, 2, 4)
        max_lv1 = block_data.abs().reshape(8, -1).max(dim=-1).values
        base_sf = _compute_base_scale_factor(max_lv1)
        base_e6m2 = float_to_e6m2(base_sf)
        max_lv2 = torch.rand(8, 8)
        max_lv3 = torch.rand(8, 8, 2)
        result = sweep_search_best_offset(block_data, base_e6m2, -3, 4, max_lv2, max_lv3)
        assert result.dtype == torch.int8
        assert result.shape == (8,)

    def test_search_returns_zero_when_single_candidate_zero(self):
        """正常：当搜索范围为 [0, 0] 时所有 offset 应为 0。"""
        from msmodelslim.core.quantizer.impl.hif4_sweep import float_to_e6m2

        block_data = torch.randn(4, 8, 2, 4)
        max_lv1 = block_data.abs().reshape(4, -1).max(dim=-1).values
        base_sf = _compute_base_scale_factor(max_lv1)
        base_e6m2 = float_to_e6m2(base_sf)
        max_lv2 = torch.rand(4, 8)
        max_lv3 = torch.rand(4, 8, 2)
        result = sweep_search_best_offset(block_data, base_e6m2, 0, 0, max_lv2, max_lv3)
        assert torch.all(result == 0)

    # ---------- 边界情形 ----------

    def test_search_returns_values_within_range_when_full_range(self):
        """边界：返回的 offset 应在搜索范围内。"""
        from msmodelslim.core.quantizer.impl.hif4_sweep import float_to_e6m2

        block_data = torch.randn(16, 8, 2, 4)
        max_lv1 = block_data.abs().reshape(16, -1).max(dim=-1).values
        base_sf = _compute_base_scale_factor(max_lv1)
        base_e6m2 = float_to_e6m2(base_sf)
        max_lv2 = torch.rand(16, 8)
        max_lv3 = torch.rand(16, 8, 2)
        result = sweep_search_best_offset(block_data, base_e6m2, -3, 4, max_lv2, max_lv3)
        assert torch.all(result >= -3)
        assert torch.all(result <= 4)


# ============================================================================
# TestHiFxWeightPerBlockSweep
# ============================================================================


class TestHiFxWeightPerBlockSweep:
    """Test suite for HiFxWeightPerBlockSweep — weight quantizer with sweep search."""

    def setup_class(self):
        self.config = _make_hif4_sweep_qconfig()
        self.in_features = 64
        self.out_features = 128

    def _make_linear_layer(self):
        from torch import nn

        return nn.Linear(self.in_features, self.out_features)

    def _make_quantizer(self):
        return HiFxWeightPerBlockSweep(self.config)

    def _make_weight_storage(self, layer=None):
        if layer is None:
            layer = self._make_linear_layer()
        return QStorage(QDType.FLOAT, layer.weight.data)

    # ---------- 正常情形 ----------

    def test_init_weight_stores_weight_when_called(self):
        """正常：init_weight 后应存储 weight 引用但不量化。"""
        quantizer = self._make_quantizer()
        weight = self._make_weight_storage()
        quantizer.init_weight(weight)
        assert quantizer.weight is not None
        assert quantizer.is_quantized is False
        assert quantizer.w_q_storage is None

    def test_forward_returns_same_shape_when_valid_input(self):
        """正常：forward 后输出 shape 应与原始权重一致。"""
        quantizer = self._make_quantizer()
        weight = self._make_weight_storage()
        quantizer.init_weight(weight)
        result = quantizer()
        assert result.shape == (self.out_features, self.in_features)

    def test_forward_quantizes_once_only_when_called_multiple_times(self):
        """正常：多次 forward 只量化一次（is_quantized 标志）。"""
        quantizer = self._make_quantizer()
        weight = self._make_weight_storage()
        quantizer.init_weight(weight)
        result1 = quantizer()
        assert quantizer.is_quantized is True
        assert quantizer.weight is None
        result2 = quantizer()
        assert torch.equal(result1, result2)

    def test_get_q_storage_triggers_forward_when_not_quantized(self):
        """正常：未量化时 get_q_storage 应触发 forward。"""
        quantizer = self._make_quantizer()
        weight = self._make_weight_storage()
        quantizer.init_weight(weight)
        q_storage = quantizer.get_q_storage()
        assert q_storage is not None
        assert quantizer.is_quantized is True

    def test_get_q_param_triggers_forward_when_not_quantized(self):
        """正常：未量化时 get_q_param 应触发 forward。"""
        quantizer = self._make_quantizer()
        weight = self._make_weight_storage()
        quantizer.init_weight(weight)
        q_param = quantizer.get_q_param()
        assert q_param is not None
        assert quantizer.is_quantized is True

    def test_w_sweep_offsets_stored_in_qparam_ext_when_quantized(self):
        """正常：量化后 w_sweep_offsets 应存入 QParam.ext。"""
        quantizer = self._make_quantizer()
        weight = self._make_weight_storage()
        quantizer.init_weight(weight)
        quantizer()
        assert "w_sweep_offsets" in quantizer.w_q_param.ext
        assert quantizer.w_q_param.ext["w_sweep_offsets"].dtype == torch.int8

    def test_is_data_free_returns_true_when_queried(self):
        """正常：权重量化器是 data-free 的。"""
        quantizer = self._make_quantizer()
        assert quantizer.is_data_free() is True

    def test_support_distributed_returns_true_when_queried(self):
        """正常：应支持分布式。"""
        quantizer = self._make_quantizer()
        assert quantizer.support_distributed() is True

    # ---------- 异常情形 ----------

    def test_forward_raises_spec_error_when_weight_not_set(self):
        """异常：未 init_weight 直接 forward 应抛出 SpecError。"""
        quantizer = self._make_quantizer()
        with pytest.raises(SpecError):
            quantizer()


# ============================================================================
# TestHiFxWeightPerBlockSweepDistributedSync
# ============================================================================


class TestHiFxWeightPerBlockSweepDistributedSync:
    """Test suite for distributed_sync and broadcast functionality."""

    def setup_class(self):
        self.config = _make_hif4_sweep_qconfig()
        self.in_features = 64
        self.out_features = 128

    def _make_quantized_quantizer(self):
        from torch import nn

        quantizer = HiFxWeightPerBlockSweep(self.config)
        layer = nn.Linear(self.in_features, self.out_features)
        weight = QStorage(QDType.FLOAT, layer.weight.data)
        quantizer.init_weight(weight)
        quantizer()
        return quantizer

    # ---------- 正常情形 ----------

    def test_distributed_sync_override_is_different_from_base(self):
        """正常：HiFxWeightPerBlockSweep 应覆盖基类的 distributed_sync。"""
        assert HiFxWeightPerBlockSweep.distributed_sync is not AutoWeightQuantizer.distributed_sync

    @mock.patch("msmodelslim.core.quantizer.impl.hif4_sweep.dist")
    def test_distributed_sync_single_rank_skips_broadcast(self, mock_dist):
        """正常：单卡环境（dist 未初始化）下 distributed_sync 不执行 broadcast。"""
        mock_dist.is_initialized.return_value = False
        from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext

        quantizer = self._make_quantized_quantizer()
        record = TaskExecutionRecord(task_id="test", executor_rank=0)
        sync_ctx = TaskSyncContext(model=quantizer, rank=0, world_size=1)
        quantizer.distributed_sync(record, sync_ctx)
        mock_dist.broadcast_object_list.assert_not_called()

    def test_load_quantized_from_broadcast_tensors_sets_quantized_state(self):
        """正常：load_quantized_from_broadcast_tensors 正确重建量化状态。"""
        src_quantizer = self._make_quantized_quantizer()
        sf = src_quantizer.w_q_param.ext["weight_scale_factor"].clone()
        lv2 = src_quantizer.w_q_param.ext["weight_scale_lv2"].clone()
        lv3 = src_quantizer.w_q_param.ext["weight_scale_lv3"].clone()
        offsets = src_quantizer.w_q_param.ext["w_sweep_offsets"].clone()
        storage = src_quantizer.w_q_storage.value.clone()
        orig_shape = src_quantizer.orig_shape
        orig_dtype = src_quantizer.orig_dtype

        new_quantizer = HiFxWeightPerBlockSweep(self.config)
        new_quantizer.load_quantized_from_broadcast_tensors(sf, lv2, lv3, offsets, storage, orig_shape, orig_dtype)
        assert new_quantizer.is_quantized is True
        assert new_quantizer.w_q_storage is not None
        assert new_quantizer.w_q_param is not None

    def test_load_quantized_from_broadcast_tensors_releases_old_weight(self):
        """正常：load 后应释放原有 weight 引用。"""
        from torch import nn

        quantizer = HiFxWeightPerBlockSweep(self.config)
        layer = nn.Linear(self.in_features, self.out_features)
        weight = QStorage(QDType.FLOAT, layer.weight.data)
        quantizer.init_weight(weight)
        assert quantizer.weight is not None

        src_quantizer = self._make_quantized_quantizer()
        sf = src_quantizer.w_q_param.ext["weight_scale_factor"].clone()
        lv2 = src_quantizer.w_q_param.ext["weight_scale_lv2"].clone()
        lv3 = src_quantizer.w_q_param.ext["weight_scale_lv3"].clone()
        offsets = src_quantizer.w_q_param.ext["w_sweep_offsets"].clone()
        storage = src_quantizer.w_q_storage.value.clone()
        orig_shape = src_quantizer.orig_shape
        orig_dtype = src_quantizer.orig_dtype

        quantizer.load_quantized_from_broadcast_tensors(sf, lv2, lv3, offsets, storage, orig_shape, orig_dtype)
        assert quantizer.weight is None
        assert quantizer.is_quantized is True

    # ---------- 异常情形 ----------

    @mock.patch("msmodelslim.core.quantizer.impl.hif4_sweep.dist")
    def test_broadcast_raises_spec_error_when_not_quantized(self, mock_dist):
        """异常：未量化时 broadcast 应抛出 SpecError。"""
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        quantizer = HiFxWeightPerBlockSweep(self.config)
        with pytest.raises(SpecError, match="Cannot broadcast quantizer state"):
            _broadcast_sweep_state(quantizer, owner_rank=0)
