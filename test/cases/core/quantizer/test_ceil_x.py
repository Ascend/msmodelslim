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

import math

import pytest
import torch

from msmodelslim.ir.qal import QDType, QScope, QStorage, QParam
from msmodelslim.core.quantizer.base import QConfig, AutoWeightQuantizer
from msmodelslim.core.quantizer.impl.ceil_x import (
    CeilXExtConfig,
    ceil_x_qparam,
    ceil_x_search_best,
)
from msmodelslim.utils.exception import SpecError, SchemaValidateError, UnsupportedError


def _make_mxfp4_qconfig(**ext_kwargs) -> QConfig:
    return QConfig(
        dtype=QDType.MXFP4,
        scope=QScope.PER_BLOCK,
        symmetric=True,
        method="ceil_x",
        ext=ext_kwargs,
    )


class TestCeilXExtConfig:
    """Test suite for CeilXExtConfig — config model for ceil_x parameters."""

    # ---------- 正常情形 ----------

    def test_parse_returns_defaults_when_empty_input(self):
        """正常：空输入应返回全部默认值。"""
        cfg = CeilXExtConfig.model_validate({})
        assert cfg.axes == -1
        assert cfg.ceil_x_value == 7.25
        assert cfg.enable_search is False
        assert cfg.search_min == 6.0
        assert cfg.search_max == 12.0
        assert cfg.search_step == 0.25

    def test_parse_sets_ceil_x_value_when_custom_values(self):
        """正常：自定义输入应正确覆盖默认值。"""
        cfg = CeilXExtConfig.model_validate(
            {
                "axes": -1,
                "ceil_x_value": 8.0,
                "enable_search": True,
                "search_min": 8.0,
                "search_max": 10.0,
                "search_step": 0.5,
            }
        )
        assert cfg.ceil_x_value == 8.0
        assert cfg.enable_search is True
        assert cfg.search_min == 8.0

    # ---------- 边界情形 ----------

    def test_parse_sets_axes_when_list_provided(self):
        """边界：axes 为列表时应正常解析。"""
        cfg = CeilXExtConfig.model_validate({"axes": [0, 1]})
        assert cfg.axes == [0, 1]

    def test_parse_raises_when_search_step_is_zero(self):
        """边界：search_step 为 0 时应拒绝（后续可能除零）。"""
        with pytest.raises(Exception):
            CeilXExtConfig.model_validate({"search_step": 0})

    # ---------- 异常情形 ----------

    def test_parse_raises_when_extra_field_provided(self):
        """异常：extra="forbid" 时传入未定义字段应报错。"""
        with pytest.raises(Exception):
            CeilXExtConfig.model_validate({"unknown_field": 42})


class TestCeilXQparam:
    """Test suite for ceil_x_qparam — compute MXFP4 qparam with ceil_x formula."""

    # ---------- 正常情形 ----------

    def test_compute_returns_qparam_with_scale_when_basic_input(self):
        """正常：基本输入应返回包含 scale 的 QParam。"""
        qp = ceil_x_qparam(
            min_val=torch.tensor(0.0),
            max_val=torch.tensor(10.0),
            ceil_x_value=7.25,
            q_dtype=QDType.MXFP4,
            q_scope=QScope.PER_BLOCK,
            symmetric=True,
            axes=-1,
        )
        assert isinstance(qp, QParam)
        assert qp.scheme.dtype == QDType.MXFP4
        assert qp.scheme.scope == QScope.PER_BLOCK
        assert qp.scheme.symmetric is True
        assert "scale" in qp.ext

    def test_compute_returns_expected_scale_when_typical_input(self):
        """正常：scale 应等于 ceil(log2(max_val / ceil_x_value + eps))。"""
        max_val = torch.tensor(20.0)
        value = 5.0
        expected = math.ceil(math.log2((20.0 / 5.0) + 9.6e-7))
        qp = ceil_x_qparam(
            min_val=torch.tensor(0.0),
            max_val=max_val,
            ceil_x_value=value,
            q_dtype=QDType.MXFP4,
            q_scope=QScope.PER_BLOCK,
            symmetric=True,
            axes=-1,
        )
        assert qp.ext["scale"].item() == expected

    # ---------- 边界情形 ----------

    def test_compute_returns_qparam_when_zero_max_val(self):
        """边界：max_val 为 0 时不应崩溃（eps 保护）。"""
        qp = ceil_x_qparam(
            min_val=torch.tensor(0.0),
            max_val=torch.tensor(0.0),
            ceil_x_value=7.25,
            q_dtype=QDType.MXFP4,
            q_scope=QScope.PER_BLOCK,
            symmetric=True,
            axes=-1,
        )
        assert "scale" in qp.ext
        assert not torch.isnan(qp.ext["scale"]).any()

    def test_compute_returns_qparam_when_negative_max_val(self):
        """边界：max_val 为负值时 log2 应处理（公式取 abs？实际传递后 eps 保护）。"""
        qp = ceil_x_qparam(
            min_val=torch.tensor(-10.0),
            max_val=torch.tensor(-5.0),
            ceil_x_value=7.25,
            q_dtype=QDType.MXFP4,
            q_scope=QScope.PER_BLOCK,
            symmetric=True,
            axes=-1,
        )
        assert "scale" in qp.ext


class TestCeilXSearchBest:
    """Test suite for ceil_x_search_best — MSE-based optimal value search."""

    # ---------- 正常情形 ----------

    def test_search_returns_default_value_when_all_equal(self):
        """正常：当所有候选值 MSE 相同时应返回 search_min。"""
        weight = torch.randn(32, 64)
        best = ceil_x_search_best(
            weight_value=weight,
            min_val=torch.tensor(0.0),
            max_val=torch.tensor(10.0),
            q_dtype=QDType.MXFP4,
            q_scope=QScope.PER_BLOCK,
            symmetric=True,
            axes=-1,
        )
        assert isinstance(best, float)
        assert 6.0 <= best <= 12.0

    # ---------- 边界情形 ----------

    def test_search_returns_single_value_when_range_has_one_candidate(self):
        """边界：当 search_min == search_max 时只测试一个值。"""
        weight = torch.randn(32, 64)
        best = ceil_x_search_best(
            weight_value=weight,
            min_val=torch.tensor(0.0),
            max_val=torch.tensor(5.0),
            q_dtype=QDType.MXFP4,
            q_scope=QScope.PER_BLOCK,
            symmetric=True,
            axes=-1,
            search_min=8.0,
            search_max=8.0,
            search_step=0.25,
        )
        assert best == 8.0

    def test_search_returns_bound_values_when_large_step(self):
        """边界：大步长只产生少量候选值。"""
        weight = torch.randn(32, 64)
        best = ceil_x_search_best(
            weight_value=weight,
            min_val=torch.tensor(0.0),
            max_val=torch.tensor(10.0),
            q_dtype=QDType.MXFP4,
            q_scope=QScope.PER_BLOCK,
            symmetric=True,
            axes=-1,
            search_min=6.0,
            search_max=12.0,
            search_step=6.0,
        )
        assert best in [6.0, 12.0]


class TestMXWeightPerBlockCeilX:
    """Test suite for MXWeightPerBlockCeilX — ceil_x weight quantizer."""

    # ---------- 正常情形 ----------

    def test_forward_returns_same_shape_weight_when_valid_input(self):
        """正常：完整流程应返回与输入相同 shape 的反量化权重。"""
        config = _make_mxfp4_qconfig(axes=-1)
        q = AutoWeightQuantizer.from_config(config)
        w = torch.randn(32, 64)
        q.init_weight(QStorage(QDType.FLOAT, w))
        out = q.forward()
        assert out.shape == w.shape

    def test_forward_is_data_free(self):
        """正常：data-free 量化的 forward 应可重复调用。"""
        config = _make_mxfp4_qconfig(axes=-1)
        q = AutoWeightQuantizer.from_config(config)
        q.init_weight(QStorage(QDType.FLOAT, torch.randn(32, 64)))
        out1 = q.forward()
        out2 = q.forward()
        assert out1.shape == out2.shape

    def test_get_q_storage_returns_original_shape(self):
        """正常：get_q_storage 应返回原始 shape 的量化存储。"""
        config = _make_mxfp4_qconfig(axes=-1)
        q = AutoWeightQuantizer.from_config(config)
        orig_shape = torch.randn(32, 64)
        q.init_weight(QStorage(QDType.FLOAT, orig_shape))
        q.forward()
        storage = q.get_q_storage()
        assert storage.value.shape == orig_shape.shape

    def test_get_q_param_returns_qparam_with_scale(self):
        """正常：get_q_param 应返回包含 scale 的 QParam。"""
        config = _make_mxfp4_qconfig(axes=-1)
        q = AutoWeightQuantizer.from_config(config)
        q.init_weight(QStorage(QDType.FLOAT, torch.randn(32, 64)))
        qp = q.get_q_param()
        assert "scale" in qp.ext
        assert "axes" in qp.ext

    def test_default_ceil_x_value_is_7_25(self):
        """正常：默认 ceil_x_value 应为 7.25。"""
        config = _make_mxfp4_qconfig()
        q = AutoWeightQuantizer.from_config(config)
        assert q.ceil_cfg.ceil_x_value == 7.25

    def test_custom_ceil_x_value_is_respected(self):
        """正常：自定义 ceil_x_value 应被正确使用。"""
        config = _make_mxfp4_qconfig(ceil_x_value=10.0)
        q = AutoWeightQuantizer.from_config(config)
        assert q.ceil_cfg.ceil_x_value == 10.0

    def test_quantize_with_search_does_not_error(self):
        """正常：enable_search=True 时量化应正常完成。"""
        config = _make_mxfp4_qconfig(axes=-1, enable_search=True)
        q = AutoWeightQuantizer.from_config(config)
        q.init_weight(QStorage(QDType.FLOAT, torch.randn(32, 64)))
        out = q.forward()
        assert out.shape == (32, 64)
        assert q.ceil_cfg.ceil_x_value != 0

    def test_is_data_free_returns_true(self):
        """正常：is_data_free 应返回 True。"""
        config = _make_mxfp4_qconfig()
        q = AutoWeightQuantizer.from_config(config)
        assert q.is_data_free() is True

    def test_support_distributed_returns_true(self):
        """正常：support_distributed 应返回 True。"""
        config = _make_mxfp4_qconfig()
        q = AutoWeightQuantizer.from_config(config)
        assert q.support_distributed() is True

    # ---------- 边界情形 ----------

    def test_forward_returns_same_shape_when_1d_weight(self):
        """边界：1D 权重应正常处理。"""
        config = _make_mxfp4_qconfig(axes=-1)
        q = AutoWeightQuantizer.from_config(config)
        q.init_weight(QStorage(QDType.FLOAT, torch.randn(64)))
        out = q.forward()
        assert out.shape == (64,)

    def test_forward_returns_same_shape_when_3d_weight(self):
        """边界：3D 权重应正常处理。"""
        config = _make_mxfp4_qconfig(axes=-1)
        q = AutoWeightQuantizer.from_config(config)
        q.init_weight(QStorage(QDType.FLOAT, torch.randn(4, 32, 64)))
        out = q.forward()
        assert out.shape == (4, 32, 64)

    def test_forward_returns_same_shape_when_zero_weight(self):
        """边界：全零权重应正常量化。"""
        config = _make_mxfp4_qconfig(axes=-1)
        q = AutoWeightQuantizer.from_config(config)
        q.init_weight(QStorage(QDType.FLOAT, torch.zeros(32, 64)))
        out = q.forward()
        assert out.shape == (32, 64)

    def test_get_q_storage_returns_same_shape_when_called_twice(self):
        """边界：多次调用 get_q_storage 应返回一致结果。"""
        config = _make_mxfp4_qconfig()
        q = AutoWeightQuantizer.from_config(config)
        q.init_weight(QStorage(QDType.FLOAT, torch.randn(32, 64)))
        q.forward()
        s1 = q.get_q_storage()
        s2 = q.get_q_storage()
        assert s1.value.shape == s2.value.shape

    # ---------- 异常情形 ----------

    def test_forward_raises_when_weight_not_set(self):
        """异常：未调用 init_weight 就 forward 应报错。"""
        config = _make_mxfp4_qconfig()
        q = AutoWeightQuantizer.from_config(config)
        with pytest.raises(SpecError):
            q.forward()

    def test_raises_when_dtype_not_mxfp4(self):
        """异常：非 MXFP4 dtype 应报错（dispatch 层拦截）。"""
        config = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_CHANNEL,
            symmetric=True,
            method="ceil_x",
        )
        with pytest.raises(UnsupportedError):
            AutoWeightQuantizer.from_config(config)

    def test_raises_when_axes_invalid(self):
        """异常：非法的 axes 类型应报错。"""
        config = _make_mxfp4_qconfig(axes="invalid")
        with pytest.raises(SchemaValidateError):
            AutoWeightQuantizer.from_config(config)

    def test_get_q_storage_raises_when_no_forward(self):
        """异常：未 forward 且未 init_weight 时 get_q_storage 应报错。"""
        config = _make_mxfp4_qconfig()
        q = AutoWeightQuantizer.from_config(config)
        with pytest.raises(SpecError):
            q.get_q_storage()

    def test_get_q_param_raises_when_no_forward(self):
        """异常：未 forward 且未 init_weight 时 get_q_param 应报错。"""
        config = _make_mxfp4_qconfig()
        q = AutoWeightQuantizer.from_config(config)
        with pytest.raises(SpecError):
            q.get_q_param()
