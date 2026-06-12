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

import pytest
import torch
from pydantic import ValidationError

from msmodelslim.ir.qal.qbase import QStorage, QDType, QScheme, QScope
from msmodelslim import ir as qir
from msmodelslim.core.observer import MsMinMaxObserver
from msmodelslim.core.quantizer.base import QConfig, AutoActQuantizer, AutoWeightQuantizer
from msmodelslim.core.quantizer.impl.minmax import (
    ActPerTensorMinmax,
    ActPerTokenMinmax,
    WeightPerChannelMinmax,
    MXWeightPerBlockMinmax,
)
from msmodelslim.utils.exception import SpecError, SchemaValidateError


def to_qconfig(q_scheme: QScheme, method: str) -> QConfig:
    q_config = QConfig(
        dtype=q_scheme.dtype.value,
        scope=q_scheme.scope.value,
        symmetric=q_scheme.symmetric,
        method=method,
    )

    if q_scheme.scope == QScope.PER_GROUP:
        q_config.ext['group_size'] = 256

    return q_config


class TestActPerTensorMinmax:
    """测试Per-Tensor激活MinMax量化器"""

    def test_initialization(self):
        """测试初始化"""
        config = QConfig(dtype="int8", scope="per_tensor", method="minmax", symmetric=True)

        quantizer = ActPerTensorMinmax(config)

        assert quantizer.config == config
        assert isinstance(quantizer.minmax_observer, MsMinMaxObserver)
        assert quantizer.q_param is None

    def test_forward_then_can_get_correct_q_param(self):
        """测试前向传播并验证量化参数"""
        config = QConfig(dtype="int8", scope="per_tensor", method="minmax", symmetric=True)

        quantizer = ActPerTensorMinmax(config)

        # 测试输入
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = quantizer(x)

        # 验证q_param被设置
        q_param = quantizer.get_q_param()
        assert q_param
        assert q_param.scheme == config.to_scheme()
        assert isinstance(q_param.ext, dict)
        assert "scale" in q_param.ext
        assert "offset" in q_param.ext
        assert isinstance(q_param.ext["scale"], torch.Tensor)
        assert isinstance(q_param.ext["offset"], torch.Tensor)
        assert q_param.ext["scale"].shape == (1,)
        assert q_param.ext["offset"].shape == (1,)

        # 验证输出形状
        assert result.shape == x.shape

    def test_forward_with_batch_input(self):
        config = QConfig(dtype="int8", scope="per_token", method="minmax", symmetric=True)

        quantizer = ActPerTokenMinmax(config)

        # 测试不同形状的输入
        x = torch.randn(2, 3, 4)  # (batch, seq, hidden)
        result = quantizer(x)
        assert result.shape == x.shape

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_tensor_sym, "minmax"),
            to_qconfig(qir.int8_per_tensor_asym, "minmax"),
        ],
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        """测试通过自动量化器创建"""
        quantizer = AutoActQuantizer.from_config(qconfig)
        assert isinstance(quantizer, ActPerTensorMinmax)


class TestActPerTokenMinmax:
    """测试Per-Token激活MinMax量化器"""

    def test_initialization(self):
        """测试初始化"""
        config = QConfig(dtype="int8", scope="per_token", method="minmax", symmetric=True)

        quantizer = ActPerTokenMinmax(config)

        assert quantizer.config == config
        assert isinstance(quantizer.minmax_observer, MsMinMaxObserver)
        assert quantizer.q_param is None

    def test_forward_then_can_get_correct_q_param(self):
        """测试前向传播并验证量化参数"""
        config = QConfig(dtype="int8", scope="per_token", method="minmax", symmetric=True)

        quantizer = ActPerTokenMinmax(config)

        # 测试输入 (batch_size, seq_len, hidden_dim)
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        original_shape = x.shape

        result = quantizer(x)

        # 验证q_param被设置
        q_param = quantizer.get_q_param()
        assert q_param
        assert q_param.scheme == config.to_scheme()

        # 验证输出形状保持不变
        assert result.shape == original_shape

    def test_forward_with_batch_input(self):
        """测试不同输入形状的处理"""
        config = QConfig(dtype="int8", scope="per_token", method="minmax", symmetric=True)

        quantizer = ActPerTokenMinmax(config)

        # 测试不同形状的输入
        x1 = torch.randn(2, 3, 4)  # (batch, seq, hidden)

        result1 = quantizer(x1)

        assert result1.shape == x1.shape

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_token_sym, "minmax"),
            to_qconfig(qir.int8_per_token_asym, "minmax"),
        ],
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        """测试通过自动量化器创建"""
        quantizer = AutoActQuantizer.from_config(qconfig)
        assert isinstance(quantizer, ActPerTokenMinmax)


class TestWeightPerChannelMinmax:  # pylint: disable=attribute-defined-outside-init
    """测试Per-Channel权重量化器"""

    def setup_class(self):
        self.config = QConfig(dtype="int8", scope="per_channel", method="minmax", symmetric=True)

    def test_initialization(self):
        """测试初始化"""
        quantizer = WeightPerChannelMinmax(self.config)

        assert quantizer.config == self.config
        assert quantizer.weight is None
        assert quantizer.bias is None

    def test_init_weight_validation(self):
        """测试权重初始化验证"""
        quantizer = WeightPerChannelMinmax(self.config)

        # 测试无效权重类型
        with pytest.raises(ValidationError, match="instance of QStorage"):
            quantizer.init_weight(torch.randn(10, 20))

        # 测试无效bias类型
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        with pytest.raises(ValidationError, match="instance of Tensor"):
            quantizer.init_weight(weight, bias="invalid")

    def test_init_weight_then_forward(self):
        """测试权重初始化并前向传播"""
        quantizer = WeightPerChannelMinmax(self.config)

        # 初始化权重
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        bias = torch.randn(20)

        quantizer.init_weight(weight, bias)

        assert quantizer.weight == weight
        assert quantizer.bias is bias

        # 前向传播
        result = quantizer()

        # 验证q_param被设置
        q_param = quantizer.get_q_param()
        assert q_param
        assert q_param.scheme == self.config.to_scheme()
        assert isinstance(q_param.ext, dict)
        assert "scale" in q_param.ext
        assert "offset" in q_param.ext
        assert isinstance(q_param.ext["scale"], torch.Tensor)
        assert isinstance(q_param.ext["offset"], torch.Tensor)
        # Per-channel的scale和offset应该与输出通道数匹配
        assert q_param.ext["scale"].shape == (weight.value.shape[0],)
        assert q_param.ext["offset"].shape == (weight.value.shape[0],)

        # 验证q_storage被设置
        q_storage = quantizer.get_q_storage()
        assert q_storage is not None

        # 验证输出形状
        assert result.shape == weight.value.shape

    def test_different_weight_shapes(self):
        """测试不同权重形状的处理"""

        # 测试不同形状的权重
        weight_shapes = [(10, 20), (32, 64), (128, 256)]

        for shape in weight_shapes:
            quantizer = WeightPerChannelMinmax(self.config)
            weight = QStorage(QDType.FLOAT, torch.randn(*shape))
            bias = torch.randn(shape[1])

            quantizer.init_weight(weight, bias)
            result = quantizer()
            q_param = quantizer.get_q_param()

            assert result.shape == weight.value.shape
            assert q_param is not None
            assert q_param.scheme == self.config.to_scheme()
            assert q_param.ext["scale"].shape == (shape[0],)
            assert q_param.ext["offset"].shape == (shape[0],)

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_channel_sym, "minmax"),
        ],
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        """测试通过自动量化器创建"""
        quantizer = AutoWeightQuantizer.from_config(qconfig)
        assert isinstance(quantizer, WeightPerChannelMinmax)

    def test_get_q_storage_returns_storage_when_not_forwarded(self):
        """边界：未执行 forward 时 get_q_storage 应自动触发量化。"""
        config = QConfig(dtype="int8", scope="per_channel", method="minmax", symmetric=True)
        quantizer = WeightPerChannelMinmax(config)
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        quantizer.init_weight(weight)
        storage = quantizer.get_q_storage()
        assert storage is not None

    def test_get_q_param_returns_qparam_when_not_forwarded(self):
        """边界：未执行 forward 时 get_q_param 应自动触发量化。"""
        config = QConfig(dtype="int8", scope="per_channel", method="minmax", symmetric=True)
        quantizer = WeightPerChannelMinmax(config)
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        quantizer.init_weight(weight)
        qp = quantizer.get_q_param()
        assert qp is not None


def _make_mxfp_qconfig(dtype: QDType, **ext_kwargs) -> QConfig:
    return QConfig(
        dtype=dtype,
        scope=QScope.PER_BLOCK,
        symmetric=True,
        method="minmax",
        ext=ext_kwargs,
    )


class TestMXWeightPerBlockMinmax:
    """Test suite for MXWeightPerBlockMinmax — MXFP8/MXFP4 per-block weight quantizer."""

    # ---------- 正常情形 ----------

    def test_forward_returns_same_shape_when_mxfp8(self):
        """正常：MXFP8 权重量化应返回与输入相同 shape。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = MXWeightPerBlockMinmax(config)
        w = torch.randn(64, 128)
        wq.init_weight(QStorage(QDType.FLOAT, w))
        out = wq.forward(None)
        assert out.shape == w.shape

    def test_forward_returns_same_shape_when_mxfp4(self):
        """正常：MXFP4 权重量化应返回与输入相同 shape。"""
        config = _make_mxfp_qconfig(QDType.MXFP4)
        wq = AutoWeightQuantizer.from_config(config)
        w = torch.randn(64, 128)
        wq.init_weight(QStorage(QDType.FLOAT, w))
        out = wq.forward(None)
        assert out.shape == w.shape

    def test_forward_returns_same_shape_when_called_twice(self):
        """正常：多次调用 forward 结果应稳定。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        wq.init_weight(QStorage(QDType.FLOAT, torch.randn(32, 64)))
        out1 = wq.forward(None)
        out2 = wq.forward(None)
        assert out1.shape == out2.shape

    def test_get_q_storage_returns_original_shape_when_quantized(self):
        """正常：量化后 get_q_storage 应返回原始 shape。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        orig = torch.randn(32, 64)
        wq.init_weight(QStorage(QDType.FLOAT, orig))
        wq.forward(None)
        storage = wq.get_q_storage()
        assert storage.value.shape == orig.shape

    def test_get_q_param_returns_qparam_with_scale_when_quantized(self):
        """正常：量化后 get_q_param 应返回包含 scale 和 axes 的 QParam。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        wq.init_weight(QStorage(QDType.FLOAT, torch.randn(32, 64)))
        qp = wq.get_q_param()
        assert "scale" in qp.ext
        assert "axes" in qp.ext

    def test_is_data_free_returns_true_when_default_config(self):
        """正常：is_data_free 应返回 True。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        assert wq.is_data_free() is True

    def test_support_distributed_returns_true_when_default_config(self):
        """正常：support_distributed 应返回 True。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        assert wq.support_distributed() is True

    # ---------- 重构等价性验证 ----------

    def test_lazy_quantize_matches_eager_quantize_when_mxfp4(self):
        """重构等价：MXFP4 惰性量化应与 eager 量化产生完全一致的 scale 和 storage。"""
        dtype, scope = QDType.MXFP4, QScope.PER_BLOCK
        config = _make_mxfp_qconfig(dtype)
        weight = torch.randn(64, 128)
        block_size = dtype.mx_finfo.block_size

        # eager flow (original init_weight logic)
        from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
        from msmodelslim.ir.api import calculate_qparam, quantize
        from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks

        obs = MsMinMaxBlockObserver(MinMaxBlockObserverConfig(axes=-1))
        proc_axes = [-1]
        proc_axes = [x + weight.ndim if x < 0 else x for x in proc_axes]
        wb, _, os_, ps_ = reshape_to_blocks(weight.detach(), proc_axes, block_size)
        sa = [x + 1 for x in _] if block_size > 0 else []
        obs.update(wb, sync=False, shared_exp_axes=sa)
        mv, xv = obs.get_min_max()
        eager_qp = calculate_qparam(mv, xv, q_dtype=QDType(dtype), q_scope=QScope(scope), symmetric=True, axes=-1)
        eager_qp.ext['axes'] = proc_axes
        eager_qs = quantize(QStorage(QDType.FLOAT, wb), eager_qp)
        eager_qs.value = undo_reshape_to_blocks(eager_qs.value, ps_, os_, proc_axes)

        # lazy flow (refactored)
        wq = AutoWeightQuantizer.from_config(config)
        wq.init_weight(QStorage(QDType.FLOAT, weight))
        wq.forward(None)
        lazy_qp = wq.get_q_param()
        lazy_qs = wq.get_q_storage()

        assert torch.allclose(eager_qp.ext['scale'], lazy_qp.ext['scale']), "scale mismatch"
        assert torch.allclose(eager_qs.value, lazy_qs.value, atol=1e-7), "storage mismatch"

    def test_lazy_quantize_matches_eager_quantize_when_mxfp8(self):
        """重构等价：MXFP8 惰性量化应与 eager 量化产生完全一致的 scale 和 storage。"""
        dtype, scope = QDType.MXFP8, QScope.PER_BLOCK
        config = _make_mxfp_qconfig(dtype)
        weight = torch.randn(64, 128)
        block_size = dtype.mx_finfo.block_size

        from msmodelslim.core.observer import MsMinMaxBlockObserver, MinMaxBlockObserverConfig
        from msmodelslim.ir.api import calculate_qparam, quantize
        from msmodelslim.ir.utils import reshape_to_blocks, undo_reshape_to_blocks

        obs = MsMinMaxBlockObserver(MinMaxBlockObserverConfig(axes=-1))
        proc_axes = [-1]
        proc_axes = [x + weight.ndim if x < 0 else x for x in proc_axes]
        wb, _, os_, ps_ = reshape_to_blocks(weight.detach(), proc_axes, block_size)
        sa = [x + 1 for x in _] if block_size > 0 else []
        obs.update(wb, sync=False, shared_exp_axes=sa)
        mv, xv = obs.get_min_max()
        eager_qp = calculate_qparam(mv, xv, q_dtype=QDType(dtype), q_scope=QScope(scope), symmetric=True, axes=-1)
        eager_qp.ext['axes'] = proc_axes
        eager_qs = quantize(QStorage(QDType.FLOAT, wb), eager_qp)
        eager_qs.value = undo_reshape_to_blocks(eager_qs.value, ps_, os_, proc_axes)

        wq = AutoWeightQuantizer.from_config(config)
        wq.init_weight(QStorage(QDType.FLOAT, weight))
        wq.forward(None)
        lazy_qp = wq.get_q_param()
        lazy_qs = wq.get_q_storage()

        assert torch.allclose(eager_qp.ext['scale'], lazy_qp.ext['scale']), "scale mismatch"
        assert torch.allclose(eager_qs.value, lazy_qs.value, atol=1e-7), "storage mismatch"

    # ---------- 边界情形 ----------

    def test_forward_returns_same_shape_when_1d_weight(self):
        """边界：1D 权重应正常处理。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        wq.init_weight(QStorage(QDType.FLOAT, torch.randn(64)))
        out = wq.forward(None)
        assert out.shape == (64,)

    def test_forward_returns_same_shape_when_3d_weight(self):
        """边界：3D 权重应正常处理。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        wq.init_weight(QStorage(QDType.FLOAT, torch.randn(4, 32, 64)))
        out = wq.forward(None)
        assert out.shape == (4, 32, 64)

    def test_forward_returns_same_shape_when_zero_weight(self):
        """边界：全零权重应正常量化。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        wq.init_weight(QStorage(QDType.FLOAT, torch.zeros(32, 64)))
        out = wq.forward(None)
        assert out.shape == (32, 64)

    def test_forward_returns_same_shape_when_axes_is_zero(self):
        """边界：axes=0 时应正常量化。"""
        config = _make_mxfp_qconfig(QDType.MXFP8, axes=0)
        wq = AutoWeightQuantizer.from_config(config)
        wq.init_weight(QStorage(QDType.FLOAT, torch.randn(32, 64)))
        out = wq.forward(None)
        assert out.shape == (32, 64)

    def test_get_q_storage_returns_same_shape_when_called_twice(self):
        """边界：多次调用 get_q_storage 应返回一致结果。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        wq.init_weight(QStorage(QDType.FLOAT, torch.randn(32, 64)))
        wq.forward(None)
        s1 = wq.get_q_storage()
        s2 = wq.get_q_storage()
        assert s1.value.shape == s2.value.shape

    def test_get_q_storage_returns_original_shape_when_not_forwarded(self):
        """边界：未 forward 时 get_q_storage 应自动触发量化并返回原始 shape。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        w = torch.randn(32, 64)
        wq.init_weight(QStorage(QDType.FLOAT, w))
        storage = wq.get_q_storage()
        assert storage.value.shape == w.shape

    def test_get_q_param_returns_qparam_with_scale_when_not_forwarded(self):
        """边界：未 forward 时 get_q_param 应自动触发量化并返回 scale。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        wq.init_weight(QStorage(QDType.FLOAT, torch.randn(32, 64)))
        qp = wq.get_q_param()
        assert "scale" in qp.ext

    # ---------- 异常情形 ----------

    def test_forward_raises_spec_error_when_weight_not_set(self):
        """异常：未 init_weight 时 forward 应报 SpecError。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        with pytest.raises(SpecError):
            wq.forward(None)

    def test_from_config_raises_schema_validate_error_when_axes_is_invalid(self):
        """异常：非法的 axes 类型应报 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            AutoWeightQuantizer.from_config(_make_mxfp_qconfig(QDType.MXFP8, axes="invalid"))

    def test_get_q_storage_raises_spec_error_when_weight_not_set(self):
        """异常：未 init_weight 时 get_q_storage 应报 SpecError。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        with pytest.raises(SpecError):
            wq.get_q_storage()

    def test_get_q_param_raises_spec_error_when_weight_not_set(self):
        """异常：未 init_weight 时 get_q_param 应报 SpecError。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        wq = AutoWeightQuantizer.from_config(config)
        with pytest.raises(SpecError):
            wq.get_q_param()

    # ---------- 自动创建 ----------

    @pytest.mark.parametrize(
        "qconfig",
        [
            _make_mxfp_qconfig(QDType.MXFP8),
            _make_mxfp_qconfig(QDType.MXFP4),
        ],
    )
    def test_from_config_returns_mx_weight_per_block_minmax_when_mxfp_config(self, qconfig):
        """自动：from_config 应正确创建 MXWeightPerBlockMinmax 实例。"""
        quantizer = AutoWeightQuantizer.from_config(qconfig)
        assert isinstance(quantizer, MXWeightPerBlockMinmax)


class TestMXActPerBlockMinmax:
    """Test suite for MXActPerBlockMinmax — MXFP8/MXFP4 per-block activation quantizer."""

    def test_init_returns_qparam_with_axes_when_mxfp8(self):
        """正常：MXFP8 初始化应设置含 axes 的 q_param。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        quantizer = AutoActQuantizer.from_config(config)
        qp = quantizer.get_q_param()
        assert qp.scheme.dtype == QDType.MXFP8
        assert qp.scheme.scope == QScope.PER_BLOCK
        assert "axes" in qp.ext

    def test_init_returns_qparam_with_axes_when_mxfp4(self):
        """正常：MXFP4 初始化应设置含 axes 的 q_param。"""
        config = _make_mxfp_qconfig(QDType.MXFP4)
        quantizer = AutoActQuantizer.from_config(config)
        qp = quantizer.get_q_param()
        assert qp.scheme.dtype == QDType.MXFP4

    def test_forward_returns_input_unchanged_when_any_input(self):
        """正常：forward 应直接返回输入不做修改。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        quantizer = AutoActQuantizer.from_config(config)
        x = torch.randn(2, 4)
        result = quantizer(x)
        assert torch.equal(result, x)

    def test_is_data_free_returns_true_when_default_config(self):
        """正常：is_data_free 应返回 True。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        quantizer = AutoActQuantizer.from_config(config)
        assert quantizer.is_data_free() is True

    def test_from_config_raises_schema_validate_error_when_axes_is_invalid(self):
        """异常：非法的 axes 类型应报 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError):
            config = QConfig(
                dtype=QDType.MXFP8, scope=QScope.PER_BLOCK, symmetric=True, method="minmax", ext={"axes": "bad"}
            )
            AutoActQuantizer.from_config(config)

    def test_get_q_param_returns_default_scheme_when_q_param_is_none(self):
        """边界：q_param 为 None 时应返回兜底 QParam。"""
        config = _make_mxfp_qconfig(QDType.MXFP8)
        quantizer = AutoActQuantizer.from_config(config)
        quantizer.q_param = None
        qp = quantizer.get_q_param()
        assert qp is not None
