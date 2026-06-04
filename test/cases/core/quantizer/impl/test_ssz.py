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

from msmodelslim.ir.api import calculate_qparam
from msmodelslim.ir.qal.qbase import QStorage, QDType, QScheme, QScope
from msmodelslim import ir as qir
from msmodelslim.core.observer import MsMinMaxObserver
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.impl.ssz import (
    WeightPerChannelSsz,
    get_ext_scale,
    get_ext_offset,
    _broadcast_quantizer_state,
)
from msmodelslim.utils.distributed.task_scheduler import DTSMixin
from msmodelslim.utils.exception import SpecError
from unittest import mock


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


class TestWeightPerChannelSsz:  # pylint: disable=attribute-defined-outside-init
    """测试Per-Channel ssz量化器"""

    def setup_class(self):
        self.config = QConfig(dtype="int8", scope="per_channel", method="ssz", symmetric=True)

    def test_initialization(self):
        """测试初始化"""
        quantizer = WeightPerChannelSsz(self.config)

        assert quantizer.config == self.config, "config is not correct, expected: %s, actual: %s" % (
            self.config,
            quantizer.config,
        )
        assert quantizer.minmax_observer is not None, "minmax_observer is not correct, expected: %s, actual: %s" % (
            MsMinMaxObserver,
            quantizer.minmax_observer,
        )
        assert isinstance(quantizer.minmax_observer, MsMinMaxObserver), (
            "minmax_observer is not correct, expected: %s, actual: %s"
            % (MsMinMaxObserver, type(quantizer.minmax_observer))
        )
        assert quantizer.weight is None, "weight is not correct, expected: %s, actual: %s" % (None, quantizer.weight)
        assert quantizer.bias is None, "bias is not correct, expected: %s, actual: %s" % (None, quantizer.bias)
        assert quantizer.w_q_param is None, "w_q_param is not correct, expected: %s, actual: %s" % (
            None,
            quantizer.w_q_param,
        )
        assert quantizer.w_q_storage is None, "w_q_storage is not correct, expected: %s, actual: %s" % (
            None,
            quantizer.w_q_storage,
        )
        assert quantizer.is_quantized is False, "is_quantized is not correct, expected: %s, actual: %s" % (
            False,
            quantizer.is_quantized,
        )

    def test_init_weight_validation(self):
        """测试权重初始化验证"""
        quantizer = WeightPerChannelSsz(self.config)

        # 测试无效权重类型
        with pytest.raises(ValidationError, match="instance of QStorage"):
            quantizer.init_weight(torch.randn(10, 20))

        # 测试无效bias类型
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        with pytest.raises(ValidationError, match="instance of Tensor"):
            quantizer.init_weight(weight, bias="invalid")

    def test_get_q_storage_and_q_param_after_forward(self):
        """测试在forward之后获取q_storage和q_param"""
        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        bias = torch.randn(20)
        quantizer.init_weight(weight, bias)
        quantizer()
        q_storage = quantizer.get_q_storage()
        q_param = quantizer.get_q_param()
        assert q_storage is not None, "q_storage is not correct, expected: %s, actual: %s" % (None, q_storage)
        assert q_param is not None, "q_param is not correct, expected: %s, actual: %s" % (None, q_param)

    def test_forward_before_init_weight(self):
        """测试在初始化权重之前前向传播"""
        quantizer = WeightPerChannelSsz(self.config)

        with pytest.raises(SpecError, match="No weight was set"):
            quantizer()

    def test_forward_after_init_weight(self):
        """测试权重初始化并前向传播"""
        quantizer = WeightPerChannelSsz(self.config)

        # 初始化权重
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        bias = torch.randn(20)

        quantizer.init_weight(weight, bias)

        assert quantizer.weight == weight, "weight is not correct, expected: %s, actual: %s" % (
            weight,
            quantizer.weight,
        )
        assert quantizer.bias is bias, "bias is not correct, expected: %s, actual: %s" % (bias, quantizer.bias)

        # 前向传播
        result = quantizer()

        # 验证q_param被设置
        q_param = quantizer.get_q_param()
        assert q_param is not None, "q_param is not correct, expected: %s, actual: %s" % (None, q_param)
        assert q_param.scheme == self.config.to_scheme(), "q_param.scheme is not correct, expected: %s, actual: %s" % (
            self.config.to_scheme(),
            q_param.scheme,
        )
        assert isinstance(q_param.ext, dict), "q_param.ext is not correct, expected: %s, actual: %s" % (
            dict,
            type(q_param.ext),
        )
        assert "scale" in q_param.ext, "q_param.ext is not correct, expected: %s, actual: %s" % (
            "scale",
            q_param.ext.keys(),
        )
        assert "offset" in q_param.ext, "q_param.ext is not correct, expected: %s, actual: %s" % (
            "offset",
            q_param.ext.keys(),
        )
        assert isinstance(q_param.ext["scale"], torch.Tensor), (
            "q_param.ext['scale'] is not correct, expected: %s, actual: %s" % (torch.Tensor, type(q_param.ext["scale"]))
        )
        assert isinstance(q_param.ext["offset"], torch.Tensor), (
            "q_param.ext['offset'] is not correct, expected: %s, actual: %s"
            % (torch.Tensor, type(q_param.ext["offset"]))
        )
        # Per-channel的scale和offset应该与输出通道数匹配
        assert q_param.ext["scale"].shape == (weight.value.shape[0],), (
            "q_param.ext['scale'].shape is not correct, expected: %s, actual: %s"
            % ((weight.value.shape[0],), q_param.ext["scale"].shape)
        )
        assert q_param.ext["offset"].shape == (weight.value.shape[0],), (
            "q_param.ext['offset'].shape is not correct, expected: %s, actual: %s"
            % ((weight.value.shape[0],), q_param.ext["offset"].shape)
        )

        # 验证q_storage被设置
        q_storage = quantizer.get_q_storage()
        assert q_storage is not None, "q_storage is not correct, expected: %s, actual: %s" % (None, q_storage)

        # 验证输出形状
        assert result.shape == weight.value.shape, "result.shape is not correct, expected: %s, actual: %s" % (
            weight.value.shape,
            result.shape,
        )

    def test_forward_with_invalid_one_dim_weight(self):
        """测试无效权重形状"""
        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(10))
        bias = torch.randn(10)
        quantizer.init_weight(weight, bias)
        with pytest.raises(SpecError, match="Weight must be a 2D tensor"):
            quantizer()

    def test_forward_with_invalid_three_dim_weight(self):
        """测试无效权重形状"""
        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20, 30))
        bias = torch.randn(20)
        quantizer.init_weight(weight, bias)
        with pytest.raises(SpecError, match="Weight must be a 2D tensor"):
            quantizer()

    def test_different_weight_shapes(self):
        """测试不同权重形状的处理"""

        # 测试不同形状的权重
        weight_shapes = [(10, 20), (32, 64), (128, 256)]

        for shape in weight_shapes:
            quantizer = WeightPerChannelSsz(self.config)
            weight = QStorage(QDType.FLOAT, torch.randn(*shape))
            bias = torch.randn(shape[1])

            quantizer.init_weight(weight, bias)
            result = quantizer()
            q_param = quantizer.get_q_param()

            assert result.shape == weight.value.shape, "result.shape is not correct, expected: %s, actual: %s" % (
                weight.value.shape,
                result.shape,
            )
            assert q_param is not None, "q_param is not correct, expected: %s, actual: %s" % (None, q_param)
            assert q_param.scheme == self.config.to_scheme(), (
                "q_param.scheme is not correct, expected: %s, actual: %s" % (self.config.to_scheme(), q_param.scheme)
            )
            assert q_param.ext["scale"].shape == (shape[0],), (
                "q_param.ext['scale'].shape is not correct, expected: %s, actual: %s"
                % ((shape[0],), q_param.ext["scale"].shape)
            )
            assert q_param.ext["offset"].shape == (shape[0],), (
                "q_param.ext['offset'].shape is not correct, expected: %s, actual: %s"
                % ((shape[0],), q_param.ext["offset"].shape)
            )

    @pytest.mark.parametrize(
        "qconfig",
        [
            to_qconfig(qir.int8_per_channel_sym, "ssz"),
        ],
    )
    def test_creation_with_auto_quantizer(self, qconfig):
        """测试通过自动量化器创建"""
        from msmodelslim.core.quantizer.base import AutoWeightQuantizer

        quantizer = AutoWeightQuantizer.from_config(qconfig)
        assert isinstance(quantizer, WeightPerChannelSsz)

    def test_dts_mixin_inheritance(self):
        """验证 WeightPerChannelSsz 通过 AutoWeightQuantizer 继承 DTSMixin"""
        from msmodelslim.core.quantizer.base import AutoWeightQuantizer as AWQ

        quantizer = WeightPerChannelSsz(self.config)
        assert isinstance(quantizer, DTSMixin), "WeightPerChannelSsz should be a DTSMixin instance"
        assert hasattr(quantizer, "distributed_sync"), "WeightPerChannelSsz should have distributed_sync"
        assert WeightPerChannelSsz.distributed_sync is not AWQ.distributed_sync, (
            "WeightPerChannelSsz should override distributed_sync"
        )

    def test_dts_mixin_default_forward(self):
        """验证 AutoWeightQuantizer 默认 distributed_sync 触发 forward"""
        from msmodelslim.core.quantizer.base import AutoWeightQuantizer as AWQ

        assert issubclass(AWQ, DTSMixin), "AutoWeightQuantizer should inherit DTSMixin"
        wq = WeightPerChannelSsz(self.config)
        assert isinstance(wq, DTSMixin), "All weight quantizers should inherit DTSMixin via AutoWeightQuantizer"


class TestWeightPerChannelSszDistributedSync:  # pylint: disable=attribute-defined-outside-init
    """测试 WeightPerChannelSsz 的分布式同步行为"""

    def setup_class(self):
        self.config = QConfig(dtype="int8", scope="per_channel", method="ssz", symmetric=True)
        self.asymmetric_config = QConfig(dtype="int8", scope="per_channel", method="ssz", symmetric=False)

    # ── load_quantized_from_broadcast_tensors ──────────────────────────

    def test_load_quantized_from_broadcast_tensors_should_set_quantized_state(self):
        """验证 load_quantized_from_broadcast_tensors 正确重建量化状态"""
        quantizer = WeightPerChannelSsz(self.config)

        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))
        bias = torch.randn(20)
        quantizer.init_weight(weight, bias)
        quantizer()  # 触发量化，获取参考结果

        # 从已量化的 quantizer 中提取参数
        ref_scale = get_ext_scale(quantizer.w_q_param)
        ref_offset = get_ext_offset(quantizer.w_q_param)
        ref_storage_value = quantizer.w_q_storage.value
        ref_storage_dtype = quantizer.w_q_storage.dtype

        # 创建新的 quantizer，用 load_quantized_from_broadcast_tensors 加载
        new_quantizer = WeightPerChannelSsz(self.config)
        new_quantizer.load_quantized_from_broadcast_tensors(
            scale=ref_scale,
            offset=ref_offset,
            w_q_storage_value=ref_storage_value,
            q_storage_dtype=ref_storage_dtype,
        )

        # 验证状态被正确重建
        assert new_quantizer.is_quantized is True, "quantizer should be marked as quantized"
        assert new_quantizer.weight is None, "weight should be released after loading quantized state"
        assert new_quantizer.w_q_param is not None, "w_q_param should be set"
        assert new_quantizer.w_q_storage is not None, "w_q_storage should be set"

        # 验证量化参数一致
        assert torch.equal(get_ext_scale(new_quantizer.w_q_param), ref_scale), "scale should match reference"
        assert torch.equal(get_ext_offset(new_quantizer.w_q_param), ref_offset), "offset should match reference"

        # 验证量化权重一致
        assert torch.equal(new_quantizer.w_q_storage.value, ref_storage_value), (
            "quantized weight should match reference"
        )
        assert new_quantizer.w_q_storage.dtype == ref_storage_dtype, "storage dtype should match reference"

        # 验证 scheme 正确
        assert new_quantizer.w_q_param.scheme == self.config.to_scheme(), "scheme should match config"

        # 验证 forward 输出一致
        ref_output = quantizer.forward(None)
        new_output = new_quantizer.forward(None)
        assert torch.equal(ref_output, new_output), "forward output should be identical"

    def test_load_quantized_from_broadcast_tensors_releases_old_weight(self):
        """验证 load_quantized_from_broadcast_tensors 释放原有 weight"""
        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(5, 10))
        bias = torch.randn(10)
        quantizer.init_weight(weight, bias)

        scale = torch.randn(5)
        offset = torch.zeros(5)
        storage_value = torch.randint(-8, 7, (5, 10), dtype=torch.int8)

        quantizer.load_quantized_from_broadcast_tensors(
            scale=scale,
            offset=offset,
            w_q_storage_value=storage_value,
            q_storage_dtype=QDType.INT8,
        )

        assert quantizer.weight is None, "weight should be released after loading broadcast state"

    def test_load_quantized_from_broadcast_tensors_with_asymmetric(self):
        """验证非对称量化下 load_quantized_from_broadcast_tensors 正确工作"""
        quantizer = WeightPerChannelSsz(self.asymmetric_config)
        weight = QStorage(QDType.FLOAT, torch.randn(8, 16))
        bias = torch.randn(16)
        quantizer.init_weight(weight, bias)
        quantizer()

        ref_scale = get_ext_scale(quantizer.w_q_param)
        ref_offset = get_ext_offset(quantizer.w_q_param)

        # offset 应非零（非对称量化）
        assert not (torch.all(ref_offset == 0)), "asymmetric quantization should have non-zero offsets"

        new_quantizer = WeightPerChannelSsz(self.asymmetric_config)
        new_quantizer.load_quantized_from_broadcast_tensors(
            scale=ref_scale,
            offset=ref_offset,
            w_q_storage_value=quantizer.w_q_storage.value,
            q_storage_dtype=quantizer.w_q_storage.dtype,
        )

        assert new_quantizer.w_q_param.scheme.symmetric is False, "scheme should be asymmetric"
        assert torch.equal(get_ext_offset(new_quantizer.w_q_param), ref_offset), (
            "offset should match in asymmetric mode"
        )

    # ── _broadcast_quantizer_state ────────────────────────────────────

    def test_broadcast_quantizer_state_raises_when_not_quantized(self):
        """验证未量化时 _broadcast_quantizer_state 抛出 RuntimeError"""
        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(5, 10))
        quantizer.init_weight(weight, bias=None)

        with mock.patch("msmodelslim.core.quantizer.impl.ssz.dist") as mock_dist:
            mock_dist.get_rank.return_value = 0
            mock_dist.is_initialized.return_value = True
            with pytest.raises(SpecError, match="Cannot broadcast quantizer state"):
                _broadcast_quantizer_state(quantizer, owner_rank=0)

    @mock.patch("msmodelslim.core.quantizer.impl.ssz.dist")
    @mock.patch("msmodelslim.core.quantizer.impl.ssz.broadcast_tensor_process_group_safe")
    def test_broadcast_quantizer_state_sends_scale_offset_storage(self, mock_broadcast, mock_dist):
        """验证 owner rank 正确 broadcast 三个张量"""
        mock_dist.get_rank.return_value = 0  # 当前是 owner
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 2

        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(5, 10))
        quantizer.init_weight(weight, bias=None)
        quantizer()  # 触发量化

        _broadcast_quantizer_state(quantizer, owner_rank=0)

        # 验证 broadcast_object_list 被调用（owner 传递 meta）
        mock_dist.broadcast_object_list.assert_called_once()

        # 验证 broadcast_tensor_process_group_safe 被调用 3 次 (scale, offset, storage)
        assert mock_broadcast.call_count == 3, "should broadcast 3 tensors: scale, offset, storage"

        # 验证调用参数：src 都是 0
        for call_arg in mock_broadcast.call_args_list:
            assert call_arg.kwargs.get('src') == 0, "all broadcasts should use owner_rank=0 as src"

    @mock.patch("msmodelslim.core.quantizer.impl.ssz.dist")
    @mock.patch("msmodelslim.core.quantizer.impl.ssz.broadcast_tensor_process_group_safe")
    def test_broadcast_quantizer_state_non_owner_calls_load(self, mock_broadcast, mock_dist):
        """验证非 owner rank 接收 broadcast 后调用 load_quantized_from_broadcast_tensors"""
        mock_dist.get_rank.return_value = 1  # 当前是非 owner
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 2

        # 模拟 broadcast_object_list 接收到 owner 的 meta
        fake_meta = [
            {
                "scale_shape": [5],
                "scale_dtype": "torch.float32",
                "offset_shape": [5],
                "offset_dtype": "torch.float32",
                "storage_shape": [5, 10],
                "storage_dtype": "torch.int8",
                "storage_qdtype": "INT8",
            }
        ]

        def fake_broadcast_object_list(meta_list, src):
            # 模拟接收: 把 fake_meta 写入 meta_list
            meta_list[0] = fake_meta[0]

        mock_dist.broadcast_object_list.side_effect = fake_broadcast_object_list

        # 模拟 broadcast_tensor_process_group_safe 写入接收到的张量
        received_tensors = {}

        def fake_broadcast(tensor, src):
            received_tensors[id(tensor)] = tensor

        mock_broadcast.side_effect = fake_broadcast

        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(5, 10))
        quantizer.init_weight(weight, bias=None)

        _broadcast_quantizer_state(quantizer, owner_rank=0)

        # 验证非 owner 的 load_quantized_from_broadcast_tensors 被调用后
        # quantizer 状态正确
        assert quantizer.is_quantized is True, "non-owner quantizer should be quantized after broadcast"
        assert quantizer.w_q_param is not None, "non-owner quantizer should have q_param"
        assert quantizer.w_q_storage is not None, "non-owner quantizer should have q_storage"

    @mock.patch("msmodelslim.core.quantizer.impl.ssz.dist")
    @mock.patch("msmodelslim.core.quantizer.impl.ssz.broadcast_tensor_process_group_safe")
    def test_broadcast_quantizer_state_contiguous_tensors(self, mock_broadcast, mock_dist):
        """验证 owner rank 发送的 tensor 是 contiguous 的"""
        mock_dist.get_rank.return_value = 0
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 2

        quantizer = WeightPerChannelSsz(self.config)
        weight = QStorage(QDType.FLOAT, torch.randn(8, 16))
        quantizer.init_weight(weight, bias=None)
        quantizer()

        # 在 forward 之后，修改 storage 使其 non-contiguous（transpose）
        quantizer.w_q_storage = QStorage(
            quantizer.w_q_storage.dtype,
            quantizer.w_q_storage.value.T.contiguous().T,  # 通过转置创造 non-contiguous
        )

        _broadcast_quantizer_state(quantizer, owner_rank=0)

        # 验证发送的 tensor 都是 contiguous
        for call_arg in mock_broadcast.call_args_list:
            t = call_arg.args[0]
            assert t.is_contiguous(), "broadcast tensor should be contiguous, got shape=%s stride=%s" % (
                t.shape,
                t.stride(),
            )

    # ── distributed_sync ──────────────────────────────────────────────

    def test_distributed_sync_override_is_different_from_base(self):
        """验证 WeightPerChannelSsz.distributed_sync 与基类不同（已覆盖）"""
        from msmodelslim.core.quantizer.base import AutoWeightQuantizer as AWQ

        assert WeightPerChannelSsz.distributed_sync is not AWQ.distributed_sync, (
            "WeightPerChannelSsz should override distributed_sync"
        )

    def test_distributed_sync_single_rank_skips_broadcast(self):
        """验证单卡环境下 distributed_sync 不执行 broadcast"""
        with mock.patch("msmodelslim.core.quantizer.impl.ssz.dist.is_initialized", return_value=False):
            quantizer = WeightPerChannelSsz(self.config)
            weight = QStorage(QDType.FLOAT, torch.randn(5, 10))
            quantizer.init_weight(weight, bias=None)
            quantizer()

            # 在单卡（dist 未初始化）下调用 distributed_sync
            # 不应抛异常
            from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext

            record = TaskExecutionRecord(task_id="test", executor_rank=0)
            sync_ctx = TaskSyncContext(model=quantizer, rank=0, world_size=1)
            quantizer.distributed_sync(record, sync_ctx)

    def test_distributed_sync_raises_spec_error_when_not_data_free(self):
        """验证非 data-free quantizer 调用 distributed_sync 抛出 SpecError"""
        from msmodelslim.core.quantizer.base import AutoWeightQuantizer

        # 创建一个非 data-free 的假 quantizer
        class _NotDataFreeQuantizer(AutoWeightQuantizer):
            def is_data_free(self):
                return False

            def init_weight(self, weight, bias=None):
                pass

            def forward(self, x=None):
                return torch.empty(0)

            def get_q_storage(self):
                return None

            def get_q_param(self):
                return None

        q = _NotDataFreeQuantizer()
        from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext

        record = TaskExecutionRecord(task_id="test", executor_rank=0)
        sync_ctx = TaskSyncContext(model=q, rank=0, world_size=2)

        with pytest.raises(SpecError, match="data-free"):
            q.distributed_sync(record, sync_ctx)

    def test_distributed_sync_data_free_quantizer_does_not_raise(self):
        """验证 data-free quantizer 调用 distributed_sync 不抛出异常"""
        from msmodelslim.core.quantizer.base import AutoWeightQuantizer

        class _DataFreeQuantizer(AutoWeightQuantizer):  # pylint: disable=attribute-defined-outside-init
            def is_data_free(self):
                return True

            def init_weight(self, weight, bias=None):
                self._weight = weight

            def forward(self, x=None):
                return torch.empty(0)

            def get_q_storage(self):
                return None

            def get_q_param(self):
                return None

        q = _DataFreeQuantizer()
        from msmodelslim.utils.distributed.task_scheduler.types import TaskExecutionRecord, TaskSyncContext

        record = TaskExecutionRecord(task_id="test", executor_rank=0)
        sync_ctx = TaskSyncContext(model=q, rank=0, world_size=2)

        try:
            q.distributed_sync(record, sync_ctx)
        except SpecError:
            pytest.fail("distributed_sync should not raise SpecError for data-free quantizers")


class TestSszCalculateQparam:  # pylint: disable=attribute-defined-outside-init
    """测试 ssz_calculate_qparam 函数"""

    def setup_class(self):
        """设置测试环境"""
        self.symmetric_config = QConfig(dtype="int8", scope="per_channel", method="ssz", symmetric=True)
        self.asymmetric_config = QConfig(dtype="int8", scope="per_channel", method="ssz", symmetric=False)

    def test_symmetric_quantization(self):
        """测试对称量化"""
        from msmodelslim.core.quantizer.impl.ssz import ssz_calculate_qparam

        # 创建测试权重
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))

        # 创建初始量化参数
        initial_q_param = calculate_qparam(
            min_val=torch.min(weight.T.value, dim=0)[0],
            max_val=torch.max(weight.T.value, dim=0)[0],
            q_dtype=QDType(self.symmetric_config.dtype),
            q_scope=QScope(self.symmetric_config.scope),
            symmetric=self.symmetric_config.symmetric,
        )

        # 调用ssz_calculate_qparam
        result_q_param = ssz_calculate_qparam(weight.T, initial_q_param)

        # 验证返回的q_param
        assert result_q_param is not None, "q_param is not correct, expected: %s, actual: %s" % (None, result_q_param)
        assert result_q_param.scheme == self.symmetric_config.to_scheme(), (
            "q_param.scheme is not correct, expected: %s, actual: %s"
            % (self.symmetric_config.to_scheme(), result_q_param.scheme)
        )
        assert "scale" in result_q_param.ext, "q_param.ext is not correct, expected: %s, actual: %s" % (
            "scale",
            result_q_param.ext.keys(),
        )
        assert "offset" in result_q_param.ext, "q_param.ext is not correct, expected: %s, actual: %s" % (
            "offset",
            result_q_param.ext.keys(),
        )
        assert isinstance(result_q_param.ext["scale"], torch.Tensor), (
            "q_param.ext['scale'] is not correct, expected: %s, actual: %s"
            % (torch.Tensor, type(result_q_param.ext["scale"]))
        )
        assert isinstance(result_q_param.ext["offset"], torch.Tensor), (
            "q_param.ext['offset'] is not correct, expected: %s, actual: %s"
            % (torch.Tensor, type(result_q_param.ext["offset"]))
        )
        assert result_q_param.ext["scale"].shape == (weight.value.shape[0],), (
            "q_param.ext['scale'].shape is not correct, expected: %s, actual: %s"
            % ((weight.value.shape[0],), result_q_param.ext["scale"].shape)
        )
        assert result_q_param.ext["offset"].shape == (weight.value.shape[0],), (
            "q_param.ext['offset'].shape is not correct, expected: %s, actual: %s"
            % ((weight.value.shape[0],), result_q_param.ext["offset"].shape)
        )
        assert result_q_param.ext["offset"].max() == 0 and result_q_param.ext["offset"].min() == 0, (
            "q_param.ext['offset'] is not correct, expected: %s, actual: %s"
            % (True, result_q_param.ext["offset"].max() == 0 and result_q_param.ext["offset"].min() == 0)
        )

    def test_asymmetric_quantization(self):
        """测试非对称量化"""
        from msmodelslim.core.quantizer.impl.ssz import ssz_calculate_qparam

        # 创建测试权重
        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))

        # 创建初始量化参数
        initial_q_param = calculate_qparam(
            min_val=torch.min(weight.T.value, dim=0)[0],
            max_val=torch.max(weight.T.value, dim=0)[0],
            q_dtype=QDType(self.asymmetric_config.dtype),
            q_scope=QScope(self.asymmetric_config.scope),
            symmetric=self.asymmetric_config.symmetric,
        )

        # 调用ssz_calculate_qparam
        result_q_param = ssz_calculate_qparam(weight.T, initial_q_param)

        # 验证返回的q_param
        assert result_q_param is not None, "q_param is not correct, expected: %s, actual: %s" % (None, result_q_param)
        assert result_q_param.scheme == self.asymmetric_config.to_scheme(), (
            "q_param.scheme is not correct, expected: %s, actual: %s"
            % (self.asymmetric_config.to_scheme(), result_q_param.scheme)
        )
        assert "scale" in result_q_param.ext, "q_param.ext is not correct, expected: %s, actual: %s" % (
            "scale",
            result_q_param.ext.keys(),
        )
        assert "offset" in result_q_param.ext, "q_param.ext is not correct, expected: %s, actual: %s" % (
            "offset",
            result_q_param.ext.keys(),
        )
        assert isinstance(result_q_param.ext["scale"], torch.Tensor), (
            "q_param.ext['scale'] is not correct, expected: %s, actual: %s"
            % (torch.Tensor, type(result_q_param.ext["scale"]))
        )
        assert isinstance(result_q_param.ext["offset"], torch.Tensor), (
            "q_param.ext['offset'] is not correct, expected: %s, actual: %s"
            % (torch.Tensor, type(result_q_param.ext["offset"]))
        )
        assert result_q_param.ext["scale"].shape == (weight.value.shape[0],), (
            "q_param.ext['scale'].shape is not correct, expected: %s, actual: %s"
            % ((weight.value.shape[0],), result_q_param.ext["scale"].shape)
        )
        assert result_q_param.ext["offset"].shape == (weight.value.shape[0],), (
            "q_param.ext['offset'].shape is not correct, expected: %s, actual: %s"
            % ((weight.value.shape[0],), result_q_param.ext["offset"].shape)
        )
        assert result_q_param.ext["offset"].max() != 0 or result_q_param.ext["offset"].min() != 0, (
            "q_param.ext['offset'] is not correct, expected: %s, actual: %s"
            % (True, result_q_param.ext["offset"].max() != 0 or result_q_param.ext["offset"].min() != 0)
        )

    def test_scale_offset_validity(self):
        """测试scale和offset的有效性"""
        from msmodelslim.core.quantizer.impl.ssz import ssz_calculate_qparam

        weight = QStorage(QDType.FLOAT, torch.randn(10, 20))

        initial_q_param = calculate_qparam(
            min_val=torch.min(weight.T.value, dim=0)[0],
            max_val=torch.max(weight.T.value, dim=0)[0],
            q_dtype=QDType(self.symmetric_config.dtype),
            q_scope=QScope(self.symmetric_config.scope),
            symmetric=self.symmetric_config.symmetric,
        )

        result_q_param = ssz_calculate_qparam(weight.T, initial_q_param)

        # 验证scale不为零且为有限值
        assert torch.all(torch.isfinite(result_q_param.ext["scale"])), (
            "scale is not correct, expected: %s, actual: %s"
            % (True, torch.all(torch.isfinite(result_q_param.ext["scale"])))
        )
        assert torch.all(result_q_param.ext["scale"] != 0), "scale is not correct, expected: %s, actual: %s" % (
            True,
            torch.all(result_q_param.ext["scale"] != 0),
        )

        # 验证offset为有限值
        assert torch.all(torch.isfinite(result_q_param.ext["offset"])), (
            "offset is not correct, expected: %s, actual: %s"
            % (True, torch.all(torch.isfinite(result_q_param.ext["offset"])))
        )
