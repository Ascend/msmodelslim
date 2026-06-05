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
# pylint: disable=redefined-outer-name

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.core.quant_service.modelslim_v1.save.interface import AscendV1GlobalModelDtypeInterface
from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config, AscendV1Saver
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.ir.qal import QParam, QScheme, QStorage, QScope, QDType


def _make_w8a8_static_module(out_features=4, in_features=8):
    """Build a minimal W8A8StaticFakeQuantLinear for testing."""
    input_scale = torch.tensor([0.5], dtype=torch.float32)
    input_offset = torch.tensor([0.0], dtype=torch.float32)
    weight_scale = torch.ones(out_features, dtype=torch.float32) * 0.1
    weight = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
    bias = torch.zeros(out_features, dtype=torch.float32)

    x_q_param = QParam(
        scheme=QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=False),
        ext={"scale": input_scale, "offset": input_offset},
    )
    w_q_param = QParam(
        scheme=QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True),
        ext={"scale": weight_scale},
    )
    w_q = QStorage(dtype=QDType.INT8, value=weight)
    return qir.W8A8StaticFakeQuantLinear(x_q_param, w_q_param, w_q, bias)


class AdapterBf16(AscendV1GlobalModelDtypeInterface):
    """Adapter that reports bfloat16."""

    def __init__(self, model_path):
        self._model_path = Path(model_path)

    @property
    def model_path(self):
        return self._model_path

    def get_global_model_torch_dtype(self):
        return torch.bfloat16


class AdapterFloat32(AscendV1GlobalModelDtypeInterface):
    """Adapter that reports float32."""

    def __init__(self, model_path):
        self._model_path = Path(model_path)

    @property
    def model_path(self):
        return self._model_path

    def get_global_model_torch_dtype(self):
        return torch.float32


class TestResolveIsBf16FromAdapter:
    """Tests for AscendV1Saver._resolve_is_bf16_from_adapter (and thus _global_torch_dtype_is_bf16)."""

    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        if os.path.exists(d):
            shutil.rmtree(d)

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized")
    def test_adapter_bf16_returns_true(self, mock_dist_init, temp_dir):
        mock_dist_init.return_value = False
        config = AscendV1Config(save_directory=temp_dir)
        model = nn.Linear(2, 2)
        adapter = AdapterBf16(temp_dir)
        saver = AscendV1Saver(model, config, adapter)
        assert saver._global_torch_dtype_is_bf16 is True

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized")
    def test_adapter_float32_returns_false(self, mock_dist_init, temp_dir):
        mock_dist_init.return_value = False
        config = AscendV1Config(save_directory=temp_dir)
        model = nn.Linear(2, 2)
        adapter = AdapterFloat32(temp_dir)
        saver = AscendV1Saver(model, config, adapter)
        assert saver._global_torch_dtype_is_bf16 is False

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized")
    def test_adapter_non_interface_returns_false(self, mock_dist_init, temp_dir):
        mock_dist_init.return_value = False
        config = AscendV1Config(save_directory=temp_dir)
        model = nn.Linear(2, 2)
        adapter = MagicMock()
        adapter.model_path = Path(temp_dir)
        saver = AscendV1Saver(model, config, adapter)
        assert saver._global_torch_dtype_is_bf16 is False


class TestW8A8DeqScaleWriteDtype:
    """W8A8 on_w8a8_static writes deq_scale as float32 when bf16, int64 when not bf16."""

    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        if os.path.exists(d):
            shutil.rmtree(d)

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized")
    def test_w8a8_static_deq_scale_int64_when_not_bf16(self, mock_dist_init, temp_dir):
        mock_dist_init.return_value = False
        config = AscendV1Config(save_directory=temp_dir)
        model = nn.Linear(2, 2)
        adapter = AdapterFloat32(temp_dir)
        saver = AscendV1Saver(model, config, adapter)
        w8a8_module = _make_w8a8_static_module()

        with patch.object(saver, "write_tensor") as mock_write:
            saver.on_w8a8_static("layer.linear", w8a8_module)

        deq_calls = [c for c in mock_write.call_args_list if c[0][0].endswith(".deq_scale")]
        assert len(deq_calls) == 1
        _, _, tensor = deq_calls[0][0]
        assert tensor.dtype == torch.int64

    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized")
    def test_w8a8_static_deq_scale_float32_when_bf16(self, mock_dist_init, temp_dir):
        mock_dist_init.return_value = False
        config = AscendV1Config(save_directory=temp_dir)
        model = nn.Linear(2, 2)
        adapter = AdapterBf16(temp_dir)
        saver = AscendV1Saver(model, config, adapter)
        w8a8_module = _make_w8a8_static_module()

        with patch.object(saver, "write_tensor") as mock_write:
            saver.on_w8a8_static("layer.linear", w8a8_module)

        deq_calls = [c for c in mock_write.call_args_list if c[0][0].endswith(".deq_scale")]
        assert len(deq_calls) == 1
        _, _, tensor = deq_calls[0][0]
        assert tensor.dtype == torch.float32


def _make_mock_saver():
    """
    创建一个 AscendV1Saver 实例，但跳过 __init__，
    并手动注入 mock 属性，方便测试各个方法。
    """
    with patch.object(AscendV1Saver, '__init__', return_value=None):
        saver = AscendV1Saver(model=None, config=None, adapter=None)
    saver.write_tensor = MagicMock()
    saver.json_append = {}
    saver.json_writer = MagicMock()
    saver.json_writer.write = MagicMock()
    saver.fa_quant_states = {}  # 用于 update_fa_quant_type 的状态记录
    return saver


def _make_int8_per_head_module(input_scale=None, dtype='int8', scope='STATIC'):
    module = MagicMock(spec=qir.INT8FakeQuantActivationPerHead)
    if input_scale is None:
        module.input_scale = torch.tensor([0.1], dtype=torch.float32)
    else:
        module.input_scale = torch.tensor(input_scale, dtype=torch.float32)
    module.x_q_scheme = MagicMock()
    module.x_q_scheme.dtype = dtype
    module.x_q_scheme.scope = QScope.PER_TOKEN if scope.upper() == 'DYNAMIC' else QScope.PER_TENSOR
    return module


def _make_fp8_per_head_module(input_scale=None, dtype='fp8_e4m3', scope='STATIC'):
    module = MagicMock(spec=qir.FP8FakeQuantActivationPerHead)
    if input_scale is None:
        module.input_scale = torch.tensor([0.2], dtype=torch.float32)
    else:
        module.input_scale = torch.tensor(input_scale, dtype=torch.float32)
    module.x_q_scheme = MagicMock()
    module.x_q_scheme.dtype = dtype
    module.x_q_scheme.scope = QScope.PER_TOKEN if scope.upper() == 'DYNAMIC' else QScope.PER_TENSOR
    return module


def _make_per_token_module(dtype='fp8_e4m3', scope='DYNAMIC'):
    module = MagicMock(spec=qir.FakeQuantActivationPerToken)
    module.x_q_scheme = MagicMock()
    module.x_q_scheme.dtype = dtype
    module.x_q_scheme.scope = QScope.PER_TOKEN if scope.upper() == 'DYNAMIC' else QScope.PER_TENSOR
    return module


class TestOnInt8ActivationPerHead:
    """on_int8_activation_per_head 方法的行为"""

    def test_writes_scale_and_int8_offset_when_called(self):
        """正常调用时应写入 scale 和 int8 offset，并设置 fa_quant_type"""
        saver = _make_mock_saver()
        prefix = "model.layers.0.self_attn.fa_k"
        module = _make_int8_per_head_module(input_scale=[0.15])

        saver.on_int8_activation_per_head(prefix, module)

        # 检查 write_tensor 被调用两次
        assert saver.write_tensor.call_count == 2
        # 第一次：scale
        args1 = saver.write_tensor.call_args_list[0][0]
        assert args1[0] == prefix + ".scale"
        assert args1[1] == "FAQuant"
        # scale 应该被 unsqueeze(-1) 过，从 1D 变成 2D (1,1)
        expected_scale = torch.tensor([0.15], dtype=torch.float32).unsqueeze(-1)
        assert torch.equal(args1[2], expected_scale)

        # 第二次：offset，dtype 为 int8
        args2 = saver.write_tensor.call_args_list[1][0]
        assert args2[0] == prefix + ".offset"
        assert args2[1] == "FAQuant"
        assert args2[2].dtype == torch.int8
        assert args2[2].shape == expected_scale.shape
        assert (args2[2] == 0).all()

        # json_append 中应设置 'fa_quant_type': 'FAKQuant'
        assert saver.json_append.get('fa_quant_type') == "FAKQuant"


class TestOnFp8ActivationPerHead:
    """on_fp8_activation_per_head 方法的行为"""

    def test_writes_scale_and_float32_offset_when_called(self):
        saver = _make_mock_saver()
        prefix = "model.layers.1.self_attn.fa_v"
        module = _make_fp8_per_head_module(input_scale=[0.3, 0.4])

        saver.on_fp8_activation_per_head(prefix, module)

        assert saver.write_tensor.call_count == 2
        # scale
        args_scale = saver.write_tensor.call_args_list[0][0]
        assert args_scale[0] == prefix + ".scale"
        assert args_scale[1] == "FAQuant"
        # offset
        args_off = saver.write_tensor.call_args_list[1][0]
        assert args_off[0] == prefix + ".offset"
        assert args_off[1] == "FAQuant"
        # offset 应为 float32，而非 int8
        assert args_off[2].dtype == torch.float32
        assert args_off[2].shape == args_scale[2].shape
        # fa_quant_type 同样设置为 FAKQuant
        assert saver.json_append.get('fa_quant_type') == "FAKQuant"


class TestOnActivationPerToken:
    """on_activation_per_token 方法应仅触发 update_fa_quant_type"""

    def test_only_calls_update_fa_quant_type_when_called(self):
        saver = _make_mock_saver()
        prefix = "model.layers.2.self_attn.fa_q"
        module = _make_per_token_module(dtype='fp8_e4m3', scope='DYNAMIC')

        with patch.object(saver, 'update_fa_quant_type') as mock_update:
            saver.on_activation_per_token(prefix, module)

        mock_update.assert_called_once_with(prefix, module)
        # 不应调用 write_tensor 或修改 json_append
        saver.write_tensor.assert_not_called()
        assert 'fa_quant_type' not in saver.json_append


class TestUpdateFaQuantType:
    """update_fa_quant_type 的 quant_type 字符串生成规则"""

    def _call_update(self, saver, prefix, module):
        """快捷调用，直接执行被测方法"""
        saver.update_fa_quant_type(prefix, module)

    def _assert_json_write(self, saver, expected_key, expected_value):
        """验证 json_writer.write 被调用并传入正确的键值"""
        saver.json_writer.write.assert_called_once_with(expected_key, expected_value)
        saver.json_writer.write.reset_mock()

    def test_single_q_static_fp8_generates_fp8(self):
        """单个 Q 激活，FP8 静态 -> 'FP8'"""
        saver = _make_mock_saver()
        prefix = "layers.0.self_attn.fa_q"
        module = _make_fp8_per_head_module(dtype='fp8_e4m3', scope='STATIC')

        self._call_update(saver, prefix, module)
        self._assert_json_write(saver, "layers.0.self_attn.quant_type", "Q_FP8")

    def test_single_q_dynamic_int8_generates_int8_dynamic(self):
        """单个 Q 激活，INT8 动态 -> 'INT8_DYNAMIC'"""
        saver = _make_mock_saver()
        prefix = "layers.0.self_attn.fa_q"
        module = _make_int8_per_head_module(dtype='int8', scope='DYNAMIC')

        self._call_update(saver, prefix, module)
        self._assert_json_write(saver, "layers.0.self_attn.quant_type", "Q_INT8_DYNAMIC")

    def test_qkv_all_static_fp8_generates_fp8(self):
        """Q、K、V 都是 FP8 STATIC，合并为 QKV 前缀省略 -> 'FP8'"""
        saver = _make_mock_saver()
        # 依次添加 Q, K, V
        for act in ['Q', 'K', 'V']:
            prefix = f"layers.1.self_attn.fa_{act.lower()}"
            module = _make_fp8_per_head_module(dtype='fp8_e4m3', scope='STATIC')
            self._call_update(saver, prefix, module)
            # 中间调用会多次写 json，最后一个调用决定最终值
        # 最后一次写入应该覆盖为合并结果
        saver.json_writer.write.assert_called_with("layers.1.self_attn.quant_type", "FP8")

    def test_qkv_all_dynamic_fp8_generates_fp8_dynamic(self):
        """Q、K、V 都是 FP8 DYNAMIC，合并后为 'FP8_DYNAMIC'"""
        saver = _make_mock_saver()
        for act in ['Q', 'K', 'V']:
            prefix = f"layers.2.self_attn.fa_{act.lower()}"
            module = _make_fp8_per_head_module(dtype='fp8_e4m3', scope='DYNAMIC')
            self._call_update(saver, prefix, module)
        saver.json_writer.write.assert_called_with("layers.2.self_attn.quant_type", "FP8_DYNAMIC")

    def test_q_int8_k_fp8_v_int8_generates_mixed_parts(self):
        """
        Q: INT8 STATIC, K: FP8 STATIC, V: INT8 STATIC
        组1 (INT8): Q,V -> 'QV_INT8'
        组2 (FP8): K   -> 'K_FP8'
        最终: 'QV_INT8_K_FP8'
        """
        saver = _make_mock_saver()
        # Q: int8 static
        self._call_update(saver, "l.s.fa_q", _make_int8_per_head_module(dtype='int8', scope='STATIC'))
        # K: fp8 static
        self._call_update(saver, "l.s.fa_k", _make_fp8_per_head_module(dtype='fp8_e4m3', scope='STATIC'))
        # V: int8 static
        self._call_update(saver, "l.s.fa_v", _make_int8_per_head_module(dtype='int8', scope='STATIC'))

        saver.json_writer.write.assert_called_with("l.s.quant_type", "QV_INT8_K_FP8")

    def test_qkvp_mixed_with_dynamic_and_static_omits_proper_prefixes(self):
        """
        Q: FP8 DYNAMIC, K: FP8 STATIC, V: FP8 STATIC, P: FP8 DYNAMIC
        组1 (FP8 DYNAMIC): Q,P -> 'QP_FP8_DYNAMIC'
        组2 (FP8 STATIC):  K,V -> 'KV_FP8'
        最终: 'QP_FP8_DYNAMIC_KV_FP8'
        """
        saver = _make_mock_saver()
        self._call_update(saver, "l.s.fa_q", _make_fp8_per_head_module(dtype='fp8_e4m3', scope='DYNAMIC'))
        self._call_update(saver, "l.s.fa_k", _make_fp8_per_head_module(dtype='fp8_e4m3', scope='STATIC'))
        self._call_update(saver, "l.s.fa_v", _make_fp8_per_head_module(dtype='fp8_e4m3', scope='STATIC'))
        self._call_update(saver, "l.s.fa_p", _make_fp8_per_head_module(dtype='fp8_e4m3', scope='DYNAMIC'))

        saver.json_writer.write.assert_called_with("l.s.quant_type", "QP_FP8_DYNAMIC_KV_FP8")

    def test_unsupported_dtype_raises_schema_error(self):
        """当 dtype 不在 DTYPE_PREFIX_MAP 中时，应抛出 SchemaValidateError"""
        saver = _make_mock_saver()
        # 使用一个不存在的 dtype
        module = _make_int8_per_head_module(dtype='unknown_dtype', scope='STATIC')
        with pytest.raises(SchemaValidateError, match="Unsupported dtype"):
            self._call_update(saver, "l.s.fa_q", module)

    def test_same_config_merges_activations_in_correct_order(self):
        """
        确保合并时按 Q, K, V, P固定顺序收集激活。
        例如添加顺序为 V, Q, K 均为 INT8 DYNAMIC，最终应为 'INT8_DYNAMIC'（前缀省略，因为 QKV 齐全）。
        """
        saver = _make_mock_saver()
        # 故意乱序添加
        self._call_update(saver, "l.s.fa_v", _make_int8_per_head_module(dtype='int8', scope='DYNAMIC'))
        self._call_update(saver, "l.s.fa_q", _make_int8_per_head_module(dtype='int8', scope='DYNAMIC'))
        self._call_update(saver, "l.s.fa_k", _make_int8_per_head_module(dtype='int8', scope='DYNAMIC'))

        saver.json_writer.write.assert_called_with("l.s.quant_type", "INT8_DYNAMIC")

    def test_partial_activations_keep_correct_prefix(self):
        """
        只有 K 和 V 都是 FP8 STATIC，应为 'KV_FP8'。
        """
        saver = _make_mock_saver()
        self._call_update(saver, "l.s.fa_k", _make_fp8_per_head_module(dtype='fp8_e4m3', scope='STATIC'))
        self._call_update(saver, "l.s.fa_v", _make_fp8_per_head_module(dtype='fp8_e4m3', scope='STATIC'))

        saver.json_writer.write.assert_called_with("l.s.quant_type", "KV_FP8")


class TestUpdateGlobalFaQuantType:
    """测试 AscendV1Saver.update_global_fa_quant_type 方法"""

    def test_sets_value_when_value_not_none(self):
        """fa_quant_states 不为 None 时，应设置 states 值"""
        saver = _make_mock_saver()
        saver.fa_quant_states["model.layer.0.self_attn.fa_q"] = {"Q", "FP8"}
        AscendV1Saver.update_global_fa_quant_type(saver, states="FAQuant")
        assert saver.json_append["fa_quant_type"] == "FAQuant"

    def test_sets_to_none_when_states_is_none_and_fa_quant_states_not_none(self):
        """fa_quant_states 不为 None 但 states 为 None 时，fa_quant_type 应设为 None"""
        saver = _make_mock_saver()
        saver.fa_quant_states["model.layer.0.self_attn.fa_q"] = {"Q", "FP8"}
        AscendV1Saver.update_global_fa_quant_type(saver)
        assert saver.json_append["fa_quant_type"] is None

    def test_does_nothing_when_fa_quant_states_is_none(self):
        """fa_quant_states 为 None 时，json_append 不应有任何改动"""
        saver = _make_mock_saver()
        original_json = saver.json_append.copy()
        AscendV1Saver.update_global_fa_quant_type(saver, states="any_value")
        assert "fa_quant_type" not in saver.json_append
        assert saver.json_append == original_json

    def test_overwrites_with_none_when_states_is_none_and_fa_quant_states_not_none(self):
        """已有 fa_quant_type 且 states 为 None 时，值应变为 None"""
        saver = _make_mock_saver()
        saver.fa_quant_states["model.layer.0.self_attn.fa_q"] = {"Q", "FP8"}
        AscendV1Saver.update_global_fa_quant_type(saver, states="any_value")
        AscendV1Saver.update_global_fa_quant_type(saver)
        assert saver.json_append["fa_quant_type"] is None
