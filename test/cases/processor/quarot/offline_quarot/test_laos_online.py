#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.processor.quarot.offline_quarot.laos_online 模块的单元测试
"""

# pylint: disable=abstract-method

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import msmodelslim.ir as qir
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.quarot.offline_quarot.laos_online import (
    LAOSOnlineRotationProcessor,
    _convert_hookir_to_wrapper,
    _get_available_device,
    _get_full_module_name,
)
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import LAOSOnlineRotationInterface
from msmodelslim.utils.exception import UnsupportedError


class _Adapter(LAOSOnlineRotationInterface):
    def __init__(self, num_attention_heads=4, head_dim=8):
        self._num_attention_heads = num_attention_heads
        self._head_dim = head_dim

    def get_head_dim(self):
        return self._head_dim

    def get_num_attention_heads(self):
        return self._num_attention_heads

    def get_layer_wise_ov_pair(self, decoder_module):
        return {}

    def get_layer_wise_up_down_pair(self, decoder_module):
        return {}


class _FakeModel(nn.Module):
    def __init__(self, intermediate_size=16):
        super().__init__()
        self.config = SimpleNamespace(intermediate_size=intermediate_size)
        self.device = torch.device("cpu")


class _DummyHookIR(qir.HookIR):
    def __init__(self):
        super().__init__()
        self.wrapped = None

    def wrapper_module(self, module):
        self.wrapped = module
        return nn.Identity()

    def __call__(self, module, args, kwargs):
        return None


def _config(max_tp_size=4):
    return SimpleNamespace(max_tp_size=max_tp_size, down_proj_online_layers=[0])


class TestGetAvailableDevice:
    def test_returns_cpu_when_no_accelerator(self):
        assert _get_available_device() == torch.device("cpu")

    def test_index_does_not_change_cpu_result(self):
        assert _get_available_device(index=3) == torch.device("cpu")


class TestGetFullModuleName:
    def test_prefixes_request_name_for_child_module(self):
        model = nn.Sequential(nn.Linear(2, 2))
        request = BatchProcessRequest(name="model.layers.3", module=model, datas=None)

        assert _get_full_module_name(model[0], request) == "model.layers.3.0"

    def test_returns_request_name_for_root_module(self):
        model = nn.Sequential(nn.Linear(2, 2))
        request = BatchProcessRequest(name="model.layers.3", module=model, datas=None)

        assert _get_full_module_name(model, request) == "model.layers.3"

    def test_raises_when_target_is_not_in_tree(self):
        model = nn.Sequential(nn.Linear(2, 2))
        request = BatchProcessRequest(name="model.layers.3", module=model, datas=None)

        with pytest.raises(UnsupportedError):
            _get_full_module_name(nn.Linear(2, 2), request)


class TestConvertHookIRToWrapper:
    def test_replaces_module_that_carries_hook_ir(self):
        linear = nn.Linear(2, 2)
        hook = _DummyHookIR()
        linear.register_forward_pre_hook(hook)
        model = nn.Sequential(linear)

        _convert_hookir_to_wrapper(model)

        assert isinstance(model[0], nn.Identity)
        assert hook.wrapped is linear

    def test_leaves_plain_pre_hook_in_place(self):
        linear = nn.Linear(2, 2)
        linear.register_forward_pre_hook(lambda module, inputs: None)
        model = nn.Sequential(linear)

        _convert_hookir_to_wrapper(model)

        assert model[0] is linear


class TestLAOSOnlineRotationProcessorInit:
    def test_rejects_adapter_without_interface(self):
        with pytest.raises(UnsupportedError):
            LAOSOnlineRotationProcessor(nn.Module(), _config(), adapter=object())

    def test_reads_num_attention_heads_from_adapter(self):
        processor = LAOSOnlineRotationProcessor(_FakeModel(), _config(), _Adapter(num_attention_heads=6))

        assert processor.num_attn_heads == 6


class TestAddOnlineRotations:
    def test_returns_three_square_matrices_from_intermediate_size(self):
        model = _FakeModel(intermediate_size=16)
        processor = LAOSOnlineRotationProcessor(model, _config(max_tp_size=4), _Adapter(num_attention_heads=4))

        rot1, rot2, rot_o_proj = processor._add_online_rotations(model)

        assert tuple(rot1.shape) == (4, 4)
        assert tuple(rot2.shape) == (4, 4)
        assert tuple(rot_o_proj.shape) == (4, 4)

    def test_requires_intermediate_size_in_config(self):
        model = _FakeModel()
        del model.config.intermediate_size
        processor = LAOSOnlineRotationProcessor(model, _config(max_tp_size=4), _Adapter(num_attention_heads=4))

        with pytest.raises(ValueError):
            processor._add_online_rotations(model)
