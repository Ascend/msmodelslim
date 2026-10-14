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
# pylint: disable=no-member

import torch
import types
import pytest

import msmodelslim.pytorch.llm_ptq.llm_ptq_tools.kv_cache_utils as kv


class DummyCfg:
    def __init__(self):
        self.kv_sym = True


class DummyMod:
    def __init__(self):
        self.config = types.SimpleNamespace()
        self.config.num_layers = 2


def test_set_kvcache_vari_func_sets_attributes():
    mod = DummyMod()
    cache_sub_dict = {
        'k_proj_scale': torch.tensor([1.0]),
        'k_proj_offset': torch.tensor([0.0]),
        'v_proj_scale': torch.tensor([2.0]),
        'v_proj_offset': torch.tensor([0.0]),
    }
    cfg = DummyCfg()
    kv.set_kvcache_vari_func(mod, cache_sub_dict, cfg, num_layers=3)
    # 检查属性
    assert hasattr(mod, 'is_calib')
    assert hasattr(mod, 'k_scale')
    assert hasattr(mod, 'v_scale')
    assert hasattr(mod, 'disable_calib')
    assert hasattr(mod, 'enable_calib')
    assert hasattr(mod, 'any_none')
    assert hasattr(mod, 'get_kvcache_scale_offset')
    assert hasattr(mod, 'num_hidden_layers')
    assert mod.num_hidden_layers == 2  # 优先config.num_layers


def test_set_kvcache_vari_func_num_layers():
    mod = DummyMod()
    delattr(mod.config, 'num_layers')
    cache_sub_dict = {}
    cfg = DummyCfg()
    kv.set_kvcache_vari_func(mod, cache_sub_dict, cfg, num_layers=5)
    assert mod.num_hidden_layers == 5


def test_update_extremum():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([2.0, 1.0])
    # max
    out = kv.update_extremum(a, torch.max, b)
    assert torch.allclose(out, torch.tensor([2.0, 2.0]))
    # min
    out = kv.update_extremum(a, torch.min, b)
    assert torch.allclose(out, torch.tensor([1.0, 1.0]))
    # None
    out = kv.update_extremum(None, torch.max, b)
    assert torch.allclose(out, b)


def test_process_tensor():
    t = torch.arange(24).float().reshape(2, 3, 4)
    channel_max, channel_min = kv.process_tensor(2, 3, None, None, t)
    assert channel_max.shape == (4,)
    assert channel_min.shape == (4,)


def test_split_kvcache_tuple(monkeypatch):
    t = torch.arange(12).reshape(1, 2, 6).float()
    monkeypatch.setattr("torch.chunk", lambda x, chunks, dim: t)
    k, v = kv.split_kvcache(t)
    assert isinstance(k, list) and isinstance(v, list)
    assert k[0].shape == v[0].shape


def test_split_kvcache_tensor():
    t = torch.stack([torch.ones(2, 2), torch.zeros(2, 2)])
    kvcache = torch.stack([t, t])
    k, v = kv.split_kvcache(kvcache)
    assert isinstance(k, list) and isinstance(v, list)


class DummyCache:
    def __init__(self):
        self.key_cache = [torch.ones(2, 2)]
        self.value_cache = [torch.zeros(2, 2)]


def test_split_kvcache_object():
    kvcache = DummyCache()
    k, v = kv.split_kvcache(kvcache)
    assert isinstance(k, list) and isinstance(v, list)


def test_get_var_index():
    x = [torch.zeros(2, 3)]
    cache_k = [torch.zeros(2, 3)]
    seq_idx, bz_idx = kv.get_var_index(x, cache_k)
    assert seq_idx.numel() > 0
    assert bz_idx.numel() > 0


def test_update_bz_seq():
    t = torch.zeros(2, 3, 4)
    bz, seq = kv.update_bz_seq(t, 0, 1)
    assert bz == 2 and seq == 3


def test_fake_quantize_process(monkeypatch):
    t = torch.ones(2, 3, 4)
    monkeypatch.setattr(kv, "fake_quantize", lambda tensor, scale, offset, bits, is_signed, dequant: (None, tensor + 1))
    out = kv.fake_quantize_process(2, 3, 1.0, 0.0, t)
    assert torch.allclose(out, t + 1)


def test_fake_quantize_cache(monkeypatch):
    monkeypatch.setattr(kv, "fake_quantize_process", lambda b, s, scale, offset, t: t + 1)
    k = torch.ones(2, 3, 4)
    v = torch.ones(2, 3, 4)
    kv_scale_offset = [[1.0, 0.0], [2.0, 0.0]]
    out_k, out_v = kv.fake_quantize_cache(2, 3, kv_scale_offset, k, v)
    assert torch.allclose(out_k, k + 1)
    assert torch.allclose(out_v, v + 1)


def test_disable_enable_calib():
    class M:
        pass

    m = M()
    kv.disable_calib(m)
    assert m.is_calib is False
    kv.enable_calib(m)
    assert m.is_calib is True


def test_get_kvcache_scale_offset(monkeypatch):
    class M:
        def __init__(self):
            self.k_min = 0
            self.k_max = 1
            self.v_min = 0
            self.v_max = 1
            self.kv_sym = True

        def __setattr__(self, k, v):
            object.__setattr__(self, k, v)

    m = M()
    monkeypatch.setattr(kv, "linear_quantization_params", lambda *a, **k: (1.0, 0.0))
    kv.get_kvcache_scale_offset(m)
    assert m.k_scale == 1.0
    assert m.v_scale == 1.0


def test_any_none():
    class M:
        k_scale = None
        k_offset = 1
        v_scale = 1
        v_offset = 1

    m = M()
    assert kv.any_none(m)
    m.k_scale = 1
    assert not kv.any_none(m)


def test_new_forward_missing_use_cache():
    class M:
        def __init__(self):
            self.original_forward = lambda *a, **k: (1, 2, 3, None)

    m = M()
    with pytest.raises(Exception):
        kv.new_forward(m, 1, 2, a=1)


def test_new_forward_with_cache(monkeypatch):
    class M:
        def __init__(self):
            self.original_forward = lambda *a, **k: (torch.zeros(2, 3, 4), 2, 3, (torch.zeros(2, 3, 4),))
            self.is_calib = False
            self.cache_seq_index = None
            self.cache_bz_index = None
            self.any_none = lambda: True
            self.get_kvcache_scale_offset = lambda: None
            self.k_scale = 1
            self.k_offset = 0
            self.v_scale = 1
            self.v_offset = 0
            self.num_hidden_layers = 1
            self.layer_idx = 0

        def __setattr__(self, k, v):
            object.__setattr__(self, k, v)

    m = M()
    monkeypatch.setattr(kv, "split_kvcache", lambda kvcache: ([torch.zeros(2, 3, 4)], [torch.zeros(2, 3, 4)]))
    monkeypatch.setattr(kv, "update_bz_seq", lambda t, b, s: (2, 3))
    monkeypatch.setattr(kv, "fake_quantize_cache", lambda b, s, so, k, v: (k, v))
    out = kv.new_forward(m, torch.zeros(2, 3, 4), use_cache=True)
    assert out is not None
