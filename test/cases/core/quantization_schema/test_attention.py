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

import unittest

import torch

from msmodelslim.core.quantization_schema.attention import FakeQuantCacheINT8PerChannel
from msmodelslim.core.quantization_schema.attention_dynamic import FakeQuantDynamicCache
from msmodelslim.core.quantization_schema.attention_fp8 import FakeQuantCacheFP8PerChannel, FakeQuantCacheFP8PerHead
from msmodelslim.core.quantization_schema.attention_fp4 import FakeQuantCacheFP4PerChannel, FakeQuantCacheFP4PerHead
from msmodelslim.core.quantization_schema.qal import QParam, QScope, QDType
from msmodelslim.core.quantization_schema.const import (
    int8_per_channel_sym,
    int8_per_token_sym,
    int8_per_tile_sym,
    fp8_e4m3_per_channel_sym,
    fp8_e4m3_per_head_sym,
    fp8_e4m3_per_token_sym,
    fp8_e4m3_per_tile_sym,
    fp4_e2m1_per_channel_sym,
    fp4_e2m1_per_head_sym,
    fp4_e2m1_per_block_sym,
    fp4_e2m1_per_token_sym,
    mxfp4_per_block_sym,
    mxfp8_per_block_sym,
    mxfp8_per_token_sym,
    mxfp8_per_tile_sym,
    hif4_per_block_sym,
)


class TestFakeQuantCacheINT8PerChannel(unittest.TestCase):
    """Test FakeQuantCacheINT8PerChannel (INT8 per-channel static KV cache)"""

    def setUp(self):
        self.num_heads = 4
        self.head_dim = 16
        self.channels = self.num_heads * self.head_dim
        self.scale = torch.ones(self.channels, dtype=torch.float32)
        self.offset = torch.zeros(self.channels, dtype=torch.int32)
        self.q_param = QParam(
            scheme=int8_per_channel_sym,
            ext={
                "scale": self.scale,
                "offset": self.offset,
            },
        )

    def test_init(self):
        ir_module = FakeQuantCacheINT8PerChannel(self.q_param)
        self.assertEqual(ir_module.x_q_scheme, int8_per_channel_sym)
        self.assertIsNotNone(ir_module.kv_cache_scale)
        self.assertIsNotNone(ir_module.kv_cache_offset)

    def test_forward_preserves_shape(self):
        ir_module = FakeQuantCacheINT8PerChannel(self.q_param)
        x = torch.randn(2, self.num_heads, 10, self.head_dim)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_produces_finite_output(self):
        ir_module = FakeQuantCacheINT8PerChannel(self.q_param)
        x = torch.randn(2, self.num_heads, 10, self.head_dim)
        with torch.no_grad():
            output = ir_module(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_differs_from_input(self):
        ir_module = FakeQuantCacheINT8PerChannel(self.q_param)
        x = torch.randn(2, self.num_heads, 10, self.head_dim)
        with torch.no_grad():
            output = ir_module(x)
        self.assertFalse(torch.equal(output, x))

    def test_forward_small_head_dim(self):
        channels = 16
        scale = torch.ones(channels, dtype=torch.float32)
        offset = torch.zeros(channels, dtype=torch.int32)
        q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": scale, "offset": offset})
        ir_module = FakeQuantCacheINT8PerChannel(q_param)
        x = torch.randn(2, 4, 10, 4)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_batch_variation(self):
        ir_module = FakeQuantCacheINT8PerChannel(self.q_param)
        x = torch.randn(1, 1, 1, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_scheme_property(self):
        ir_module = FakeQuantCacheINT8PerChannel(self.q_param)
        self.assertEqual(ir_module.x_q_scheme.scope, QScope.PER_CHANNEL)
        self.assertEqual(ir_module.x_q_scheme.dtype, QDType.INT8)
        self.assertTrue(ir_module.x_q_scheme.symmetric)

    def test_kv_cache_offset_present(self):
        ir_module = FakeQuantCacheINT8PerChannel(self.q_param)
        self.assertIsNotNone(ir_module.kv_cache_offset)
        self.assertEqual(ir_module.kv_cache_offset.shape, self.offset.shape)


class TestFakeQuantCacheFP8PerChannel(unittest.TestCase):
    """Test FakeQuantCacheFP8PerChannel (FP8 per-channel static KV cache)"""

    def setUp(self):
        self.scale = torch.ones(64, dtype=torch.float32)
        self.q_param = QParam(
            scheme=fp8_e4m3_per_channel_sym,
            ext={
                "scale": self.scale,
            },
        )

    def test_init(self):
        ir_module = FakeQuantCacheFP8PerChannel(self.q_param)
        self.assertEqual(ir_module.x_q_scheme, fp8_e4m3_per_channel_sym)
        self.assertIsNotNone(ir_module.kv_cache_scale)
        self.assertIsNone(ir_module.kv_cache_offset)

    def test_init_raises_when_scale_missing(self):
        q_param = QParam(scheme=fp8_e4m3_per_channel_sym, ext={})
        with self.assertRaises(ValueError):
            FakeQuantCacheFP8PerChannel(q_param)

    def test_forward_preserves_shape(self):
        ir_module = FakeQuantCacheFP8PerChannel(self.q_param)
        x = torch.randn(2, 4, 10, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_produces_finite_output(self):
        ir_module = FakeQuantCacheFP8PerChannel(self.q_param)
        x = torch.randn(2, 4, 10, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_differs_from_input(self):
        scale = torch.full((64,), 0.01, dtype=torch.float32)
        q_param = QParam(scheme=fp8_e4m3_per_channel_sym, ext={"scale": scale})
        ir_module = FakeQuantCacheFP8PerChannel(q_param)
        x = torch.randn(2, 4, 10, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertFalse(torch.equal(output, x))

    def test_forward_small_head_dim(self):
        scale = torch.ones(16, dtype=torch.float32)
        q_param = QParam(scheme=fp8_e4m3_per_channel_sym, ext={"scale": scale})
        ir_module = FakeQuantCacheFP8PerChannel(q_param)
        x = torch.randn(2, 4, 10, 16)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_single_element(self):
        ir_module = FakeQuantCacheFP8PerChannel(self.q_param)
        x = torch.randn(1, 1, 1, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_scheme_property(self):
        ir_module = FakeQuantCacheFP8PerChannel(self.q_param)
        self.assertEqual(ir_module.x_q_scheme.scope, QScope.PER_CHANNEL)
        self.assertEqual(ir_module.x_q_scheme.dtype, QDType.FP8_E4M3)
        self.assertTrue(ir_module.x_q_scheme.symmetric)


class TestFakeQuantCacheFP8PerHead(unittest.TestCase):
    """Test FakeQuantCacheFP8PerHead (FP8 per-head static KV cache)"""

    def setUp(self):
        self.scale = torch.ones(8, dtype=torch.float32)
        self.q_param = QParam(
            scheme=fp8_e4m3_per_head_sym,
            ext={
                "scale": self.scale,
            },
        )

    def test_init(self):
        ir_module = FakeQuantCacheFP8PerHead(self.q_param)
        self.assertEqual(ir_module.x_q_scheme, fp8_e4m3_per_head_sym)
        self.assertIsNotNone(ir_module.kv_cache_scale)
        self.assertIsNone(ir_module.kv_cache_offset)

    def test_init_raises_when_scale_missing(self):
        q_param = QParam(scheme=fp8_e4m3_per_head_sym, ext={})
        with self.assertRaises(ValueError):
            FakeQuantCacheFP8PerHead(q_param)

    def test_forward_preserves_shape(self):
        ir_module = FakeQuantCacheFP8PerHead(self.q_param)
        x = torch.randn(2, 8, 10, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_produces_finite_output(self):
        ir_module = FakeQuantCacheFP8PerHead(self.q_param)
        x = torch.randn(2, 8, 10, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_scheme_property(self):
        ir_module = FakeQuantCacheFP8PerHead(self.q_param)
        self.assertEqual(ir_module.x_q_scheme.scope, QScope.PER_HEAD)
        self.assertEqual(ir_module.x_q_scheme.dtype, QDType.FP8_E4M3)


def _verify_dynamic_cache(scheme):
    """验证传入 QScheme 后 FakeQuantDynamicCache 能正确创建和推理。"""
    q_param = QParam(scheme=scheme)
    ir_module = FakeQuantDynamicCache(q_param)
    assert ir_module.x_q_scheme == scheme
    assert ir_module.kv_cache_scale is not None
    assert ir_module.kv_cache_offset is None
    x = torch.randn(2, 4, 10, 64)
    with torch.no_grad():
        output = ir_module(x)
    assert torch.equal(output, x)
    assert output.shape == x.shape


class TestFakeQuantDynamicCache(unittest.TestCase):
    """Test FakeQuantDynamicCache (dynamic KV cache — per-token/per-tile/per-block)"""

    def test_int8_per_token(self):
        _verify_dynamic_cache(int8_per_token_sym)

    def test_int8_per_tile(self):
        _verify_dynamic_cache(int8_per_tile_sym)

    def test_fp8_e4m3_per_token(self):
        _verify_dynamic_cache(fp8_e4m3_per_token_sym)

    def test_fp8_e4m3_per_tile(self):
        _verify_dynamic_cache(fp8_e4m3_per_tile_sym)

    def test_fp4_e2m1_per_token(self):
        _verify_dynamic_cache(fp4_e2m1_per_token_sym)

    def test_fp4_e2m1_per_block(self):
        _verify_dynamic_cache(fp4_e2m1_per_block_sym)

    def test_mxfp4_per_block(self):
        _verify_dynamic_cache(mxfp4_per_block_sym)

    def test_mxfp8_per_block(self):
        _verify_dynamic_cache(mxfp8_per_block_sym)

    def test_mxfp8_per_token(self):
        _verify_dynamic_cache(mxfp8_per_token_sym)

    def test_mxfp8_per_tile(self):
        _verify_dynamic_cache(mxfp8_per_tile_sym)

    def test_hif4_per_block(self):
        _verify_dynamic_cache(hif4_per_block_sym)

    def test_init_with_scale_in_ext(self):
        q_param = QParam(scheme=int8_per_token_sym, ext={"scale": torch.ones(1)})
        ir_module = FakeQuantDynamicCache(q_param)
        self.assertEqual(ir_module.x_q_scheme, int8_per_token_sym)
        self.assertIsNotNone(ir_module.kv_cache_scale)

    def test_init_without_scale_uses_fallback(self):
        q_param = QParam(scheme=int8_per_token_sym)
        ir_module = FakeQuantDynamicCache(q_param)
        self.assertIsNotNone(ir_module.kv_cache_scale)

    def test_forward_no_op(self):
        q_param = QParam(scheme=int8_per_token_sym)
        ir_module = FakeQuantDynamicCache(q_param)
        x = torch.randn(2, 4, 10, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertTrue(torch.equal(output, x))

    def test_forward_preserves_shape(self):
        q_param = QParam(scheme=int8_per_token_sym)
        ir_module = FakeQuantDynamicCache(q_param)
        x = torch.randn(2, 4, 10, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_scheme_property(self):
        q_param = QParam(scheme=int8_per_token_sym)
        ir_module = FakeQuantDynamicCache(q_param)
        self.assertEqual(ir_module.x_q_scheme, int8_per_token_sym)

    def test_kv_cache_offset_is_none(self):
        q_param = QParam(scheme=int8_per_token_sym)
        ir_module = FakeQuantDynamicCache(q_param)
        self.assertIsNone(ir_module.kv_cache_offset)


class TestFakeQuantCacheFP4PerChannel(unittest.TestCase):
    """Test FakeQuantCacheFP4PerChannel (FP4 per-channel static KV cache)"""

    def setUp(self):
        self.num_heads = 4
        self.head_dim = 64
        self.channels = self.num_heads * self.head_dim
        self.scale = torch.ones(self.channels, dtype=torch.float32)
        self.q_param = QParam(
            scheme=fp4_e2m1_per_channel_sym,
            ext={
                "scale": self.scale,
            },
        )

    def test_init(self):
        ir_module = FakeQuantCacheFP4PerChannel(self.q_param)
        self.assertEqual(ir_module.x_q_scheme, fp4_e2m1_per_channel_sym)
        self.assertIsNotNone(ir_module.kv_cache_scale)
        self.assertIsNone(ir_module.kv_cache_offset)

    def test_init_raises_when_scale_missing(self):
        q_param = QParam(scheme=fp4_e2m1_per_channel_sym, ext={})
        with self.assertRaises(ValueError):
            FakeQuantCacheFP4PerChannel(q_param)

    def test_forward_preserves_shape(self):
        ir_module = FakeQuantCacheFP4PerChannel(self.q_param)
        x = torch.randn(2, self.num_heads, 10, self.head_dim)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_produces_finite_output(self):
        ir_module = FakeQuantCacheFP4PerChannel(self.q_param)
        x = torch.randn(2, self.num_heads, 10, self.head_dim)
        with torch.no_grad():
            output = ir_module(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_differs_from_input(self):
        scale = torch.full((self.channels,), 0.01, dtype=torch.float32)
        q_param = QParam(scheme=fp4_e2m1_per_channel_sym, ext={"scale": scale})
        ir_module = FakeQuantCacheFP4PerChannel(q_param)
        x = torch.randn(2, self.num_heads, 10, self.head_dim)
        with torch.no_grad():
            output = ir_module(x)
        self.assertFalse(torch.equal(output, x))

    def test_forward_small_head_dim(self):
        scale = torch.ones(16, dtype=torch.float32)
        q_param = QParam(scheme=fp4_e2m1_per_channel_sym, ext={"scale": scale})
        ir_module = FakeQuantCacheFP4PerChannel(q_param)
        x = torch.randn(2, 4, 10, 4)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_single_element(self):
        ir_module = FakeQuantCacheFP4PerChannel(self.q_param)
        x = torch.randn(1, 1, 1, self.channels)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_scheme_property(self):
        ir_module = FakeQuantCacheFP4PerChannel(self.q_param)
        self.assertEqual(ir_module.x_q_scheme.scope, QScope.PER_CHANNEL)
        self.assertEqual(ir_module.x_q_scheme.dtype, QDType.FP4_E2M1)
        self.assertTrue(ir_module.x_q_scheme.symmetric)


class TestFakeQuantCacheFP4PerHead(unittest.TestCase):
    """Test FakeQuantCacheFP4PerHead (FP4 per-head static KV cache)"""

    def setUp(self):
        self.num_heads = 8
        self.scale = torch.ones(self.num_heads, dtype=torch.float32)
        self.q_param = QParam(
            scheme=fp4_e2m1_per_head_sym,
            ext={
                "scale": self.scale,
            },
        )

    def test_init(self):
        ir_module = FakeQuantCacheFP4PerHead(self.q_param)
        self.assertEqual(ir_module.x_q_scheme, fp4_e2m1_per_head_sym)
        self.assertIsNotNone(ir_module.kv_cache_scale)
        self.assertIsNone(ir_module.kv_cache_offset)

    def test_init_raises_when_scale_missing(self):
        q_param = QParam(scheme=fp4_e2m1_per_head_sym, ext={})
        with self.assertRaises(ValueError):
            FakeQuantCacheFP4PerHead(q_param)

    def test_forward_preserves_shape(self):
        ir_module = FakeQuantCacheFP4PerHead(self.q_param)
        x = torch.randn(2, self.num_heads, 10, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertEqual(output.shape, x.shape)

    def test_forward_produces_finite_output(self):
        ir_module = FakeQuantCacheFP4PerHead(self.q_param)
        x = torch.randn(2, self.num_heads, 10, 64)
        with torch.no_grad():
            output = ir_module(x)
        self.assertTrue(torch.isfinite(output).all())

    def test_scheme_property(self):
        ir_module = FakeQuantCacheFP4PerHead(self.q_param)
        self.assertEqual(ir_module.x_q_scheme.scope, QScope.PER_HEAD)
        self.assertEqual(ir_module.x_q_scheme.dtype, QDType.FP4_E2M1)
        self.assertTrue(ir_module.x_q_scheme.symmetric)


class TestFakeQuantCacheINT8PerChannelForwardNumerics(unittest.TestCase):
    """Forward numerics of FakeQuantCacheINT8PerChannel (synced from upstream ut)"""

    @staticmethod
    def _make_cache(scale, offset=None):
        if offset is None:
            offset = torch.zeros_like(scale)
        q_param = QParam(scheme=int8_per_channel_sym, ext={"scale": scale, "offset": offset})
        return FakeQuantCacheINT8PerChannel(q_param)

    def test_is_atomic_returns_true(self):
        module = self._make_cache(torch.full((2,), 0.5, dtype=torch.float32))
        self.assertTrue(module.is_atomic())

    def test_init_keeps_scale_and_offset_as_frozen_parameters(self):
        scale = torch.full((2,), 0.5, dtype=torch.float32)
        offset = torch.zeros(2, dtype=torch.float32)
        module = self._make_cache(scale, offset)
        self.assertIsInstance(module.kv_cache_scale, torch.nn.Parameter)
        self.assertFalse(module.kv_cache_scale.requires_grad)
        self.assertFalse(module.kv_cache_offset.requires_grad)
        self.assertTrue(torch.equal(module.kv_cache_scale, scale))
        self.assertTrue(torch.equal(module.kv_cache_offset, offset))

    def test_forward_preserves_shape_and_dtype_of_input(self):
        module = self._make_cache(torch.full((2,), 0.5, dtype=torch.float32))
        x = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        output = module(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, torch.float32)

    def test_forward_quantizes_values_with_configured_scale(self):
        # scale=0.5: value 1.0/0.5=2 -> round 2 -> 2*0.5 = 1.0
        module = self._make_cache(torch.full((2,), 0.5, dtype=torch.float32))
        x = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        output = module(x)
        self.assertTrue(torch.allclose(output, torch.ones_like(x)))

    def test_forward_clamps_small_values_to_zero_with_large_scale(self):
        # scale=3.0: value 1.0/3.0=0.33 -> round 0 -> 0.0
        module = self._make_cache(torch.full((2,), 3.0, dtype=torch.float32))
        x = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        output = module(x)
        self.assertTrue(torch.equal(output, torch.zeros_like(x)))

    def test_forward_rounds_value_up_to_single_step_for_unit_scale(self):
        # scale=1.0: value 0.6 -> round 1 -> 1.0
        module = self._make_cache(torch.ones(2, dtype=torch.float32))
        x = torch.full((1, 1, 2, 2), 0.6, dtype=torch.float32)
        output = module(x)
        self.assertTrue(torch.allclose(output, torch.ones_like(x)))

    def test_forward_raises_when_scale_size_does_not_match_input_channels(self):
        module = self._make_cache(torch.ones(3, dtype=torch.float32))
        x = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            _ = module(x)


if __name__ == '__main__':
    unittest.main()
