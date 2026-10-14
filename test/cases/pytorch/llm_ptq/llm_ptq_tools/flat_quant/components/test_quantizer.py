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
# pylint: disable=consider-using-from-import

import pytest
import torch

from unittest.mock import Mock, patch  # noqa: F401
import torch.nn as nn  # noqa: F401
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.quantizer import (
    ActivationQuantizer,
    WeightQuantizer,
    asym_quant,
    asym_dequant,
    sym_quant,
    sym_dequant,
    round_ste,
    get_qmin_qmax,
    get_maxq,
    sym_quant_dequant,
    asym_quant_dequant,
)


class TestQuantizationFunctions:
    def test_round_ste(self):
        x = torch.tensor([1.3, 2.7, -1.6])
        result = round_ste(x)
        expected = torch.tensor([1.0, 3.0, -2.0])
        assert torch.allclose(result, expected)

    def test_round_ste_gradient(self):
        x = torch.tensor([1.3, 2.7], requires_grad=True)
        result = round_ste(x)
        loss = result.sum()
        loss.backward()

        # Gradient should pass through unchanged
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_get_qmin_qmax_symmetric(self):
        q_max, q_min = get_qmin_qmax(8, sym=True)
        assert q_max == 127
        assert q_min == -128

    def test_get_qmin_qmax_asymmetric(self):
        q_max, q_min = get_qmin_qmax(8, sym=False)
        assert q_max == 255
        assert q_min == 0

    def test_get_maxq(self):
        maxq_sym = get_maxq(8, sym=True)
        maxq_asym = get_maxq(8, sym=False)
        assert maxq_sym == 127
        assert maxq_asym == 255

    def test_sym_quant_dequant(self):
        x = torch.tensor([1.0, -2.0, 3.0])
        scale = torch.tensor(0.1)

        q, _ = sym_quant(x, scale, 8)
        dequant = sym_dequant(q, scale)

        assert torch.allclose(dequant, x, atol=0.1)

    def test_sym_quant_dequant_function(self):
        x = torch.tensor([1.0, -2.0, 3.0])
        scale = torch.tensor(0.1)

        result = sym_quant_dequant(x, scale, 8)

        assert result.shape == x.shape

    def test_asym_quant_dequant(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(0.1)
        zero = torch.tensor(10.0)

        q, _, _ = asym_quant(x, scale, zero, 8)
        dequant = asym_dequant(q, scale, zero)

        assert torch.allclose(dequant, x, atol=0.1)

    def test_asym_quant_dequant_function(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(0.1)
        zero = torch.tensor(10.0)

        result = asym_quant_dequant(x, scale, zero, 8)

        assert result.shape == x.shape

    def test_sym_quant_unsigned(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(0.1)

        q, _ = sym_quant(x, scale, 8, is_signed=False)

        assert torch.all(q >= 0)

    def test_asym_quant_unsigned(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(0.1)
        zero = torch.tensor(10.0)

        q, _, _ = asym_quant(x, scale, zero, 8, is_signed=False)

        assert torch.all(q >= 0)


class TestActivationQuantizer:
    def test_init_default(self):
        quantizer = ActivationQuantizer(bits=8)

        assert quantizer.bits == 8
        assert quantizer.sym is False
        assert quantizer.lac is False
        assert quantizer.per_tensor is False
        assert quantizer.enable is True

    def test_init_with_lac(self):
        quantizer = ActivationQuantizer(bits=8, lac=True)

        assert quantizer.lac is True
        assert hasattr(quantizer, 'clip_factor')
        assert hasattr(quantizer, 'sigmoid')

    def test_init_per_tensor(self):
        quantizer = ActivationQuantizer(bits=8, per_tensor=True)

        assert quantizer.per_tensor is True
        assert hasattr(quantizer, 'observer')

    def test_repr(self):
        quantizer = ActivationQuantizer(bits=8, sym=True, lac=True, per_tensor=True)
        repr_str = repr(quantizer)

        assert "ActivationQuantizer" in repr_str
        assert "bits=8" in repr_str
        assert "sym=True" in repr_str
        assert "lac=True" in repr_str
        assert "per_tensor=True" in repr_str

    def test_reparameterize_without_lac(self):
        quantizer = ActivationQuantizer(bits=8, lac=False)

        # Should not raise error
        quantizer.reparameterize()

    def test_forward_no_quantization_16bit(self):
        quantizer = ActivationQuantizer(bits=16)
        x = torch.randn(2, 10)

        result = quantizer(x)

        assert torch.equal(result, x)

    def test_forward_no_quantization_disabled(self):
        quantizer = ActivationQuantizer(bits=8)
        quantizer.enable = False
        x = torch.randn(2, 10)

        result = quantizer(x)

        assert torch.equal(result, x)

    def test_forward_calibration_mode(self):
        quantizer = ActivationQuantizer(bits=8, per_tensor=True)
        x = torch.randn(2, 10)

        result = quantizer(x, quantize=False)

        assert torch.equal(result, x)

    def test_forward_with_quantization(self):
        quantizer = ActivationQuantizer(bits=8)
        x = torch.randn(2, 10)

        result = quantizer(x)

        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_forward_with_quantization_symmetric(self):
        quantizer = ActivationQuantizer(bits=8, sym=True)
        x = torch.randn(2, 10)

        result = quantizer(x)

        assert result.shape == x.shape

    def test_get_clip_ratio_with_lac(self):
        quantizer = ActivationQuantizer(bits=8, lac=True)

        clip_ratio = quantizer.get_clip_ratio()

        assert clip_ratio is not None
        assert torch.all(clip_ratio > 0)
        assert torch.all(clip_ratio < 1)

    def test_get_clip_ratio_without_lac(self):
        quantizer = ActivationQuantizer(bits=8, lac=False, clip_ratio=0.8)

        clip_ratio = quantizer.get_clip_ratio()

        assert clip_ratio == 0.8

    def test_get_scale_zero_per_tensor(self):
        quantizer = ActivationQuantizer(bits=8, per_tensor=True)
        x = torch.randn(2, 10)

        # First call to update observer
        quantizer(x, quantize=False)

        scale, zero = quantizer.get_scale_zero(x)

        assert scale is not None
        assert zero is not None

    def test_get_scale_zero_per_tensor_none_input(self):
        quantizer = ActivationQuantizer(bits=8, per_tensor=True)
        x = torch.randn(2, 10)

        # First call to update observer
        quantizer(x, quantize=False)

        scale, zero = quantizer.get_scale_zero(None)

        assert scale is not None
        assert zero is not None

    def test_get_scale_zero_per_token_symmetric(self):
        quantizer = ActivationQuantizer(bits=8, sym=True)
        x = torch.randn(2, 10)

        scale, zero = quantizer.get_scale_zero(x)

        assert scale.shape == x.shape
        assert zero.shape == x.shape
        assert torch.all(zero == 0)

    def test_get_scale_zero_per_token_asymmetric(self):
        quantizer = ActivationQuantizer(bits=8, sym=False)
        x = torch.randn(2, 10)

        scale, zero = quantizer.get_scale_zero(x)

        assert scale.shape == x.shape
        assert zero.shape == x.shape

    def test_get_scale_zero_with_lac_clipping(self):
        quantizer = ActivationQuantizer(bits=8, lac=True)
        x = torch.randn(2, 10)

        scale, zero = quantizer.get_scale_zero(x)

        assert scale.shape == x.shape
        assert zero.shape == x.shape


class TestWeightQuantizer:
    def test_init_default(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64)

        assert quantizer.in_size == 128
        assert quantizer.out_size == 64
        assert quantizer.bits == 8
        assert quantizer.perchannel is False
        assert quantizer.sym is True
        assert quantizer.enable is True

    def test_init_with_lwc(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, lwc=True)

        assert quantizer.lwc is True
        assert hasattr(quantizer, 'clip_factor_w_max')
        assert hasattr(quantizer, 'clip_factor_w_min')
        assert hasattr(quantizer, 'sigmoid')

    def test_repr(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, bits=4, sym=False, lwc=True)
        repr_str = repr(quantizer)

        assert "WeightQuantizer" in repr_str
        assert "bits=4" in repr_str
        assert "sym=False" in repr_str
        assert "lwc=True" in repr_str

    def test_reparameterize_with_lwc(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, lwc=True)

        quantizer.reparameterize()

    def test_reparameterize_without_lwc(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, lwc=False)

        # Should not raise error
        quantizer.reparameterize()

    def test_forward_no_quantization_16bit(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, bits=16)
        x = torch.randn(64, 128)

        result = quantizer(x)

        assert torch.equal(result, x)

    def test_forward_no_quantization_disabled(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64)
        quantizer.enable = False
        x = torch.randn(64, 128)

        result = quantizer(x)

        assert torch.equal(result, x)

    def test_forward_calibration_mode(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64)
        x = torch.randn(64, 128)

        result = quantizer(x, quantize=False)

        assert torch.equal(result, x)

    def test_forward_with_quantization(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64)
        x = torch.randn(64, 128)

        result = quantizer(x)

        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_forward_with_quantization_asymmetric(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, sym=False)
        x = torch.randn(64, 128)

        result = quantizer(x)

        assert result.shape == x.shape

    def test_find_params_symmetric(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, sym=True)
        x = torch.randn(64, 128)

        quantizer.find_params(x)

        assert quantizer.ready()
        assert quantizer.scale.shape[0] == 64
        assert torch.all(quantizer.zero == 0)

    def test_find_params_asymmetric(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, sym=False)
        x = torch.randn(64, 128)

        quantizer.find_params(x)

        assert quantizer.ready()
        assert quantizer.scale.shape[0] == 64
        assert quantizer.zero.shape[0] == 64

    def test_find_params_perchannel(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, perchannel=True)
        x = torch.randn(64, 128)

        quantizer.find_params(x)

        assert quantizer.ready()

    def test_find_params_disabled_skip(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64)
        quantizer.enable = False
        x = torch.randn(64, 128)

        quantizer.find_params(x)

        # Should not set scale/zero when disabled
        assert not quantizer.ready()

    def test_apply_wclip(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, lwc=True)
        x = torch.randn(64, 128)

        result = quantizer.apply_wclip(x)

        assert result.shape == x.shape

    def test_enable_find_params(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64)

        quantizer.enable_find_params(False)
        assert quantizer.enable_find is False

        quantizer.enable_find_params(True)
        assert quantizer.enable_find is True

    def test_enable_quant(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64)

        quantizer.enable_quant(False)
        assert quantizer.enable is False

        quantizer.enable_quant(True)
        assert quantizer.enable is True

    def test_get_fake_quant_weight(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64)
        x = torch.randn(64, 128)

        result = quantizer.get_fake_quant_weight(x)

        assert result.shape == x.shape
        assert quantizer.enable is False

    def test_get_scale_zero(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64)
        x = torch.randn(64, 128)

        scale, zero = quantizer.get_scale_zero(x)

        assert scale is not None
        assert zero is not None

    def test_quantize_not_ready_error(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64)
        quantizer.enable_find_params(False)
        x = torch.randn(64, 128)

        with pytest.raises(ValueError, match="WeightQuantizer is not ready"):
            quantizer.quantize(x)

    def test_quantize_symmetric(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, sym=True)
        x = torch.randn(64, 128)

        result = quantizer.quantize(x)

        assert result.shape == x.shape

    def test_quantize_asymmetric(self):
        quantizer = WeightQuantizer(in_size=128, out_size=64, sym=False)
        x = torch.randn(64, 128)

        result = quantizer.quantize(x)

        assert result.shape == x.shape
