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

MXWeightPerBlockMseRound 量化器单元测试。

要求：
- 每个被测类对应一个测试类
- 单测方法命名：test_对象_断言_when_条件（对象覆盖外部接口与核心内部方法）
- 每个用例只覆盖一种情形（正常、边界、异常）
"""

import math
import unittest

import torch

from msmodelslim.core.observer import MinMaxBlockObserverConfig, MsMinMaxBlockObserver
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.impl.minmax import MXWeightPerBlockMinmax
from msmodelslim.core.quantizer.impl.mse_round import (
    FP32_MIN_NORMAL,
    MXWeightPerBlockMseRound,
)
from msmodelslim.ir.api import dequantize, quantize
from msmodelslim.ir.qal import QDType, QScope, QStorage
from msmodelslim.ir.utils import reshape_to_blocks
from msmodelslim.utils.exception import SpecError


# 固定 32 维权重（单 MX block，block_size=32）。
# max≈7.7017 → log2≈2.945 → ceil 档 shared_exp=-5，floor 档 shared_exp=-6；
# ceil 档块内 MSE 更小，mse_round 应选 -5，而 minmax 仅用 floor 得 -6。
_CONCRETE_WEIGHT_32 = torch.tensor(
    [
        -1.7077441215515137,
        2.759913921356201,
        3.3324484825134277,
        3.869621515274048,
        -4.434523105621338,
        7.701698303222656,
        -1.4193594455718994,
        1.0066521167755127,
        -4.8879780769348145,
        -1.64923095703125,
        -1.4395027160644531,
        -1.4990464448928833,
        -3.2009401321411133,
        3.344818592071533,
        -0.4220142960548401,
        2.4172608852386475,
        -0.28004467487335205,
        2.061150550842285,
        -2.5149459838867188,
        0.0026754653081297874,
        2.525682210922241,
        -1.2001036405563354,
        3.1183857917785645,
        1.0744593143463135,
        -0.7380028367042542,
        6.9075493812561035,
        -5.6450676918029785,
        -0.149181067943573,
        -3.1349358558654785,
        -2.8695015907287598,
        0.10059557855129242,
        2.1302597522735596,
    ],
    dtype=torch.float32,
)

_EXPECTED_BLOCK_MAX = 7.701698303222656
_EXPECTED_LOG2_MAX = 2.945176601409912
_EXPECTED_SHARED_EXP_UP = -5.0
_EXPECTED_SHARED_EXP_DOWN = -6.0
_EXPECTED_SELECTED_SCALE = -5.0
_EXPECTED_MINMAX_SCALE = -6.0
# MSE 经 quantize/dequantize 后随 PyTorch/硬件略有浮点差异，不做硬编码 golden 值
_MSE_RTOL = 1e-5
_MSE_ATOL = 1e-7


def _mse_round_config() -> QConfig:
    return QConfig(
        dtype=QDType.MXFP8,
        scope=QScope.PER_BLOCK,
        symmetric=True,
        method="mse_round",
        ext={"axes": -1},
    )


def _minmax_config() -> QConfig:
    return QConfig(
        dtype=QDType.MXFP8,
        scope=QScope.PER_BLOCK,
        symmetric=True,
        method="minmax",
        ext={"axes": -1},
    )


def _block_max_and_candidate_scales(weight: torch.Tensor):
    """按 init_weight 相同路径计算块内 max 与 ceil/floor 两档 shared_exp。"""
    mx_finfo = QDType.MXFP8.mx_finfo
    weight_blocks, _, _, _ = reshape_to_blocks(weight.detach(), [0], mx_finfo.block_size)
    observer = MsMinMaxBlockObserver(MinMaxBlockObserverConfig(axes=-1))
    observer.update(weight_blocks, sync=False, shared_exp_axes=[1])
    _, max_val = observer.get_min_max()

    log_arg = max_val + FP32_MIN_NORMAL * (max_val == 0).to(max_val.dtype)
    log2v = torch.log2(log_arg)
    shared_exp_up = torch.ceil(log2v) - mx_finfo.emax
    shared_exp_down = torch.floor(log2v) - mx_finfo.emax
    return weight_blocks, max_val, shared_exp_up, shared_exp_down


def _block_mse(
    weight_blocks: torch.Tensor, shared_exp: torch.Tensor, quantizer: MXWeightPerBlockMseRound
) -> torch.Tensor:
    q_param = quantizer._build_qparam(shared_exp)
    float_storage = QStorage(QDType.FLOAT, weight_blocks)
    dequant = dequantize(quantize(float_storage, q_param), q_param).value
    return (weight_blocks - dequant).pow(2).mean(dim=-1, keepdim=True)


class TestMXWeightPerBlockMseRound(unittest.TestCase):
    """测试 MXWeightPerBlockMseRound"""

    def test_select_shared_exp_by_mse_return_up_scale_when_mse_up_is_smaller(self):
        mse_up = torch.tensor([[0.1]])
        mse_down = torch.tensor([[0.2]])
        shared_exp_up = torch.tensor([[2.0]])
        shared_exp_down = torch.tensor([[1.0]])

        result = MXWeightPerBlockMseRound._select_shared_exp_by_mse(mse_up, mse_down, shared_exp_up, shared_exp_down)

        self.assertEqual(result.item(), 2.0)

    def test_select_shared_exp_by_mse_return_down_scale_when_mse_down_is_smaller(self):
        mse_up = torch.tensor([[0.3]])
        mse_down = torch.tensor([[0.1]])
        shared_exp_up = torch.tensor([[5.0]])
        shared_exp_down = torch.tensor([[4.0]])

        result = MXWeightPerBlockMseRound._select_shared_exp_by_mse(mse_up, mse_down, shared_exp_up, shared_exp_down)

        self.assertEqual(result.item(), 4.0)

    def test_select_shared_exp_by_mse_return_expected_scales_when_some_mse_candidates_are_nan(self):
        mse_up = torch.tensor([[float("nan")], [0.1], [0.5], [float("nan")]])
        mse_down = torch.tensor([[0.2], [0.3], [float("nan")], [float("nan")]])
        shared_exp_up = torch.tensor([[float("nan")], [2.0], [5.0], [float("nan")]])
        shared_exp_down = torch.tensor([[1.0], [1.0], [float("nan")], [float("nan")]])

        result = MXWeightPerBlockMseRound._select_shared_exp_by_mse(mse_up, mse_down, shared_exp_up, shared_exp_down)

        expected = torch.tensor([[1.0], [2.0], [5.0], [float("nan")]])
        self.assertTrue(torch.allclose(result, expected, equal_nan=True))

    def test_select_shared_exp_by_mse_return_ceil_scale_when_concrete_weight_ceil_mse_is_smaller(self):
        quantizer = MXWeightPerBlockMseRound(_mse_round_config())
        weight_blocks, _, shared_exp_up, shared_exp_down = _block_max_and_candidate_scales(_CONCRETE_WEIGHT_32)
        mse_up = _block_mse(weight_blocks, shared_exp_up, quantizer)
        mse_down = _block_mse(weight_blocks, shared_exp_down, quantizer)

        result = MXWeightPerBlockMseRound._select_shared_exp_by_mse(mse_up, mse_down, shared_exp_up, shared_exp_down)

        self.assertEqual(result.item(), _EXPECTED_SELECTED_SCALE)

    def test_build_qparam_mark_scale_nan_when_shared_exp_exceeds_scale_emax(self):
        quantizer = MXWeightPerBlockMseRound(_mse_round_config())
        mx_finfo = QDType.MXFP8.mx_finfo
        scale_emax = 2 ** (mx_finfo.scale_bits - 1) - 1

        overflow = torch.tensor([[scale_emax + 1.0]])
        q_param = quantizer._build_qparam(overflow)

        self.assertTrue(torch.isnan(q_param.ext["scale"]).all())

    def test_build_qparam_clip_scale_to_neg_scale_emax_when_shared_exp_below_neg_scale_emax(self):
        quantizer = MXWeightPerBlockMseRound(_mse_round_config())
        mx_finfo = QDType.MXFP8.mx_finfo
        scale_emax = 2 ** (mx_finfo.scale_bits - 1) - 1

        underflow = torch.tensor([[-scale_emax - 1.0]])
        q_param = quantizer._build_qparam(underflow)

        self.assertEqual(q_param.ext["scale"].item(), -scale_emax)

    def test_init_weight_derive_expected_block_max_and_candidate_scales_when_concrete_weight(self):
        _, max_val, shared_exp_up, shared_exp_down = _block_max_and_candidate_scales(_CONCRETE_WEIGHT_32)

        self.assertEqual(max_val.numel(), 1)
        self.assertTrue(math.isclose(max_val.item(), _EXPECTED_BLOCK_MAX, rel_tol=0, abs_tol=1e-5))
        self.assertTrue(
            math.isclose(
                torch.log2(max_val + FP32_MIN_NORMAL * (max_val == 0).to(max_val.dtype)).item(),
                _EXPECTED_LOG2_MAX,
                rel_tol=0,
                abs_tol=1e-5,
            )
        )
        self.assertEqual(shared_exp_up.item(), _EXPECTED_SHARED_EXP_UP)
        self.assertEqual(shared_exp_down.item(), _EXPECTED_SHARED_EXP_DOWN)

    def test_init_weight_compute_smaller_mse_for_ceil_than_floor_when_concrete_weight(self):
        quantizer = MXWeightPerBlockMseRound(_mse_round_config())
        weight_blocks, _, shared_exp_up, shared_exp_down = _block_max_and_candidate_scales(_CONCRETE_WEIGHT_32)

        mse_up = _block_mse(weight_blocks, shared_exp_up, quantizer)
        mse_down = _block_mse(weight_blocks, shared_exp_down, quantizer)

        self.assertTrue(torch.isfinite(mse_up).all())
        self.assertTrue(torch.isfinite(mse_down).all())
        self.assertLess(mse_up.item(), mse_down.item())

    def test_init_weight_set_ceil_scale_when_ceil_candidate_has_smaller_mse(self):
        quantizer = MXWeightPerBlockMseRound(_mse_round_config())
        quantizer.init_weight(QStorage(QDType.FLOAT, _CONCRETE_WEIGHT_32.clone()))

        self.assertEqual(
            quantizer.get_q_param().ext["scale"].reshape(-1)[0].item(),
            _EXPECTED_SELECTED_SCALE,
        )

    def test_init_weight_produce_larger_scale_than_minmax_when_concrete_weight(self):
        mse_round = MXWeightPerBlockMseRound(_mse_round_config())
        minmax = MXWeightPerBlockMinmax(_minmax_config())
        weight = QStorage(QDType.FLOAT, _CONCRETE_WEIGHT_32.clone())

        mse_round.init_weight(weight)
        minmax.init_weight(QStorage(QDType.FLOAT, _CONCRETE_WEIGHT_32.clone()))

        self.assertEqual(
            mse_round.get_q_param().ext["scale"].reshape(-1)[0].item(),
            _EXPECTED_SELECTED_SCALE,
        )
        self.assertEqual(
            minmax.get_q_param().ext["scale"].reshape(-1)[0].item(),
            _EXPECTED_MINMAX_SCALE,
        )

    def test_forward_return_dequant_matching_selected_scale_mse_when_weight_initialized(self):
        quantizer = MXWeightPerBlockMseRound(_mse_round_config())
        weight = _CONCRETE_WEIGHT_32.clone()
        quantizer.init_weight(QStorage(QDType.FLOAT, weight))

        weight_blocks, _, shared_exp_up, _ = _block_max_and_candidate_scales(weight)
        mse_up = _block_mse(weight_blocks, shared_exp_up, quantizer).item()

        dequant = quantizer.forward().reshape_as(weight)
        mse_forward = (weight - dequant).pow(2).mean().item()

        self.assertTrue(
            math.isclose(mse_forward, mse_up, rel_tol=_MSE_RTOL, abs_tol=_MSE_ATOL),
            msg=f"forward mse={mse_forward}, block mse_up={mse_up}",
        )

    def test_forward_raise_spec_error_when_weight_not_initialized(self):
        quantizer = MXWeightPerBlockMseRound(_mse_round_config())

        with self.assertRaises(SpecError):
            quantizer.forward()

    def test_get_q_param_raise_spec_error_when_not_initialized(self):
        quantizer = MXWeightPerBlockMseRound(_mse_round_config())

        with self.assertRaises(SpecError):
            quantizer.get_q_param()

    def test_get_q_storage_raise_spec_error_when_not_initialized(self):
        quantizer = MXWeightPerBlockMseRound(_mse_round_config())

        with self.assertRaises(SpecError):
            quantizer.get_q_storage()


if __name__ == "__main__":
    unittest.main()
