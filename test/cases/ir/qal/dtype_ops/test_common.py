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

from msmodelslim.ir.qal.dtype_ops.common import (
    align_block_param,
    as_shared_exp,
    blocked_apply,
    cast_q,
    clamp_ste,
    elem_cfg,
    floor_ste,
    grouped_apply,
    pack_e8m0,
    reduce_amax,
    reduce_minmax,
    restore_blocks,
    restore_groups,
    round_ste,
    unpack_e8m0,
    view_blocks,
    view_groups,
)


class TestElemCfg(unittest.TestCase):
    """被测目标：elem_cfg。"""

    def test_elem_cfg_max_norm_equals_448_when_fmt_is_e4m3(self):
        """场景：e4m3。预期：max_norm=448，emax=8。"""
        cfg = elem_cfg("e4m3")
        self.assertEqual(cfg["emax"], 8)
        self.assertEqual(cfg["max_norm"], 448.0)

    def test_elem_cfg_max_norm_equals_6_when_fmt_is_e2m1(self):
        """场景：e2m1。预期：max_norm=6，emax=2。"""
        cfg = elem_cfg("e2m1")
        self.assertEqual(cfg["emax"], 2)
        self.assertEqual(cfg["max_norm"], 6.0)

    def test_elem_cfg_raises_value_error_when_fmt_unknown(self):
        """场景：非法格式。预期：ValueError。"""
        with self.assertRaises(ValueError) as ctx:
            elem_cfg("e3m4")
        self.assertIn("unsupported element format", str(ctx.exception))


class TestCastQ(unittest.TestCase):
    """被测目标：cast_q。"""

    def test_cast_q_returns_same_tensor_when_dtype_is_none(self):
        """场景：dtype 省略。预期：返回原张量。"""
        q = torch.tensor([1.5], requires_grad=True)
        self.assertIs(cast_q(q, None), q)

    def test_cast_q_returns_same_tensor_when_dtype_is_float32(self):
        """场景：dtype=float32。预期：仍在计算图里。"""
        q = torch.tensor([1.5], requires_grad=True)
        out = cast_q(q, torch.float32)
        self.assertIs(out, q)
        self.assertTrue(out.requires_grad)

    def test_cast_q_detaches_when_dtype_is_int8(self):
        """场景：转 int8。预期：断开计算图。"""
        q = torch.tensor([1.5], requires_grad=True)
        out = cast_q(q, torch.int8)
        self.assertEqual(out.dtype, torch.int8)
        self.assertFalse(out.requires_grad)


class TestViewGroups(unittest.TestCase):
    """被测目标：view_groups / restore_groups / grouped_apply。"""

    def test_view_groups_raises_value_error_when_length_not_divisible(self):
        """场景：长度 5、组 4。预期：ValueError 含 not divisible。"""
        with self.assertRaises(ValueError) as ctx:
            view_groups(torch.arange(5, dtype=torch.float32), axis=-1, group_size=4)
        self.assertIn("not divisible", str(ctx.exception))

    def test_view_groups_raises_value_error_when_group_size_not_positive(self):
        """场景：group_size=0。预期：ValueError。"""
        with self.assertRaises(ValueError) as ctx:
            view_groups(torch.arange(4, dtype=torch.float32), axis=-1, group_size=0)
        self.assertIn("must be positive", str(ctx.exception))

    def test_restore_groups_returns_original_shape_when_group_size_4(self):
        """场景：切组再还原。预期：形状回到 (8,)。"""
        x = torch.arange(8, dtype=torch.float32)
        grouped, meta = view_groups(x, axis=-1, group_size=4)
        y = restore_groups(grouped, meta)
        self.assertEqual(tuple(y.shape), (8,))
        self.assertTrue(torch.equal(y, x))

    def test_grouped_apply_multiplies_when_scale_is_two(self):
        """场景：切组后乘 2。预期：输出为 2x。"""
        x = torch.arange(8, dtype=torch.float32)
        y = grouped_apply(lambda t, s: t * s, x, -1, 4, torch.tensor(2.0))
        self.assertTrue(torch.equal(y, x * 2))

    def test_grouped_apply_raises_value_error_when_param_cannot_broadcast(self):
        """场景：scale 形状无法广播到切组布局。预期：ValueError。"""
        x = torch.arange(8, dtype=torch.float32)
        with self.assertRaises(ValueError):
            grouped_apply(lambda t, s: t * s, x, -1, 4, torch.ones(3, 1))


class TestReduce(unittest.TestCase):
    """被测目标：reduce_minmax / reduce_amax。"""

    def test_reduce_minmax_returns_scalars_when_axis_is_none(self):
        """场景：全张量归约。预期：min=-2，max=3。"""
        mn, mx = reduce_minmax(torch.tensor([1.0, -2.0, 3.0]), axis=None)
        self.assertEqual(mn.item(), -2.0)
        self.assertEqual(mx.item(), 3.0)

    def test_reduce_amax_equals_10_when_axis_is_none(self):
        """场景：absmax。预期：10。"""
        self.assertEqual(reduce_amax(torch.tensor([-10.0, 3.0]), axis=None).item(), 10.0)


class TestSte(unittest.TestCase):
    """被测目标：round_ste / floor_ste / clamp_ste。"""

    def test_round_ste_rounds_to_even_when_half_integer(self):
        """场景：半整数。预期：[-2,0,0,2,2]。"""
        x = torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5])
        y = round_ste(x)
        self.assertTrue(torch.equal(y, torch.tensor([-2.0, 0.0, 0.0, 2.0, 2.0])))

    def test_floor_ste_floors_when_positive_fraction(self):
        """场景：1.9。预期：1。"""
        self.assertEqual(floor_ste(torch.tensor([1.9])).item(), 1.0)

    def test_clamp_ste_forward_equals_clamp_when_out_of_range(self):
        """场景：clamp 到 [-1,1]。预期：[-1,0,1]。"""
        y = clamp_ste(torch.tensor([-2.0, 0.0, 3.0]), min=-1.0, max=1.0)
        self.assertTrue(torch.equal(y, torch.tensor([-1.0, 0.0, 1.0])))

    def test_clamp_ste_x_grad_equals_one_at_sat_when_backward(self):
        """场景：饱和处反向。预期：梯度仍为 1。"""
        x = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
        clamp_ste(x, min=-1.0, max=1.0).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))


class TestE8M0(unittest.TestCase):
    """被测目标：pack_e8m0 / unpack_e8m0 / as_shared_exp。"""

    def test_pack_e8m0_equals_127_when_shared_exp_is_zero(self):
        """场景：共享指数 0。预期：uint8 编码 127。"""
        raw = pack_e8m0(torch.tensor([0.0]))
        self.assertEqual(raw.dtype, torch.uint8)
        self.assertEqual(raw.item(), 127)

    def test_unpack_e8m0_equals_zero_when_raw_is_127(self):
        """场景：编码 127。预期：指数 0。"""
        self.assertEqual(unpack_e8m0(torch.tensor([127], dtype=torch.uint8)).item(), 0.0)

    def test_as_shared_exp_unpacks_when_dtype_is_uint8(self):
        """场景：uint8 输入。预期：按 E8M0 还原。"""
        self.assertEqual(as_shared_exp(torch.tensor([128], dtype=torch.uint8)).item(), 1.0)

    def test_pack_e8m0_clamps_to_254_when_exp_exceeds_127(self):
        """场景：指数 200。预期：编码钳到 254，不写 255。"""
        self.assertEqual(pack_e8m0(torch.tensor([200.0])).item(), 254)


class TestBlockedApply(unittest.TestCase):
    """被测目标：view_blocks / restore_blocks / blocked_apply。"""

    def test_restore_blocks_returns_length_33_when_padded_to_32(self):
        """场景：长度 33 切块再还原。预期：形状 (33,)。"""
        x = torch.arange(33, dtype=torch.float32)
        blocked, meta = view_blocks(x, axis=-1)
        y = restore_blocks(blocked, meta)
        self.assertEqual(tuple(y.shape), (33,))
        self.assertTrue(torch.equal(y, x))

    def test_align_block_param_expands_when_param_is_scalar(self):
        """场景：标量参数。预期：能广播到切块张量。"""
        blocked, _meta = view_blocks(torch.arange(32, dtype=torch.float32), axis=-1)
        aligned = align_block_param(torch.tensor(1.0), blocked)
        self.assertEqual(tuple(aligned.shape), tuple(blocked.shape))

    def test_blocked_apply_restores_shape_when_fn_is_identity(self):
        """场景：块上恒等。预期：输出等于输入。"""
        x = torch.arange(40, dtype=torch.float32)
        y = blocked_apply(lambda t, p: t + p, x, -1, torch.zeros(()))
        self.assertTrue(torch.equal(y, x))


if __name__ == "__main__":
    unittest.main()
