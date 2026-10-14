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

import unittest
import torch
import torch.nn as nn

from msmodelslim.ir.quarot import (
    QuarotOfflineRotationInfo,
    QuarotOnlineRotationInfo,
    QuarotOnlineHeadRotationWrapper,
    QuarotOnlineKroneckerRotationWrapper,
    QuarotHeadsRotationHookIR,
    QuarotKroneckerRotationHookIR,
    QuaRotExtraInfoHookIR,
    QuaRotExtraInfoWrapperIR,
    RotationType,
    OnlineRotationInfo,
    BaseOnlineRotation,
    OnlineRotationWrapper,
    OnlineRotationInputHookIR,
    OnlineRotationOutputHookIR,
)


class TestQuarotOnlineRotationInfo(unittest.TestCase):
    """测试 QuarotOnlineRotationInfo 类"""

    def setUp(self):
        self.rotation_o_proj = torch.randn(4, 4)
        self.rotation_o_proj_eye = torch.randn(8, 8)
        self.rotation_down_proj_m = torch.randn(4, 4)
        self.rotation_down_proj_n = torch.randn(8, 8)
        self.max_tp_size = 8

    def test_init_and_methods(self):
        """测试初始化、添加层和获取保存信息"""
        info = QuarotOnlineRotationInfo(
            rotation_o_proj=self.rotation_o_proj,
            rotation_o_proj_eye=self.rotation_o_proj_eye,
            rotation_down_proj_m=self.rotation_down_proj_m,
            rotation_down_proj_n=self.rotation_down_proj_n,
            max_tp_size=self.max_tp_size,
        )
        self.assertTrue(torch.equal(info.heads_rotation, self.rotation_o_proj))
        self.assertEqual(info.max_tp_size, self.max_tp_size)
        self.assertEqual(len(info.heads_rotation_layers), 0)

        info.add_rotation_layer("layer1")
        info.add_kronecker_rotation_layer("layer2")
        self.assertEqual(len(info.heads_rotation_layers), 1)
        self.assertEqual(len(info.kronecker_rotation_layers), 1)

        save_info = info.get_quarot_save_info()
        self.assertEqual(save_info["max_tp_size"], self.max_tp_size)
        self.assertEqual(save_info["heads_rotation"]["layers"], ["layer1"])


class TestQuarotRotationWrappers(unittest.TestCase):
    """测试旋转包装器类"""

    def setUp(self):
        self.rotation_o_proj = torch.eye(4)
        self.rotation_o_proj_eye = torch.eye(8)
        self.rotation_m = torch.eye(4)
        self.rotation_n = torch.eye(8)
        self.wrapped_module = nn.Linear(32, 16)
        self.layer_name = "test_layer"

    def test_head_rotation_wrapper(self):
        """测试头部旋转包装器"""
        rotation_info = QuarotOnlineRotationInfo(
            rotation_o_proj=self.rotation_o_proj,
            rotation_o_proj_eye=self.rotation_o_proj_eye,
            rotation_down_proj_m=None,
            rotation_down_proj_n=None,
            max_tp_size=8,
        )
        wrapper = QuarotOnlineHeadRotationWrapper(self.wrapped_module, self.layer_name, rotation_info)
        x = torch.randn(2, 32)
        output = wrapper(x)
        self.assertEqual(output.shape, (2, 16))
        self.assertIn("heads_rotation", wrapper.extra_repr())

    def test_kronecker_rotation_wrapper(self):
        """测试Kronecker旋转包装器"""
        rotation_info = QuarotOnlineRotationInfo(
            rotation_o_proj=None,
            rotation_o_proj_eye=None,
            rotation_down_proj_m=self.rotation_m,
            rotation_down_proj_n=self.rotation_n,
            max_tp_size=8,
        )
        wrapper = QuarotOnlineKroneckerRotationWrapper(self.wrapped_module, self.layer_name, rotation_info)
        x = torch.randn(2, 32)
        output = wrapper(x)
        self.assertEqual(output.shape, (2, 16))
        self.assertIn("kronecker_rotation", wrapper.extra_repr())


class TestQuarotRotationHookIR(unittest.TestCase):
    """测试旋转HookIR类"""

    def setUp(self):
        self.rotation_o_proj = torch.eye(4)
        self.rotation_o_proj_eye = torch.eye(8)
        self.rotation_m = torch.eye(4)
        self.rotation_n = torch.eye(8)
        self.layer_name = "test_layer"

    def test_heads_rotation_hook_ir(self):
        """测试头部旋转HookIR"""
        rotation_info = QuarotOnlineRotationInfo(
            rotation_o_proj=self.rotation_o_proj,
            rotation_o_proj_eye=self.rotation_o_proj_eye,
            rotation_down_proj_m=None,
            rotation_down_proj_n=None,
            max_tp_size=8,
        )
        hook_ir = QuarotHeadsRotationHookIR(self.layer_name, rotation_info)
        module = nn.Linear(32, 16)
        x = torch.randn(2, 32)
        result = hook_ir(module, (x,))
        self.assertEqual(result[0].shape, x.shape)

        wrapper = hook_ir.wrapper_module(module)
        self.assertIsInstance(wrapper, QuarotOnlineHeadRotationWrapper)

    def test_kronecker_rotation_hook_ir(self):
        """测试Kronecker旋转HookIR"""
        rotation_info = QuarotOnlineRotationInfo(
            rotation_o_proj=None,
            rotation_o_proj_eye=None,
            rotation_down_proj_m=self.rotation_m,
            rotation_down_proj_n=self.rotation_n,
            max_tp_size=8,
        )
        hook_ir = QuarotKroneckerRotationHookIR(self.layer_name, rotation_info)
        module = nn.Linear(32, 16)
        x = torch.randn(2, 32)
        result = hook_ir(module, (x,))
        self.assertEqual(result[0].shape, x.shape)


class TestRotationTypeAndInfo(unittest.TestCase):
    """测试RotationType枚举和OnlineRotationInfo"""

    def test_rotation_type_enum(self):
        """测试枚举值"""
        self.assertEqual(RotationType.INPUT, "input")
        self.assertEqual(RotationType.OUTPUT, "output")
        self.assertEqual(RotationType.REPLACE, "replace")
        self.assertEqual(RotationType.OFFLINE, "offline")

    def test_online_rotation_info(self):
        """测试OnlineRotationInfo"""
        rotation_matrix = torch.randn(4, 4)
        info = OnlineRotationInfo(
            rotation_matrix=rotation_matrix, rotation_type=RotationType.INPUT, layer_name="test_layer"
        )
        self.assertTrue(torch.equal(info.rotation_matrix, rotation_matrix))
        self.assertEqual(info.rotation_type, RotationType.INPUT)

        info_with_side = OnlineRotationInfo(
            rotation_matrix=rotation_matrix,
            rotation_type=RotationType.OFFLINE,
            layer_name="test_layer",
            rotation_side="right",
        )
        self.assertEqual(info_with_side.rotation_side, "right")


class TestBaseOnlineRotation(unittest.TestCase):
    """测试 BaseOnlineRotation 类"""

    def setUp(self):
        self.rotation_matrix = torch.eye(4)
        self.rotation_info = OnlineRotationInfo(
            rotation_matrix=self.rotation_matrix, rotation_type=RotationType.INPUT, layer_name="test_layer"
        )
        self.base_rotation = BaseOnlineRotation()
        self.base_rotation._init_rotation_common("test_layer", self.rotation_info)

    def test_apply_rotation(self):
        """测试旋转应用（不同形状）"""
        for shape in [(4,), (2, 4), (2, 3, 4)]:
            x = torch.randn(*shape)
            rotated = self.base_rotation._apply_rotation(x)
            self.assertEqual(rotated.shape, x.shape)


class TestOnlineRotationWrapper(unittest.TestCase):
    """测试 OnlineRotationWrapper 类"""

    def test_wrapper(self):
        """测试包装器初始化和前向传播"""
        rotation_matrix = torch.eye(4)
        rotation_info = OnlineRotationInfo(
            rotation_matrix=rotation_matrix, rotation_type=RotationType.REPLACE, layer_name="test_layer"
        )
        wrapped_module = nn.Linear(4, 4)
        wrapper = OnlineRotationWrapper(wrapped_module, "test_layer", rotation_info)
        x = torch.randn(2, 4)
        output = wrapper(x)
        self.assertEqual(output.shape, x.shape)


class TestOnlineRotationHookIR(unittest.TestCase):
    """测试在线旋转HookIR类"""

    def setUp(self):
        self.rotation_matrix = torch.eye(4)
        self.rotation_info = OnlineRotationInfo(
            rotation_matrix=self.rotation_matrix, rotation_type=RotationType.INPUT, layer_name="test_layer"
        )

    def test_input_hook_ir(self):
        """测试输入HookIR"""
        hook_ir = OnlineRotationInputHookIR("test_layer", self.rotation_info)
        module = nn.Linear(4, 4)
        x = torch.randn(2, 4)
        result = hook_ir(module, (x,))
        self.assertEqual(result[0].shape, x.shape)

        with self.assertRaises(ValueError):
            hook_ir(module, ())

    def test_output_hook_ir(self):
        """测试输出HookIR"""
        rotation_info = OnlineRotationInfo(
            rotation_matrix=self.rotation_matrix, rotation_type=RotationType.OUTPUT, layer_name="test_layer"
        )
        hook_ir = OnlineRotationOutputHookIR("test_layer", rotation_info)
        module = nn.Linear(4, 4)
        x = torch.randn(2, 4)
        output = module(x)
        rotated_output = hook_ir(module, (x,), output)
        self.assertEqual(rotated_output.shape, output.shape)

        # 测试wrapper_module处理元组输出
        class TupleOutputModule(nn.Module):
            def forward(self, x):
                return (x, x)

        module = TupleOutputModule()
        wrapper = hook_ir.wrapper_module(module)
        output = wrapper(x)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)


class TestQuarotOfflineRotationInfo(unittest.TestCase):
    """Unit tests for QuarotOfflineRotationInfo (PR #141)."""

    def test_QuarotOfflineRotationInfo_after_init_global_rotation_equals_input(self):
        """QuarotOfflineRotationInfo-初始化后-global_rotation 与传入张量一致"""
        rot = torch.randn(8, 8)
        info = QuarotOfflineRotationInfo(global_rotation=rot)
        self.assertIs(info.global_rotation, rot)
        self.assertTrue(torch.equal(info.global_rotation, rot))


class TestQuaRotExtraInfoHookIR(unittest.TestCase):
    """Unit tests for QuaRotExtraInfoHookIR (PR #141)."""

    def setUp(self):
        self.global_rotation = torch.eye(4)
        self.rotation_info = QuarotOfflineRotationInfo(global_rotation=self.global_rotation)

    def test_QuaRotExtraInfoHookIR_after_init_holds_rotation_info(self):
        """QuaRotExtraInfoHookIR-初始化后-正确持有 rotation_info"""
        hook_ir = QuaRotExtraInfoHookIR(self.rotation_info)
        self.assertIs(hook_ir.rotation_info, self.rotation_info)

    def test_QuaRotExtraInfoHookIR_on_call_returns_args_unchanged(self):
        """QuaRotExtraInfoHookIR-调用 __call__ 时-不改变前向参数"""
        hook_ir = QuaRotExtraInfoHookIR(self.rotation_info)
        module = nn.Linear(4, 4)
        x = torch.randn(2, 4)
        result = hook_ir(module, (x,))
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], x)
        self.assertTrue(torch.equal(result[0], x))

    def test_QuaRotExtraInfoHookIR_on_wrapper_module_returns_QuaRotExtraInfoWrapperIR_and_removes_hook(self):
        """QuaRotExtraInfoHookIR-调用 wrapper_module 后-返回 QuaRotExtraInfoWrapperIR 且移除 hook"""
        hook_ir = QuaRotExtraInfoHookIR(self.rotation_info)
        module = nn.Linear(4, 4)
        wrapper = hook_ir.wrapper_module(module)
        self.assertIsInstance(wrapper, QuaRotExtraInfoWrapperIR)
        self.assertIs(wrapper.wrapped_module, module)
        self.assertIs(wrapper.rotation_info, self.rotation_info)
        self.assertIsNone(hook_ir.hook_handle)


class TestQuaRotExtraInfoWrapperIR(unittest.TestCase):
    """Unit tests for QuaRotExtraInfoWrapperIR (PR #141)."""

    def setUp(self):
        self.global_rotation = torch.eye(4)
        self.rotation_info = QuarotOfflineRotationInfo(global_rotation=self.global_rotation)
        self.wrapped_module = nn.Linear(8, 4)

    def test_QuaRotExtraInfoWrapperIR_after_init_holds_wrapped_module_and_rotation_info(self):
        """QuaRotExtraInfoWrapperIR-初始化后-正确持有 wrapped_module 与 rotation_info"""
        wrapper = QuaRotExtraInfoWrapperIR(self.wrapped_module, self.rotation_info)
        self.assertIs(wrapper.wrapped_module, self.wrapped_module)
        self.assertIs(wrapper.rotation_info, self.rotation_info)

    def test_QuaRotExtraInfoWrapperIR_is_atomic_returns_false(self):
        """QuaRotExtraInfoWrapperIR-调用 is_atomic()-返回 False 表示额外导出模式"""
        self.assertFalse(QuaRotExtraInfoWrapperIR.is_atomic())

    def test_QuaRotExtraInfoWrapperIR_on_forward_delegates_to_wrapped_module(self):
        """QuaRotExtraInfoWrapperIR-调用 forward 时-透传到 wrapped_module 且输出一致"""
        wrapper = QuaRotExtraInfoWrapperIR(self.wrapped_module, self.rotation_info)
        x = torch.randn(2, 8)
        out = wrapper(x)
        expected = self.wrapped_module(x)
        self.assertEqual(out.shape, expected.shape)
        self.assertTrue(torch.allclose(out, expected))


if __name__ == '__main__':
    unittest.main()
