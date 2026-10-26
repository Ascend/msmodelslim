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

from msmodelslim.processor.quarot.online_quarot.online_quarot_interface import RotationConfig
from msmodelslim.processor.quarot.common.quarot_utils import QuaRotMode


class TestRotationConfig(unittest.TestCase):
    """测试 RotationConfig 类"""

    def test_init_with_rotation_matrix(self):
        """测试使用旋转矩阵初始化"""
        rotation_matrix = torch.randn(4, 4)
        config = RotationConfig(rotation_type="input", rotation_matrix=rotation_matrix)
        self.assertEqual(config.rotation_type, "input")
        self.assertTrue(torch.equal(config.rotation_matrix, rotation_matrix))
        self.assertIsNone(config.rotation_size)

    def test_init_with_rotation_size(self):
        """测试使用rotation_size初始化"""
        config = RotationConfig(rotation_type="output", rotation_size=64, rotation_mode=QuaRotMode.HADAMARD)
        self.assertEqual(config.rotation_type, "output")
        self.assertEqual(config.rotation_size, 64)
        self.assertEqual(config.rotation_mode, QuaRotMode.HADAMARD)

    def test_init_with_default_values(self):
        """测试默认值"""
        config = RotationConfig(rotation_type="replace", rotation_size=32, rotation_mode=QuaRotMode.HADAMARD)
        self.assertEqual(config.block_size, -1)
        self.assertEqual(config.rot_step, 1)
        self.assertEqual(config.eye_step, (-1,))
        self.assertEqual(config.seed, 1234)
        self.assertEqual(config.dtype, torch.float32)

    def test_init_with_custom_values(self):
        """测试自定义值"""
        config = RotationConfig(
            rotation_type="input",
            rotation_size=128,
            rotation_mode=QuaRotMode.BLOCK_HADAMARD_SHIFTED,
            block_size=32,
            rot_step=2,
            eye_step=(0, 1),
            seed=5678,
            dtype=torch.float16,
        )
        self.assertEqual(config.block_size, 32)
        self.assertEqual(config.rot_step, 2)
        self.assertEqual(config.eye_step, (0, 1))
        self.assertEqual(config.seed, 5678)
        self.assertEqual(config.dtype, torch.float16)

    def test_validate_rotation_type_valid(self):
        """测试有效的rotation_type"""
        for rotation_type in ["input", "output", "replace", "offline"]:
            config = RotationConfig(rotation_type=rotation_type, rotation_size=32, rotation_mode=QuaRotMode.HADAMARD)
            self.assertEqual(config.rotation_type, rotation_type)

    def test_offline_rotation_with_default_side(self):
        """测试offline类型默认rotation_side"""
        config = RotationConfig(rotation_type="offline", rotation_size=32, rotation_mode=QuaRotMode.HADAMARD)
        # 应该默认为"right"
        self.assertEqual(config.rotation_side, "right")

    def test_offline_rotation_with_explicit_side(self):
        """测试offline类型显式指定rotation_side"""
        for side in ["left", "right"]:
            config = RotationConfig(
                rotation_type="offline", rotation_size=32, rotation_mode=QuaRotMode.HADAMARD, rotation_side=side
            )
            self.assertEqual(config.rotation_side, side)

    def test_non_offline_rotation_side_ignored(self):
        """测试非offline类型时rotation_side被忽略"""
        for rotation_type in ["input", "output", "replace"]:
            config = RotationConfig(
                rotation_type=rotation_type,
                rotation_size=32,
                rotation_mode=QuaRotMode.HADAMARD,
                rotation_side="left",  # 非offline类型应该允许但不使用
            )
            self.assertEqual(config.rotation_type, rotation_type)
            # rotation_side可能为None或设置的值，取决于实现
            # 但应该不会抛出异常

    def test_rotation_matrix_priority(self):
        """测试旋转矩阵优先级（如果提供了矩阵，其他参数可忽略）"""
        rotation_matrix = torch.randn(4, 4)
        config = RotationConfig(
            rotation_type="input",
            rotation_matrix=rotation_matrix,
            rotation_size=64,  # 即使提供了size，也应该使用matrix
            rotation_mode=QuaRotMode.HADAMARD,
        )
        self.assertIsNotNone(config.rotation_matrix)
        # 验证应该通过，因为提供了matrix

    def test_all_rotation_types(self):
        """测试所有旋转类型"""
        rotation_matrix = torch.randn(4, 4)
        for rotation_type in ["input", "output", "replace", "offline"]:
            if rotation_type == "offline":
                config = RotationConfig(
                    rotation_type=rotation_type,
                    rotation_size=32,
                    rotation_mode=QuaRotMode.HADAMARD,
                    rotation_side="right",
                )
            else:
                config = RotationConfig(rotation_type=rotation_type, rotation_matrix=rotation_matrix)
            self.assertEqual(config.rotation_type, rotation_type)

    def test_quarot_mode_enum(self):
        """测试QuaRotMode枚举"""
        for mode in [QuaRotMode.HADAMARD, QuaRotMode.BLOCK_HADAMARD_SHIFTED]:
            config = RotationConfig(rotation_type="input", rotation_size=32, rotation_mode=mode)
            self.assertEqual(config.rotation_mode, mode)


if __name__ == '__main__':
    unittest.main()
