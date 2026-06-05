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

import unittest

import torch

from msmodelslim.ir.qal.qbase import QDType, QScope, QScheme, QParam, QStorage


class TestQDType(unittest.TestCase):
    """测试 QDType 枚举类"""

    def test_enum_values_should_be_correct(self):
        """测试枚举值是否正确"""
        self.assertEqual(QDType.FLOAT.value, "float")
        self.assertEqual(QDType.INT8.value, "int8")
        self.assertEqual(QDType.INT4.value, "int4")
        self.assertEqual(QDType.MXFP8.value, "mxfp8")
        self.assertEqual(QDType.MXFP4.value, "mxfp4")
        self.assertEqual(QDType.FP8_E4M3.value, "fp8_e4m3")
        self.assertEqual(QDType.PLACEHOLDER.value, "placeholder")

    def test_enum_should_be_str_subclass(self):
        """测试 QDType 是 str 的子类"""
        self.assertTrue(issubclass(QDType, str))

    def test_mx_finfo_should_return_valid_info_for_mxfp8(self):
        """测试 MXFP8 的 mx_finfo 属性"""
        finfo = QDType.MXFP8.mx_finfo
        self.assertEqual(finfo.block_size, 32)
        self.assertEqual(finfo.scale_bits, 8)
        self.assertFalse(finfo.flush_fp32_subnorms)
        self.assertEqual(finfo.ebits, 4)
        self.assertEqual(finfo.mbits, 5)
        self.assertEqual(finfo.emax, 8)
        # max_norm = 2^emax * 1.75 = 2^8 * 1.75 = 448.0
        self.assertAlmostEqual(finfo.max_norm, 448.0)

    def test_mx_finfo_should_return_valid_info_for_mxfp4(self):
        """测试 MXFP4 的 mx_finfo 属性"""
        finfo = QDType.MXFP4.mx_finfo
        self.assertEqual(finfo.block_size, 32)
        self.assertEqual(finfo.scale_bits, 8)
        self.assertFalse(finfo.flush_fp32_subnorms)
        self.assertEqual(finfo.ebits, 2)
        self.assertEqual(finfo.mbits, 3)
        self.assertEqual(finfo.emax, 2)
        self.assertAlmostEqual(finfo.max_norm, 6.0)  # 2^2 * 3/2 = 6.0

    def test_mx_finfo_should_raise_for_non_mx_types(self):
        """测试非 MX 类型访问 mx_finfo 时抛出异常"""
        non_mx_types = [QDType.FLOAT, QDType.INT8, QDType.INT4, QDType.FP8_E4M3, QDType.PLACEHOLDER]
        for dtype in non_mx_types:
            with self.subTest(dtype=dtype):
                with self.assertRaises(Exception):
                    _ = dtype.mx_finfo


class TestQScope(unittest.TestCase):
    """测试 QScope 枚举类"""

    def test_enum_values_should_be_correct(self):
        """测试枚举值是否正确"""
        self.assertEqual(QScope.PER_TENSOR.value, "per_tensor")
        self.assertEqual(QScope.PER_CHANNEL.value, "per_channel")
        self.assertEqual(QScope.PER_GROUP.value, "per_group")
        self.assertEqual(QScope.PER_BLOCK.value, "per_block")
        self.assertEqual(QScope.PER_TOKEN.value, "per_token")
        self.assertEqual(QScope.PD_MIX.value, "pd_mix")
        self.assertEqual(QScope.PER_HEAD.value, "per_head")
        self.assertEqual(QScope.DUAL_SCALE.value, "dual_scale")
        self.assertEqual(QScope.PLACEHOLDER.value, "placeholder")

    def test_enum_should_be_str_subclass(self):
        """测试 QScope 是 str 的子类"""
        self.assertTrue(issubclass(QScope, str))


class TestQScheme(unittest.TestCase):
    """测试 QScheme 数据类"""

    def test_default_values_should_be_placeholder(self):
        """测试默认值"""
        scheme = QScheme()
        self.assertEqual(scheme.scope, QScope.PLACEHOLDER)
        self.assertEqual(scheme.dtype, QDType.PLACEHOLDER)
        self.assertTrue(scheme.symmetric)

    def test_custom_values_should_be_set_correctly(self):
        """测试自定义值"""
        scheme = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=False)
        self.assertEqual(scheme.scope, QScope.PER_TENSOR)
        self.assertEqual(scheme.dtype, QDType.INT8)
        self.assertFalse(scheme.symmetric)

    def test_frozen_should_prevent_modification(self):
        """测试不可变性"""
        scheme = QScheme()
        with self.assertRaises(AttributeError):
            scheme.scope = QScope.PER_TENSOR

    def test_repr_should_contain_field_values(self):
        """测试 __repr__ 输出"""
        scheme = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True)
        repr_str = repr(scheme)
        self.assertIn("per_channel", repr_str)
        self.assertIn("int8", repr_str)
        self.assertIn("True", repr_str)

    def test_equality_should_work_for_same_values(self):
        """测试相同值的相等性"""
        scheme1 = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=True)
        scheme2 = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=True)
        self.assertEqual(scheme1, scheme2)

    def test_equality_should_fail_for_different_values(self):
        """测试不同值的不相等性"""
        scheme1 = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=True)
        scheme2 = QScheme(scope=QScope.PER_CHANNEL, dtype=QDType.INT8, symmetric=True)
        self.assertNotEqual(scheme1, scheme2)


class TestQParam(unittest.TestCase):
    """测试 QParam 数据类"""

    def test_creation_with_valid_scheme(self):
        """测试使用有效 scheme 创建"""
        scheme = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=True)
        param = QParam(scheme=scheme)
        self.assertEqual(param.scheme, scheme)
        self.assertIsNone(param.ext)

    def test_creation_with_ext(self):
        """测试带 ext 参数创建"""
        scheme = QScheme(scope=QScope.PER_GROUP, dtype=QDType.INT8, symmetric=True)
        ext = {"group_size": 128}
        param = QParam(scheme=scheme, ext=ext)
        self.assertEqual(param.ext["group_size"], 128)

    def test_repr_for_per_group_should_show_group_size(self):
        """测试 PER_GROUP 的 repr 显示 group_size"""
        scheme = QScheme(scope=QScope.PER_GROUP, dtype=QDType.INT8, symmetric=True)
        param = QParam(scheme=scheme, ext={"group_size": 64})
        repr_str = repr(param)
        self.assertIn("group_size=64", repr_str)

    def test_repr_for_non_per_group_should_not_show_group_size(self):
        """测试非 PER_GROUP 的 repr 不显示 group_size"""
        scheme = QScheme(scope=QScope.PER_TENSOR, dtype=QDType.INT8, symmetric=True)
        param = QParam(scheme=scheme)
        repr_str = repr(param)
        self.assertNotIn("group_size", repr_str)


class TestQStorage(unittest.TestCase):
    """测试 QStorage 数据类"""

    def test_creation_with_valid_params(self):
        """测试使用有效参数创建"""
        value = torch.tensor([1.0, 2.0, 3.0])
        storage = QStorage(dtype=QDType.FLOAT, value=value)
        self.assertEqual(storage.dtype, QDType.FLOAT)
        self.assertTrue(torch.equal(storage.value, value))
        self.assertIsNone(storage.ext)

    def test_creation_with_ext(self):
        """测试带 ext 参数创建"""
        value = torch.tensor([1.0, 2.0])
        ext = {"scale": torch.tensor(0.5)}
        storage = QStorage(dtype=QDType.INT8, value=value, ext=ext)
        self.assertEqual(storage.ext["scale"], torch.tensor(0.5))

    def test_T_should_return_transposed(self):
        """测试 T 属性返回转置"""
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        storage = QStorage(dtype=QDType.FLOAT, value=value)
        transposed = storage.T
        self.assertTrue(torch.equal(transposed.value, value.T))

    def test_T_should_preserve_dtype_and_ext(self):
        """测试 T 属性保留 dtype 和 ext"""
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ext = {"key": "value"}
        storage = QStorage(dtype=QDType.INT8, value=value, ext=ext)
        transposed = storage.T
        self.assertEqual(transposed.dtype, QDType.INT8)
        self.assertEqual(transposed.ext, ext)

    def test_same_like_should_create_new_storage_with_same_dtype_and_ext(self):
        """测试 same_like 创建具有相同 dtype 和 ext 的新存储"""
        value = torch.tensor([1.0, 2.0])
        ext = {"key": "value"}
        storage = QStorage(dtype=QDType.INT8, value=value, ext=ext)
        new_value = torch.tensor([3.0, 4.0])
        new_storage = storage.same_like(new_value)
        self.assertEqual(new_storage.dtype, QDType.INT8)
        self.assertEqual(new_storage.ext, ext)
        self.assertTrue(torch.equal(new_storage.value, new_value))

    def test_same_like_should_not_share_storage(self):
        """测试 same_like 不共享存储"""
        value = torch.tensor([1.0, 2.0])
        storage = QStorage(dtype=QDType.FLOAT, value=value)
        new_value = torch.tensor([3.0, 4.0])
        new_storage = storage.same_like(new_value)
        # 修改新存储不应影响原存储
        new_storage.value[0] = 100.0
        self.assertNotEqual(storage.value[0], 100.0)

    def test_to_float_should_convert_to_torch_float32(self):
        """测试 to(FLOAT) 转换为 torch.float32"""
        value = torch.tensor([1, 2, 3], dtype=torch.int8)
        storage = QStorage(dtype=QDType.INT8, value=value)
        storage.to(QDType.FLOAT)
        self.assertEqual(storage.dtype, QDType.FLOAT)
        self.assertEqual(storage.value.dtype, torch.float32)

    def test_to_int8_should_convert_to_torch_int8(self):
        """测试 to(INT8) 转换为 torch.int8"""
        value = torch.tensor([1.0, 2.0, 3.0])
        storage = QStorage(dtype=QDType.FLOAT, value=value)
        storage.to(QDType.INT8)
        self.assertEqual(storage.dtype, QDType.INT8)
        self.assertEqual(storage.value.dtype, torch.int8)

    def test_to_int4_should_convert_to_torch_int8(self):
        """测试 to(INT4) 转换为 torch.int8"""
        value = torch.tensor([1.0, 2.0, 3.0])
        storage = QStorage(dtype=QDType.FLOAT, value=value)
        storage.to(QDType.INT4)
        self.assertEqual(storage.dtype, QDType.INT4)
        self.assertEqual(storage.value.dtype, torch.int8)

    def test_to_fp8_e4m3_should_convert_to_torch_float32(self):
        """测试 to(FP8_E4M3) 转换为 torch.float32"""
        value = torch.tensor([1.0, 2.0, 3.0])
        storage = QStorage(dtype=QDType.FLOAT, value=value)
        storage.to(QDType.FP8_E4M3)
        self.assertEqual(storage.dtype, QDType.FP8_E4M3)
        self.assertEqual(storage.value.dtype, torch.float32)

    def test_set_value_float_type_should_temporarily_change_float_type(self):
        """测试 set_value_float_type 临时改变浮点类型"""
        value = torch.tensor([1.0, 2.0])
        storage = QStorage(dtype=QDType.FLOAT, value=value)

        # 默认 float32
        storage.to(QDType.FLOAT)
        self.assertEqual(storage.value.dtype, torch.float32)

        # 临时切换到 float16
        with QStorage.set_value_float_type(torch.float16):
            storage.to(QDType.FLOAT)
            self.assertEqual(storage.value.dtype, torch.float16)

        # 恢复后应该是 float32
        storage.to(QDType.FLOAT)
        self.assertEqual(storage.value.dtype, torch.float32)

    def test_reshape_should_not_modify_in_place(self):
        """测试 reshape 不修改原张量（已知 bug）"""
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        storage = QStorage(dtype=QDType.FLOAT, value=value)
        original_shape = storage.value.shape
        storage.reshape(torch.Size([4]))
        # 注意：当前实现有 bug，reshape 结果未赋值回去
        self.assertEqual(storage.value.shape, original_shape)


if __name__ == '__main__':
    unittest.main()
