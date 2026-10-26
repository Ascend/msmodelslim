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

from msmodelslim.processor.quant.autoround_utils.utils import (
    SHARED_CACHE_KEYS,
    SPECIAL_SHARED_CACHE_KEYS,
    get_shared_keys,
    get_quant_func,
    reshape_pad_tensor_by_group_size,
    round_ste,
    revert_tensor_by_pad,
)


class TestSharedCacheKeys(unittest.TestCase):
    """测试共享缓存键常量"""

    def test_shared_cache_keys_should_contain_expected_keys(self):
        """测试 SHARED_CACHE_KEYS 包含预期的键"""
        self.assertIn("position_ids", SHARED_CACHE_KEYS)
        self.assertIn("cache_position", SHARED_CACHE_KEYS)
        self.assertIn("position_embeddings", SHARED_CACHE_KEYS)

    def test_special_shared_cache_keys_should_contain_expected_models(self):
        """测试 SPECIAL_SHARED_CACHE_KEYS 包含预期的模型"""
        self.assertIn("Gemma3ForConditionalGeneration", SPECIAL_SHARED_CACHE_KEYS)
        self.assertIn("MiniMaxText01ForCausalLM", SPECIAL_SHARED_CACHE_KEYS)


class TestGetSharedKeys(unittest.TestCase):
    """测试 get_shared_keys 函数"""

    def test_should_return_base_keys_for_unknown_model(self):
        """测试未知模型返回基础键"""

        class UnknownModel:
            pass

        model = UnknownModel()
        keys = get_shared_keys(model)
        for key in SHARED_CACHE_KEYS:
            self.assertIn(key, keys)

    def test_should_return_extended_keys_for_gemma3(self):
        """测试 Gemma3 模型返回扩展键"""

        class Gemma3ForConditionalGeneration:
            pass

        model = Gemma3ForConditionalGeneration()
        keys = get_shared_keys(model)
        self.assertIn("position_embeddings_global", keys)
        self.assertIn("position_embeddings_local", keys)

    def test_should_return_extended_keys_for_minimax(self):
        """测试 MiniMax 模型返回扩展键"""

        class MiniMaxText01ForCausalLM:
            pass

        model = MiniMaxText01ForCausalLM()
        keys = get_shared_keys(model)
        self.assertIn("slope_rate", keys)


class TestGetQuantFunc(unittest.TestCase):
    """测试 get_quant_func 函数"""

    def test_should_return_symmetric_func(self):
        """测试返回对称量化函数"""
        func, key = get_quant_func("int", 4, True)
        self.assertIsNotNone(func)
        self.assertEqual(key, "int_sym")

    def test_should_return_asymmetric_func(self):
        """测试返回非对称量化函数"""
        func, key = get_quant_func("int", 4, False)
        self.assertIsNotNone(func)
        self.assertEqual(key, "int_asym")

    def test_should_raise_for_unsupported_dtype(self):
        """测试不支持的数据类型抛出异常"""
        with self.assertRaises(ValueError):
            get_quant_func("unsupported", 4, True)


class TestReshapePadTensorByGroupSize(unittest.TestCase):
    """测试 reshape_pad_tensor_by_group_size 函数"""

    def test_group_size_0_should_reshape_to_2d(self):
        """测试 group_size=0 时重塑为 2D"""
        data = torch.randn(2, 3, 4)
        result, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, 0)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 24)  # 2 * 3 * 4
        self.assertEqual(orig_shape, (2, 3, 4))
        self.assertEqual(pad_len, 0)

    def test_group_size_minus1_should_not_reshape(self):
        """测试 group_size=-1 时不重塑"""
        data = torch.randn(2, 4)
        result, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, -1)
        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(pad_len, 0)

    def test_group_size_greater_than_last_dim_should_not_reshape(self):
        """测试 group_size 大于最后一维时不重塑"""
        data = torch.randn(2, 4)
        result, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, 8)
        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(pad_len, 0)

    def test_group_size_divisible_should_reshape(self):
        """测试 group_size 可整除时重塑"""
        data = torch.randn(2, 8)
        result, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, 4)
        self.assertEqual(result.shape, (4, 4))  # (2*8/4, 4)
        self.assertEqual(pad_len, 0)

    def test_group_size_not_divisible_should_pad_and_reshape(self):
        """测试 group_size 不可整除时填充并重塑"""
        data = torch.randn(2, 10)
        result, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, 4)
        self.assertEqual(result.shape[1], 4)
        self.assertEqual(pad_len, 2)  # 12 - 10 = 2
        # 2*10=20 elements, padded to 24, reshaped to (6, 4)
        self.assertEqual(result.shape[0], 6)

    def test_3d_tensor_should_be_flattened_to_2d(self):
        """测试 3D 张量被展平为 2D"""
        data = torch.randn(2, 3, 8)
        result, orig_shape, pad_len = reshape_pad_tensor_by_group_size(data, 4)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[1], 4)


class TestRoundSte(unittest.TestCase):
    """测试 round_ste 函数"""

    def test_should_round_values(self):
        """测试四舍五入"""
        x = torch.tensor([0.3, 0.7, 1.2, 1.8, -0.3, -0.7])
        result = round_ste(x)
        expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 0.0, -1.0])
        self.assertTrue(torch.equal(result, expected))

    def test_should_pass_through_gradients(self):
        """测试梯度直通"""
        x = torch.tensor([0.5, 1.5], requires_grad=True)
        result = round_ste(x)
        result.sum().backward()
        # 梯度应该直通，即 grad = [1.0, 1.0]
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.allclose(x.grad, torch.tensor([1.0, 1.0])))

    def test_integer_values_should_unchanged(self):
        """测试整数值不变"""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = round_ste(x)
        self.assertTrue(torch.equal(result, x))


class TestRevertTensorByPad(unittest.TestCase):
    """测试 revert_tensor_by_pad 函数"""

    def test_no_padding_should_reshape(self):
        """测试无填充时重塑"""
        data = torch.tensor([1, 2, 3, 4, 5, 6])
        result = revert_tensor_by_pad(data, (2, 3), 0)
        self.assertEqual(result.shape, (2, 3))

    def test_with_padding_should_remove_padding(self):
        """测试有填充时移除填充"""
        # 创建一个 3x4 的张量，模拟从 (3, 3) 填充 1 个元素后的情况
        # 原始形状 (3, 3) = 9 个元素，填充到 12 个元素 (3x4)
        data = torch.arange(12).reshape(3, 4)
        # orig_shape = (3, 3), pad_len = 1
        # 需要从 3x4 恢复到 (3, 3)，移除最后 1 个元素
        result = revert_tensor_by_pad(data, (3, 3), 1)
        self.assertEqual(result.shape, (3, 3))
        # 验证数据正确恢复
        self.assertEqual(result[0, 0].item(), 0)
        self.assertEqual(result[0, 2].item(), 2)
        self.assertEqual(result[1, 0].item(), 4)
        self.assertEqual(result[2, 2].item(), 10)


class TestQuantTensorAsym(unittest.TestCase):
    """测试 quant_tensor_asym 函数"""

    def test_should_return_qdq_result(self):
        """测试返回量化反量化结果"""
        from msmodelslim.processor.quant.autoround_utils.utils import quant_tensor_asym

        tensor = torch.randn(4, 8)
        result, scale, zp = quant_tensor_asym(tensor, bits=4, group_size=-1)

        self.assertEqual(result.shape, tensor.shape)
        self.assertIsNotNone(scale)
        self.assertIsNotNone(zp)

    def test_should_return_quantized_when_output_qdq_false(self):
        """测试 output_qdq=False 返回量化值"""
        from msmodelslim.processor.quant.autoround_utils.utils import quant_tensor_asym

        tensor = torch.randn(4, 8)
        result, scale, zp = quant_tensor_asym(tensor, bits=4, group_size=-1, output_qdq=False)

        # 量化值应该是整数
        self.assertTrue(torch.all(result == result.round()))

    def test_with_tensor_min_max(self):
        """测试提供 tensor_min/max"""
        from msmodelslim.processor.quant.autoround_utils.utils import quant_tensor_asym

        tensor = torch.randn(4, 8)
        tensor_min = torch.clamp(tensor.min(-1)[0], max=0)
        tensor_max = torch.clamp(tensor.max(-1)[0], min=0)
        result, scale, zp = quant_tensor_asym(
            tensor, bits=4, group_size=-1, tensor_min=tensor_min, tensor_max=tensor_max
        )

        self.assertEqual(result.shape, tensor.shape)

    def test_with_group_size(self):
        """测试带 group_size"""
        from msmodelslim.processor.quant.autoround_utils.utils import quant_tensor_asym

        tensor = torch.randn(4, 8)
        result, scale, zp = quant_tensor_asym(tensor, bits=4, group_size=4)

        self.assertEqual(result.shape, tensor.shape)


class TestQuantTensorSym(unittest.TestCase):
    """测试 quant_tensor_sym 函数"""

    def test_should_return_qdq_result(self):
        """测试返回量化反量化结果"""
        from msmodelslim.processor.quant.autoround_utils.utils import quant_tensor_sym

        tensor = torch.randn(4, 8)
        result, scale, zp = quant_tensor_sym(tensor, bits=4, group_size=-1)

        self.assertEqual(result.shape, tensor.shape)
        self.assertIsNotNone(scale)
        self.assertIsNotNone(zp)

    def test_should_return_quantized_when_output_qdq_false(self):
        """测试 output_qdq=False 返回量化值"""
        from msmodelslim.processor.quant.autoround_utils.utils import quant_tensor_sym

        tensor = torch.randn(4, 8)
        result, scale, zp = quant_tensor_sym(tensor, bits=4, group_size=-1, output_qdq=False)

        # 量化值应该是整数
        self.assertTrue(torch.all(result == result.round()))

    def test_with_tensor_min_max(self):
        """测试提供 tensor_min/max"""
        from msmodelslim.processor.quant.autoround_utils.utils import quant_tensor_sym

        tensor = torch.randn(4, 8)
        tensor_min = torch.clamp(tensor.min(-1)[0], max=0)
        tensor_max = torch.clamp(tensor.max(-1)[0], min=0)
        result, scale, zp = quant_tensor_sym(
            tensor, bits=4, group_size=-1, tensor_min=tensor_min, tensor_max=tensor_max
        )

        self.assertEqual(result.shape, tensor.shape)

    def test_with_group_size(self):
        """测试带 group_size"""
        from msmodelslim.processor.quant.autoround_utils.utils import quant_tensor_sym

        tensor = torch.randn(4, 8)
        result, scale, zp = quant_tensor_sym(tensor, bits=4, group_size=4)

        self.assertEqual(result.shape, tensor.shape)

    def test_with_8_bits(self):
        """测试 8 位量化"""
        from msmodelslim.processor.quant.autoround_utils.utils import quant_tensor_sym

        tensor = torch.randn(4, 8)
        result, scale, zp = quant_tensor_sym(tensor, bits=8, group_size=-1)

        self.assertEqual(result.shape, tensor.shape)

    def test_with_4_bits(self):
        """测试 4 位量化"""
        from msmodelslim.processor.quant.autoround_utils.utils import quant_tensor_sym

        tensor = torch.randn(4, 8)
        result, scale, zp = quant_tensor_sym(tensor, bits=4, group_size=-1)

        self.assertEqual(result.shape, tensor.shape)


if __name__ == '__main__':
    unittest.main()
