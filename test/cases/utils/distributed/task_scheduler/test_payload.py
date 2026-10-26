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

import functools
import unittest

import torch
from torch import nn

from msmodelslim.utils.distributed.task_scheduler.payload import (
    validate_dependency_paths,
    normalize_hash_value,
    stable_callable_identifier,
    task_semantic_hash,
    wave_semantic_hash,
)
from msmodelslim.utils.exception import SchemaValidateError


# ==================== validate_dependency_paths ====================
class TestValidateDependencyPathsWhenValid(unittest.TestCase):
    """测试validate_dependency_paths在有效路径时的行为"""

    def test_validate_dependency_paths_passes_when_all_paths_exist(self):
        """正常：所有路径存在时不应抛出异常"""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        # 不应抛出异常
        validate_dependency_paths(model, ["0", "1"])

    def test_validate_dependency_paths_passes_when_empty_list(self):
        """边界：空列表不应抛出异常"""
        model = nn.Linear(4, 8)
        validate_dependency_paths(model, [])

    def test_validate_dependency_paths_passes_when_none(self):
        """边界：None不应抛出异常"""
        model = nn.Linear(4, 8)
        validate_dependency_paths(model, None)


class TestValidateDependencyPathsWhenInvalid(unittest.TestCase):
    """测试validate_dependency_paths在无效路径时的行为"""

    def test_validate_dependency_paths_raises_when_path_not_found(self):
        """异常：路径不存在时应抛出SchemaValidateError"""
        model = nn.Linear(4, 8)
        with self.assertRaises(SchemaValidateError) as ctx:
            validate_dependency_paths(model, ["nonexistent"])
        self.assertIn("nonexistent", str(ctx.exception))

    def test_validate_dependency_paths_raises_when_path_is_none_in_list(self):
        """异常：列表中包含None时应抛出SchemaValidateError"""
        model = nn.Linear(4, 8)
        with self.assertRaises(SchemaValidateError) as ctx:
            validate_dependency_paths(model, [None])
        self.assertIn("must not be None", str(ctx.exception))

    def test_validate_dependency_paths_raises_when_path_is_empty_string(self):
        """异常：空字符串路径应抛出SchemaValidateError"""
        model = nn.Linear(4, 8)
        with self.assertRaises(SchemaValidateError) as ctx:
            validate_dependency_paths(model, [""])
        self.assertIn("non-empty", str(ctx.exception))

    def test_validate_dependency_paths_raises_when_path_is_whitespace_only(self):
        """异常：仅空白字符的路径应抛出SchemaValidateError"""
        model = nn.Linear(4, 8)
        with self.assertRaises(SchemaValidateError):
            validate_dependency_paths(model, ["   "])


# ==================== normalize_hash_value ====================
class TestNormalizeHashValueWhenNone(unittest.TestCase):
    """测试normalize_hash_value对None的处理"""

    def test_normalize_hash_value_returns_none_when_value_is_none(self):
        """正常：None应原样返回"""
        result = normalize_hash_value(None, "test")
        self.assertIsNone(result)


class TestNormalizeHashValueWhenScalar(unittest.TestCase):
    """测试normalize_hash_value对标量的处理"""

    def test_normalize_hash_value_returns_int_when_int(self):
        """正常：int应原样返回"""
        result = normalize_hash_value(42, "test")
        self.assertEqual(result, 42)

    def test_normalize_hash_value_returns_float_when_float(self):
        """正常：float应原样返回"""
        result = normalize_hash_value(3.14, "test")
        self.assertAlmostEqual(result, 3.14)

    def test_normalize_hash_value_returns_str_when_str(self):
        """正常：str应原样返回"""
        result = normalize_hash_value("hello", "test")
        self.assertEqual(result, "hello")

    def test_normalize_hash_value_returns_bool_when_true(self):
        """正常：bool应原样返回"""
        result = normalize_hash_value(True, "test")
        self.assertTrue(result)


class TestNormalizeHashValueWhenTensor(unittest.TestCase):
    """测试normalize_hash_value对Tensor的处理"""

    def test_normalize_hash_value_returns_dict_when_tensor(self):
        """正常：Tensor应返回包含shape/dtype/device的字典"""
        t = torch.randn(3, 4)
        result = normalize_hash_value(t, "test")
        self.assertIsInstance(result, dict)
        self.assertTrue(result["__tensor__"])
        self.assertEqual(result["shape"], (3, 4))
        self.assertIn("float32", result["dtype"])

    def test_normalize_hash_value_includes_requires_grad_when_tensor(self):
        """正常：Tensor应包含requires_grad信息"""
        t = torch.randn(2, requires_grad=True)
        result = normalize_hash_value(t, "test")
        self.assertTrue(result["requires_grad"])


class TestNormalizeHashValueWhenCollection(unittest.TestCase):
    """测试normalize_hash_value对集合类型的处理"""

    def test_normalize_hash_value_returns_list_when_tuple(self):
        """正常：tuple应转为list"""
        result = normalize_hash_value((1, 2, 3), "test")
        self.assertEqual(result, [1, 2, 3])

    def test_normalize_hash_value_returns_list_when_list(self):
        """正常：list应原样返回结构"""
        result = normalize_hash_value([1, "a", None], "test")
        self.assertEqual(result, [1, "a", None])

    def test_normalize_hash_value_returns_sorted_list_when_set(self):
        """正常：set应转为排序后的list"""
        result = normalize_hash_value({3, 1, 2}, "test")
        self.assertEqual(result, [1, 2, 3])

    def test_normalize_hash_value_returns_sorted_dict_when_dict(self):
        """正常：dict应按键排序"""
        result = normalize_hash_value({"b": 2, "a": 1}, "test")
        self.assertIsInstance(result, dict)
        keys = list(result.keys())
        self.assertEqual(keys, sorted(keys))


class TestNormalizeHashValueWhenUnsupported(unittest.TestCase):
    """测试normalize_hash_value对不支持类型的行为"""

    def test_normalize_hash_value_raises_when_unsupported_type(self):
        """异常：不支持的类型应抛出SchemaValidateError"""
        with self.assertRaises(SchemaValidateError) as ctx:
            normalize_hash_value(lambda x: x, "test")
        self.assertIn("unsupported", str(ctx.exception))


# ==================== stable_callable_identifier ====================
class TestStableCallableIdentifierWhenNone(unittest.TestCase):
    """测试stable_callable_identifier对None的处理"""

    def test_stable_callable_identifier_returns_none_when_cb_is_none(self):
        """正常：None应返回None"""
        result = stable_callable_identifier(None, "test")
        self.assertIsNone(result)


class TestStableCallableIdentifierWhenFunction(unittest.TestCase):
    """测试stable_callable_identifier对普通函数的处理"""

    def test_stable_callable_identifier_returns_module_qualname_when_function(self):
        """正常：普通函数应返回module:qualname格式"""
        result = stable_callable_identifier(len, "test")
        self.assertIsInstance(result, str)
        self.assertIn("builtins", result)
        self.assertIn("len", result)


class TestStableCallableIdentifierWhenPartial(unittest.TestCase):
    """测试stable_callable_identifier对functools.partial的处理"""

    def test_stable_callable_identifier_returns_dict_when_partial(self):
        """正常：partial应返回包含__partial__标记的字典"""
        p = functools.partial(len, [1, 2, 3])
        result = stable_callable_identifier(p, "test")
        self.assertIsInstance(result, dict)
        self.assertTrue(result["__partial__"])
        self.assertIn("func", result)


# ==================== task_semantic_hash ====================
class TestTaskSemanticHash(unittest.TestCase):
    """测试task_semantic_hash函数"""

    def test_task_semantic_hash_returns_hex_string_when_called(self):
        """正常：应返回十六进制哈希字符串"""

        def dummy_fn():
            pass

        result = task_semantic_hash(
            fn=dummy_fn,
            args=(),
            kwargs={},
            dependencies=[],
            parallel=True,
            sync_fn=None,
        )
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)  # SHA-256 hex length

    def test_task_semantic_hash_is_deterministic_when_same_input(self):
        """正常：相同输入应产生相同哈希"""

        def dummy_fn():
            pass

        hash1 = task_semantic_hash(dummy_fn, (1, 2), {"k": "v"}, ["dep"], True, None)
        hash2 = task_semantic_hash(dummy_fn, (1, 2), {"k": "v"}, ["dep"], True, None)
        self.assertEqual(hash1, hash2)

    def test_task_semantic_hash_differs_when_args_differ(self):
        """正常：不同参数应产生不同哈希"""

        def dummy_fn():
            pass

        hash1 = task_semantic_hash(dummy_fn, (1,), {}, [], True, None)
        hash2 = task_semantic_hash(dummy_fn, (2,), {}, [], True, None)
        self.assertNotEqual(hash1, hash2)

    def test_task_semantic_hash_differs_when_parallel_differs(self):
        """正常：不同parallel标志应产生不同哈希"""

        def dummy_fn():
            pass

        hash1 = task_semantic_hash(dummy_fn, (), {}, [], True, None)
        hash2 = task_semantic_hash(dummy_fn, (), {}, [], False, None)
        self.assertNotEqual(hash1, hash2)


# ==================== wave_semantic_hash ====================
class TestWaveSemanticHash(unittest.TestCase):
    """测试wave_semantic_hash函数"""

    def test_wave_semantic_hash_returns_hex_string_when_called(self):
        """正常：应返回十六进制哈希字符串"""
        result = wave_semantic_hash(["abc", "def"])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)

    def test_wave_semantic_hash_is_deterministic_when_same_input(self):
        """正常：相同输入应产生相同哈希"""
        hash1 = wave_semantic_hash(["a", "b", "c"])
        hash2 = wave_semantic_hash(["a", "b", "c"])
        self.assertEqual(hash1, hash2)

    def test_wave_semantic_hash_differs_when_order_differs(self):
        """正常：不同顺序应产生不同哈希"""
        hash1 = wave_semantic_hash(["a", "b"])
        hash2 = wave_semantic_hash(["b", "a"])
        self.assertNotEqual(hash1, hash2)

    def test_wave_semantic_hash_handles_empty_list(self):
        """边界：空列表应返回有效哈希"""
        result = wave_semantic_hash([])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
