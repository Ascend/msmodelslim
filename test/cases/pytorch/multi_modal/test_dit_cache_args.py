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
# pylint: disable=no-member,pointless-string-statement

import sys
import unittest
from unittest.mock import Mock

from torch import nn


# 基类：封装通用的setUp和tearDown逻辑
class BaseDitCacheTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_used = False  # 标记是否使用了mock
        self.original_torch_npu = None  # 保存真实模块（如果存在）

        # 1. 尝试导入真实的torch_npu
        try:
            # 导入成功：使用真实模块，无需mock
            import torch_npu  # noqa: F401
        except ImportError:
            # 导入失败：使用mock
            self.mock_used = True
            self.mock_torch_npu = Mock()

            if 'torch_npu' in sys.modules:
                self.original_torch_npu = sys.modules['torch_npu']

            # 注册mock模块
            sys.modules['torch_npu'] = self.mock_torch_npu
            self.mock_torch_npu.__version__ = '2.1.0'
            self.mock_torch_npu.npu_init.return_value = None

        from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheSearchConfig, DitCacheAdaptor, DitCacheConfig

        self.dit_cache_search_config = DitCacheSearchConfig
        self.dit_cache_adaptor = DitCacheAdaptor
        self.dit_cache_config = DitCacheConfig

    def tearDown(self):
        # 仅当使用了mock时才需要清理
        if self.mock_used:
            if 'torch_npu' in sys.modules:
                del sys.modules['torch_npu']

            if self.original_torch_npu is not None:
                sys.modules['torch_npu'] = self.original_torch_npu


class TestDitCacheConfig(BaseDitCacheTestCase):
    def test_valid_config(self):
        """测试有效的配置参数"""
        config = self.dit_cache_config(
            cache_step_start=10, cache_step_interval=5, cache_block_start=2, cache_num_blocks=4
        )
        self.assertEqual(config.cache_step_start, 10)
        self.assertEqual(config.cache_step_interval, 5)
        self.assertEqual(config.cache_block_start, 2)
        self.assertEqual(config.cache_num_blocks, 4)

    def test_invalid_cache_step_start(self):
        """测试 cache_step_start 不能为负数"""
        with self.assertRaises(ValueError) as context:
            self.dit_cache_config(cache_step_start=-1, cache_step_interval=5, cache_block_start=2, cache_num_blocks=4)
        self.assertIn("cache_step_start must be a non-negative integer", str(context.exception))

    def test_invalid_cache_step_interval(self):
        """测试 cache_step_interval 不能小于等于 0"""
        with self.assertRaises(ValueError) as context:
            self.dit_cache_config(cache_step_start=10, cache_step_interval=0, cache_block_start=2, cache_num_blocks=4)
        self.assertIn("cache_step_interval must be a positive integer", str(context.exception))

    def test_invalid_cache_block_start(self):
        """测试 cache_block_start 不能为负数"""
        with self.assertRaises(ValueError) as context:
            self.dit_cache_config(cache_step_start=10, cache_step_interval=5, cache_block_start=-1, cache_num_blocks=4)
        self.assertIn("cache_block_start must be a non-negative integer", str(context.exception))

    def test_invalid_cache_num_blocks(self):
        """测试 cache_num_blocks 不能小于等于 0"""
        with self.assertRaises(ValueError) as context:
            self.dit_cache_config(cache_step_start=10, cache_step_interval=5, cache_block_start=2, cache_num_blocks=-1)
        self.assertIn("cache_num_blocks must be", str(context.exception))


class TestDitCacheSearchConfig(BaseDitCacheTestCase):
    def test_valid_search_config(self):
        """测试有效的配置参数"""
        config = self.dit_cache_search_config(cache_ratio=1.5, dit_block_num=12, num_sampling_steps=100)
        self.assertEqual(config.cache_ratio, 1.5)
        self.assertEqual(config.dit_block_num, 12)
        self.assertEqual(config.num_sampling_steps, 100)

    def test_invalid_cache_ratio(self):
        """测试 cache_ratio 必须在 [1.0, 2.0] 范围内"""
        with self.assertRaises(ValueError) as context:
            self.dit_cache_search_config(cache_ratio=2.5, num_sampling_steps=100)
        self.assertIn("cache_ratio should be in the range of (1.0, 2.0]", str(context.exception))

    def test_invalid_dit_block_num(self):
        """测试 dit_block_num 不能为负数或 0"""
        with self.assertRaises(ValueError) as context:
            self.dit_cache_search_config(cache_ratio=1.5, dit_block_num=0, num_sampling_steps=100)
        self.assertIn("dit_block_num must be positive", str(context.exception))

    def test_invalid_num_sampling_steps(self):
        """测试 num_sampling_steps 不能为空且必须为正整数"""
        with self.assertRaises(ValueError) as context:
            self.dit_cache_search_config(cache_ratio=1.3, num_sampling_steps=None)
        self.assertIn("num_sampling_steps must be set to search", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.dit_cache_search_config(cache_ratio=1.3, num_sampling_steps=-10)
        self.assertIn("num_sampling_steps must be positive", str(context.exception))

    def test_invalid_config(self):
        with self.assertRaises(ValueError) as context:
            self.dit_cache_search_config(cache_ratio="1.3", dit_block_num='10', num_sampling_steps=100)
        with self.assertRaises(ValueError) as context:
            self.dit_cache_search_config(cache_ratio=1.3, dit_block_num=10)
        with self.assertRaises(ValueError) as context:  # noqa: F841
            self.dit_cache_search_config(cache_ratio=1.3, dit_block_num=10, num_sampling_steps="100")


class TestDitCacheAdaptor(BaseDitCacheTestCase):
    def setUp(self):
        # 先调用基类的setUp()，确保torch_npu的mock逻辑执行
        super().setUp()
        """初始化一个假设的 pipeline 用于测试"""

        class FakeBlock(nn.Module):
            def forward(self, x):
                return x

        class FakePipeline:
            def __init__(self):
                self.transformer = type('TempClass', (), {})()
                self.transformer.transformer_blocks = nn.ModuleList([FakeBlock() for _ in range(5)])

        self.fake_pipeline = FakePipeline
        self.pipeline = FakePipeline()
        self.config = self.dit_cache_search_config(
            cache_ratio=1.5,
            dit_block_num=len(self.pipeline.transformer.transformer_blocks),
            num_sampling_steps=100,
            num_hidden_states=1,
        )
        self.adaptor = self.dit_cache_adaptor(
            pipeline=self.pipeline, config=self.config, dit_block_path="transformer.transformer_blocks"
        )

    def test_set_timestep_idx(self):
        """测试 set_timestep_idx() 正常工作"""
        self.adaptor.set_timestep_idx(5)
        self.assertEqual(self.adaptor.get_timestep_idx(), 5)

    def test_get_timestep_idx_without_setting(self):
        """测试未设置时间步时，get_timestep_idx() 抛出异常"""
        adaptor = self.dit_cache_adaptor(pipeline=self.fake_pipeline(), dit_block_path="transformer.transformer_blocks")
        self.dit_cache_adaptor._timestep_idx = None
        with self.assertRaises(ValueError) as context:
            adaptor.get_timestep_idx()
        self.assertIn("Please call DitCacheAdaptor.set_timestep_idx", str(context.exception))

    def test_get_and_check_blocks_valid(self):
        """测试正确提取 transformer_blocks"""
        blocks = self.adaptor.get_and_check_blocks(self.pipeline, "transformer.transformer_blocks")
        self.assertIsInstance(blocks, nn.ModuleList)
        self.assertEqual(len(blocks), 5)

    def test_get_and_check_blocks_invalid_path(self):
        """测试错误的路径抛出异常"""
        with self.assertRaises(ValueError) as context:
            self.adaptor.get_and_check_blocks(self.pipeline, "transformer.invalid_blocks")
        self.assertIn("Failed to access 'transformer.invalid_blocks'", str(context.exception))

    def test_get_and_check_blocks_wrong_type(self):
        """测试路径正确但类型错误的情况"""
        self.pipeline.transformer.transformer_blocks = []
        with self.assertRaises(TypeError) as context:
            self.adaptor.get_and_check_blocks(self.pipeline, "transformer.transformer_blocks")
        self.assertIn("must be type nn.ModuleList", str(context.exception))


if __name__ == "__main__":
    unittest.main()
