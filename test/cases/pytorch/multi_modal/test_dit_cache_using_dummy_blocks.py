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

import sys
import logging
import unittest
from unittest.mock import Mock

import torch
import torch.nn as nn


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

            # 保存原始模块（如果已在sys.modules中）
            if 'torch_npu' in sys.modules:
                self.original_torch_npu = sys.modules['torch_npu']

            # 注册mock模块
            sys.modules['torch_npu'] = self.mock_torch_npu

            # 配置mock属性
            self.mock_torch_npu.__version__ = '2.1.0'
            self.mock_torch_npu.npu_init.return_value = None

        from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheSearchConfig, DitCacheAdaptor, DitCacheConfig

        self.DitCacheSearchConfig = DitCacheSearchConfig
        self.DitCacheAdaptor = DitCacheAdaptor
        self.DitCacheConfig = DitCacheConfig

    def tearDown(self):
        # 仅当使用了mock时才需要清理
        if self.mock_used:
            # 移除mock模块
            if 'torch_npu' in sys.modules:
                del sys.modules['torch_npu']

            # 恢复原始模块（如果存在）
            if self.original_torch_npu is not None:
                sys.modules['torch_npu'] = self.original_torch_npu


class DummyPipeline:
    """
    模拟一个简单单流 pipeline，包含 transformer 模块
    """

    def __init__(self, num_blocks, dit_cache_adaptor):
        self.name = "DummyPipeline"
        self.transformer = DummyTransformer(num_blocks)
        self.dit_cache_adaptor = dit_cache_adaptor

    def __call__(self, *args, **kwargs):
        # 定义各个参数的 dummy 值
        attention_mask = torch.tensor(1.0)
        encoder_hidden_states = torch.tensor(1.0)
        encoder_attention_mask = torch.tensor(1.0)
        cross_attention_kwargs = {}
        class_labels = 0
        frame = 0
        height = 1
        width = 1

        output = None
        # 模拟 5 个时间步的前向传递
        for t in range(5):
            print(f"Time step {t}:")
            hidden_states = torch.tensor([0.0, 0.0])
            input_val = hidden_states
            for i, block in enumerate(self.transformer.transformer_blocks):
                block.t_idx = t
                self.dit_cache_adaptor.set_timestep_idx(t)

                output = block.forward(
                    input_val,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=t * 0.1,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame,
                    height=height,
                    width=width,
                )
                print(f"  Block {i} output: {output}")
                input_val = output  # 将当前 block 输出作为下一个 block 的输入
            print("------")
        print("----- End of System Test -----\n")

        if output is None:
            raise ValueError("No output found")
        return output


# ----------------- 构造简易的单流 pipeline 与 transformer 模块 -----------------
class DummyBlock(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        self.t_idx = -1

    def forward(self, hidden_states, *args, **kwargs):
        """
        为便于观察，这里仅返回一个计算值：
        hidden_states + self.idx + timestep
        其他参数仅作为占位存在
        """
        return hidden_states + torch.tensor([self.t_idx, self.idx])


class DummyTransformer(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        # 构造 num_blocks 个 DummyBlock
        self.transformer_blocks = nn.ModuleList([DummyBlock(i) for i in range(num_blocks)])


# ----------------- 双流 pipeline 与 Transfomer模块 -----------------
class DummyBlockDual(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        self.t_idx = -1

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        """
        为便于观察，这里仅返回一个计算值：
        hidden_states + self.idx + timestep
        其他参数仅作为占位存在
        """
        return hidden_states + torch.tensor([self.t_idx, self.idx]), encoder_hidden_states + torch.tensor(
            [self.t_idx, self.idx]
        )


class DummyTransformerDual(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        # 构造 num_blocks 个 DummyBlock
        self.transformer_blocks = nn.ModuleList([DummyBlockDual(i) for i in range(num_blocks)])


class DummyPipelineDual:
    """
    模拟双流 pipeline，包含 transformer 模块
    """

    def __init__(self, num_blocks, dit_cache_adaptor):
        self.name = "DummyPipeline"
        self.transformer = DummyTransformerDual(num_blocks)
        self.dit_cache_adaptor = dit_cache_adaptor

    def __call__(self, *args, **kwargs):
        # 定义各个参数的 dummy 值
        attention_mask = torch.tensor(1.0)
        encoder_attention_mask = torch.tensor(1.0)
        cross_attention_kwargs = {}
        class_labels = 0
        frame = 0
        height = 1
        width = 1

        output = None
        # 模拟 5 个时间步的前向传递
        for t in range(5):
            print(f"Time step {t}:")
            hidden_states = torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])
            input_val = hidden_states
            for i, block in enumerate(self.transformer.transformer_blocks):
                block.t_idx = t
                self.dit_cache_adaptor.set_timestep_idx(t)

                output = block.forward(
                    *input_val,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=t * 0.1,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame,
                    height=height,
                    width=width,
                )
                print(f"  Block {i} output: {output}")
                input_val = output  # 将当前 block 输出作为下一个 block 的输入
            print("------")
        print("----- End of System Test -----\n")

        if output is None:
            raise ValueError("No output found")
        return output


# 为了让类型提示通过，这里将 OpenSoraPipelineV1_2 定义为 DummyPipeline 的别名
OpenSoraPipelineV1_2 = DummyPipeline

import os  # noqa: E402
import torch.distributed as dist  # noqa: E402


class TestDitCacheAdaptor(BaseDitCacheTestCase):
    def setUp(self):
        super().setUp()
        # 设置分布式环境变量
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

        # 初始化分布式进程组
        dist.init_process_group(
            backend='gloo',  # 单机建议使用gloo，GPU用nccl
            init_method='tcp://localhost:29599',
            rank=0,
            world_size=1,
        )

    def tearDown(self):
        # 释放分布式资源
        dist.destroy_process_group()
        super().tearDown()

    def test_search_with_dit_cache(self):
        self.run_search_test()

    def test_forward_with_dit_cache(self):
        self.run_forward_test()

    def test_search_with_dit_cache_dual(self):
        self.run_search_test_dual()

    def test_forward_with_dit_cache_dual(self):
        self.run_forward_test_dual()

    def run_search_test(self):
        print("----- Running System Test -----")
        num_blocks = 5  # 假设 transformer 有 4 个 block
        pipeline = DummyPipeline(num_blocks, self.DitCacheAdaptor)
        # 配置 DitCache：指定总 block 数和采样步数

        config = self.DitCacheSearchConfig(dit_block_num=num_blocks, num_sampling_steps=20)
        cache_adaptor = self.DitCacheAdaptor(pipeline, config)

        def run_pipeline_and_save_videos(pipeline: DummyPipeline):
            tensor_ouput = pipeline('default input')
            vid = torch.zeros((30, 64, 64, 3)) + tensor_ouput[0]
            return [vid.to(torch.uint8).cpu().numpy()]

        cache_adaptor.search(run_pipeline_and_save_videos)

    def run_search_test_dual(self):
        print("----- Running System Test -----")
        num_blocks = 5  # 假设 transformer 有 4 个 block
        pipeline = DummyPipelineDual(num_blocks, self.DitCacheAdaptor)
        # 配置 DitCache：指定总 block 数和采样步数

        config = self.DitCacheSearchConfig(dit_block_num=num_blocks, num_sampling_steps=20, num_hidden_states=2)
        cache_adaptor = self.DitCacheAdaptor(pipeline, config)

        def run_pipeline_and_save_videos(pipeline: DummyPipelineDual):
            tensor_ouput = pipeline('default input')
            print(f"Warning!!! {tensor_ouput}")
            vid = torch.zeros((30, 64, 64, 3)) + tensor_ouput[0] @ tensor_ouput[1]
            return [vid.to(torch.uint8).cpu().numpy()]

        cache_adaptor.search(run_pipeline_and_save_videos)

    # ----------------- 系统测试 (ST) -----------------
    def run_forward_test(self):
        print("----- Running System Test -----")
        num_blocks = 5  # 假设 transformer 有 4 个 block
        pipeline = DummyPipeline(num_blocks, self.DitCacheAdaptor)
        # 配置 DitCache：指定总 block 数和采样步数

        config = self.DitCacheSearchConfig(dit_block_num=num_blocks, num_sampling_steps=5)

        dit_cache_config = self.DitCacheConfig(
            # Timestep start：开始缓存的时间步
            cache_step_start=1,
            # Timestep interval：每隔 n 个时间步计算一次，其他复用 cache
            cache_step_interval=3,
            # Block 缓存起始索引 idx， 若值为 0 表示第一个 block 开始
            cache_block_start=1,
            # Block 设定缓存区域 block 数量：
            cache_num_blocks=3,
        )

        cache_adaptor = self.DitCacheAdaptor(pipeline, config)
        cache_adaptor.dit_cache_config = dit_cache_config

        # 定义各个参数的 dummy 值
        attention_mask = torch.tensor(1.0)
        encoder_hidden_states = torch.tensor(1.0)
        encoder_attention_mask = torch.tensor(1.0)
        cross_attention_kwargs = {}
        class_labels = 0
        frame = 0
        height = 1
        width = 1

        # 模拟 5 个时间步的前向传递
        for t in range(5):
            print(f"Time step {t}:")
            hidden_states = torch.tensor([0.0, 0.0])
            input_val = hidden_states
            for i, block in enumerate(pipeline.transformer.transformer_blocks):
                block.t_idx = t
                self.DitCacheAdaptor.set_timestep_idx(t)

                output = block.forward(
                    input_val,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=t * 0.1,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame,
                    height=height,
                    width=width,
                )
                print(f"  Block {i} output: {output}")
                input_val = output  # 将当前 block 输出作为下一个 block 的输入
            print("------")
        print("----- End of System Test -----\n")

    def run_forward_test_dual(self):
        print("----- Running System Test -----")
        num_blocks = 5  # 假设 transformer 有 4 个 block
        pipeline = DummyPipelineDual(num_blocks, self.DitCacheAdaptor)
        # 配置 DitCache：指定总 block 数和采样步数

        config = self.DitCacheSearchConfig(dit_block_num=num_blocks, num_sampling_steps=5, num_hidden_states=2)

        dit_cache_config = self.DitCacheConfig(
            # Timestep start：开始缓存的时间步
            cache_step_start=1,
            # Timestep interval：每隔 n 个时间步计算一次，其他复用 cache
            cache_step_interval=3,
            # Block 缓存起始索引 idx， 若值为 0 表示第一个 block 开始
            cache_block_start=1,
            # Block 设定缓存区域 block 数量：
            cache_num_blocks=3,
        )

        cache_adaptor = self.DitCacheAdaptor(pipeline, config)
        cache_adaptor.dit_cache_config = dit_cache_config

        # 定义各个参数的 dummy 值
        attention_mask = torch.tensor(1.0)
        encoder_attention_mask = torch.tensor(1.0)
        cross_attention_kwargs = {}
        class_labels = 0
        frame = 0
        height = 1
        width = 1

        # 模拟 5 个时间步的前向传递
        for t in range(5):
            print(f"Time step {t}:")
            hidden_states = torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])
            input_val = hidden_states
            for i, block in enumerate(pipeline.transformer.transformer_blocks):
                block.t_idx = t
                self.DitCacheAdaptor.set_timestep_idx(t)

                output = block.forward(
                    *input_val,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=t * 0.1,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame,
                    height=height,
                    width=width,
                )
                print(f"  Block {i} output: {output}")
                input_val = output  # 将当前 block 输出作为下一个 block 的输入
            print("------")
        print("----- End of System Test -----\n")


# ----------------- 主入口 -----------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # 先运行系统测试，观察缓存适配器在多步前向传递中的效果
    # run_system_test()
    # 再运行单元测试
    unittest.main(verbosity=2)
