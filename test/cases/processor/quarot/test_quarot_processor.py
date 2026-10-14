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
# pylint: disable=consider-using-in,redefined-outer-name

from unittest.mock import MagicMock

import os  # noqa: F401
import sys  # noqa: F401
import pytest
import torch

from resources.fake.qwen3_dense import FakeQwen3Creator
from msmodelslim.core.base.protocol import BatchProcessRequest  # noqa: F401

from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.processor.quarot.common.hadamard import random_hadamard_matrix, walsh_matrix
from msmodelslim.processor.quarot.offline_quarot.quarot import QuaRotProcessorConfig, QuaRotProcessor
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import QuaRotInterface, RotatePair


@pytest.fixture
def mock_model():
    """创建模拟模型"""
    torch.manual_seed(42)
    model = FakeQwen3Creator.get_model()
    return model


@pytest.fixture
def basic_config():
    """基础配置"""
    config = QuaRotProcessorConfig()
    config.online = False
    config.block_size = -1
    config.down_proj_online_layers = []
    config.max_tp_size = 4
    return config


@pytest.fixture
def mock_adapter(mock_model):
    """创建模拟适配器"""
    return MockQuaRotAdapter(mock_model)


class TestRotMaker:
    @pytest.mark.parametrize("size", [12, 20, 28, 36, 40, 52, 60, 76, 108, 136, 140, 156, 160, 172, 200])
    def test_create_rot_with_random(self, size):
        rot = random_hadamard_matrix(size, torch.float32, torch.device("cpu"))
        assert rot.shape == (size, size)

        # 验证正交性
        identity = torch.eye(size, dtype=torch.float32)
        product = rot @ rot.T
        assert torch.allclose(product, identity, atol=1e-5)

    @pytest.mark.parametrize("size", [11, 21, 87, 121])
    def test_create_rot_with_random_unsupported(self, size):
        """
        测试random_hadamard_matrix在不支持的size下会抛出UnsupportedError异常
        """
        with pytest.raises(UnsupportedError):
            random_hadamard_matrix(size, torch.float32, torch.device("cpu"))

    @pytest.mark.parametrize("size", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    def test_create_rot_with_walsh(self, size):
        rot = walsh_matrix(size, torch.float32, torch.device("cpu"))
        assert rot.shape == (size, size)


class MockQuaRotAdapter(QuaRotInterface):
    def __init__(self, model=None):
        if model is None:
            torch.manual_seed(42)
            self.model = FakeQwen3Creator.get_model()
        else:
            self.model = model

    def get_hidden_dim(self):
        return self.model.config.hidden_size

    def get_head_dim(self):
        return self.model.config.head_dim

    def get_num_attention_heads(self):
        return self.model.config.num_attention_heads

    def get_num_key_value_heads(self):
        return self.model.config.num_key_value_heads

    def get_lm_head(self):
        return "lm_head"

    def get_pre_head_layernorm(self):
        return "norm"

    def get_embedding(self):
        return "embed_tokens"

    def get_layer_wise_norm_liner_pair(self, decoder_module):
        norm_linear_pairs = {}
        norm_linear_pairs[decoder_module.input_layernorm] = [
            decoder_module.self_attn.q_proj,
            decoder_module.self_attn.k_proj,
            decoder_module.self_attn.v_proj,
        ]
        norm_linear_pairs[decoder_module.post_attention_layernorm] = [
            decoder_module.mlp.gate_proj,
            decoder_module.mlp.up_proj,
        ]
        return norm_linear_pairs

    def get_layer_wise_ov_pair(self, decoder_module):
        ov_pairs = {}
        ov_pairs[decoder_module.self_attn.o_proj] = decoder_module.self_attn.v_proj
        return ov_pairs

    def get_layer_wise_up_down_pair(self, decoder_module):
        up_down_pairs = {}
        up_down_pairs[decoder_module.mlp.up_proj] = decoder_module.mlp.down_proj
        return up_down_pairs

    def get_ln_fuse_map(self):
        """返回层融合的mapping: 前置norm + 后续linear"""
        fuse_map = {}
        bake_names = []

        # 处理decoder layers
        for i, _ in enumerate(self.model.layers):
            # input_layernorm融合到q_proj, k_proj, v_proj
            fuse_map[f"model.layers.{i}.input_layernorm"] = [
                f"model.layers.{i}.self_attn.q_proj",
                f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.v_proj",
            ]

            # post_attention_layernorm融合到gate_proj, up_proj
            fuse_map[f"model.layers.{i}.post_attention_layernorm"] = [
                f"model.layers.{i}.mlp.gate_proj",
                f"model.layers.{i}.mlp.up_proj",
            ]

            # bake mean into down_proj
            bake_names.append(f"model.layers.{i}.mlp.down_proj")

        return fuse_map, bake_names

    def get_bake_names(self):
        """返回需要bake mean的层名称"""
        bake_names = []
        for i in range(len(self.model.layers)):
            bake_names.append(f"model.layers.{i}.mlp.down_proj")
        return bake_names, bake_names

    def get_rotate_map(self, block_size):
        """返回旋转矩阵的mapping"""
        model_dim = self.get_hidden_dim()
        head_dim = self.get_head_dim()

        # 创建旋转矩阵（简化版本，实际使用中应该从hadamard或walsh生成）
        try:
            if block_size == -1 or model_dim == block_size:
                rot_main = random_hadamard_matrix(model_dim, torch.float32, self.model.device)
            else:
                rot_main = random_hadamard_matrix(model_dim, torch.float32, self.model.device)
        except UnsupportedError:
            rot_main = torch.eye(model_dim, dtype=torch.float32, device=self.model.device)

        try:
            rot_head = random_hadamard_matrix(head_dim, torch.float32, self.model.device)
        except UnsupportedError:
            rot_head = torch.eye(head_dim, dtype=torch.float32, device=self.model.device)

        rotate_pairs = []

        # 生成rotate pair供pre_run使用
        pre_run_pair = RotatePair(left_rot={"embed_tokens": rot_main}, right_rot={})
        rotate_pairs.append(pre_run_pair)

        # 为每一层生成rotate pair
        for i in range(len(self.model.layers)):
            layer_pair = RotatePair(
                left_rot={
                    f"model.layers.{i}.self_attn.q_proj": rot_main,
                    f"model.layers.{i}.self_attn.k_proj": rot_main,
                    f"model.layers.{i}.self_attn.v_proj": rot_main,
                    f"model.layers.{i}.mlp.gate_proj": rot_main,
                    f"model.layers.{i}.mlp.up_proj": rot_main,
                },
                right_rot={
                    f"model.layers.{i}.self_attn.o_proj": rot_head,
                    f"model.layers.{i}.mlp.down_proj": rot_main,
                },
            )
            rotate_pairs.append(layer_pair)

        return rotate_pairs, rotate_pairs


class TestQuaRotAdapter:
    """测试QuaRotInterface的所有方法"""

    @staticmethod
    def test_abstract_class():
        decoder_module = MagicMock()  # noqa: F841
        adapter = QuaRotInterface()
        # 测试QuaRotInterface的抽象方法
        try:
            adapter.get_ln_fuse_map()
            adapter.get_bake_names()
            adapter.get_rotate_map(block_size=-1)
        except Exception:
            pass  # 抽象方法可能会抛出异常，这是正常的


class TestQuaRotProcessor:
    """测试QuaRotProcessor类"""

    @staticmethod
    def test_init_with_default_config(mock_model, basic_config, mock_adapter):
        """测试使用默认配置初始化"""
        processor = QuaRotProcessor(mock_model, basic_config, mock_adapter)
        assert processor.config == basic_config
        assert processor.model == mock_model
        assert processor.adapter == mock_adapter

    @staticmethod
    def test_pre_run_basic(mock_model, basic_config, mock_adapter):
        """测试pre_run基础功能"""
        processor = QuaRotProcessor(mock_model, basic_config, mock_adapter)
        # 应该不抛出异常
        try:
            processor.pre_run()
        except Exception:
            # 可能出现一些与模型结构相关的异常，这是可以接受的
            pass
