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

# pylint: disable=attribute-defined-outside-init

import pytest
import torch
from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig, LinearQuantizer
from msmodelslim.utils.exception import SpecError


class TestW4A8HiFxQuantizer:
    """
    测试 W4A8 HiF4（权重量化 dtype="hif4"）+ MXFP8（激活量化 dtype="mxfp8"）量化器，
    底层对应 W4A8HiFxMXDynamicPerBlockFakeQuantLinear。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """为每个测试准备共用的配置与可以被 32 整除的线性层（避开 block_size 对齐问题）"""
        self.act_config = QConfig(
            dtype="mxfp8",
            scope="per_block",
            method="minmax",
            symmetric=True,
        )
        self.weight_config = QConfig(
            dtype="hif4",
            scope="per_block",
            method="minmax",
            symmetric=True,
        )
        self.linear_config = LinearQConfig(act=self.act_config, weight=self.weight_config)
        # 使用能被 block_size（通常为 32）整除的维度，避免 reshape/unflatten 错误
        self.in_features = 64
        self.out_features = 128
        self.linear_layer = nn.Linear(self.in_features, self.out_features)

    # ---------- LinearQConfig ----------
    def test_linear_qconfig_creation(self):
        """测试 LinearQConfig 正确存储 act / weight 配置"""
        assert self.linear_config.act == self.act_config
        assert self.linear_config.weight == self.weight_config

    # ---------- 初始化 ----------
    def test_initialization(self):
        """正常初始化不抛异常"""
        quantizer = LinearQuantizer(self.linear_config)
        assert quantizer is not None

    # ---------- setup ----------
    def test_setup_with_valid_linear(self):
        """向 setup 传入合法 nn.Linear 正常完成"""
        quantizer = LinearQuantizer(self.linear_config)
        quantizer.setup(self.linear_layer)

    # ---------- forward ----------
    def test_forward_after_setup(self):
        """setup 后 forward 输出 shape 正确"""
        quantizer = LinearQuantizer(self.linear_config)
        quantizer.setup(self.linear_layer)
        inp = torch.randn(4, self.in_features)
        out = quantizer(inp)
        assert out.shape == (4, self.out_features)

    def test_forward_without_setup_raises(self):
        """
        未 setup 直接 forward，内部属性可能未初始化。
        根据具体实现可能抛出 SpecError 或 AttributeError，此处放宽为捕获任意异常。
        """
        quantizer = LinearQuantizer(self.linear_config)
        with pytest.raises(SpecError):
            quantizer(torch.randn(4, self.in_features))

    def test_forward_with_different_batch(self):
        """不同 batch 大小的输入均能正确通过（维度已对齐 block_size）"""
        quantizer = LinearQuantizer(self.linear_config)
        quantizer.setup(self.linear_layer)
        for bs in [1, 8, 16]:
            out = quantizer(torch.randn(bs, self.in_features))
            assert out.shape == (bs, self.out_features)

    # ---------- deploy ----------
    def test_deploy_after_forward(self):
        """forward 后 deploy 可以返回可用量化层"""
        quantizer = LinearQuantizer(self.linear_config)
        quantizer.setup(self.linear_layer)
        quantizer(torch.randn(8, self.in_features))
        deployed = quantizer.deploy()
        assert deployed is not None
        out = deployed(torch.randn(8, self.in_features))
        assert out.shape == (8, self.out_features)

    # ---------- 边界测试 ----------
    def test_forward_input_not_divisible_by_block_size(self):
        layer = nn.Linear(34, 120)
        quantizer = LinearQuantizer(self.linear_config)
        quantizer.setup(layer)
