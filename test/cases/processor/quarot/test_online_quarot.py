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
from unittest.mock import Mock

import torch
import torch.nn as nn

from msmodelslim.processor.quarot.online_quarot.online_quarot import (
    OnlineQuaRotProcessor,
    OnlineQuaRotProcessorConfig,
)
from msmodelslim.processor.quarot.online_quarot.online_quarot_interface import (
    OnlineQuaRotInterface,
    RotationConfig,
)
from msmodelslim.processor.quarot.common.quarot_utils import QuaRotMode
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.utils.exception import UnsupportedError, SchemaValidateError
from msmodelslim.ir.quarot import OnlineRotationWrapper


class MockOnlineQuaRotAdapter(OnlineQuaRotInterface):
    """Mock适配器用于测试"""

    def __init__(self, rotation_configs=None):
        self.rotation_configs = rotation_configs or {}

    def get_online_rotation_configs(self, model=None):
        return self.rotation_configs


class TestOnlineQuaRotProcessorConfig(unittest.TestCase):
    """测试 OnlineQuaRotProcessorConfig 类"""

    def test_init_and_validation(self):
        """测试初始化和验证"""
        # 默认初始化
        config = OnlineQuaRotProcessorConfig()
        self.assertEqual(config.type, "online_quarot")
        self.assertEqual(config.block_size, -1)

        # 带值初始化
        config = OnlineQuaRotProcessorConfig(include=["layer1"], exclude=["layer2"], block_size=32)
        self.assertEqual(config.include, ["layer1"])
        self.assertEqual(config.block_size, 32)

        # 有效值：-1和2的幂
        OnlineQuaRotProcessorConfig(block_size=-1)
        OnlineQuaRotProcessorConfig(block_size=16)

        # 无效值
        with self.assertRaises(SchemaValidateError):
            OnlineQuaRotProcessorConfig(block_size=0)
        with self.assertRaises(SchemaValidateError):
            OnlineQuaRotProcessorConfig(block_size=-2)
        with self.assertRaises(SchemaValidateError):
            OnlineQuaRotProcessorConfig(block_size=3)


class TestOnlineQuaRotProcessor(unittest.TestCase):
    """测试 OnlineQuaRotProcessor 类"""

    def setUp(self):
        self.model = nn.Module()
        self.model.linear = nn.Linear(4, 4)

    def test_init(self):
        """测试初始化"""
        config = OnlineQuaRotProcessorConfig()
        rotation_configs = {
            "linear": RotationConfig(rotation_type="input", rotation_size=4, rotation_mode=QuaRotMode.HADAMARD)
        }
        adapter = MockOnlineQuaRotAdapter(rotation_configs)
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        self.assertEqual(processor.config, config)
        self.assertEqual(processor.adapter, adapter)

        # 测试异常情况
        with self.assertRaises(UnsupportedError):
            OnlineQuaRotProcessor(self.model, config, adapter=None)
        with self.assertRaises(UnsupportedError):
            OnlineQuaRotProcessor(self.model, config, adapter=Mock())

    def test_basic_properties(self):
        """测试基本属性"""
        config = OnlineQuaRotProcessorConfig()
        adapter = MockOnlineQuaRotAdapter()
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        self.assertTrue(processor.support_distributed())
        self.assertTrue(processor.is_data_free())

    def test_pre_run(self):
        """测试pre_run"""
        config = OnlineQuaRotProcessorConfig()

        # 使用提供的旋转矩阵
        rotation_matrix = torch.eye(4)
        rotation_configs = {"linear": RotationConfig(rotation_type="input", rotation_matrix=rotation_matrix)}
        adapter = MockOnlineQuaRotAdapter(rotation_configs)
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        processor.pre_run()
        self.assertIn("linear", processor.rotation_infos)
        self.assertTrue(torch.equal(processor.rotation_infos["linear"].rotation_matrix, rotation_matrix))

        # 自动生成旋转矩阵
        rotation_configs = {
            "linear": RotationConfig(rotation_type="input", rotation_size=4, rotation_mode=QuaRotMode.HADAMARD)
        }
        adapter = MockOnlineQuaRotAdapter(rotation_configs)
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        processor.pre_run()
        self.assertIsNotNone(processor.rotation_infos["linear"].rotation_matrix)

        # 缺少rotation_mode时抛出异常
        rotation_configs = {"linear": RotationConfig(rotation_type="input", rotation_size=4)}
        adapter = MockOnlineQuaRotAdapter(rotation_configs)
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        with self.assertRaises(UnsupportedError):
            processor.pre_run()

    def test_preprocess_rotation_types(self):
        """测试preprocess应用各种旋转类型"""
        config = OnlineQuaRotProcessorConfig()
        rotation_matrix = torch.eye(4)
        request = BatchProcessRequest(name="", module=self.model, datas=None, outputs=None)

        # input旋转
        rotation_configs = {"linear": RotationConfig(rotation_type="input", rotation_matrix=rotation_matrix)}
        adapter = MockOnlineQuaRotAdapter(rotation_configs)
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        processor.pre_run()
        processor.preprocess(request)
        self.assertTrue(len(self.model.linear._forward_pre_hooks) > 0)

        # output旋转
        self.model.linear = nn.Linear(4, 4)
        rotation_configs = {"linear": RotationConfig(rotation_type="output", rotation_matrix=rotation_matrix)}
        adapter = MockOnlineQuaRotAdapter(rotation_configs)
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        processor.pre_run()
        processor.preprocess(request)
        self.assertTrue(len(self.model.linear._forward_hooks) > 0)

        # replace旋转
        self.model.linear = nn.Linear(4, 4)
        rotation_configs = {"linear": RotationConfig(rotation_type="replace", rotation_matrix=rotation_matrix)}
        adapter = MockOnlineQuaRotAdapter(rotation_configs)
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        processor.pre_run()
        processor.preprocess(request)
        self.assertIsInstance(self.model.linear, OnlineRotationWrapper)

        # offline旋转
        self.model.linear = nn.Linear(4, 4)
        rotation_configs = {
            "linear": RotationConfig(rotation_type="offline", rotation_matrix=rotation_matrix, rotation_side="right")
        }
        adapter = MockOnlineQuaRotAdapter(rotation_configs)
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        processor.pre_run()
        processor.preprocess(request)
        self.assertIsNotNone(self.model.linear.weight)

    def test_include_exclude_and_post_run(self):
        """测试include/exclude过滤和post_run"""
        config = OnlineQuaRotProcessorConfig(include=["linear"], exclude=["other"])
        rotation_matrix = torch.eye(4)
        rotation_configs = {"linear": RotationConfig(rotation_type="input", rotation_matrix=rotation_matrix)}
        adapter = MockOnlineQuaRotAdapter(rotation_configs)
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        processor.pre_run()

        request = BatchProcessRequest(name="", module=self.model, datas=None, outputs=None)
        processor.preprocess(request)
        processor.post_run()

    def test_should_apply_rotation(self):
        """测试_should_apply_rotation方法"""
        # 基本过滤
        config = OnlineQuaRotProcessorConfig(include=["layer1", "layer2"], exclude=["layer2"])
        adapter = MockOnlineQuaRotAdapter()
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        self.assertTrue(processor._should_apply_rotation("layer1"))
        self.assertFalse(processor._should_apply_rotation("layer2"))
        self.assertFalse(processor._should_apply_rotation("layer3"))

        # 通配符
        config = OnlineQuaRotProcessorConfig(include=["*"])
        processor = OnlineQuaRotProcessor(self.model, config, adapter)
        self.assertTrue(processor._should_apply_rotation("layer1"))
        self.assertTrue(processor._should_apply_rotation("any_layer"))


if __name__ == '__main__':
    unittest.main()
