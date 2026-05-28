#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
# msmodelslim.processor.adapt_rotation.adapt_rotation 模块的单元测试
import unittest
from unittest.mock import MagicMock

from msmodelslim.processor.adapt_rotation.adapt_rotation import (
    AdaptRotationProcessor,
    AdaptRotationProcessorConfig,
)
from msmodelslim.processor.adapt_rotation.adapt_rotation_stage1 import AdaptRotationStage1Processor
from msmodelslim.processor.adapt_rotation.adapt_rotation_stage2 import AdaptRotationStage2Processor
from msmodelslim.utils.exception import SchemaValidateError

from .common import MockQuaRotAdapter


class TestAdaptRotationProcessorConfig(unittest.TestCase):
    """测试 AdaptRotationProcessorConfig 类"""

    def test_build_stage_config_return_stage1_config_when_stage_is_1_and_type_adapt_rotation(self):
        """测试 stage=1 且 type=adapt_rotation 时正确构建 stage1 配置"""
        data = {"type": "adapt_rotation", "stage": 1, "steps": 10}
        result = AdaptRotationProcessorConfig.model_validate(data)
        self.assertEqual(result.type, "adapt_rotation")
        self.assertEqual(result.stage, 1)
        self.assertIsNotNone(result.stage_config)
        self.assertEqual(result.stage_config.type, "_adapt_rotation_stage1")
        self.assertEqual(result.stage_config.steps, 10)

    def test_build_stage_config_return_stage2_config_when_stage_is_2_and_type_adapt_rotation(self):
        """测试 stage=2 且 type=adapt_rotation 时正确构建 stage2 配置"""
        data = {"type": "adapt_rotation", "stage": 2}
        result = AdaptRotationProcessorConfig.model_validate(data)
        self.assertEqual(result.type, "adapt_rotation")
        self.assertEqual(result.stage, 2)
        self.assertIsNotNone(result.stage_config)
        self.assertEqual(result.stage_config.type, "_adapt_rotation_stage2")

    def test_build_stage_config_raise_error_when_stage_not_in_1_or_2(self):
        """测试 stage 不为 1 或 2 时抛出 SchemaValidateError"""
        data = {"type": "adapt_rotation", "stage": 3}
        with self.assertRaises(Exception):
            AdaptRotationProcessorConfig.model_validate(data)

    def test_build_stage_config_raise_error_when_stage1_has_extra_disallowed_fields(self):
        """测试 stage1 配置包含不允许的 stage2 字段时抛出 SchemaValidateError"""
        data = {"type": "adapt_rotation", "stage": 1, "online": True}
        with self.assertRaises(SchemaValidateError):
            AdaptRotationProcessorConfig.model_validate(data)

    def test_build_stage_config_raise_error_when_stage2_has_extra_disallowed_fields(self):
        """测试 stage2 配置包含不允许的 stage1 字段时抛出 SchemaValidateError"""
        data = {"type": "adapt_rotation", "stage": 2, "steps": 20}
        with self.assertRaises(SchemaValidateError):
            AdaptRotationProcessorConfig.model_validate(data)

    def test_getattr_delegate_to_stage_config_when_accessing_stage_specific_field(self):
        """测试 __getattr__ 将 stage 特有字段委托给 stage_config"""
        data = {"type": "adapt_rotation", "stage": 1, "steps": 15}
        config = AdaptRotationProcessorConfig.model_validate(data)
        self.assertEqual(config.steps, 15)


class TestAdaptRotationProcessor(unittest.TestCase):
    """测试 AdaptRotationProcessor 类"""

    def test_init_create_stage1_inner_processor_when_config_stage_is_1(self):
        """测试 config.stage=1 时创建 AdaptRotationStage1Processor 作为内部处理器"""
        mock_model = MagicMock()
        config = AdaptRotationProcessorConfig.model_validate(
            {"type": "adapt_rotation", "stage": 1}
        )
        mock_adapter = MockQuaRotAdapter()
        with unittest.mock.patch.object(AdaptRotationProcessor, '__init__', lambda s, m, c, a, **kw: None):
            proc = AdaptRotationProcessor.__new__(AdaptRotationProcessor)
            proc.model = mock_model
            proc.config = config
            proc._inner = AdaptRotationStage1Processor(mock_model, config.stage_config, mock_adapter)
        self.assertIsInstance(proc._inner, AdaptRotationStage1Processor)

    def test_init_create_stage2_inner_processor_when_config_stage_is_2(self):
        """测试 config.stage=2 时创建 AdaptRotationStage2Processor 作为内部处理器"""
        mock_model = MagicMock()
        config = AdaptRotationProcessorConfig.model_validate(
            {"type": "adapt_rotation", "stage": 2}
        )
        mock_adapter = MockQuaRotAdapter()
        with unittest.mock.patch.object(AdaptRotationProcessor, '__init__', lambda s, m, c, a, **kw: None):
            proc = AdaptRotationProcessor.__new__(AdaptRotationProcessor)
            proc.model = mock_model
            proc.config = config
            proc._inner = AdaptRotationStage2Processor(mock_model, config.stage_config, mock_adapter)
        self.assertIsInstance(proc._inner, AdaptRotationStage2Processor)

    def test_all_delegate_to_inner_processor(self):
        """测试各方法均委托给内部处理器"""
        mock_request = MagicMock()
        cases = [
            ("support_distributed", True, lambda p, m: (p.support_distributed(), m.support_distributed), None),
            ("is_data_free", False, lambda p, m: (p.is_data_free(), m.is_data_free), None),
            ("preprocess", None, lambda p, m: (p.preprocess(mock_request), m.preprocess), mock_request),
            ("postprocess", None, lambda p, m: (p.postprocess(mock_request), m.postprocess), mock_request),
            ("pre_run", None, lambda p, m: (p.pre_run(), m.pre_run), None),
            ("post_run", None, lambda p, m: (p.post_run(), m.post_run), None),
            ("process", None, lambda p, m: (p.process(mock_request), m.process), mock_request),
            ("need_kv_cache", True, lambda p, m: (p.need_kv_cache(), m.need_kv_cache), None),
        ]
        for name, expected_return, run, call_arg in cases:
            with self.subTest(method=name):
                mock_inner = MagicMock()
                if expected_return is not None:
                    getattr(mock_inner, name).return_value = expected_return
                with unittest.mock.patch.object(
                    AdaptRotationProcessor, "__init__", lambda s, m, c, a, **kw: None
                ):
                    proc = AdaptRotationProcessor.__new__(AdaptRotationProcessor)
                    proc._inner = mock_inner
                result, inner_method = run(proc, mock_inner)
                if expected_return is not None:
                    self.assertEqual(result, expected_return, msg=name)
                if call_arg is None:
                    inner_method.assert_called_once()
                else:
                    inner_method.assert_called_once_with(call_arg)
