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
from unittest.mock import MagicMock, patch

from torch import nn

from msmodelslim.processor.save.processor import (
    QuantSaveProcessorConfig,
    QuantSaveProcessor,
    _convert_hookir_to_wrapper,
)
from msmodelslim.format.base import QuantFormatConfig
from msmodelslim.utils.exception import SchemaValidateError


class TestQuantSaveProcessorConfig(unittest.TestCase):
    """测试 QuantSaveProcessorConfig 类"""

    def test_type_should_be_saver(self):
        """测试 type 字段默认值"""
        mock_format = MagicMock(spec=QuantFormatConfig)
        config = QuantSaveProcessorConfig(format=mock_format)
        self.assertEqual(config.type, "saver")

    def test_save_directory_should_default_to_empty(self):
        """测试 save_directory 默认为空"""
        mock_format = MagicMock(spec=QuantFormatConfig)
        config = QuantSaveProcessorConfig(format=mock_format)
        self.assertEqual(config.save_directory, "")

    def test_set_save_directory_should_update(self):
        """测试 set_save_directory 更新目录"""
        mock_format = MagicMock(spec=QuantFormatConfig)
        config = QuantSaveProcessorConfig(format=mock_format)
        config.set_save_directory("/tmp/test")
        self.assertEqual(config.save_directory, "/tmp/test")

    def test_format_via_parse_save_config_with_config_instance(self):
        """测试 format 直接传入配置实例"""
        mock_config = MagicMock(spec=QuantFormatConfig)
        config = QuantSaveProcessorConfig(format=mock_config)
        self.assertEqual(config.format, mock_config)

    def test_format_via_parse_save_config_should_raise_for_invalid_type(self):
        """测试 format 无效类型抛出异常"""
        with self.assertRaises(SchemaValidateError):
            QuantSaveProcessorConfig(format=123)


class TestConvertHookirToWrapper(unittest.TestCase):
    """测试 _convert_hookir_to_wrapper 函数"""

    def test_should_handle_module_without_hooks(self):
        """测试没有 hook 的模块"""
        model = nn.Module()
        linear = nn.Linear(4, 2)
        model.sub = linear

        # 不应该抛出异常
        _convert_hookir_to_wrapper(model)

    def test_should_handle_module_with_empty_hooks(self):
        """测试空 hook 的模块"""
        model = nn.Module()
        linear = nn.Linear(4, 2)
        model.sub = linear
        linear._forward_pre_hooks = {}

        _convert_hookir_to_wrapper(model)


def _make_format_mock():
    """创建 mock format 及对应的 QuantFormatFactory mock"""
    mock_format = MagicMock()
    mock_factory = MagicMock()
    mock_factory.create.return_value = mock_format
    return mock_format, mock_factory


class TestQuantSaveProcessor(unittest.TestCase):
    """测试 QuantSaveProcessor 类"""

    @patch('msmodelslim.processor.save.processor.QuantFormatFactory')
    def test_should_initialize_with_format(self, mock_factory_cls):
        """测试初始化时创建格式"""
        mock_format, mock_factory = _make_format_mock()
        mock_factory_cls.return_value = mock_factory

        model = nn.Module()
        config = MagicMock()
        config.save_directory = "/tmp/test"
        config.format = MagicMock(spec=QuantFormatConfig)
        config.format.set_save_directory = MagicMock()

        adapter = MagicMock()
        adapter.model_path = "/model/path"

        processor = QuantSaveProcessor(model, config, adapter)

        self.assertEqual(processor._format, mock_format)
        config.format.set_save_directory.assert_called_once_with("/tmp/test")

    @patch('msmodelslim.processor.save.processor.QuantFormatFactory')
    def test_support_distributed_should_return_false_by_default(self, mock_factory_cls):
        """测试 support_distributed 默认返回 False（未重写）"""
        mock_format, mock_factory = _make_format_mock()
        mock_factory_cls.return_value = mock_factory

        model = nn.Module()
        config = MagicMock()
        config.save_directory = ""
        config.format = MagicMock(spec=QuantFormatConfig)
        config.format.set_save_directory = MagicMock()

        adapter = MagicMock()
        adapter.model_path = ""

        processor = QuantSaveProcessor(model, config, adapter)
        result = processor.support_distributed()

        self.assertFalse(result)

    @patch('msmodelslim.processor.save.processor.QuantFormatFactory')
    def test_pre_run_should_call_prepare_export(self, mock_factory_cls):
        """测试 pre_run 调用 prepare_export"""
        mock_format, mock_factory = _make_format_mock()
        mock_factory_cls.return_value = mock_factory

        model = nn.Module()
        config = MagicMock()
        config.save_directory = ""
        config.format = MagicMock(spec=QuantFormatConfig)
        config.format.set_save_directory = MagicMock()

        adapter = MagicMock()
        adapter.model_path = ""

        processor = QuantSaveProcessor(model, config, adapter)
        processor.pre_run()

        mock_format.prepare_export.assert_called_once()

    @patch('msmodelslim.processor.save.processor.QuantFormatFactory')
    @patch('msmodelslim.processor.save.processor._convert_hookir_to_wrapper')
    def test_postprocess_should_convert_hooks_and_process(self, mock_convert, mock_factory_cls):
        """测试 postprocess 转换 hooks 并处理"""
        mock_format, mock_factory = _make_format_mock()
        mock_factory_cls.return_value = mock_factory

        model = nn.Module()
        config = MagicMock()
        config.save_directory = ""
        config.format = MagicMock(spec=QuantFormatConfig)
        config.format.set_save_directory = MagicMock()

        adapter = MagicMock()
        adapter.model_path = ""

        processor = QuantSaveProcessor(model, config, adapter)

        request = MagicMock()
        request.name = "layer1"
        request.module = nn.Linear(4, 2)

        processor.postprocess(request)

        mock_convert.assert_called_once_with(request.module)
        mock_format.process_module_tensors.assert_called_once_with("layer1", request.module)

    @patch('msmodelslim.processor.save.processor.QuantFormatFactory')
    def test_post_run_should_finalize_export(self, mock_factory_cls):
        """测试 post_run 调用 finalize_export"""
        mock_format, mock_factory = _make_format_mock()
        mock_format.support_distributed.return_value = False
        mock_factory_cls.return_value = mock_factory

        model = nn.Module()
        config = MagicMock()
        config.save_directory = ""
        config.format = MagicMock(spec=QuantFormatConfig)
        config.format.set_save_directory = MagicMock()

        adapter = MagicMock()
        adapter.model_path = ""

        processor = QuantSaveProcessor(model, config, adapter)
        processor.post_run()

        mock_format.finalize_export.assert_called_once_with(model)

    @patch('msmodelslim.processor.save.processor.QuantFormatFactory')
    def test_post_run_should_only_finalize_when_no_merge(self, mock_factory_cls):
        """测试 post_run 仅调用 finalize_export（merge_ranks 由 format 内部处理）"""
        mock_format, mock_factory = _make_format_mock()
        mock_factory_cls.return_value = mock_factory

        model = nn.Module()
        config = MagicMock()
        config.save_directory = ""
        config.format = MagicMock(spec=QuantFormatConfig)
        config.format.set_save_directory = MagicMock()

        adapter = MagicMock()
        adapter.model_path = ""

        processor = QuantSaveProcessor(model, config, adapter)
        processor.post_run()

        mock_format.finalize_export.assert_called_once_with(model)


if __name__ == '__main__':
    unittest.main()
