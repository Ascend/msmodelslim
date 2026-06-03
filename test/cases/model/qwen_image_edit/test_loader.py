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
msmodelslim.model.qwen_image_edit.loader 模块的单元测试
"""
# pylint: disable=duplicate-code

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from msmodelslim.model.qwen_image_edit.loader import QwenImageEditAdapterLoader
from msmodelslim.utils.exception import UnsupportedError


class TestQwenImageEditAdapterLoader:
    """测试 QwenImageEditAdapterLoader 类"""

    def test_adapter_class_path_point_to_qwen_image_edit_adapter_when_defined(self):
        """ADAPTER_CLASS_PATH 应指向 QwenImageEditModelAdapter"""
        assert (
            QwenImageEditAdapterLoader.ADAPTER_CLASS_PATH
            == "msmodelslim.model.qwen_image_edit.model_adapter:QwenImageEditModelAdapter"
        )

    def test_load_return_adapter_instance_when_class_path_valid(self):
        """load 在类路径有效时应返回适配器实例"""
        loader = QwenImageEditAdapterLoader()

        class DummyAdapter:
            def __init__(self, model_type, model_path, trust_remote_code):
                self.model_type = model_type
                self.model_path = model_path
                self.trust_remote_code = trust_remote_code

        with patch("msmodelslim.model.plugin_factory.base_loader.import_module") as mock_import:
            mock_import.return_value = SimpleNamespace(QwenImageEditModelAdapter=DummyAdapter)
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin"):
                with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                    result = loader.load(
                        model_type="qwen-image-edit",
                        model_path=Path("/tmp/model"),
                        trust_remote_code=True,
                    )

        assert isinstance(result, DummyAdapter)
        assert result.model_type == "qwen-image-edit"
        assert result.trust_remote_code is True

    def test_load_raise_unsupported_when_adapter_class_path_invalid(self):
        """ADAPTER_CLASS_PATH 格式非法时应抛出 UnsupportedError"""
        loader = QwenImageEditAdapterLoader()
        loader.ADAPTER_CLASS_PATH = "invalid_without_colon"

        with pytest.raises(UnsupportedError, match="ADAPTER_CLASS_PATH"):
            loader.load(model_type="qwen-image-edit", model_path=Path("/tmp/model"))
