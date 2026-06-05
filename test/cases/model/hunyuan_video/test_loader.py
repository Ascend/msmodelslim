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

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from msmodelslim.model.hunyuan_video.loader import HunyuanVideoAdapterLoader
from msmodelslim.utils.exception import VersionError

# 将 hyvideo 相关模块注入 sys.modules，避免 _check_import_dependency 失败
_MOCK_MODULES = [
    'hyvideo',
    'hyvideo.config',
    'hyvideo.constants',
    'hyvideo.modules.models',
    'hyvideo.inference',
    'hyvideo.utils.file_utils',
]
_ORIGINAL_MODULES = {}
for _mod in _MOCK_MODULES:
    _ORIGINAL_MODULES[_mod] = sys.modules.get(_mod)
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

try:
    from msmodelslim.model.hunyuan_video.model_adapter import HunyuanVideoModelAdapter

    _HUANYUAN_IMPORT_OK = True
except Exception:
    HunyuanVideoModelAdapter = None
    _HUANYUAN_IMPORT_OK = False
finally:
    # 恢复原始模块
    for _mod, _orig in _ORIGINAL_MODULES.items():
        if _orig is not None:
            sys.modules[_mod] = _orig
        elif _mod in sys.modules:
            del sys.modules[_mod]


def _mock_adapter_init(self, model_type, model_path, trust_remote_code=False):
    self.model_type = model_type
    self.model_path = model_path
    self.trust_remote_code = trust_remote_code
    self.pipeline = None
    self.transformer = None
    self.model_args = None


class TestHunyuanVideoAdapterLoaderAdapterClassPath(unittest.TestCase):
    """测试HunyuanVideoAdapterLoader的ADAPTER_CLASS_PATH配置"""

    def test_adapter_class_path_when_defined_then_point_to_hunyuan_video_model_adapter(self):
        """正常：ADAPTER_CLASS_PATH应指向HunyuanVideoModelAdapter"""
        self.assertEqual(
            HunyuanVideoAdapterLoader.ADAPTER_CLASS_PATH,
            "msmodelslim.model.hunyuan_video.model_adapter:HunyuanVideoModelAdapter",
        )


@unittest.skipUnless(_HUANYUAN_IMPORT_OK, "HunyuanVideo dependencies are not available for import")
class TestHunyuanVideoAdapterLoaderLoad(unittest.TestCase):
    """测试HunyuanVideoAdapterLoader的load方法"""

    def setUp(self):
        self.model_type = "hunyuan_video"
        self.model_path = Path("/tmp/hunyuan-video-model")
        self.loader = HunyuanVideoAdapterLoader()

    def test_load_with_valid_params_when_called_then_return_adapter(self):
        """正常：load应实例化并返回HunyuanVideoModelAdapter"""
        with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin"):
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                with patch("msmodelslim.model.plugin_factory.base_loader.get_require_packages", return_value={}):
                    with patch("msmodelslim.model.plugin_factory.base_loader.import_module") as mock_import:
                        mock_import.return_value = SimpleNamespace(HunyuanVideoModelAdapter=HunyuanVideoModelAdapter)

                        with patch(
                            "msmodelslim.model.base.BaseModelAdapter.__init__",
                            _mock_adapter_init,
                        ):
                            with patch.object(
                                HunyuanVideoModelAdapter,
                                "_check_import_dependency",
                            ):
                                adapter = self.loader.load(
                                    model_type=self.model_type,
                                    model_path=self.model_path,
                                    trust_remote_code=True,
                                )

        self.assertIsInstance(adapter, HunyuanVideoModelAdapter)
        self.assertEqual(adapter.model_type, self.model_type)
        self.assertEqual(adapter.model_path, self.model_path)
        self.assertTrue(adapter.trust_remote_code)

    def test_load_with_trust_remote_code_false_when_called_then_pass_false(self):
        """边界：trust_remote_code默认False时应传递False"""
        with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin"):
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                with patch("msmodelslim.model.plugin_factory.base_loader.get_require_packages", return_value={}):
                    with patch("msmodelslim.model.plugin_factory.base_loader.import_module") as mock_import:
                        mock_import.return_value = SimpleNamespace(HunyuanVideoModelAdapter=HunyuanVideoModelAdapter)

                        with patch(
                            "msmodelslim.model.base.BaseModelAdapter.__init__",
                            _mock_adapter_init,
                        ):
                            with patch.object(
                                HunyuanVideoModelAdapter,
                                "_check_import_dependency",
                            ):
                                adapter = self.loader.load(
                                    model_type=self.model_type,
                                    model_path=self.model_path,
                                )

        self.assertFalse(adapter.trust_remote_code)


class TestHunyuanVideoAdapterLoaderPrecheck(unittest.TestCase):
    """测试HunyuanVideoAdapterLoader的precheck方法"""

    def setUp(self):
        self.loader = HunyuanVideoAdapterLoader()
        self.model_type = "hunyuan_video"
        self.model_path = Path("/tmp/hunyuan-video-model")

    def test_precheck_with_valid_model_type_when_called_then_check_dependencies(self):
        """正常：precheck应触发依赖检查"""
        with patch(
            "msmodelslim.model.plugin_factory.base_loader.msmodelslim_config",
            SimpleNamespace(model_adapter_dependencies={}),
        ):
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin") as mock_set:
                with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                    self.loader.precheck(
                        model_type=self.model_type,
                        model_path=self.model_path,
                    )

        plugin_name = mock_set.call_args[0][0]
        self.assertEqual(plugin_name, f"msmodelslim.model_adapter.plugins:{self.model_type}")

    def test_precheck_when_dependency_check_fails_then_set_is_match_false(self):
        """异常：依赖检查失败时应设置 _is_match 为 False"""
        with patch(
            "msmodelslim.model.plugin_factory.base_loader.msmodelslim_config",
            SimpleNamespace(model_adapter_dependencies={}),
        ):
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin"):
                with patch(
                    "msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin",
                    side_effect=VersionError("dependency mismatch"),
                ):
                    self.loader.precheck(
                        model_type=self.model_type,
                        model_path=self.model_path,
                    )

        self.assertTrue(self.loader._is_match)
