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

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from msmodelslim.model.qwen2_5_vl.loader import Qwen2_5VlAdapterLoader
from msmodelslim.utils.exception import VersionError

try:
    from msmodelslim.model.qwen2_5_vl.model_adapter import Qwen25VLModelAdapter

    _QWEN25_VL_IMPORT_OK = True
except Exception:
    Qwen25VLModelAdapter = None
    _QWEN25_VL_IMPORT_OK = False


def _mock_adapter_init(self, model_type, model_path, trust_remote_code=False):
    self.model_type = model_type
    self.model_path = model_path
    self.trust_remote_code = trust_remote_code
    self._processor = None
    self._tokenizer = None


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen2_5VlAdapterLoaderAdapterClassPath(unittest.TestCase):
    """测试Qwen2_5VlAdapterLoader的ADAPTER_CLASS_PATH配置"""

    def test_adapter_class_path_when_defined_then_point_to_qwen25_vl_model_adapter(self):
        """正常：ADAPTER_CLASS_PATH应指向Qwen25VLModelAdapter"""
        self.assertEqual(
            Qwen2_5VlAdapterLoader.ADAPTER_CLASS_PATH, "msmodelslim.model.qwen2_5_vl.model_adapter:Qwen25VLModelAdapter"
        )


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen2_5VlAdapterLoaderLoad(unittest.TestCase):
    """测试Qwen2_5VlAdapterLoader的load方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path("/tmp/qwen2_5_vl-model")
        self.loader = Qwen2_5VlAdapterLoader()

    def test_load_with_valid_params_when_called_then_return_qwen25_vl_model_adapter(self):
        """正常：load应实例化并返回Qwen25VLModelAdapter"""
        with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin"):
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                with patch("msmodelslim.model.plugin_factory.base_loader.get_require_packages", return_value={}):
                    with patch("msmodelslim.model.plugin_factory.base_loader.import_module") as mock_import:
                        mock_import.return_value = SimpleNamespace(Qwen25VLModelAdapter=Qwen25VLModelAdapter)

                        with patch(
                            "msmodelslim.model.common.vlm_base.VLMBaseModelAdapter.__init__", _mock_adapter_init
                        ):
                            adapter = self.loader.load(
                                model_type=self.model_type,
                                model_path=self.model_path,
                                trust_remote_code=True,
                            )

        self.assertIsInstance(adapter, Qwen25VLModelAdapter)
        self.assertEqual(adapter.model_type, self.model_type)
        self.assertEqual(adapter.model_path, self.model_path)
        self.assertTrue(adapter.trust_remote_code)

    def test_load_with_trust_remote_code_false_when_called_then_pass_false(self):
        """边界：trust_remote_code默认False时应传递False"""
        with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin"):
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                with patch("msmodelslim.model.plugin_factory.base_loader.get_require_packages", return_value={}):
                    with patch("msmodelslim.model.plugin_factory.base_loader.import_module") as mock_import:
                        mock_import.return_value = SimpleNamespace(Qwen25VLModelAdapter=Qwen25VLModelAdapter)

                        with patch(
                            "msmodelslim.model.common.vlm_base.VLMBaseModelAdapter.__init__", _mock_adapter_init
                        ):
                            adapter = self.loader.load(
                                model_type=self.model_type,
                                model_path=self.model_path,
                            )

        self.assertFalse(adapter.trust_remote_code)


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen2_5VlAdapterLoaderPrecheck(unittest.TestCase):
    """测试Qwen2_5VlAdapterLoader的precheck方法"""

    def setUp(self):
        self.loader = Qwen2_5VlAdapterLoader()
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path("/tmp/qwen2_5_vl-model")

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

    def test_precheck_when_dependency_check_fails_then_raise_version_error(self):
        """异常：依赖检查失败时应抛出VersionError"""
        with patch(
            "msmodelslim.model.plugin_factory.base_loader.msmodelslim_config",
            SimpleNamespace(model_adapter_dependencies={}),
        ):
            with patch(
                "msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin",
                side_effect=VersionError("dependency mismatch"),
            ):
                with self.assertRaises(VersionError) as context:
                    self.loader.precheck(
                        model_type=self.model_type,
                        model_path=self.model_path,
                    )

                self.assertIn("Dependency check failed for plugin", str(context.exception))
