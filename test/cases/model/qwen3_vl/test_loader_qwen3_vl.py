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

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import torch  # noqa: F401
except ModuleNotFoundError as e:
    raise unittest.SkipTest("torch is not installed; skip qwen3_vl loader unit tests") from e

from msmodelslim.model.qwen3_vl.loader import Qwen3VlAdapterLoader
from msmodelslim.utils.exception import VersionError

if "transformers.masking_utils" not in sys.modules:
    masking_utils = types.ModuleType("transformers.masking_utils")
    masking_utils.create_causal_mask = MagicMock()
    sys.modules["transformers.masking_utils"] = masking_utils

if "transformers.models.qwen3_vl.modeling_qwen3_vl" not in sys.modules:
    modeling = types.ModuleType("transformers.models.qwen3_vl.modeling_qwen3_vl")
    modeling.Qwen3VLTextDecoderLayer = MagicMock
    sys.modules["transformers.models.qwen3_vl.modeling_qwen3_vl"] = modeling

try:
    from msmodelslim.model.qwen3_vl.model_adapter import Qwen3VLModelAdapter

    _QWEN3_VL_IMPORT_OK = True
except Exception:
    Qwen3VLModelAdapter = None
    _QWEN3_VL_IMPORT_OK = False


def _mock_adapter_init(self, model_type, model_path, trust_remote_code=False):
    self.model_type = model_type
    self.model_path = model_path
    self.trust_remote_code = trust_remote_code
    self._processor = None
    self._tokenizer = None


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VlAdapterLoaderAdapterClassPath(unittest.TestCase):
    """测试Qwen3VlAdapterLoader的ADAPTER_CLASS_PATH配置"""

    def test_adapter_class_path_when_defined_then_point_to_qwen3_vl_model_adapter(self):
        """正常：ADAPTER_CLASS_PATH应指向Qwen3VLModelAdapter"""
        self.assertEqual(
            Qwen3VlAdapterLoader.ADAPTER_CLASS_PATH, "msmodelslim.model.qwen3_vl.model_adapter:Qwen3VLModelAdapter"
        )


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VlAdapterLoaderLoad(unittest.TestCase):
    """测试Qwen3VlAdapterLoader的load方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path("/tmp/qwen3_vl-model")
        self.loader = Qwen3VlAdapterLoader()

    def test_load_with_valid_params_when_called_then_return_qwen3_vl_model_adapter(self):
        """正常：load应实例化并返回Qwen3VLModelAdapter"""
        with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin"):
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                with patch("msmodelslim.model.plugin_factory.base_loader.get_require_packages", return_value={}):
                    with patch("msmodelslim.model.plugin_factory.base_loader.import_module") as mock_import:
                        mock_import.return_value = SimpleNamespace(Qwen3VLModelAdapter=Qwen3VLModelAdapter)

                        with patch(
                            "msmodelslim.model.common.vlm_base.VLMBaseModelAdapter.__init__", _mock_adapter_init
                        ):
                            adapter = self.loader.load(
                                model_type=self.model_type,
                                model_path=self.model_path,
                                trust_remote_code=True,
                            )

        self.assertIsInstance(adapter, Qwen3VLModelAdapter)
        self.assertEqual(adapter.model_type, self.model_type)
        self.assertEqual(adapter.model_path, self.model_path)
        self.assertTrue(adapter.trust_remote_code)

    def test_load_with_trust_remote_code_false_when_called_then_pass_false(self):
        """边界：trust_remote_code默认False时应传递False"""
        with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin"):
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                with patch("msmodelslim.model.plugin_factory.base_loader.get_require_packages", return_value={}):
                    with patch("msmodelslim.model.plugin_factory.base_loader.import_module") as mock_import:
                        mock_import.return_value = SimpleNamespace(Qwen3VLModelAdapter=Qwen3VLModelAdapter)

                        with patch(
                            "msmodelslim.model.common.vlm_base.VLMBaseModelAdapter.__init__", _mock_adapter_init
                        ):
                            adapter = self.loader.load(
                                model_type=self.model_type,
                                model_path=self.model_path,
                            )

        self.assertFalse(adapter.trust_remote_code)


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VlAdapterLoaderPrecheck(unittest.TestCase):
    """测试Qwen3VlAdapterLoader的precheck方法"""

    def setUp(self):
        self.loader = Qwen3VlAdapterLoader()
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path("/tmp/qwen3_vl-model")

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
