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

from msmodelslim.model.transformers.detect import LLM_ADAPTER_CLASS_PATH
from msmodelslim.model.transformers.loader import TransformersAdapterLoader
from msmodelslim.model.transformers.model_adapter import LLMTransformersModel
from msmodelslim.utils.exception import UnsupportedError, VersionError


def _mock_adapter_init(self, model_type, model_path, trust_remote_code=False):
    self.model_type = model_type
    self.model_path = model_path
    self.trust_remote_code = trust_remote_code


class TestTransformersAdapterLoader(unittest.TestCase):
    def setUp(self):
        self.model_type = "transformers"
        self.model_path = Path("/tmp/transformers-model")
        self.loader = TransformersAdapterLoader()

    def test_adapter_class_path_points_to_llm_transformers_model_when_defined(self):
        self.assertEqual(
            TransformersAdapterLoader.ADAPTER_CLASS_PATH,
            LLM_ADAPTER_CLASS_PATH,
        )
        self.assertEqual(
            TransformersAdapterLoader.ADAPTER_CLASS_PATH,
            "msmodelslim.model.transformers.model_adapter:LLMTransformersModel",
        )

    def _load_with_patches(self, config=None, **load_kwargs):
        if config is None:
            config = {}
        with patch("msmodelslim.model.transformers.detect.json_safe_load", return_value=config):
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin"):
                with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker._check_single"):
                    with patch("msmodelslim.model.plugin_factory.base_loader.get_require_packages", return_value={}):
                        with patch("msmodelslim.model.plugin_factory.base_loader.import_module") as mock_import:
                            mock_import.return_value = SimpleNamespace(LLMTransformersModel=LLMTransformersModel)
                            with patch(
                                "msmodelslim.model.transformers.model_adapter.TransformersModel.__init__",
                                _mock_adapter_init,
                            ):
                                return self.loader.load(
                                    model_type=self.model_type,
                                    model_path=self.model_path,
                                    **load_kwargs,
                                )

    def test_load_returns_llm_transformers_model_when_trust_remote_code_true(self):
        adapter = self._load_with_patches(trust_remote_code=True)

        self.assertIsInstance(adapter, LLMTransformersModel)
        self.assertEqual(adapter.model_type, self.model_type)
        self.assertEqual(adapter.model_path, self.model_path)
        self.assertTrue(adapter.trust_remote_code)

    def test_load_passes_false_when_trust_remote_code_omitted(self):
        adapter = self._load_with_patches()
        self.assertFalse(adapter.trust_remote_code)

    def test_load_raises_unsupported_error_when_vlm_config(self):
        with self.assertRaises(UnsupportedError) as ctx:
            self._load_with_patches(config={"vision_config": {}})

        self.assertIn("VLM", str(ctx.exception))
        self.assertIn("transformers", str(ctx.exception))

    def test_load_raises_unsupported_error_when_dit_config(self):
        with self.assertRaises(UnsupportedError) as ctx:
            self._load_with_patches(config={"_diffusers_version": "0.30.0"})

        self.assertIn("DiT", str(ctx.exception))
        self.assertIn("transformers", str(ctx.exception))

    def test_precheck_sets_plugin_name_when_model_type_valid(self):
        with patch(
            "msmodelslim.model.plugin_factory.base_loader.msmodelslim_config",
            SimpleNamespace(model_adapter_dependencies={}),
        ):
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin") as mock_set:
                with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker._check_single"):
                    self.loader.precheck(
                        model_type=self.model_type,
                        model_path=self.model_path,
                    )

        plugin_name = mock_set.call_args[0][0]
        self.assertEqual(plugin_name, f"msmodelslim.model_adapter.plugins:{self.model_type}")

    def test_precheck_sets_is_match_false_when_dependency_check_fails(self):
        self.loader._require_packages = {"numpy": ">=1.26"}
        with patch(
            "msmodelslim.model.plugin_factory.base_loader.msmodelslim_config",
            SimpleNamespace(model_adapter_dependencies={}),
        ):
            with patch(
                "msmodelslim.model.plugin_factory.base_loader.DependencyChecker._check_single",
                side_effect=VersionError("dependency mismatch"),
            ):
                self.loader.precheck(
                    model_type=self.model_type,
                    model_path=self.model_path,
                )

        self.assertFalse(self.loader._is_match)
