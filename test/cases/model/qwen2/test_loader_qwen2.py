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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from msmodelslim.model.qwen2.loader import Qwen2AdapterLoader
from msmodelslim.model.qwen2.model_adapter import Qwen2ModelAdapter
from msmodelslim.utils.exception import VersionError


def _mock_adapter_init(self, model_type, model_path, trust_remote_code=False):
    self.model_type = model_type
    self.model_path = model_path
    self.trust_remote_code = trust_remote_code


class TestQwen2AdapterLoaderAdapterClassPath(unittest.TestCase):
    """测试Qwen2AdapterLoader的ADAPTER_CLASS_PATH配置"""

    def test_adapter_class_path_when_defined_then_point_to_qwen2_model_adapter(self):
        """正常：ADAPTER_CLASS_PATH应指向Qwen2ModelAdapter"""
        self.assertEqual(
            Qwen2AdapterLoader.ADAPTER_CLASS_PATH, "msmodelslim.model.qwen2.model_adapter:Qwen2ModelAdapter"
        )


class TestQwen2AdapterLoaderLoad(unittest.TestCase):
    """测试Qwen2AdapterLoader的load方法"""

    def setUp(self):
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('/tmp/qwen2-model')
        self.loader = Qwen2AdapterLoader()

    def test_load_with_valid_params_when_called_then_return_qwen2_model_adapter(self):
        """正常：load应实例化并返回Qwen2ModelAdapter"""
        with patch('msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin'):
            with patch('msmodelslim.model.plugin_factory.base_loader.DependencyChecker._check_single'):
                with patch('msmodelslim.model.plugin_factory.base_loader.get_require_packages', return_value={}):
                    with patch('msmodelslim.model.plugin_factory.base_loader.import_module') as mock_import:
                        mock_import.return_value = SimpleNamespace(Qwen2ModelAdapter=Qwen2ModelAdapter)

                        with patch(
                            'msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', _mock_adapter_init
                        ):
                            adapter = self.loader.load(
                                model_type=self.model_type,
                                model_path=self.model_path,
                                trust_remote_code=True,
                            )

        self.assertIsInstance(adapter, Qwen2ModelAdapter)
        self.assertEqual(adapter.model_type, self.model_type)
        self.assertEqual(adapter.model_path, self.model_path)
        self.assertTrue(adapter.trust_remote_code)

    def test_load_with_trust_remote_code_false_when_called_then_pass_false(self):
        """边界：trust_remote_code默认False时应传递False"""
        with patch('msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin'):
            with patch('msmodelslim.model.plugin_factory.base_loader.DependencyChecker._check_single'):
                with patch('msmodelslim.model.plugin_factory.base_loader.get_require_packages', return_value={}):
                    with patch('msmodelslim.model.plugin_factory.base_loader.import_module') as mock_import:
                        mock_import.return_value = SimpleNamespace(Qwen2ModelAdapter=Qwen2ModelAdapter)

                        with patch(
                            'msmodelslim.model.qwen2.model_adapter.DefaultModelAdapter.__init__', _mock_adapter_init
                        ):
                            adapter = self.loader.load(
                                model_type=self.model_type,
                                model_path=self.model_path,
                            )

        self.assertFalse(adapter.trust_remote_code)


class TestQwen2AdapterLoaderPrecheck(unittest.TestCase):
    """测试Qwen2AdapterLoader的precheck方法"""

    def setUp(self):
        self.loader = Qwen2AdapterLoader()
        self.model_type = 'Qwen2-7B-Instruct'
        self.model_path = Path('/tmp/qwen2-model')

    def test_precheck_with_valid_model_type_when_called_then_check_dependencies(self):
        """正常：precheck应触发依赖检查"""
        with patch(
            'msmodelslim.model.plugin_factory.base_loader.msmodelslim_config',
            SimpleNamespace(model_adapter_dependencies={}),
        ):
            with patch('msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin') as mock_set:
                with patch('msmodelslim.model.plugin_factory.base_loader.DependencyChecker._check_single'):
                    self.loader.precheck(
                        model_type=self.model_type,
                        model_path=self.model_path,
                    )

        plugin_name = mock_set.call_args[0][0]
        self.assertEqual(plugin_name, f"msmodelslim.model_adapter.plugins:{self.model_type}")

    def test_precheck_when_dependency_check_fails_then_set_is_match_false(self):
        """异常：依赖检查失败时应设置 _is_match 为 False"""
        self.loader._require_packages = {"numpy": ">=1.26"}
        with patch(
            'msmodelslim.model.plugin_factory.base_loader.msmodelslim_config',
            SimpleNamespace(model_adapter_dependencies={}),
        ):
            with patch(
                'msmodelslim.model.plugin_factory.base_loader.DependencyChecker._check_single',
                side_effect=VersionError('dependency mismatch'),
            ):
                self.loader.precheck(
                    model_type=self.model_type,
                    model_path=self.model_path,
                )

        self.assertFalse(self.loader._is_match)
