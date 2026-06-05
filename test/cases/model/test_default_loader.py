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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader as DefaultModelAdapterLoader
from msmodelslim.utils.exception import UnexpectedError, UnsupportedError, VersionError


def test_precheck_should_merge_metadata_and_config_with_config_priority():
    loader = DefaultModelAdapterLoader()
    loader._require_packages = {
        "torch": ">=2.0",
        "transformers": ">=4.56",
    }
    plugin_name = "msmodelslim.model_adapter.plugins:GLM-4.5-w8a8"

    config_requirements = {
        "transformers": "==4.57.3",
        "accelerate": ">=0.30",
    }

    with patch(
        "msmodelslim.model.plugin_factory.base_loader.msmodelslim_config",
        SimpleNamespace(model_adapter_dependencies={plugin_name: config_requirements}),
    ):
        with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin") as mock_set_plugin:
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                loader.precheck(
                    model_type="GLM-4.5-w8a8",
                    model_path=Path("/tmp/path"),
                )

    merged_requirements = mock_set_plugin.call_args[0][1]
    assert merged_requirements["torch"] == ">=2.0"
    assert merged_requirements["transformers"] == "==4.57.3"
    assert merged_requirements["accelerate"] == ">=0.30"


def test_precheck_should_fallback_when_metadata_missing():
    loader = DefaultModelAdapterLoader()

    with patch(
        "msmodelslim.model.plugin_factory.base_loader.msmodelslim_config",
        SimpleNamespace(model_adapter_dependencies={}),
    ):
        with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin") as mock_set_plugin:
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                loader.precheck(
                    model_type="fallback-model",
                    model_path=Path("/tmp/path"),
                )

    plugin_name, requirements = mock_set_plugin.call_args[0]
    assert plugin_name == "msmodelslim.model_adapter.plugins:fallback-model"
    assert requirements == {}


def test_load_should_keep_post_import_decorator_check():
    loader = DefaultModelAdapterLoader()
    loader.ADAPTER_CLASS_PATH = "fake.module:DummyAdapter"

    class DummyAdapter:
        def __init__(self, model_type, model_path, trust_remote_code):
            self.model_type = model_type
            self.model_path = model_path
            self.trust_remote_code = trust_remote_code

        _require_packages = {"einops": ">=0.8.0"}

    with patch("msmodelslim.model.plugin_factory.base_loader.import_module") as mock_import_module:
        mock_import_module.return_value = SimpleNamespace(DummyAdapter=DummyAdapter)
        with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin") as mock_set_plugin:
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                adapter_instance = loader.load(
                    model_type="test-model",
                    model_path=Path("/tmp/path"),
                    trust_remote_code=True,
                )

    assert isinstance(adapter_instance, DummyAdapter)
    assert adapter_instance.model_type == "test-model"
    assert adapter_instance.model_path == Path("/tmp/path")
    assert adapter_instance.trust_remote_code is True
    plugin_name, requirements = mock_set_plugin.call_args[0]
    assert plugin_name == "msmodelslim.model_adapter.plugins:test-model"
    assert requirements == {"einops": ">=0.8.0"}


def test_precheck_should_support_loader_class_decorator_requirements():
    class DecoratedLoader(DefaultModelAdapterLoader):
        _require_packages = {"numpy": ">=1.26"}

    loader = DecoratedLoader()

    with patch(
        "msmodelslim.model.plugin_factory.base_loader.msmodelslim_config",
        SimpleNamespace(model_adapter_dependencies={}),
    ):
        with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.set_plugin") as mock_set_plugin:
            with patch("msmodelslim.model.plugin_factory.base_loader.DependencyChecker.check_plugin"):
                loader.precheck(
                    model_type="external-decorated-loader",
                    model_path=Path("/tmp/path"),
                )

    plugin_name, requirements = mock_set_plugin.call_args[0]
    assert plugin_name == "msmodelslim.model_adapter.plugins:external-decorated-loader"
    assert requirements == {"numpy": ">=1.26"}


def test_precheck_should_not_raise_error_when_dependency_check_fails():
    loader = DefaultModelAdapterLoader()
    loader._require_packages = {"numpy": ">=1.26"}

    with patch(
        "msmodelslim.model.plugin_factory.base_loader.msmodelslim_config",
        SimpleNamespace(model_adapter_dependencies={}),
    ):
        with patch(
            "msmodelslim.model.plugin_factory.base_loader.DependencyChecker._check_single",
            side_effect=VersionError("dependency mismatch"),
        ):
            loader.precheck(
                model_type="test-model",
                model_path=Path("/tmp/path"),
            )
    assert loader._is_match is False


def test_load_should_raise_when_adapter_path_not_configured():
    loader = DefaultModelAdapterLoader()
    loader.ADAPTER_CLASS_PATH = ""

    with pytest.raises(UnsupportedError, match="must define ADAPTER_CLASS_PATH"):
        loader.load(
            model_type="missing-model",
            model_path=Path("/tmp/path"),
        )


def test_load_should_decorate_when_adapter_dependency_check_fails():
    loader = DefaultModelAdapterLoader()
    loader.ADAPTER_CLASS_PATH = "fake.module:DummyAdapter"

    class DummyAdapter:
        def __init__(self, model_type, model_path, trust_remote_code):
            self.model_type = model_type
            self.model_path = model_path
            self.trust_remote_code = trust_remote_code

        def dummy_method(self):
            raise RuntimeError("mock inner error")

        _require_packages = {"einops": ">=0.8.0"}

    with patch("msmodelslim.model.plugin_factory.base_loader.import_module") as mock_import_module:
        mock_import_module.return_value = SimpleNamespace(DummyAdapter=DummyAdapter)
        with patch(
            "msmodelslim.model.plugin_factory.base_loader.DependencyChecker._check_single",
            side_effect=VersionError("mock error"),
        ):
            adapter_instance = loader.load(
                model_type="test-model",
                model_path=Path("/tmp/path"),
                trust_remote_code=True,
            )

    # Check that dummy_method raises VersionError because the second validation failed
    with pytest.raises(VersionError) as exc_info:
        adapter_instance.dummy_method()
    assert "Recommended dependencies: einops>=0.8.0" in str(exc_info.value)
    assert UnexpectedError.tips
    UnexpectedError.clear_tips()
