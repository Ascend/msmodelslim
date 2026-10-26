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

import importlib.util
import json
import sys
from configparser import ConfigParser
from pathlib import Path

import pytest

from msmodelslim.utils.exception import ConfigError, EnvVarError


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "[URL]\n"
        "repository_url=https://repo.example\n"
        "question_and_answer_url=https://qa.example\n"
        "[ModelAdapter]\n"
        "v0=model_a, model_b\n"
        "[ModelAdapterDependencies]\n"
        'v0={"torch": ">=2.0"}\n',
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def _loaded_config_fixture(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    security_path = sys.modules["msmodelslim.utils.security.path"]
    monkeypatch.setattr(security_path, "get_valid_read_path", lambda _: str(config_path), raising=False)
    monkeypatch.setenv("MSMODELSLIM_LOG_LEVEL", "debug")
    monkeypatch.setenv("MSMODELSLIM_CUSTOM_PRACTICE_REPO", "custom/repo")

    module_name = "b1_config_module_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).parents[3] / "msmodelslim" / "utils" / "config.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, config_path


def test_init_config_loads_urls_env_and_model_adapter_dependencies(_loaded_config_fixture):
    module, _ = _loaded_config_fixture

    assert module.msmodelslim_config.urls.repository == "https://repo.example"
    assert module.msmodelslim_config.urls.question_and_answer == "https://qa.example"
    assert module.msmodelslim_config.env_vars.log_level == "debug"
    assert module.msmodelslim_config.env_vars.custom_practice_repo == "custom/repo"
    assert module.msmodelslim_config.model_adapter_dependencies == {
        "msmodelslim.model_adapter.plugins:model_a": {"torch": ">=2.0"},
        "msmodelslim.model_adapter.plugins:model_b": {"torch": ">=2.0"},
    }


def test_load_model_adapter_dependencies_expands_models_for_each_dependency(_loaded_config_fixture):
    module, _ = _loaded_config_fixture
    parser = ConfigParser()
    parser.read_dict(
        {
            "ModelAdapter": {"v0": "alpha, beta"},
            "ModelAdapterDependencies": {"v0": json.dumps({"transformers": ">=4.0"})},
        }
    )

    assert module.load_model_adapter_dependencies(parser) == {
        "msmodelslim.model_adapter.plugins:alpha": {"transformers": ">=4.0"},
        "msmodelslim.model_adapter.plugins:beta": {"transformers": ">=4.0"},
    }


def test_invalid_log_level_raises_env_var_error(_loaded_config_fixture, monkeypatch):
    module, config_path = _loaded_config_fixture
    monkeypatch.setenv("MSMODELSLIM_LOG_LEVEL", "trace")
    monkeypatch.setattr(module, "get_valid_read_path", lambda _: str(config_path))

    with pytest.raises(EnvVarError, match="Invalid log level: trace"):
        module.init_config()


def test_load_model_adapter_dependencies_returns_empty_when_no_adapters_configured(_loaded_config_fixture):
    module, _ = _loaded_config_fixture
    parser = ConfigParser()
    parser.read_dict({"ModelAdapter": {}, "ModelAdapterDependencies": {}})

    assert module.load_model_adapter_dependencies(parser) == {}


def test_load_model_adapter_dependencies_rejects_invalid_json(_loaded_config_fixture):
    module, _ = _loaded_config_fixture
    parser = ConfigParser()
    parser.read_dict(
        {
            "ModelAdapter": {"v0": "model_a"},
            "ModelAdapterDependencies": {"v0": "{not-json"},
        }
    )

    with pytest.raises(ValueError, match="Invalid JSON for model v0"):
        module.load_model_adapter_dependencies(parser)


def test_init_config_raises_config_error_when_url_key_missing(_loaded_config_fixture):
    module, config_path = _loaded_config_fixture
    config_path.write_text(
        "[URL]\nquestion_and_answer_url=https://qa.example\n[ModelAdapter]\n[ModelAdapterDependencies]\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Invalid config key: repository_url"):
        module.init_config()
