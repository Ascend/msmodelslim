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

import json
import os
from configparser import ConfigParser
from typing import Optional

from pydantic import BaseModel

from msmodelslim.utils.exception import ConfigError, EnvVarError
from msmodelslim.utils.security.path import get_valid_read_path

CONFIG_PATH = '../config/config.ini'

SECTION_URL = 'URL'
KEY_REPO_URL = 'repository_url'
KEY_QUESTION_AND_ANSWER_URL = 'question_and_answer_url'

ENV_VAR_LOG_LEVEL = 'MSMODELSLIM_LOG_LEVEL'
ENV_VAR_CUSTOM_PRACTICE_REPO = 'MSMODELSLIM_CUSTOM_PRACTICE_REPO'

VALID_LOG_LEVELS = ['info', 'debug']

SECTION_MODEL_ADAPTER = "ModelAdapter"
SECTION_MODEL_ADAPTER_DEPENDENCIES = "ModelAdapterDependencies"

MODEL_ADAPTER_ENTRY_POINTS = "msmodelslim.model_adapter.plugins"


class URLs(BaseModel):
    repository: str
    question_and_answer: str


class EnvVars(BaseModel):
    log_level: str
    custom_practice_repo: Optional[str] = None


class ModelSlimConfig(BaseModel):
    urls: URLs
    env_vars: EnvVars
    model_adapter_dependencies: dict[str, dict[str, str]]


def load_model_adapter_dependencies(
    file_config: ConfigParser,
) -> dict[str, dict[str, str]]:
    deps: dict[str, dict[str, str]] = {}

    for model_type, models in file_config.items(SECTION_MODEL_ADAPTER):
        if file_config.has_option(SECTION_MODEL_ADAPTER_DEPENDENCIES, model_type):
            dep_str = file_config.get(SECTION_MODEL_ADAPTER_DEPENDENCIES, model_type)
            try:
                model_dependency = json.loads(dep_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for model {model_type}: {dep_str}") from e

            model_list = [m.strip() for m in models.split(",")]

            for model in model_list:
                deps[f"{MODEL_ADAPTER_ENTRY_POINTS}:{model}"] = model_dependency

    return deps


def init_config():
    file_config = ConfigParser()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(cur_dir, CONFIG_PATH))
    config_path = get_valid_read_path(config_path)
    file_config.read(config_path)

    if not file_config.has_section(SECTION_URL):
        raise ConfigError(
            f"Invalid config section: {SECTION_URL}", action=f'Please check the config file {config_path}'
        )

    if not file_config.has_option(SECTION_URL, KEY_REPO_URL):
        raise ConfigError(f"Invalid config key: {KEY_REPO_URL}", action=f'Please check the config file {config_path}')

    if not file_config.has_option(SECTION_URL, KEY_QUESTION_AND_ANSWER_URL):
        raise ConfigError(
            f"Invalid config key: {KEY_QUESTION_AND_ANSWER_URL}", action=f'Please check the config file {config_path}'
        )

    if not file_config.has_section(SECTION_MODEL_ADAPTER):
        raise ConfigError(
            f"Invalid config section: {SECTION_MODEL_ADAPTER}", action=f"Please check the config file {config_path}"
        )

    if not file_config.has_section(SECTION_MODEL_ADAPTER_DEPENDENCIES):
        raise ConfigError(
            f"Invalid config section: {SECTION_MODEL_ADAPTER_DEPENDENCIES}",
            action=f"Please check the config file {config_path}",
        )

    logger_level = os.getenv(ENV_VAR_LOG_LEVEL, "INFO").lower()
    if logger_level not in VALID_LOG_LEVELS:
        raise EnvVarError(
            f"Invalid log level: {logger_level}, must be in {VALID_LOG_LEVELS}",
            action=f'Please check the environment variable {ENV_VAR_LOG_LEVEL}',
        )

    custom_practice_repo = os.getenv(ENV_VAR_CUSTOM_PRACTICE_REPO)

    modelslim_config = ModelSlimConfig(
        urls=URLs(
            repository=file_config.get(SECTION_URL, KEY_REPO_URL),
            question_and_answer=file_config.get(SECTION_URL, KEY_QUESTION_AND_ANSWER_URL),
        ),
        env_vars=EnvVars(log_level=logger_level, custom_practice_repo=custom_practice_repo),
        model_adapter_dependencies=load_model_adapter_dependencies(file_config),
    )
    return modelslim_config


msmodelslim_config = init_config()
