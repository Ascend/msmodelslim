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

import json
import os
import shutil
import stat
import sys
import yaml

from unittest.mock import MagicMock


def _mock_json_safe_dump(obj, path, indent=None, extensions="json", check_user_stat=True):
    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as json_file:
        json.dump(obj, json_file, indent=indent)


def _mock_get_valid_write_path(path: str, *args, **kwarg) -> str:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    # 若路径无扩展名或以分隔符结尾，视为目录并创建。Windows 上若 path 已存在且为文件，
    # makedirs(path, exist_ok=True) 会触发 FileExistsError(WinError 183)，故仅在实际需要时创建。
    base = os.path.basename(path)
    if not base or not os.path.splitext(base)[1] or path.endswith(os.sep):
        if not os.path.exists(path) or os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
    return path


def _mock_yaml_safe_load(path, *args, **kwargs):
    """Mock yaml_safe_load function that reads yaml file from path and converts to dict"""
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def _mock_yaml_safe_dump(obj, path, extensions=("yml", "yaml"), check_user_stat=True):
    """Mock yaml_safe_dump that persists YAML for infra unit tests."""
    write_path = _mock_get_valid_write_path(path, extensions, check_user_stat)
    default_mode = stat.S_IWUSR | stat.S_IRUSR  # 600
    with os.fdopen(os.open(write_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=default_mode), "w") as yaml_file:
        yaml.safe_dump(obj, yaml_file, sort_keys=False)


def _mock_json_safe_load(path, *args, **kwargs):
    """Mock json_safe_load function that reads json file from path and converts to dict"""
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _mock_safe_copy_file(src_path, dest_path, **kwargs):
    shutil.copy2(src_path, dest_path, follow_symlinks=False)


class _MockSafeWriteUmask:
    def __init__(self, umask=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def mock_kia_library():
    sys.modules['msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils'] = MagicMock()
    sys.modules['msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs'] = MagicMock()
    sys.modules['msmodelslim.pytorch.llm_sparsequant.atomic_power_outlier'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.atomic_power_outlier'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.calibration'] = MagicMock()
    sys.modules['msmodelslim.pytorch.lowbit.quant_modules'] = MagicMock()


def mock_security_library():
    sys.modules['msmodelslim.utils.security.path'] = MagicMock()
    sys.modules['msmodelslim.utils.security.path'].json_safe_dump = _mock_json_safe_dump
    sys.modules['msmodelslim.utils.security.path'].json_safe_load = _mock_json_safe_load
    sys.modules['msmodelslim.utils.security.path'].yaml_safe_load = _mock_yaml_safe_load
    sys.modules['msmodelslim.utils.security.path'].yaml_safe_dump = _mock_yaml_safe_dump
    sys.modules['msmodelslim.utils.security.path'].get_valid_write_path = _mock_get_valid_write_path
    sys.modules['msmodelslim.utils.security.path'].get_valid_path = _mock_get_valid_write_path
    sys.modules['msmodelslim.utils.security.path'].get_valid_read_path = _mock_get_valid_write_path
    sys.modules['msmodelslim.utils.security.path'].get_write_directory = _mock_get_valid_write_path
    sys.modules['msmodelslim.utils.security.path'].safe_copy_file = _mock_safe_copy_file
    sys.modules['msmodelslim.utils.security.path'].SafeWriteUmask = _MockSafeWriteUmask

    sys.modules['ascend_utils.common.security.path'] = MagicMock()
    sys.modules['ascend_utils.common.security.path'].json_safe_dump = _mock_json_safe_dump
    sys.modules['ascend_utils.common.security.path'].json_safe_load = _mock_json_safe_load
    sys.modules['ascend_utils.common.security.path'].yaml_safe_load = _mock_yaml_safe_load
    sys.modules['ascend_utils.common.security.path'].get_valid_write_path = _mock_get_valid_write_path
    sys.modules['ascend_utils.common.security.path'].get_valid_path = _mock_get_valid_write_path
    sys.modules['ascend_utils.common.security.path'].get_write_directory = _mock_get_valid_write_path


def mock_init_config():
    """Mock init_config function and related config modules"""
    # Mock the config module
    config_mock = MagicMock()

    # Create a mock config object that mimics the structure of ModelSlimConfig
    mock_config = MagicMock()
    mock_config.urls.repository = "mocked_url"
    mock_config.env_vars.log_level = "info"
    mock_config.env_vars.custom_practice_repo = None

    # Mock the init_config function to return the mock config
    config_mock.init_config.return_value = mock_config
    config_mock.msmodelslim_config = mock_config

    # Mock the config classes
    config_mock.ModelSlimConfig = MagicMock()
    config_mock.URLs = MagicMock()
    config_mock.EnvVars = MagicMock()

    # Register the mock module
    sys.modules['msmodelslim.utils.config'] = config_mock
