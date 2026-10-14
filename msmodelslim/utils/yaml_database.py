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

from pathlib import Path

from pydantic import BaseModel

from msmodelslim.utils.exception import SchemaValidateError, SecurityError, UnsupportedError
from msmodelslim.utils.security import (
    yaml_safe_dump,
    yaml_safe_load,
    get_write_directory,
    get_valid_read_path,
)


class YamlDatabase:
    def __init__(self, config_dir: Path, read_only: bool = True):
        """Load all configs from modelType subdirectories"""
        if read_only:
            get_valid_read_path(str(config_dir), is_dir=True)
        else:
            get_write_directory(str(config_dir))
        self.config_dir = config_dir
        self.read_only = read_only

    def __iter__(self):
        return (config_file.stem for config_file in self.config_dir.glob("*.yaml"))

    def __getitem__(self, item):
        """Load value from a YAML file"""
        if not isinstance(item, str):
            raise SchemaValidateError(
                f"yaml database key must be a string, but got {type(item)}",
                action='Please make sure the key is a string',
            )

        try:
            value_file = self.config_dir / f"{item}.yaml"
            return yaml_safe_load(str(value_file))
        except FileNotFoundError as e:
            raise UnsupportedError(
                f"yaml database key {item} not found", action='Please check the yaml database'
            ) from e

    def __setitem__(self, key, value):
        """Save value to a YAML file"""
        if self.read_only:
            raise SecurityError(
                f"yaml database {self.config_dir} is read-only", action='Writing operation is forbidden'
            )

        if not isinstance(key, str):
            raise SchemaValidateError(
                f"yaml database key must be a string, but got {type(key)}",
                action='Please make sure the key is a string',
            )

        if isinstance(value, BaseModel):
            # mode='json'：Decimal 等在 PyYAML safe_dump 中会 RepresenterError
            value = value.model_dump(mode="json")

        value_file = self.config_dir / f"{key}.yaml"
        yaml_safe_dump(value, str(value_file))

    def __contains__(self, item):
        """Check if a YAML file exists"""
        return (self.config_dir / f"{item}.yaml").exists()

    def values(self):
        """Load values from a YAML directory"""
        for key in self:
            yield self[key]
