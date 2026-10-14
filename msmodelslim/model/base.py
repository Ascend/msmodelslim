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

from .interface import IModel


class BaseModelAdapter(IModel):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._model_type = model_type
        self._model_path = model_path
        self._trust_remote_code = trust_remote_code

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def trust_remote_code(self) -> bool:
        return self._trust_remote_code

    @model_type.setter
    def model_type(self, model_type: str):
        self._model_type = model_type

    @model_path.setter
    def model_path(self, model_path: Path):
        self._model_path = model_path

    @trust_remote_code.setter
    def trust_remote_code(self, trust_remote_code: bool):
        self._trust_remote_code = trust_remote_code
