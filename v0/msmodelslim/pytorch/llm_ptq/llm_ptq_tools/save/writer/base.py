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

from abc import ABC, abstractmethod


class BaseWriter(ABC):
    def __init__(self, logger):
        self.logger = logger
        self.is_closed = False

    def __setitem__(self, key, value):
        self.write(key, value)

    @abstractmethod
    def _write(self, key, value) -> None:
        ...

    @abstractmethod
    def _close(self) -> None:
        ...

    def write(self, key, value):
        if self.is_closed:
            return

        self._write(key, value)

    def close(self):
        if self.is_closed:
            return

        self.is_closed = True
        self._close()