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

import os

from msmodelslim.utils.security import json_safe_dump


class JsonWriter:
    def __init__(self, save_directory: str, file_name: str):
        self.save_directory = save_directory
        self.file_name = file_name
        self.value_map = {}

    def write(self, prefix: str, desc: object):
        self.value_map[prefix] = desc

    def close(self):
        json_safe_dump(self.value_map, os.path.join(self.save_directory, self.file_name), indent=4)
