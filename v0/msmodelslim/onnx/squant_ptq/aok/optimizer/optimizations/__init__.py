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
from inspect import isclass
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations.abstract_optimization import AbstractOptimization


def import_all_optimizations(file: str, name: str) -> None:
    from pkgutil import iter_modules
    from pathlib import Path
    from importlib import import_module

    _package_dir = str(Path(file).resolve().parent)
    for (_, module_name, _) in iter_modules([_package_dir]):
        module = import_module(f'{name}.{module_name}')
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isclass(attribute) and issubclass(attribute, AbstractOptimization):
                globals()[attribute_name] = attribute


import_all_optimizations(__file__, __name__)
