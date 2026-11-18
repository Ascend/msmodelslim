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

import importlib
import inspect
import logging


class FunctionReplace:
    def __init__(self, ori_function_define, new_function):
        self.new_function = new_function
        self.ori_function = None
        self.location = None
        self.attr_name = None
        if ori_function_define is None:
            return
        elif inspect.isfunction(ori_function_define):
            self.ori_function = ori_function_define
            self.location, self.attr_name = self.get_location(self.ori_function)
        elif callable(ori_function_define):
            self.ori_function = ori_function_define.__call__
            self.location, self.attr_name = self.get_location(self.ori_function)
        elif isinstance(ori_function_define, tuple) and len(ori_function_define) == 2:
            self.location, self.attr_name = ori_function_define
            self.ori_function = getattr(self.location, self.attr_name)

        if not all((self.ori_function, self.location, self.attr_name, self.new_function)):
            warn_msg = "{} replace failed.".format(ori_function_define)
            logging.warning(warn_msg)
            raise ValueError(warn_msg)

    def __enter__(self):
        self.replace()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.recover()

    @staticmethod
    def nothing():
        return FunctionReplace(None, None)

    @staticmethod
    def get_location(function_ins):
        if not hasattr(function_ins, "__module__"):
            warning_msg = "function {} do not has attr __module__.".format(str(function_ins))
            logging.warning(warning_msg)
            raise ValueError(warning_msg)
        module = importlib.import_module(function_ins.__module__)
        qualified_name = function_ins.__qualname__.split('.')
        classes = qualified_name[:-1]
        attr_name = qualified_name[-1]
        location = module
        for class_name in classes:
            location = getattr(location, class_name, None)
            if location is None:
                break

        if location is None:
            warning_msg = "{} do not exists".format('.'.join(classes))
            logging.warning(warning_msg)
            raise ValueError(warning_msg)
        return location, attr_name

    def replace(self):
        if all((self.ori_function, self.location, self.attr_name, self.new_function)):
            setattr(self.location, self.attr_name, self._get_method(self.new_function))

    def recover(self):
        if all((self.ori_function, self.location, self.attr_name, self.new_function)):
            setattr(self.location, self.attr_name, self._get_method(self.ori_function))

    def _get_method(self, func):
        if inspect.isclass(self.location):
            func_cls_name = inspect.getattr_static(self.location, self.attr_name).__class__.__name__
            if func_cls_name in ('staticmethod', 'classmethod'):
                return staticmethod(func)
        return func
