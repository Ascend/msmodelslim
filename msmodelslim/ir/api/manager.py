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
# pylint: disable=logging-fstring-interpolation

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Optional, Dict

from packaging import version

from msmodelslim.utils.exception import VersionError
from msmodelslim.utils.logging import get_logger


@dataclass
class FuncMark:
    min_version: str
    max_version: str


class APIManager:
    MAX_VERSION: str = "9999.9999.9999"
    API_MODULE: Optional[ModuleType] = None
    INITIALIZED: bool = False
    MARKED_FUNC: Dict[str, FuncMark] = {}

    @classmethod
    def init_module(cls):
        if cls.INITIALIZED:
            return

        if cls.API_MODULE is not None:
            return

        try:
            cls.API_MODULE = importlib.import_module(".api_main", __name__)
        except ImportError as e:
            get_logger().warning(
                f"[Core] Load api module failed, please check if the api_impl package is installed, "
                f"exception info : {e}"
            )

        cls.INITIALIZED = True

    @classmethod
    def get_module(cls) -> ModuleType:
        APIManager.init_module()
        return cls.API_MODULE

    @classmethod
    def get_version(cls) -> str:
        APIManager.init_module()
        return cls.API_MODULE.__version__ if cls.API_MODULE is not None else "0.0.0"

    @classmethod
    def check_version(cls):
        kia_version = version.parse(cls.get_version())
        for func_name, mark in cls.MARKED_FUNC.items():
            if kia_version < version.parse(mark.min_version) or kia_version > version.parse(mark.max_version):
                get_logger().warning(
                    f"Function {func_name} requires KIA version {mark.min_version}-{mark.max_version}, "
                    f"but the current KIA version is {cls.get_version()}"
                )

    @classmethod
    def mark_require_version(cls, min_version: str, max_version: str = MAX_VERSION):
        """
        检查KIA版本是否满足要求。

        通过装饰器，可以在函数定义、函数运行两个阶段检查版本是否符合要求。

        在定义函数阶段，如果版本不满足要求，会抛出ImportError异常。

        在函数运行阶段，如果版本不满足要求，会抛出RuntimeError异常。

        Args:
            min_version: 最小版本
            max_version: 最大版本

        Raises:
            ImportError: 函数定义阶段，版本不满足时抛出异常
            RuntimeError: 函数运行阶段，版本不满足时抛出异常
        """

        def decorator(func):
            cls.MARKED_FUNC[func.__name__] = FuncMark(min_version, max_version)

            def wrapped_func(*args, **kwargs):
                kia_version = version.parse(cls.get_version())
                func_name = func.__name__

                # Runtime Check

                if kia_version < version.parse(min_version) or kia_version > version.parse(max_version):
                    raise VersionError(
                        f"Trying to invoke function {func_name}, "
                        f"which requires KIA version {min_version}-{max_version}, "
                        f"but the current KIA version is {cls.get_version()}",
                        action=f"Please upgrade CANN to {min_version}-{max_version}",
                    )

                return func(*args, **kwargs)

            return wrapped_func

        return decorator
