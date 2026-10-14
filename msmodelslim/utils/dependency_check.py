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

from __future__ import annotations
import os  # noqa: F401
import importlib
from typing import Dict

from importlib.metadata import PackageNotFoundError, version as metadata_version
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from packaging.version import Version, InvalidVersion

from msmodelslim.utils.exception import VersionError


class DependencyChecker:
    """
    一个全局依赖检查器：

    用法：
        1. set_plugin(plugin, reqs)
       - 为某个插件登记依赖
       - 多次调用会自动 merge，不会覆盖其他插件的数据

        DependencyChecker.set_plugin(
            "msmodelslim.model.qwen3_next.model_adapter",
            {"torch": ">=2.0", "transformers": ">=4.35,<4.40"}
        )

        2. require_packages(requirements)
           - 类装饰器，用于给类声明依赖
           - 装饰器不会立即写入全局缓存
           - 需要调用 get_require_packages() 才会加入全局缓存

            @DependencyChecker.require_packages({"torch": ">=2.1"})
            class MyAdapter: ...

        3. get_require_packages(target_class)
           - 读取类上通过 require_packages 注册的依赖
           - 自动把依赖写入 DependencyChecker 全局缓存（按模块名分类）

            DependencyChecker.get_require_packages(MyAdapter)

        4. check_plugin(plugin)
           - 根据插件名执行依赖检查（版本解析 + import 检查）

            DependencyChecker.check_plugin("msmodelslim.model.qwen3_next.model_adapter")
    """

    # 全局缓存：plugin -> pkg_name -> version_specifier
    _requirements: dict[str, dict[str, str]] = {}

    @staticmethod
    def _check_single(pkg_name: str, constraint: str):
        """检查单个包"""

        # 解析版本约束
        try:
            spec = SpecifierSet(constraint)
        except InvalidSpecifier as e:
            raise VersionError(f"Invalid version specifier {constraint!r} for package '{pkg_name}'") from e

        installed_version = None

        # 优先尝试 distribution 名称
        try:
            installed_version = metadata_version(pkg_name)
        except PackageNotFoundError:
            pass

        # 回退到 import module
        if installed_version is None:
            try:
                mod = importlib.import_module(pkg_name)
                installed_version = getattr(mod, "__version__", None)
            except ImportError:
                raise VersionError(f"Required package '{pkg_name}{constraint}' is not installed.") from None

        if installed_version is None:
            raise VersionError(f"Cannot determine version for '{pkg_name}'.")

        # 校验版本合法性
        try:
            parsed_version = Version(installed_version)
        except InvalidVersion as e:
            raise VersionError(f"Installed version for '{pkg_name}' is invalid: {installed_version!r}") from e

        # 最终版本约束匹配
        if parsed_version not in spec:
            raise VersionError(
                f"Package '{pkg_name}' version {installed_version} does not satisfy requirement '{constraint}'."
            )

    @classmethod
    def set_plugin(cls, plugin: str, reqs: dict[str, str]):
        if not isinstance(reqs, dict):
            raise TypeError("requirements must be dict {package: version_spec}")

        if plugin not in cls._requirements:
            cls._requirements[plugin] = {}

        for pkg, spec in reqs.items():
            cls._requirements[plugin][pkg] = spec

    @classmethod
    def get_plugin_requirements(cls, plugin: str) -> dict[str, str]:
        if not isinstance(plugin, str):
            raise TypeError("requirement must be str {package: version_spec}")
        if plugin not in cls._requirements:
            cls._requirements[plugin] = {}
        return cls._requirements[plugin]

    @classmethod
    def check_plugin(cls, plugin: str):
        if plugin in cls._requirements:
            for pkg, spec in cls._requirements[plugin].items():
                cls._check_single(pkg, spec)


def require_packages(requirements: Dict[str, str]):
    if not isinstance(requirements, dict):
        raise TypeError("config must be a dict")

    def decorator(target_class):
        if not hasattr(target_class, "_require_packages"):
            setattr(target_class, "_require_packages", {})

        target_class._require_packages.update(requirements)
        return target_class

    return decorator


def get_require_packages(target_class):
    require_pkgs = getattr(target_class, "_require_packages", {})
    return require_pkgs
