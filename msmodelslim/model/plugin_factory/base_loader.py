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

from importlib import import_module
from pathlib import Path
from typing import Dict

from msmodelslim.model.plugin_factory.loader_interface import AdapterLoaderInterface
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.dependency_check import (
    DependencyChecker,
    get_require_packages,
)
from msmodelslim.utils.exception import UnexpectedError, UnsupportedError, VersionError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.exception_decorator import exception_handler

MODEL_ADAPTER_ENTRY_POINTS = "msmodelslim.model_adapter.plugins"


class BaseModelAdapterLoader(AdapterLoaderInterface):
    ADAPTER_CLASS_PATH = ""
    _logged_plugins = set()

    def __init__(self):
        self._is_match = True
        self._requirements = {}
        self._failed_requirements = {}

    def get_loader_requirements(self) -> Dict[str, str]:
        return get_require_packages(self)

    def check_requirements(
        self,
        plugin_name: str,
        requirements: Dict[str, str],
    ) -> bool:
        DependencyChecker.set_plugin(plugin_name, requirements)
        failed = {}
        for pkg, spec in requirements.items():
            try:
                DependencyChecker._check_single(pkg, spec)
            except VersionError:
                failed[pkg] = spec

        if not failed:
            return True

        self._failed_requirements.update(failed)

        if plugin_name not in BaseModelAdapterLoader._logged_plugins:
            dep_list = " ".join([f"{pkg}{spec}" for pkg, spec in failed.items()])
            get_logger().warning(
                "Environment does not meet dependency requirements: %s. Compatibility issues may occur.",
                dep_list,
            )
            BaseModelAdapterLoader._logged_plugins.add(plugin_name)
        return False

    def precheck(
        self,
        model_type: str,
        model_path: Path,
    ) -> None:
        del model_path
        plugin_name = f"{MODEL_ADAPTER_ENTRY_POINTS}:{model_type}"

        dependency_map = getattr(msmodelslim_config, "model_adapter_dependencies", {})
        if not isinstance(dependency_map, dict):
            dependency_map = {}
        config_requirements = dependency_map.get(plugin_name, {})
        if not isinstance(config_requirements, dict):
            config_requirements = {}

        loader_requirements = self.get_loader_requirements()

        # Priority: loader decorator < platform config.
        merged_requirements = dict(loader_requirements)
        merged_requirements.update(config_requirements)

        self._requirements = merged_requirements
        self._is_match = self.check_requirements(
            plugin_name=plugin_name,
            requirements=merged_requirements,
        )

    def load(
        self,
        model_type: str,
        model_path: Path,
        trust_remote_code: bool = False,
    ):
        if ":" not in self.ADAPTER_CLASS_PATH:
            raise UnsupportedError(
                f"Loader '{type(self).__name__}' must define ADAPTER_CLASS_PATH in 'module.path:ClassName' format."
            )
        module_path, class_name = self.ADAPTER_CLASS_PATH.split(":", 1)

        def get_action_msg(failed_reqs):
            dep_list = " ".join([f"{pkg}{spec}" for pkg, spec in failed_reqs.items()])
            return f"Dependency version mismatch detected. Recommended dependencies: {dep_list}"

        if not self._is_match:
            action_msg = get_action_msg(self._failed_requirements)
            with exception_handler(err_cls=Exception, ms_err_cls=VersionError, action=action_msg):
                adapter_module = import_module(module_path)
                adapter_class = getattr(adapter_module, class_name)
        else:
            adapter_module = import_module(module_path)
            adapter_class = getattr(adapter_module, class_name)

        adapter_reqs = get_require_packages(adapter_class)
        is_adapter_match = self.check_requirements(
            plugin_name=f"{MODEL_ADAPTER_ENTRY_POINTS}:{model_type}",
            requirements=adapter_reqs,
        )

        if not is_adapter_match:
            self._is_match = False
            self._requirements.update(adapter_reqs)

        if not self._is_match:
            action_msg = get_action_msg(self._failed_requirements)
            with exception_handler(err_cls=Exception, ms_err_cls=VersionError, action=action_msg):
                adapter_instance = adapter_class(
                    model_type=model_type,
                    model_path=model_path,
                    trust_remote_code=trust_remote_code,
                )
            # 注入依赖检查失败提示给所有适配器方法
            for attr_name in dir(adapter_instance):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(adapter_instance, attr_name)
                if not callable(attr):
                    continue

                decorated_func = exception_handler(err_cls=Exception, ms_err_cls=VersionError, action=action_msg)(attr)
                try:
                    setattr(adapter_instance, attr_name, decorated_func)
                except AttributeError:
                    continue
            # 注入依赖检查失败提示给UnexpectedError类
            UnexpectedError.inject_tips(action_msg)

            return adapter_instance

        return adapter_class(
            model_type=model_type,
            model_path=model_path,
            trust_remote_code=trust_remote_code,
        )
