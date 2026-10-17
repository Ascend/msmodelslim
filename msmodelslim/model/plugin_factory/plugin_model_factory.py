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
# pylint: disable=logging-fstring-interpolation,no-member

import sys
from importlib.metadata import entry_points
from pathlib import Path

from msmodelslim.model.interface import IModel, IModelFactory
from msmodelslim.model.plugin_factory.loader_interface import AdapterLoaderInterface
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.dependency_check import DependencyChecker, get_require_packages
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger

MODEL_ADAPTER_ENTRY_POINTS = "msmodelslim.model_adapter.plugins"

DEFAULT = "default"


class PluginModelFactory(IModelFactory):
    _model_map = None

    @classmethod
    def _get_model_map(cls) -> dict:
        if cls._model_map is None:
            if sys.version_info >= (3, 10):
                eps = entry_points().select(group=MODEL_ADAPTER_ENTRY_POINTS)
            else:
                eps = entry_points().get(MODEL_ADAPTER_ENTRY_POINTS, [])
            cls._model_map = {ep.name: ep for ep in eps}
            get_logger().info(f"Found {len(cls._model_map)} model adapters: {list(cls._model_map.keys())}")
        return cls._model_map

    @staticmethod
    def _plugin_name(model_type: str) -> str:
        return f"{MODEL_ADAPTER_ENTRY_POINTS}:{model_type}"

    def get_adapter_config_class(self, model_type: str):
        """pydantic model the practice ``spec.model_adapter`` is validated against, or None."""
        model_map = self._get_model_map()
        if model_type not in model_map:
            if DEFAULT in model_map:
                model_type = DEFAULT
            else:
                return None
        loaded = model_map[model_type].load()
        if isinstance(loaded, type) and issubclass(loaded, AdapterLoaderInterface):
            loaded = loaded()
        if isinstance(loaded, AdapterLoaderInterface):
            return loaded.get_adapter_config_class()
        return None

    def create(
        self,
        model_type: str,
        model_path: Path,
        trust_remote_code: bool = False,
    ) -> IModel:
        model_map = self._get_model_map()
        model_name = model_type

        if model_type not in model_map:
            if DEFAULT in model_map:
                get_logger().warning(f"Model adapter '{model_type}' not found, trying default adapter...")
                model_type = DEFAULT
            else:
                raise UnsupportedError(
                    f"No adapter found for '{model_type}' and no default adapter registered. "
                    f"Registered adapters: {list(model_map.keys())}"
                )
        adapter_entry_point = model_map[model_type]

        loaded_obj = adapter_entry_point.load()
        if isinstance(loaded_obj, type) and issubclass(loaded_obj, AdapterLoaderInterface):
            loaded_obj = loaded_obj()

        if isinstance(loaded_obj, AdapterLoaderInterface):
            loader = loaded_obj
            loader.precheck(model_type=model_type, model_path=model_path)
            return loader.load(
                model_type=model_name,
                model_path=model_path,
                trust_remote_code=trust_remote_code,
            )
        elif isinstance(loaded_obj, type):
            adapter_class = loaded_obj
            plugin_name = self._plugin_name(model_type)
            if plugin_name in msmodelslim_config.model_adapter_dependencies:
                DependencyChecker.set_plugin(plugin_name, msmodelslim_config.model_adapter_dependencies[plugin_name])
            DependencyChecker.set_plugin(plugin_name, get_require_packages(adapter_class))
            DependencyChecker.check_plugin(plugin_name)
        else:
            raise UnsupportedError(
                f"Entry point '{model_type}' must resolve to AdapterLoaderInterface "
                f"or adapter class, "
                f"but got {type(loaded_obj)}."
            )

        adapter_instance = adapter_class(
            model_type=model_name,
            model_path=model_path,
            trust_remote_code=trust_remote_code,
        )
        return adapter_instance
