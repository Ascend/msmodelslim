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

from typing import Any

from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.plugin import TypedFactory
from msmodelslim.utils.plugin.plugin_utils import load_plugin_component_class

from .base import BaseLosslessMetric, LOSSLESS_METRIC_ENTRY_GROUP, LosslessMetricConfig


class LosslessMetricFactory:
    """Create lossless metric components from parsed plugin configurations."""

    def __init__(self):
        self._factory = TypedFactory[BaseLosslessMetric](
            config_base_class=LosslessMetricConfig,
        )

    @staticmethod
    def _validate_component_class(plugin_type: str, component_class: Any) -> None:
        """Reject an entry point component that is outside the metric contract."""
        if not isinstance(component_class, type) or not issubclass(component_class, BaseLosslessMetric):
            component_name = getattr(component_class, "__name__", type(component_class).__name__)
            raise SchemaValidateError(
                "Lossless metric plugin {!r} component {!r} must inherit from BaseLosslessMetric".format(
                    plugin_type,
                    component_name,
                )
            )

    def create(self, config: LosslessMetricConfig, **kwargs: Any) -> BaseLosslessMetric:
        """Create one metric from an already parsed ``LosslessMetricConfig``.

        ``TypedFactory`` performs the common config-to-component construction;
        this wrapper adds the lossless-specific component type guard before the
        component constructor is called.
        """
        if not isinstance(config, LosslessMetricConfig):
            raise SchemaValidateError(
                "Lossless metric config must be an instance of LosslessMetricConfig, got {!r}".format(
                    type(config).__name__,
                )
            )

        plugin_type = config.name
        component_class = load_plugin_component_class(
            LOSSLESS_METRIC_ENTRY_GROUP,
            plugin_type,
        )
        self._validate_component_class(plugin_type, component_class)

        instance = self._factory.create(config, **kwargs)
        if not isinstance(instance, BaseLosslessMetric):
            # The class guard above should make this unreachable, but retaining
            # an instance guard protects callers if a custom factory is injected.
            raise SchemaValidateError(
                "Lossless metric plugin {!r} did not create a BaseLosslessMetric instance".format(plugin_type)
            )
        return instance


lossless_metric_factory = LosslessMetricFactory()


def create_lossless_metric(config: LosslessMetricConfig, **kwargs: Any) -> BaseLosslessMetric:
    """Convenience entry point for containers creating one lossless metric."""
    return lossless_metric_factory.create(config, **kwargs)
