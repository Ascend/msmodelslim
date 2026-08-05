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

# Strategy → per-layer qconfig resolution.
#
# Supports layer-wise loading: prefer matching at wrap time via
# ``StrategyResolver``; ``resolve_layer_qconfigs`` remains for eager/full-model scans.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.processor.trainable_linear_quant.config.processor_config import QuantStrategyConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.logging import get_logger

SUPPORTED_LAYER_TYPES = (torch.nn.Linear,)


def _strategy_match(
    layer_name: str,
    include_set: ConfigSet,
    exclude_set: ConfigSet,
) -> bool:
    return layer_name in include_set and layer_name not in exclude_set


def _warn_strategy_conflict(layer_name: str, qconfigs: Dict[str, LinearQConfig]) -> None:
    get_logger().warning(
        "Layer '%s' configuration already exists, skipping to preserve original configuration: %s",
        layer_name,
        qconfigs[layer_name],
    )


def _warning_unmatched_pattern(label: str, config_set: ConfigSet) -> None:
    unmatched_keys = [key for key in config_set.unmatched_keys() if key != "*"]
    if unmatched_keys:
        get_logger().warning(
            "These %s patterns are not matched any module, please ensure this is as expected: %s",
            label,
            unmatched_keys,
        )


def _supported_layer_names(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if isinstance(module, SUPPORTED_LAYER_TYPES)]


class StrategyResolver:
    """Match ``QuantStrategyConfig`` list against layer names (first hit wins).

    Reuses config objects; only compiles ``include`` / ``exclude`` into ``ConfigSet``
    for matching and unmatched-pattern tracking across layer-wise wraps.
    """

    def __init__(self, strategies: Sequence[QuantStrategyConfig]) -> None:
        if not strategies:
            raise SchemaValidateError(
                "strategies must be a non-empty list",
                action="Please provide at least one quantization strategy",
            )
        self._strategies: List[QuantStrategyConfig] = list(strategies)
        self._include_sets: List[ConfigSet] = [ConfigSet(list(s.include)) for s in self._strategies]
        self._exclude_sets: List[ConfigSet] = [ConfigSet(list(s.exclude)) for s in self._strategies]

    def match(self, layer_name: str) -> Optional[LinearQConfig]:
        for strategy, include_set, exclude_set in zip(self._strategies, self._include_sets, self._exclude_sets):
            if _strategy_match(layer_name, include_set, exclude_set):
                return strategy.qconfig
        return None

    def warn_unmatched(self) -> None:
        for i, (include_set, exclude_set) in enumerate(zip(self._include_sets, self._exclude_sets)):
            _warning_unmatched_pattern(f"strategies[{i}].include", include_set)
            _warning_unmatched_pattern(f"strategies[{i}].exclude", exclude_set)

    def _iter_entries(self):
        return zip(self._strategies, self._include_sets, self._exclude_sets)


def resolve_layer_qconfigs(
    model: nn.Module,
    strategies: List[QuantStrategyConfig],
) -> Tuple[Dict[str, LinearQConfig], int]:
    """Eager map ``strategies`` over currently-resident linears (full-model / tests)."""
    resolver = StrategyResolver(strategies)
    qconfigs: Dict[str, LinearQConfig] = {}
    supported_layers = _supported_layer_names(model)
    get_logger().debug(
        "Resolving %d strategies against %d supported layers",
        len(strategies),
        len(supported_layers),
    )
    get_logger().debug("Supported layers: %s", supported_layers)

    # Strategy-outer order preserves first-match and conflict warnings.
    for i, (strategy, include_set, exclude_set) in enumerate(resolver._iter_entries()):
        for layer_name in supported_layers:
            if not _strategy_match(layer_name, include_set, exclude_set):
                continue
            if layer_name in qconfigs:
                _warn_strategy_conflict(layer_name, qconfigs)
                continue
            qconfigs[layer_name] = strategy.qconfig
            get_logger().debug("Applied strategy %d to layer '%s'", i, layer_name)

    resolver.warn_unmatched()

    if not qconfigs:
        raise SchemaValidateError(
            "No supported linear layer matched any quantization strategy; "
            "at least one layer must receive a trainable qconfig",
            action=("Please check strategies include/exclude patterns against the model's linear layers"),
        )

    return qconfigs, len(supported_layers)


__all__ = [
    "SUPPORTED_LAYER_TYPES",
    "StrategyResolver",
    "resolve_layer_qconfigs",
]
