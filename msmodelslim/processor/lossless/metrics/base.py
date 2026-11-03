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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
from pydantic import ConfigDict, field_validator

from msmodelslim.utils.plugin import TypedConfig


LOSSLESS_METRIC_ENTRY_GROUP = "msmodelslim.lossless_metric.plugins"


@dataclass(frozen=True)
class MetricContext:
    """Stable runtime context supplied to every metric."""

    processor: str
    decoder: str
    structure: str
    structure_type: str
    side: str
    invocation: int
    path: str


@dataclass(frozen=True)
class MetricViolation:
    """One concrete threshold condition that a metric did not satisfy."""

    name: str
    value: Any
    threshold: Any
    comparison: Optional[str] = None


@dataclass(frozen=True)
class MetricResult:
    """One metric decision and its serializable observations."""

    name: str
    passed: bool
    value: Any
    threshold: Any
    observations: Dict[str, Any] = field(default_factory=dict)
    violations: Tuple[MetricViolation, ...] = field(default_factory=tuple)


@TypedConfig.plugin_entry(entry_point_group=LOSSLESS_METRIC_ENTRY_GROUP)
class LosslessMetricConfig(TypedConfig):
    """Base configuration for lossless metric plugins.

    Concrete plugins override ``name`` with a ``Literal`` default and own all
    additional validation fields.  The base class deliberately contains no
    threshold or options fields: a metric's configuration belongs to that
    metric's concrete Pydantic model.
    """

    model_config = ConfigDict(extra="forbid")

    name: TypedConfig.TypeField

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Lossless metric name must not be empty")
        if len(value) > 200:
            raise ValueError("Lossless metric name must not exceed 200 characters")
        return value


class BaseLosslessMetric(ABC):
    """Abstract runtime contract implemented by every lossless metric."""

    def __init__(self, config: LosslessMetricConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    def evaluate(
        self,
        reference: torch.Tensor,
        processed: torch.Tensor,
        context: MetricContext,
    ) -> MetricResult:
        """Compare one tensor pair without mutating either tensor."""
