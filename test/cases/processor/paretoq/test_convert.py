#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from unittest.mock import MagicMock

import pytest
from torch import nn

from msmodelslim.core.convert.types import IRKind
from msmodelslim.processor.paretoq.convert import (
    ParetoQToW4A4DynamicProcessor,
    ParetoQToW4A8DynamicProcessor,
    get_paretoq_convert_processors,
)


@pytest.mark.parametrize(
    ("processor_cls", "dst_ir"),
    [
        (ParetoQToW4A8DynamicProcessor, IRKind.W4A8_DYNAMIC),
        (ParetoQToW4A4DynamicProcessor, IRKind.W4A4_DYNAMIC),
    ],
)
def test_ParetoQDeployProcessor_exposes_dst_ir_when_constructed(processor_cls, dst_ir):
    processor = processor_cls()
    assert processor.dst_ir == dst_ir


@pytest.mark.parametrize(
    "processor_cls",
    [ParetoQToW4A8DynamicProcessor, ParetoQToW4A4DynamicProcessor],
)
def test_ParetoQDeployProcessor_exposes_src_ir_when_constructed(processor_cls):
    assert processor_cls().src_ir == IRKind.PARETOQ_FLOAT


@pytest.mark.parametrize(
    "processor_cls",
    [ParetoQToW4A8DynamicProcessor, ParetoQToW4A4DynamicProcessor],
)
def test_ParetoQDeployProcessor_delegates_transform_when_called(processor_cls, monkeypatch):
    sentinel = nn.Linear(2, 2)
    called = {}

    def _fake_transform(module, context):
        called["module"] = module
        called["context"] = context
        return sentinel

    monkeypatch.setattr(
        "msmodelslim.processor.paretoq.impl.ParetoQImpl.transform_for_convert",
        _fake_transform,
    )
    module = nn.Linear(2, 2)
    context = MagicMock()
    out = processor_cls().transform(module, context)
    assert out is sentinel
    assert called["module"] is module
    assert called["context"] is context


def test_get_paretoq_convert_processors_returns_both_deploy_edges_when_called():
    names = {type(p).__name__ for p in get_paretoq_convert_processors()}
    assert names == {
        ParetoQToW4A8DynamicProcessor.__name__,
        ParetoQToW4A4DynamicProcessor.__name__,
    }
