#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=abstract-class-instantiated

from typing import List, Optional, Sequence, Tuple

import pytest
from torch import nn

from msmodelslim.app.train.qat.qat_linear_algorithm import QATLinearAlgorithm
from msmodelslim.core.convert.protocol import IIRTransformProcessor


class _ConcreteQATLinearAlgorithm(QATLinearAlgorithm):
    def __init__(self) -> None:
        self.last_shells: Optional[List[nn.Module]] = None

    def replace_linears(self, model: nn.Module) -> List[nn.Module]:
        return [model]

    def init_weight_clip_val(self, shells: Sequence[nn.Module]) -> None:
        self.last_shells = list(shells)

    def qparam_names(self) -> Tuple[str, ...]:
        return ("weight_clip_val",)

    @classmethod
    def get_convert_processors(cls) -> List[IIRTransformProcessor]:
        return []


def test_QATLinearAlgorithm_raises_when_abstract_methods_missing():
    class _Incomplete(QATLinearAlgorithm):
        pass

    with pytest.raises(TypeError):
        _Incomplete()


def test_QATLinearAlgorithm_requires_get_convert_processors_when_other_methods_defined():
    class _MissingConvert(QATLinearAlgorithm):
        def replace_linears(self, model: nn.Module) -> List[nn.Module]:
            return []

        def init_weight_clip_val(self, shells: Sequence[nn.Module]) -> None:
            return None

        def qparam_names(self) -> Tuple[str, ...]:
            return ()

    with pytest.raises(TypeError):
        _MissingConvert()


def test_QATLinearAlgorithm_replace_linears_returns_shells_when_implemented():
    model = nn.Linear(2, 2)
    impl = _ConcreteQATLinearAlgorithm()
    assert impl.replace_linears(model) == [model]


def test_QATLinearAlgorithm_init_weight_clip_val_receives_shells_when_called():
    shell = nn.Linear(1, 1)
    impl = _ConcreteQATLinearAlgorithm()
    impl.init_weight_clip_val([shell])
    assert impl.last_shells == [shell]


def test_QATLinearAlgorithm_qparam_names_returns_tuple_when_implemented():
    assert _ConcreteQATLinearAlgorithm().qparam_names() == ("weight_clip_val",)


def test_QATLinearAlgorithm_get_convert_processors_returns_list_when_implemented():
    processors = _ConcreteQATLinearAlgorithm.get_convert_processors()
    assert isinstance(processors, list)
    assert len(processors) == 0
