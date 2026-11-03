#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=abstract-class-instantiated

import pytest
from torch import nn

from msmodelslim.core.train.protocol import ITrainHook, QATLoadInterface, TrainHookContext


def test_TrainHookContext_defaults_extra_when_constructed():
    ctx = TrainHookContext()
    assert ctx.extra == {}


def test_TrainHookContext_defaults_model_when_constructed():
    ctx = TrainHookContext()
    assert ctx.model is None


def test_ITrainHook_raises_when_not_implemented():
    class _Bad(ITrainHook):
        pass

    with pytest.raises(TypeError):
        _Bad()


def test_ITrainHook_isinstance_true_when_pre_train_defined():
    class _Hook(ITrainHook):
        def pre_train(self, ctx: TrainHookContext) -> None:
            return None

    assert isinstance(_Hook(), ITrainHook)


def test_ITrainHook_pre_train_updates_extra_when_called():
    class _Hook(ITrainHook):
        def pre_train(self, ctx: TrainHookContext) -> None:
            ctx.extra["ok"] = True

    ctx = TrainHookContext(model=nn.Linear(2, 2))
    _Hook().pre_train(ctx)
    assert ctx.extra["ok"] is True


def test_QATLoadInterface_raises_when_not_implemented():
    class _Bad(QATLoadInterface):
        pass

    with pytest.raises(TypeError):
        _Bad()


def test_QATLoadInterface_load_tokenizer_for_qat_returns_value_when_implemented():
    class _Ok(QATLoadInterface):
        def load_tokenizer_for_qat(self, *args, **kwargs):
            return "tok"

        def load_model_for_qat(self, *args, **kwargs):
            return nn.Linear(1, 1)

    assert _Ok().load_tokenizer_for_qat() == "tok"


def test_QATLoadInterface_load_model_for_qat_returns_module_when_implemented():
    class _Ok(QATLoadInterface):
        def load_tokenizer_for_qat(self, *args, **kwargs):
            return "tok"

        def load_model_for_qat(self, *args, **kwargs):
            return nn.Linear(1, 1)

    assert isinstance(_Ok().load_model_for_qat(), nn.Linear)
