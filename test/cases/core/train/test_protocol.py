#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=abstract-class-instantiated

import pytest
from torch import nn

from msmodelslim.core.train.protocol import ITrainHook, QATLoadInterface, TrainHookContext
from msmodelslim.utils.exception import UnsupportedError


def test_TrainHookContext_defaults_extra_when_constructed():
    ctx = TrainHookContext()
    assert ctx.extra == {}


def test_TrainHookContext_defaults_model_when_constructed():
    ctx = TrainHookContext()
    assert ctx.model is None


def test_ITrainHook_is_structural_when_method_missing():
    class _Bad:
        pass

    assert not isinstance(_Bad(), ITrainHook)


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


def test_ITrainHook_pre_train_raises_when_super_not_implemented():
    class _SuperHook(ITrainHook):
        def pre_train(self, ctx: TrainHookContext) -> None:  # pylint: disable=useless-parent-delegation
            return super().pre_train(ctx)

    with pytest.raises(UnsupportedError, match="does not support pre_train"):
        _SuperHook().pre_train(TrainHookContext())


def test_QATLoadInterface_load_tokenizer_for_qat_raises_when_super_not_implemented():
    class _SuperAdapter(QATLoadInterface):
        def load_tokenizer_for_qat(self, model_max_length):  # pylint: disable=useless-parent-delegation
            return super().load_tokenizer_for_qat(model_max_length)

        def load_model_for_qat(self):
            return nn.Linear(1, 1)

    with pytest.raises(UnsupportedError, match="does not support load_tokenizer_for_qat"):
        _SuperAdapter().load_tokenizer_for_qat(16)


def test_QATLoadInterface_load_model_for_qat_raises_when_super_not_implemented():
    class _SuperAdapter(QATLoadInterface):
        def load_tokenizer_for_qat(self, model_max_length):
            return "tok"

        def load_model_for_qat(self):  # pylint: disable=useless-parent-delegation
            return super().load_model_for_qat()

    with pytest.raises(UnsupportedError, match="does not support load_model_for_qat"):
        _SuperAdapter().load_model_for_qat()
