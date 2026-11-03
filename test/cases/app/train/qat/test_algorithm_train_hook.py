#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from typing import List, Sequence, Tuple
from unittest.mock import MagicMock

import pytest
from torch import nn

from msmodelslim.app.train.qat.algorithm_train_hook import AlgorithmTrainHook
from msmodelslim.app.train.qat.qat_linear_algorithm import QATLinearAlgorithm
from msmodelslim.core.train.protocol import ITrainHook, TrainHookContext
from msmodelslim.core.train.task import CoreTrainTask, TrainAlgorithmConfig
from msmodelslim.utils.exception import SchemaValidateError


class _FakeQATLinearAlgorithm(QATLinearAlgorithm):
    def __init__(self):
        self.model = None
        self.shells: List[nn.Module] = [nn.Linear(2, 2)]
        self.calls: List[str] = []

    def replace_linears(self, model: nn.Module) -> List[nn.Module]:
        self.calls.append("replace_linears")
        return self.shells

    def init_weight_clip_val(self, shells: Sequence[nn.Module]) -> None:
        self.calls.append("init_weight_clip_val")

    def qparam_names(self) -> Tuple[str, ...]:
        return ("weight_clip_val",)

    @classmethod
    def get_convert_processors(cls):
        return []


def test_AlgorithmTrainHook_is_itrain_hook_when_constructed():
    hook = AlgorithmTrainHook(TrainAlgorithmConfig(type="paretoq"), factory=MagicMock())
    assert isinstance(hook, ITrainHook)


def test_AlgorithmTrainHook_raises_when_model_missing():
    hook = AlgorithmTrainHook(TrainAlgorithmConfig(type="paretoq"), factory=MagicMock())
    with pytest.raises(SchemaValidateError, match="ctx.model"):
        hook.pre_train(TrainHookContext(model=None))


def test_AlgorithmTrainHook_raises_when_plugin_not_qat_linear_algorithm():
    factory = MagicMock()
    factory.create.return_value = object()
    hook = AlgorithmTrainHook(TrainAlgorithmConfig(type="paretoq"), factory=factory)
    with pytest.raises(SchemaValidateError, match="QATLinearAlgorithm"):
        hook.pre_train(TrainHookContext(model=nn.Linear(4, 4), extra={}))


def test_AlgorithmTrainHook_calls_replace_before_init_when_pre_train():
    impl = _FakeQATLinearAlgorithm()
    factory = MagicMock()
    factory.create.return_value = impl
    hook = AlgorithmTrainHook(TrainAlgorithmConfig(type="paretoq"), factory=factory)

    hook.pre_train(TrainHookContext(model=nn.Linear(4, 4), extra={}))

    assert impl.calls == ["replace_linears", "init_weight_clip_val"]


def test_AlgorithmTrainHook_writes_qparam_names_to_core_task_when_pre_train():
    impl = _FakeQATLinearAlgorithm()
    factory = MagicMock()
    factory.create.return_value = impl
    core_task = CoreTrainTask(config=MagicMock())
    hook = AlgorithmTrainHook(TrainAlgorithmConfig(type="paretoq"), factory=factory)

    hook.pre_train(TrainHookContext(model=nn.Linear(4, 4), core_task=core_task, extra={}))

    assert core_task.qparam_names == ["weight_clip_val"]


def test_AlgorithmTrainHook_writes_qparam_names_to_ctx_extra_when_pre_train():
    impl = _FakeQATLinearAlgorithm()
    factory = MagicMock()
    factory.create.return_value = impl
    hook = AlgorithmTrainHook(TrainAlgorithmConfig(type="paretoq"), factory=factory)
    ctx = TrainHookContext(model=nn.Linear(4, 4), extra={})

    hook.pre_train(ctx)

    assert ctx.extra["qparam_names"] == ["weight_clip_val"]
