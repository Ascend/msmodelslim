#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Copyright (c) 2026 Huawei Technologies Co.,Ltd.
msModelSlim is licensed under Mulan PSL v2.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import DataUnit, ProcessRequest
from msmodelslim.core.runner.generated_runner import (
    GeneratedProcessUnit,
    GeneratedRunner,
    init_pipeline_generators,
)
from msmodelslim.utils.exception import (
    InvalidDatasetError,
    UnexpectedError,
)


class _AddBlock(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, hidden):
        return hidden + self.delta


class _ToyModel(nn.Module):
    def __init__(self, drift=False):
        super().__init__()
        self.first = _AddBlock(1.0)
        self.second = _AddBlock(2.0)
        self.drift = drift
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden):
        hidden = self.first(hidden)
        if not self.drift or hidden.item() < 3:
            hidden = self.second(hidden)
        return hidden


class _LegacyAdapter:
    """Hand-written generator adapter (Feature 1: no PausableForward)."""

    def __init__(self, drift=False):
        self.drift = drift

    def generate_model_forward(self, model, data):
        hidden = data
        for name, module in (("first", model.first), ("second", model.second)):
            if self.drift and name == "second" and hidden.item() >= 3:
                break
            out = yield ProcessRequest(
                name=name,
                module=module,
                args=(hidden,),
                kwargs={},
            )
            if out is None:
                out = module(hidden)
            hidden = out

    def generate_model_visit(self, model):
        for name, module in model.named_modules():
            if isinstance(module, _AddBlock):
                yield ProcessRequest(name=name, module=module, args=(), kwargs={})

    def enable_kv_cache(self, _model, _enabled):
        return None


class _Processor:
    def __init__(self, *, data_free=False, output=True, log=None):
        self.data_free = data_free
        self.output = output
        self.log = log if log is not None else []

    def __repr__(self):
        return "ToyProcessor"

    def pre_run(self):
        self.log.append("pre_run")

    def post_run(self):
        self.log.append("post_run")

    def preprocess(self, request):
        self.log.append(("preprocess", request.name))

    def process(self, request):
        self.log.append(("process", request.name))
        if self.output:
            request.outputs = [request.module(*args, **kwargs) for args, kwargs in request.datas]

    def postprocess(self, request):
        self.log.append(("postprocess", request.name))

    def is_data_free(self):
        return self.data_free

    def need_kv_cache(self):
        return False


def _unit(model, processor):
    return GeneratedProcessUnit(model, processor, DataUnit(None, None))


class TestGeneratedRunnerPropagate:
    def test_generated_schedule_sends_outputs_when_propagate_true(self):
        model = _ToyModel()
        processor = _Processor()
        runner = GeneratedRunner(_LegacyAdapter(), propagate=True)

        with patch(
            "msmodelslim.core.runner.generated_runner.get_input_datas",
            return_value=[torch.tensor(0.0), torch.tensor(1.0)],
        ):
            runner.generated_schedule([_unit(model, processor)], model, calib_data=[0, 1])

        assert processor.log[-1] == "post_run"

    def test_generated_schedule_runs_when_propagate_false(self):
        model = _ToyModel()
        processor = _Processor()
        runner = GeneratedRunner(_LegacyAdapter(), propagate=False)

        with patch(
            "msmodelslim.core.runner.generated_runner.get_input_datas",
            return_value=[torch.tensor(0.0), torch.tensor(1.0)],
        ):
            runner.generated_schedule([_unit(model, processor)], model, calib_data=[0, 1])

        assert processor.log[-1] == "post_run"


class TestGeneratedProcessUnit:
    def test_run_step_sets_recorder_output_when_processor_produces_output(self):
        processor = MagicMock()
        recorder = DataUnit(None, None)
        unit = GeneratedProcessUnit(nn.Linear(2, 2), processor, recorder)
        request = MagicMock(outputs=None)
        processor.process.side_effect = lambda current: setattr(current, "outputs", ["output"])

        unit.run_step(request)

        assert recorder.output == ["output"]
        processor.preprocess.assert_called_once_with(request)
        processor.postprocess.assert_called_once_with(request)

    def test_run_step_preserves_recorder_output_when_processor_produces_no_output(self):
        processor = MagicMock()
        recorder = DataUnit(None, ["existing"])
        unit = GeneratedProcessUnit(nn.Linear(2, 2), processor, recorder)

        unit.run_step(MagicMock(outputs=None))

        assert recorder.output == ["existing"]

    def test_run_step_propagates_processor_error_when_process_fails(self):
        processor = MagicMock()
        processor.process.side_effect = RuntimeError("processor failed")
        unit = GeneratedProcessUnit(nn.Linear(2, 2), processor, DataUnit(None, None))

        with pytest.raises(RuntimeError, match="processor failed"):
            unit.run_step(MagicMock(outputs=None))


class TestGeneratedRunner:
    def test_add_processor_preserves_append_and_prepend_behavior(self):
        runner = GeneratedRunner(Mock())
        first, second = Mock(), Mock()

        runner.add_processor(first)
        runner.add_processor(second, append=False)

        assert runner.process_config_list == [second, first]

    def test_build_process_unit_returns_one_unit_per_config_when_processors_are_valid(self):
        adapter = Mock()
        runner = GeneratedRunner(adapter)
        model = nn.Linear(2, 2)
        recorder = DataUnit(None, None)
        processors = [Mock(), Mock()]
        for processor in processors:
            processor.need_kv_cache.return_value = False

        with patch(
            "msmodelslim.core.runner.generated_runner.AutoSessionProcessor.from_config",
            side_effect=processors,
        ):
            units = runner.build_process_unit([Mock(), Mock()], model, adapter, recorder)

        assert [unit.processor for unit in units] == processors
        adapter.enable_kv_cache.assert_called_once_with(model, False)

    def test_generated_schedule_runs_multisample_pipeline_when_adapter_is_compatible(self):
        model = _ToyModel()
        processor = _Processor()
        runner = GeneratedRunner(_LegacyAdapter())

        with patch(
            "msmodelslim.core.runner.generated_runner.get_input_datas",
            return_value=[torch.tensor(0.0), torch.tensor(1.0)],
        ):
            runner.generated_schedule(
                [_unit(model, processor)],
                model,
                calib_data=[0, 1],
            )

        assert processor.log == [
            "pre_run",
            ("preprocess", "first"),
            ("process", "first"),
            ("postprocess", "first"),
            ("preprocess", "second"),
            ("process", "second"),
            ("postprocess", "second"),
            "post_run",
        ]

    def test_generated_schedule_initializes_generators_after_pre_run_when_data_free(self):
        order = []
        adapter = Mock()
        processor = Mock()
        processor.is_data_free.return_value = True
        processor.pre_run.side_effect = lambda: order.append("pre_run")
        processor.post_run.side_effect = lambda: order.append("post_run")
        unit = _unit(Mock(), processor)

        def empty_generator():
            return
            yield

        with patch(
            "msmodelslim.core.runner.generated_runner.init_pipeline_generators",
            side_effect=lambda *_args, **_kwargs: (order.append("init_generators") or [empty_generator()]),
        ):
            GeneratedRunner(adapter).generated_schedule(
                [unit],
                Mock(),
            )

        assert order == ["pre_run", "init_generators", "post_run"]

    def test_generated_schedule_preserves_empty_process_name_when_target_is_model_root(self):
        adapter = Mock()
        model = nn.Linear(2, 2)

        def root_generator():
            yield ProcessRequest(name="", module=model, args=(), kwargs={})

        adapter.generate_model_visit.return_value = root_generator()
        processor = _Processor(data_free=True, output=False)

        GeneratedRunner(adapter).generated_schedule(
            [_unit(model, processor)],
            model,
        )

        assert ("process", "") in processor.log

    def test_generated_schedule_identifies_processor_when_sample_ends_early(self):
        model = _ToyModel(drift=True)
        processor = _Processor()
        runner = GeneratedRunner(_LegacyAdapter(drift=True))

        with (
            patch(
                "msmodelslim.core.runner.generated_runner.get_input_datas",
                return_value=[torch.tensor(0.0), torch.tensor(5.0)],
            ),
            pytest.raises(
                UnexpectedError,
                match=r"shared processors=\['ToyProcessor'\]",
            ),
        ):
            runner.generated_schedule(
                [_unit(model, processor)],
                model,
                calib_data=[0, 1],
            )

        assert "post_run" not in processor.log


class TestGeneratedRunnerFunctions:
    @staticmethod
    def _generator(*names):
        for name in names:
            yield ProcessRequest(name=name, module=Mock(), args=(), kwargs={})

    def test_init_pipeline_generators_raises_when_forward_has_no_calibration_data(self):
        with pytest.raises(InvalidDatasetError, match="Calib data"):
            init_pipeline_generators(Mock(), nn.Linear(2, 2), None, True)

    def test_init_pipeline_generators_builds_one_forward_generator_per_sample(self):
        adapter = Mock()
        adapter.generate_model_forward.side_effect = lambda _model, data: iter((data,))
        model = nn.Linear(2, 2)

        with patch(
            "msmodelslim.core.runner.generated_runner.get_input_datas",
            return_value=[torch.tensor(1.0), torch.tensor(2.0)],
        ):
            generators = init_pipeline_generators(adapter, model, [1, 2], True)

        assert len(generators) == 2
        assert adapter.generate_model_forward.call_count == 2

    def test_init_pipeline_generators_builds_single_visit_generator_when_data_free(self):
        adapter = Mock()
        visit = self._generator("a")
        adapter.generate_model_visit.return_value = visit

        generators = init_pipeline_generators(adapter, nn.Linear(2, 2), None, False)

        assert generators == [visit]
