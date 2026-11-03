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

# pylint: disable=too-many-lines

import gc
import json
import random
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest, ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.context import ContextManager, LocalDictContext
from msmodelslim.core.runner.generated_runner import KEY_DATA_LOADER, GeneratedRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.processor.container.group import GroupProcessorConfig
from msmodelslim.processor.lossless.processor import (
    LosslessCheckProcessor,
    LosslessCheckProcessorConfig,
    LosslessMetricConfig,
    _StructureCapture,
)
from msmodelslim.processor.lossless.protocol import (
    LosslessCheckableProcessor,
    LosslessCheckContext,
    LosslessCheckPlan,
    LosslessCheckTarget,
)
from msmodelslim.utils.cache.memory import clear_cache
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError


class _RepeatedLinearModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        self.input_offset = 0.0
        self.forward_count = 0
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))

    def forward(self, hidden_states, *, scale=1.0, mask=None, execute=True):
        self.forward_count += 1
        if not execute:
            return {"hidden_states": hidden_states, "mask": mask}
        first = self.linear(hidden_states + self.input_offset)
        second = self.linear(hidden_states + 1.0 + self.input_offset)
        return {"hidden_states": (first, second * scale), "mask": mask}


class _StochasticLinearModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(3, 3, bias=False)
        self.forward_count = 0
        self.forward_grad_enabled = []
        self.forward_training = []
        self.raise_on_call = None
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))

    def forward(self, hidden_states, *, scale=1.0, mask=None, execute=True):
        del mask
        self.forward_count += 1
        self.forward_grad_enabled.append(torch.is_grad_enabled())
        self.forward_training.append((self.training, self.dropout.training))
        if not execute:
            return hidden_states
        randomized = self.dropout(hidden_states) + torch.rand_like(hidden_states) + random.random()
        if self.forward_count == self.raise_on_call:
            raise RuntimeError("stochastic forward failed")
        return self.linear(randomized) * scale


class _FakeConfig(AutoProcessorConfig):
    type: Literal["lossless_test_fake"] = "lossless_test_fake"
    label: str = "fake"
    target: bool = True
    fault: Literal[
        "none",
        "input",
        "output",
        "nonfinite",
        "infinite",
        "align",
        "prepare",
        "replace",
    ] = "none"
    fault_target: str = "linear"
    needs_kv_cache: bool = False
    pending_target: str = ""


@dataclass(frozen=True)
class _FakeContext(LosslessCheckContext):
    pass


def _fake_targets(processor, request):
    target_modules = getattr(request.module, "lossless_targets", None)
    if target_modules is None:
        target_modules = {"linear": request.module.linear}
    return (
        tuple(
            LosslessCheckTarget(
                name=f"{request.name}.{name}",
                path=f"{request.name}.{name}",
                decoder=request.name,
                structure_type=module.__class__.__name__,
            )
            for name, module in target_modules.items()
        )
        if processor.config.target
        else ()
    )


def _apply_fake_fault(processor, request):
    if processor.config.fault == "input":
        request.module.input_offset = 0.5
    elif processor.config.fault == "output":
        with torch.no_grad():
            getattr(request.module, processor.config.fault_target).weight.add_(0.5)
    elif processor.config.fault == "nonfinite":
        with torch.no_grad():
            getattr(request.module, processor.config.fault_target).weight[0, 0] = torch.nan
    elif processor.config.fault == "infinite":
        with torch.no_grad():
            getattr(request.module, processor.config.fault_target).weight[0, 0] = torch.inf
    elif processor.config.fault == "replace":
        original = getattr(request.module, processor.config.fault_target)
        replacement = nn.Linear(
            original.in_features,
            original.out_features,
            bias=original.bias is not None,
        )
        replacement.load_state_dict(original.state_dict())
        setattr(request.module, processor.config.fault_target, replacement)


@QABCRegistry.register(dispatch_key=_FakeConfig, abc_class=AutoSessionProcessor)
class _FakeProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter=None):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.events = getattr(model, "lossless_events", [])
        model.lossless_events = self.events
        self._lossless_check_active = False

    def prepare_lossless_check(self):
        if self.config.fault == "prepare":
            raise RuntimeError("prepare failed")
        self._lossless_check_active = True
        expected = (self.config.pending_target,) if self.config.pending_target else ()
        return LosslessCheckPlan(expected_targets=expected)

    def create_lossless_check_context(self, request, reference_datas):
        return _FakeContext(
            targets=_fake_targets(self, request),
            reference_datas=reference_datas,
            covered_targets=(),
            unresolved_targets=(),
        )

    def align_lossless_check_io(self, context, target, reference, processed):
        del context, target
        self.events.append(f"{self.config.label}.align_structure")
        if self.config.fault == "align":
            raise RuntimeError("alignment failed")
        return reference, processed

    def process(self, request):
        self.events.append(f"{self.config.label}.process")
        if not self._lossless_check_active:
            super().process(request)
            return
        _apply_fake_fault(self, request)

    def need_kv_cache(self):
        return self.config.needs_kv_cache


class _NoProtocolConfig(AutoProcessorConfig):
    type: Literal["lossless_test_no_protocol"] = "lossless_test_no_protocol"


@QABCRegistry.register(dispatch_key=_NoProtocolConfig, abc_class=AutoSessionProcessor)
class _NoProtocolProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter=None):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.events = []

    def pre_run(self):
        self.events.append("pre_run")

    def process(self, request):
        self.events.append("process")
        request.outputs = [request.module(*args, **kwargs) for args, kwargs in request.datas]

    def post_run(self):
        self.events.append("post_run")


class _PartialProtocolConfig(AutoProcessorConfig):
    type: Literal["lossless_test_partial_protocol"] = "lossless_test_partial_protocol"


@QABCRegistry.register(dispatch_key=_PartialProtocolConfig, abc_class=AutoSessionProcessor)
class _PartialProtocolProcessor(_NoProtocolProcessor):
    def prepare_lossless_check(self):
        raise AssertionError("An incomplete protocol must not be invoked")


class _Adapter:
    def generate_model_forward(self, _model, _inputs):
        yield


class _AdapterWithoutForward:
    pass


class _AdapterWithDefaultForward(_AdapterWithoutForward):
    generate_model_forward = PipelineInterface.generate_model_forward


class _MultimodalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        self.input_offset = 0.0
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))

    def forward(self, hidden_states, *, image_features, token_ids, metadata):
        audio_features = metadata["modalities"]["audio_features"]
        combined = hidden_states + image_features + audio_features + self.input_offset
        return {
            "hidden_states": self.linear(combined),
            "token_ids": token_ids,
            "metadata": metadata,
        }


class _MultimodalAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self.handled_samples = []

    def handle_dataset(self, dataset, device=DeviceType.NPU):
        self.handled_samples.append((dataset, device))
        return [
            {
                "hidden_states": torch.tensor([sample["hidden_states"]], dtype=torch.float32),
                "image_features": torch.tensor([sample["image"]], dtype=torch.float32),
                "token_ids": torch.tensor([sample["tokens"]], dtype=torch.long),
                "metadata": {
                    "source": sample["source"],
                    "modalities": {
                        "audio_features": torch.tensor([sample["audio"]], dtype=torch.float32),
                    },
                },
            }
            for sample in dataset
        ]

    def generate_model_forward(self, model, inputs):
        yield ProcessRequest(
            "multimodal.block",
            model,
            (inputs["hidden_states"],),
            {
                "image_features": inputs["image_features"],
                "token_ids": inputs["token_ids"],
                "metadata": inputs["metadata"],
            },
        )

    def enable_kv_cache(self, _model, _need_kv_cache):
        pass


class _RecordingGeneratedRunner(GeneratedRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = []

    def build_process_unit(self, *args, **kwargs):
        self.units = super().build_process_unit(*args, **kwargs)
        return self.units


def _request(module, *, execute=True, name="block"):
    return BatchProcessRequest(
        name=name,
        module=module,
        datas=[
            (
                (torch.tensor([[1.0, 2.0, 3.0]]),),
                {"scale": 2.0, "mask": torch.tensor([[1, 0, 1]]), "execute": execute},
            )
        ],
        outputs=None,
    )


def _processor(
    model,
    *configs,
    adapter=None,
    include=None,
    exclude=None,
    metrics=None,
):
    if adapter is None:
        adapter = _Adapter()
    config_kwargs = {"configs": list(configs)}
    config_kwargs["include"] = include if include is not None else ["*"]
    config_kwargs["exclude"] = exclude if exclude is not None else []
    config_kwargs["metrics"] = metrics if metrics is not None else ["numerical_equivalence"]
    return LosslessCheckProcessor(
        model,
        LosslessCheckProcessorConfig(**config_kwargs),
        adapter,
    )


def _assert_summary_has_no_runtime_objects(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_summary_has_no_runtime_objects(key)
            _assert_summary_has_no_runtime_objects(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_summary_has_no_runtime_objects(item)
        return
    assert not isinstance(value, (torch.Tensor, nn.Module, LosslessCheckContext))


def test_removed_quarot_self_check_config_is_no_longer_registered():
    with pytest.raises(SchemaValidateError, match="quarot_self_check"):
        AutoProcessorConfig.model_validate({"type": "quarot_self_check"})
    parsed = AutoProcessorConfig.model_validate(
        {
            "type": "lossless_check",
            "configs": [{"type": "lossless_test_fake"}],
            "metrics": ["numerical_equivalence"],
        }
    )
    assert isinstance(parsed, LosslessCheckProcessorConfig)
    assert isinstance(parsed.configs[0], _FakeConfig)
    with pytest.raises(SchemaValidateError):
        LosslessCheckProcessorConfig(configs=[], metrics=["numerical_equivalence"])


@pytest.mark.parametrize("stale_field", ["patterns", "default_metric_policy"])
def test_removed_lossless_config_fields_are_rejected(stale_field):
    with pytest.raises(SchemaValidateError, match="Extra inputs are not permitted"):
        LosslessCheckProcessorConfig(
            configs=[_FakeConfig()],
            metrics=["numerical_equivalence"],
            **{stale_field: ["block.*"] if stale_field == "patterns" else "required"},
        )


def test_metrics_is_a_required_non_empty_list():
    with pytest.raises(SchemaValidateError, match="numerical_equivalence"):
        LosslessCheckProcessorConfig(configs=[_FakeConfig()])
    with pytest.raises(SchemaValidateError, match="numerical_equivalence"):
        LosslessCheckProcessorConfig(configs=[_FakeConfig()], metrics=[])


def test_metric_name_shorthand_is_normalized_to_plugin_config():
    parsed = LosslessCheckProcessorConfig(
        configs=[_FakeConfig()],
        metrics=["numerical_equivalence"],
    )

    assert len(parsed.metrics) == 1
    assert isinstance(parsed.metrics[0], LosslessMetricConfig)
    assert parsed.metrics[0].name == "numerical_equivalence"


def test_metric_object_form_preserves_explicit_selection():
    parsed = LosslessCheckProcessorConfig(
        configs=[_FakeConfig()],
        metrics=[{"name": "numerical_equivalence"}],
    )

    assert [metric.name for metric in parsed.metrics] == ["numerical_equivalence"]


def test_duplicate_metric_names_are_rejected_before_model_execution():
    with pytest.raises(SchemaValidateError, match="must be unique"):
        LosslessCheckProcessorConfig(
            configs=[_FakeConfig()],
            metrics=["numerical_equivalence", {"name": "numerical_equivalence"}],
        )


def test_processor_adaptation_contract_has_exactly_three_methods():
    methods = {
        name for name, value in vars(LosslessCheckableProcessor).items() if not name.startswith("_") and callable(value)
    }

    assert methods == {
        "prepare_lossless_check",
        "create_lossless_check_context",
        "align_lossless_check_io",
    }


def test_invalid_plan_return_is_reported_as_protocol_error():
    processor = _processor(_RepeatedLinearModule(), _FakeConfig())
    processor.processor.prepare_lossless_check = lambda: None

    with pytest.raises(UnexpectedError, match="protocol_contract_invalid"):
        processor.pre_run()

    assert processor.result["anomalies"][-1]["reason"] == "protocol_contract_invalid"


def test_invalid_context_data_is_reported_before_forward():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig())
    processor.pre_run()
    processor.processor.create_lossless_check_context = lambda _request, datas: _FakeContext(
        targets=(),
        reference_datas=datas,
        covered_targets=[],
        unresolved_targets=(),
    )

    with pytest.raises(UnexpectedError, match="protocol_contract_invalid"):
        processor.process(_request(model))

    assert model.forward_count == 0
    assert processor.result["anomalies"][-1]["reason"] == "protocol_contract_invalid"


def test_invalid_alignment_return_is_a_single_protocol_error():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig())
    processor.pre_run()
    processor.processor.align_lossless_check_io = lambda *_args: ("invalid",)

    with pytest.raises(UnexpectedError, match="protocol_contract_invalid"):
        processor.process(_request(model))

    assert [item["reason"] for item in processor.result["anomalies"]] == ["protocol_contract_invalid"]


def test_rejects_child_without_lossless_protocol_and_kv_cache_child():
    model = _RepeatedLinearModule()
    with pytest.raises(SchemaValidateError, match="does not allow nested 'group'"):
        _processor(model, GroupProcessorConfig(type="group", configs=[]))
    processor = _processor(model, _FakeConfig(needs_kv_cache=True))
    with pytest.raises(SchemaValidateError, match="KV cache"):
        processor.pre_run()


@pytest.mark.parametrize(
    ("config", "processor_type"),
    [
        (_NoProtocolConfig(), _NoProtocolProcessor),
        (_PartialProtocolConfig(), _PartialProtocolProcessor),
    ],
)
def test_child_without_complete_protocol_warns_and_runs_normally(
    mocker,
    config,
    processor_type,
):
    logger = mocker.patch("msmodelslim.processor.lossless.processor.get_logger").return_value
    model = _RepeatedLinearModule()
    processor = _processor(model, config)
    request = _request(model)

    processor.pre_run()
    processor.process(request)
    processor.post_run()

    assert isinstance(processor.processor, processor_type)
    assert processor.processor.events == ["pre_run", "process", "post_run"]
    assert processor.result["passed"] is None
    assert processor.result["skipped"]
    assert processor.result["skip_reason"] == "processor_capability_missing"
    logger.warning.assert_called_once()


def test_runtime_metrics_are_created_once_per_enabled_round(mocker):
    create_metric = mocker.patch(
        "msmodelslim.processor.lossless.processor.create_lossless_metric",
        side_effect=lambda _config: object(),
    )
    processor = _processor(_RepeatedLinearModule(), _FakeConfig())

    assert create_metric.call_count == 0

    processor.pre_run()
    first_metric = processor._resolved_metrics[0]
    assert create_metric.call_count == 1

    processor.pre_run()
    assert create_metric.call_count == 2
    assert processor._resolved_metrics[0] is not first_metric

    processor.adapter = _AdapterWithoutForward()
    processor.pre_run()
    assert create_metric.call_count == 2
    assert not processor._resolved_metrics


def test_processor_capability_bypass_does_not_create_runtime_metrics(mocker):
    create_metric = mocker.patch(
        "msmodelslim.processor.lossless.processor.create_lossless_metric",
        side_effect=lambda _config: object(),
    )
    processor = _processor(_RepeatedLinearModule(), _NoProtocolConfig())

    processor.pre_run()

    assert create_metric.call_count == 0
    assert not processor._resolved_metrics


@pytest.mark.parametrize(
    "method_name",
    [
        "prepare_lossless_check",
        "create_lossless_check_context",
        "align_lossless_check_io",
    ],
)
def test_child_with_non_callable_protocol_member_warns_and_runs_normally(mocker, method_name):
    logger = mocker.patch("msmodelslim.processor.lossless.processor.get_logger").return_value
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig())
    setattr(processor.processor, method_name, None)
    request = _request(model)

    processor.pre_run()
    processor.process(request)
    processor.post_run()

    assert processor.processor.events == ["fake.process"]
    assert len(request.outputs) == 1
    assert processor.result["passed"] is None
    assert processor.result["skipped"]
    assert processor.result["skip_reason"] == "processor_capability_missing"
    logger.warning.assert_called_once()


@pytest.mark.parametrize("adapter", [_AdapterWithoutForward(), _AdapterWithDefaultForward()])
def test_adapter_without_generate_model_forward_warns_skips_check_and_runs_child(mocker, adapter):
    logger = mocker.patch("msmodelslim.processor.lossless.processor.get_logger").return_value
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig(), adapter=adapter)
    context = LocalDictContext(enable_debug=True)

    with ContextManager(context):
        processor.pre_run()
        request = _request(model)
        processor.preprocess(request)
        processor.process(request)
        processor.postprocess(request)
        processor.post_run()

    result = context["lossless_check"].state["result"]
    assert result["passed"] is None
    assert result["skipped"]
    assert result["skip_reason"] == "adapter_capability_missing"
    assert not result["anomalies"]
    assert not result["checks"]
    assert model.forward_count == 1
    logger.warning.assert_called_once_with(
        "Lossless check skipped: model adapter %s does not implement generate_model_forward; "
        "the processor will run normally",
        adapter.__class__.__name__,
    )


def test_multiple_processors_are_rejected_by_single_processor_contract():
    model = _RepeatedLinearModule()
    with pytest.raises(SchemaValidateError):
        _processor(model, _FakeConfig(label="first"), _FakeConfig(label="second"))


def test_no_target_runs_one_forward_without_installing_hooks():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig(target=False))
    processor.pre_run()
    pre_hooks = len(model.linear._forward_pre_hooks)
    forward_hooks = len(model.linear._forward_hooks)

    processor.process(_request(model))
    processor.post_run()

    assert model.forward_count == 1
    assert len(model.linear._forward_pre_hooks) == pre_hooks
    assert len(model.linear._forward_hooks) == forward_hooks


def test_target_path_is_resolved_again_after_processor_replaces_module():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig(fault="replace"))
    processor.pre_run()
    original_id = id(model.linear)

    processor.process(_request(model))
    processor.post_run()

    assert id(model.linear) != original_id
    assert processor.result["passed"] is True


def test_repeated_invocations_and_nested_io_are_compared_independently():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig())
    processor.pre_run()

    processor.process(_request(model))
    processor.post_run()

    structure_checks = [check for check in processor.result["checks"] if check["structure"] == "block.linear"]
    assert {check["invocation"] for check in structure_checks} == {0, 1}
    assert all("metric_name" in check and "threshold" in check for check in structure_checks)
    assert {check["side"] for check in structure_checks} == {"input", "output"}
    assert not any(check["side"] == "request_output" for check in processor.result["checks"])


@pytest.mark.parametrize(
    ("fault", "side"),
    [("input", "input"), ("output", "output")],
)
def test_first_mismatch_is_structured_fail_fast_and_hooks_are_released(fault, side):
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig(fault=fault))
    processor.pre_run()
    pre_hooks = len(model.linear._forward_pre_hooks)
    forward_hooks = len(model.linear._forward_hooks)

    with pytest.raises(UnexpectedError, match=f"side={side}"):
        processor.process(_request(model))

    assert len(processor.result["anomalies"]) == 1
    anomaly = processor.result["anomalies"][0]
    assert anomaly["processor"] == "_FakeProcessor"
    assert anomaly["request"] == "block"
    assert anomaly["structure"] == "block.linear"
    assert anomaly["side"] == side
    assert len(model.linear._forward_pre_hooks) == pre_hooks
    assert len(model.linear._forward_hooks) == forward_hooks
    assert processor.active_capture_count == 0


def test_alignment_failure_has_distinct_reason_and_releases_context():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig(fault="align"))
    processor.pre_run()

    with pytest.raises(UnexpectedError, match="alignment_error"):
        processor.process(_request(model))

    assert processor.result["anomalies"][0]["reason"] == "alignment_error"
    assert processor.active_capture_count == 0
    gc.collect()
    assert not any(isinstance(item, _FakeContext) for item in gc.get_objects())


@pytest.mark.parametrize("fault", ["nonfinite", "infinite"])
def test_nonfinite_processed_output_has_distinct_reason_and_is_logged_before_failure(
    mocker,
    fault,
):
    logger = mocker.patch("msmodelslim.processor.lossless.processor.get_logger").return_value
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig(fault=fault))
    processor.pre_run()

    with pytest.raises(UnexpectedError, match="non_finite_value"):
        processor.process(_request(model))

    anomaly = processor.result["anomalies"][0]
    assert anomaly["structure"] == "block.linear"
    assert anomaly["side"] == "output"
    assert anomaly["reason"] == "non_finite_value"
    messages = [call.args[0] % tuple(call.args[1:]) for call in logger.error.call_args_list]
    assert len(messages) == 3
    assert "reason=non_finite_value" in messages[0]
    assert "structure=block.linear" in messages[1]
    assert "side=output" in messages[1]
    assert "details:" in messages[2]
    assert "reference_finite" in messages[2]
    assert "processed_finite" in messages[2]


def test_tensor_and_non_tensor_leaf_mismatch_has_distinct_reason():
    processor = _processor(_RepeatedLinearModule(), _FakeConfig())

    with pytest.raises(UnexpectedError, match="tensor_type_mismatch"):
        processor._compare_tree(
            {"value": torch.ones(1)},
            {"value": 1.0},
            request_name="block",
            structure_name="block.linear",
            side="output",
            invocation=0,
        )

    assert processor.result["anomalies"][0]["reason"] == "tensor_type_mismatch"


def test_tensor_shape_mismatch_has_distinct_reason():
    processor = _processor(_RepeatedLinearModule(), _FakeConfig())

    with pytest.raises(UnexpectedError, match="shape_mismatch"):
        processor._compare_tree(
            torch.ones(1, 3),
            torch.ones(1, 2),
            request_name="block",
            structure_name="block.linear",
            side="output",
            invocation=0,
        )

    assert processor.result["anomalies"][0]["reason"] == "shape_mismatch"


def test_tensor_device_mismatch_has_distinct_reason():
    processor = _processor(_RepeatedLinearModule(), _FakeConfig())

    with pytest.raises(UnexpectedError, match="device_mismatch"):
        processor._compare_tree(
            torch.ones(1),
            torch.ones(1, device="meta"),
            request_name="block",
            structure_name="block.linear",
            side="output",
            invocation=0,
        )

    anomaly = processor.result["anomalies"][0]
    assert anomaly["reason"] == "device_mismatch"
    assert anomaly["reference_device"] == "cpu"
    assert anomaly["processed_device"] == "meta"


def test_non_floating_tensor_dtype_mismatch_has_distinct_reason():
    processor = _processor(_RepeatedLinearModule(), _FakeConfig())

    with pytest.raises(UnexpectedError, match="dtype_mismatch"):
        processor._compare_tree(
            torch.ones(1, dtype=torch.int64),
            torch.ones(1, dtype=torch.int32),
            request_name="block",
            structure_name="block.linear",
            side="output",
            invocation=0,
        )

    assert processor.result["anomalies"][0]["reason"] == "dtype_mismatch"


def test_mixed_floating_dtype_is_rejected_before_tolerance_comparison():
    processor = _processor(_RepeatedLinearModule(), _FakeConfig())

    with pytest.raises(UnexpectedError, match="dtype_mismatch"):
        processor._compare_tree(
            torch.ones(1, dtype=torch.float16),
            torch.ones(1, dtype=torch.float32),
            request_name="block",
            structure_name="block.linear",
            side="output",
            invocation=0,
        )

    anomaly = processor.result["anomalies"][0]
    assert anomaly["reference_dtype"] == "torch.float16"
    assert anomaly["processed_dtype"] == "torch.float32"


def test_context_success_summary_is_serializable_and_contains_no_runtime_objects():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig())
    context = LocalDictContext(enable_debug=True)

    with ContextManager(context):
        processor.pre_run()
        processor.process(_request(model))
        processor.post_run()

    result = context["lossless_check"].state["result"]
    assert result["passed"]
    assert result["checked_targets"] == ["block.linear"]
    _assert_summary_has_no_runtime_objects(result)
    json.dumps(result)


def test_context_failure_summary_is_written_before_raise_and_is_serializable():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig(fault="output"))
    context = LocalDictContext(enable_debug=True)

    with ContextManager(context):
        processor.pre_run()
        with pytest.raises(UnexpectedError, match="metric_threshold_exceeded"):
            processor.process(_request(model))

    result = context["lossless_check"].state["result"]
    assert not result["passed"]
    assert result["anomalies"][0]["reason"] == "metric_threshold_exceeded"
    _assert_summary_has_no_runtime_objects(result)
    json.dumps(result)


def test_context_no_target_summary_is_serializable_and_reports_zero_checks():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig(target=False))
    context = LocalDictContext(enable_debug=True)

    with ContextManager(context):
        processor.pre_run()
        processor.process(_request(model))
        processor.post_run()

    result = context["lossless_check"].state["result"]
    assert result["passed"] is None
    assert result["skipped"]
    assert result["skip_reason"] == "target_not_observed"
    assert not result["checked_targets"]
    assert result["total_checks"] == 0
    _assert_summary_has_no_runtime_objects(result)
    json.dumps(result)


def test_one_hundred_requests_over_twenty_rounds_release_all_hooks_and_captures():
    blocks = nn.ModuleList([_RepeatedLinearModule() for _ in range(100)])
    processor = _processor(blocks, _FakeConfig())

    for round_index in range(20):
        processor.pre_run()
        for block_index, block in enumerate(blocks):
            processor.process(_request(block, name=f"round{round_index}.block{block_index}"))
            assert processor.active_capture_count == 0
            assert not block.linear._forward_pre_hooks
            assert not block.linear._forward_hooks
        processor.post_run()
        assert processor.result["passed"]
        assert len(processor.result["checked_targets"]) == 100
        blocks.lossless_events.clear()
        gc.collect()
        runtime_objects = gc.get_objects()
        assert not any(isinstance(item, _StructureCapture) for item in runtime_objects)
        assert not any(isinstance(item, _FakeContext) for item in runtime_objects)


def test_target_not_invoked_by_adapter_request_is_skipped_not_failed():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig())
    processor.pre_run()

    processor.process(_request(model, execute=False))
    processor.post_run()

    assert processor.result["passed"] is None
    assert processor.result["skipped_targets"] == ["block.linear"]
    assert not processor.result["anomalies"]


@pytest.mark.parametrize("execute_order", [(False, True), (True, False)])
def test_target_observed_in_any_request_is_not_reported_as_skipped(execute_order):
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig())
    processor.pre_run()

    for execute in execute_order:
        processor.process(_request(model, execute=execute))
    processor.post_run()

    assert processor.result["passed"]
    assert not processor.result["skipped"]
    assert processor.result["checked_targets"] == ["block.linear"]
    assert not processor.result["skipped_targets"]


def test_theoretical_target_missing_from_model_has_distinct_reason():
    model = _RepeatedLinearModule()
    processor = _processor(
        model,
        _FakeConfig(target=False, pending_target="missing.structure"),
    )
    processor.pre_run()

    with pytest.raises(UnexpectedError, match="structure_not_found"):
        processor.post_run()

    assert processor.result["missing_targets"] == ["missing.structure"]
    assert processor.result["anomalies"][0]["reason"] == "structure_not_found"


def test_target_not_exposed_by_adapter_is_reported_as_skipped_not_passed(mocker):
    logger = mocker.patch("msmodelslim.processor.lossless.processor.get_logger").return_value
    model = _RepeatedLinearModule()
    processor = _processor(
        model,
        _FakeConfig(target=False, pending_target="linear"),
    )
    processor.pre_run()

    processor.post_run()

    assert processor.result["passed"] is None
    assert processor.result["skipped"]
    assert processor.result["skip_reason"] == "target_not_observed"
    assert processor.result["skipped_targets"] == ["linear"]
    assert not processor.result["missing_targets"]
    assert not processor.result["anomalies"]
    logger.warning.assert_called_once_with(
        "Lossless check skipped targets not observed in adapter-generated requests: %s",
        ["linear"],
    )


def test_out_of_scope_theoretical_target_does_not_fail_coverage():
    model = _RepeatedLinearModule()
    processor = _processor(
        model,
        _FakeConfig(target=False, pending_target="missing.structure"),
        include=["block.linear"],
    )
    processor.pre_run()

    processor.post_run()

    assert processor.result["passed"] is None
    assert processor.result["skipped"]
    assert not processor.result["missing_targets"]


def test_out_of_scope_unexecuted_actual_target_does_not_fail_coverage():
    model = _RepeatedLinearModule()
    processor = _processor(
        model,
        _FakeConfig(),
        include=["another.*"],
    )
    processor.pre_run()

    processor.process(_request(model, execute=False))
    processor.post_run()

    assert processor.result["passed"] is None
    assert processor.result["skipped"]
    assert not processor.result["checked_targets"]
    assert model.forward_count == 1


def test_prepare_failure_is_propagated():
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig(fault="prepare"))

    with pytest.raises(RuntimeError, match="prepare failed"):
        processor.pre_run()


def test_distributed_execution_is_rejected(monkeypatch):
    model = _RepeatedLinearModule()
    processor = _processor(model, _FakeConfig())
    monkeypatch.setattr("msmodelslim.processor.lossless.processor.dist.is_initialized", lambda: True)

    with pytest.raises(SchemaValidateError, match="distributed"):
        processor.pre_run()


def test_multimodal_dataset_is_owned_by_adapter_and_runs_without_input_guessing():
    clear_cache(KEY_DATA_LOADER)
    try:
        model = _MultimodalBlock()
        adapter = _MultimodalAdapter()
        runner = _RecordingGeneratedRunner(adapter)
        runner.add_processor(
            LosslessCheckProcessorConfig(
                configs=[_FakeConfig()],
                metrics=["numerical_equivalence"],
            )
        )
        for round_index in range(10):
            clear_cache(KEY_DATA_LOADER)
            offset = round_index / 100
            dataset = [
                {
                    "hidden_states": [1.0 + offset, 2.0, 3.0],
                    "image": [0.1, 0.2 + offset, 0.3],
                    "audio": [0.4, 0.5, 0.6 + offset],
                    "tokens": [11 + round_index, 12, 13],
                    "source": f"sample-{round_index}",
                }
            ]

            runner.run(model=model, calib_data=dataset, device=DeviceType.CPU)

            processor = runner.units[0].processor
            assert processor.result["passed"]
            assert adapter.handled_samples[-1] == (dataset, DeviceType.CPU)
            assert any(
                check["structure"] == "multimodal.block.linear" and check["side"] == "input"
                for check in processor.result["checks"]
            )
            assert processor.active_capture_count == 0
        assert len(adapter.handled_samples) == 10
    finally:
        clear_cache(KEY_DATA_LOADER)


def test_multimodal_generated_runner_reports_processed_input_mismatch_and_cleans_up():
    clear_cache(KEY_DATA_LOADER)
    try:
        for round_index in range(10):
            clear_cache(KEY_DATA_LOADER)
            model = _MultimodalBlock()
            adapter = _MultimodalAdapter()
            runner = _RecordingGeneratedRunner(adapter)
            runner.add_processor(
                LosslessCheckProcessorConfig(
                    configs=[_FakeConfig(fault="input")],
                    metrics=["numerical_equivalence"],
                )
            )
            offset = round_index / 100
            dataset = [
                {
                    "hidden_states": [1.0 + offset, 2.0, 3.0],
                    "image": [0.1, 0.2 + offset, 0.3],
                    "audio": [0.4, 0.5, 0.6 + offset],
                    "tokens": [11 + round_index, 12, 13],
                    "source": f"fault-sample-{round_index}",
                }
            ]

            with pytest.raises(UnexpectedError, match="multimodal.block.linear"):
                runner.run(model=model, calib_data=dataset, device=DeviceType.CPU)

            processor = runner.units[0].processor
            anomaly = processor.result["anomalies"][0]
            assert anomaly["structure"] == "multimodal.block.linear"
            assert anomaly["side"] == "input"
            assert anomaly["reason"] == "metric_threshold_exceeded"
            assert processor.active_capture_count == 0
    finally:
        clear_cache(KEY_DATA_LOADER)
