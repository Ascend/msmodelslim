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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

# pylint: disable=too-many-lines

import operator
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Sequence,
)

import torch
from torch import distributed as dist
from torch import nn
from pydantic import AfterValidator, ConfigDict, Field, SerializeAsAny, field_validator, model_validator
from typing_extensions import Annotated

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import (
    AutoProcessorConfig,
    AutoProcessorConfigList,
    AutoSessionProcessor,
)
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.validation.pydantic import non_empty_string, validate_str_length

from .metrics import (
    LosslessMetricConfig,
    MetricContext,
    MetricResult,
    MetricViolation,
    create_lossless_metric,
)
from .protocol import (
    LosslessCheckContext,
    LosslessCheckPlan,
    LosslessCheckTarget,
    StructureIORecord,
    clone_tree,
)
from .result import new_lossless_check_result
from .runtime import (
    ForwardExecutionState as _ForwardExecutionState,
    StructureCapture as _StructureCapture,
    TargetResolutionError as _TargetResolutionError,
    flatten_tree as _flatten_tree,
)


_METRIC_RESULT_RESERVED_OBSERVATION_KEYS = frozenset(
    {
        "processor",
        "decoder",
        "request",
        "structure",
        "structure_type",
        "side",
        "invocation",
        "path",
        "metric_name",
        "metric_value",
        "threshold",
        "shape",
        "dtype",
        "reference_dtype",
        "processed_dtype",
        "passed",
        "failed_metrics",
        "key_metrics",
        "reason",
    }
)


class LosslessCheckProcessorConfig(AutoProcessorConfig):
    model_config = ConfigDict(extra="forbid")

    type: Literal["lossless_check"] = "lossless_check"
    configs: AutoProcessorConfigList = Field(min_length=1, max_length=1)
    include: List[
        Annotated[
            str,
            AfterValidator(non_empty_string),
            AfterValidator(validate_str_length()),
        ]
    ] = Field(default_factory=lambda: ["*"], min_length=1)
    exclude: List[
        Annotated[
            str,
            AfterValidator(non_empty_string),
            AfterValidator(validate_str_length()),
        ]
    ] = Field(default_factory=list)
    metrics: List[SerializeAsAny[LosslessMetricConfig]] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def require_explicit_metrics(cls, value: Any) -> Any:
        if isinstance(value, dict) and not value.get("metrics"):
            raise SchemaValidateError(
                "lossless_check.metrics must contain at least one explicitly selected metric",
                action=(
                    "Add metrics: [numerical_equivalence] to use the built-in recommended metric, "
                    "or select an installed third-party metric"
                ),
            )
        return value

    @field_validator("metrics", mode="before")
    @classmethod
    def normalize_metric_configs(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        normalized = []
        for item in value:
            if isinstance(item, str):
                item = {"name": item}
            if isinstance(item, LosslessMetricConfig):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                normalized.append(LosslessMetricConfig.model_validate(item))
                continue
            raise SchemaValidateError("Each lossless metric must be a metric name or a metric configuration object")
        return normalized

    @model_validator(mode="after")
    def validate_single_processor_and_metrics(self):
        child_type = self.configs[0].type
        if child_type in {"lossless_check", "group"}:
            raise SchemaValidateError(f"lossless_check does not allow nested '{child_type}' processors")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise SchemaValidateError("Lossless metric names must be unique")
        return self


@QABCRegistry.register(dispatch_key=LosslessCheckProcessorConfig, abc_class=AutoSessionProcessor)
class LosslessCheckProcessor(AutoSessionProcessor):
    """Run request-scoped before/after checks around one processor."""

    def __init__(
        self,
        model: nn.Module,
        config: LosslessCheckProcessorConfig,
        adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        self.adapter = adapter
        self.processor = AutoSessionProcessor.from_config(model, config.configs[0], adapter)
        self.processor_name = self.processor.__class__.__name__
        self.include = ConfigSet(config.include)
        self.exclude = ConfigSet(config.exclude)
        self._protocol_state = self._inspect_processor_contract()
        self.result = self._new_result()
        self._check_run_complete = False
        self.active_capture_count = 0
        self._check_enabled = self._protocol_state == "supported" and self._adapter_supports_forward()
        self._resolved_metrics = ()
        self._metric_checks_executed = 0
        self._expected_targets = set()
        self._covered_targets = set()
        self._executed_targets = set()
        self._unobserved_targets = set()
        self._matched_targets = set()

    def __repr__(self) -> str:
        return f"LosslessCheckProcessor(processor={self.processor_name})"

    def is_data_free(self) -> bool:
        return self.processor.is_data_free() if not self._check_enabled else False

    def support_distributed(self) -> bool:
        return self.processor.support_distributed() if not self._check_enabled else False

    def need_kv_cache(self):
        return self.processor.need_kv_cache() if not self._check_enabled else False

    def preprocess(self, request: BatchProcessRequest) -> None:
        if not self._check_enabled:
            self.processor.preprocess(request)

    def postprocess(self, request: BatchProcessRequest) -> None:
        if not self._check_enabled:
            self.processor.postprocess(request)

    def pre_run(self) -> None:
        self._resolved_metrics = ()
        self.result = self._new_result()
        self._check_run_complete = False
        self._metric_checks_executed = 0
        self._expected_targets.clear()
        self._covered_targets.clear()
        self._executed_targets.clear()
        self._unobserved_targets.clear()
        self._matched_targets.clear()
        self.include = ConfigSet(self.config.include)
        self.exclude = ConfigSet(self.config.exclude)
        self._protocol_state = self._inspect_processor_contract()
        if self._protocol_state != "supported":
            self._check_enabled = False
            self._set_capability_skip(
                "processor_capability_missing",
                "Lossless check skipped: processor %s does not implement the "
                "complete lossless-check protocol; the processor will run normally",
                self.processor_name,
            )
        else:
            self._check_enabled = self._validate_adapter_forward()
        if self._check_enabled and dist.is_initialized():
            raise SchemaValidateError("Lossless check does not support distributed execution")
        try:
            if self._check_enabled:
                if self.processor.need_kv_cache():
                    raise SchemaValidateError("Lossless check does not support processors that require KV cache")
                self._resolved_metrics = self._resolve_metrics()
                plan = self.processor.prepare_lossless_check()
                self._validate_plan_contract(plan)
                self._expected_targets.update(plan.expected_targets)
            self.processor.pre_run()
        except BaseException:
            self.result["passed"] = False
            self._abort_after_failure("pre_run")
            raise

    def _inspect_processor_contract(self) -> str:
        required = (
            "prepare_lossless_check",
            "create_lossless_check_context",
            "align_lossless_check_io",
        )
        present = [hasattr(self.processor, name) for name in required]
        if not any(present):
            return "missing"
        if not all(present):
            return "invalid"
        if not all(callable(getattr(self.processor, name, None)) for name in required):
            return "invalid"
        return "supported"

    def _adapter_supports_forward(self) -> bool:
        forward = getattr(self.adapter, "generate_model_forward", None)
        implementation = getattr(forward, "__func__", None)
        return callable(forward) and implementation is not PipelineInterface.generate_model_forward

    def _validate_adapter_forward(self) -> bool:
        if self._adapter_supports_forward():
            return True
        adapter_name = self.adapter.__class__.__name__ if self.adapter is not None else "None"
        self._set_capability_skip(
            "adapter_capability_missing",
            "Lossless check skipped: model adapter %s does not implement "
            "generate_model_forward; the processor will run normally",
            adapter_name,
        )
        return False

    def _set_capability_skip(self, reason, message, *args) -> None:
        self.result.update(
            {
                "passed": None,
                "skipped": True,
                "skip_reason": reason,
            }
        )
        get_logger().warning(message, *args)
        self._record_result()

    def _resolve_metrics(self):
        names = [config.name for config in self.config.metrics]
        if len(names) != len(set(names)):
            raise SchemaValidateError("Resolved lossless metric names must be unique")
        return tuple(create_lossless_metric(config) for config in self.config.metrics)

    def process(self, request: BatchProcessRequest) -> None:
        if not self._check_enabled:
            self.processor.process(request)
            return

        context = None
        reference_records = {}
        processed_records = {}
        try:
            processed_datas = clone_tree(request.datas)
            context = self.processor.create_lossless_check_context(
                request,
                clone_tree(processed_datas),
            )
            self._validate_context_contract(context)
            if context.unresolved_targets:
                missing_targets = sorted(set(context.unresolved_targets))
                self.result["missing_targets"] = missing_targets
                self._fail(
                    {
                        "request": request.name,
                        "processor": self.processor_name,
                        "decoder": request.name,
                        "structure": missing_targets[0],
                        "structure_type": "unknown",
                        "side": "coverage",
                        "invocation": None,
                        "path": "root",
                        "reason": "structure_not_found",
                        "missing_targets": missing_targets,
                    }
                )
            self._covered_targets.update(context.covered_targets)
            all_targets = self._unique_targets(context.targets)
            targets = []
            for target in all_targets:
                if self._target_selected(target.name):
                    targets.append(target)
                    self._matched_targets.add(target.name)
            with _ForwardExecutionState(request.module) as execution_state:
                if not targets:
                    self._process_checked_child(request)
                    request.datas = processed_datas
                    execution_state.rewind_rng()
                    self._run_forward(request)
                    execution_state.commit()
                    return

                reference_request = BatchProcessRequest(
                    name=request.name,
                    module=request.module,
                    datas=context.reference_datas,
                    outputs=None,
                )
                try:
                    with _StructureCapture(reference_request, targets) as capture:
                        self.active_capture_count = len(capture.handles)
                        self._run_forward(reference_request)
                except _TargetResolutionError as error:
                    self._fail_target_resolution(error.target, request.name)
                self.active_capture_count = 0
                reference_records = capture.records()
                reference_order = tuple(capture.invocation_order)

                execution_state.rewind_rng()
                request.datas = processed_datas
                self._process_checked_child(request)
                request.datas = processed_datas
                # Calibration processors may execute an additional forward.
                # Compare both streams with the same model RNG, and commit only
                # the state consumed by the final processed forward.
                execution_state.rewind_rng()
                try:
                    with _StructureCapture(request, targets) as capture:
                        self.active_capture_count = len(capture.handles)
                        self._run_forward(request)
                except _TargetResolutionError as error:
                    self._fail_target_resolution(error.target, request.name)
                self.active_capture_count = 0
                processed_records = capture.records()
                processed_order = tuple(capture.invocation_order)
                for target in targets:
                    if not reference_records.get(target.name) and not processed_records.get(target.name):
                        if target.name not in self._executed_targets:
                            self._unobserved_targets.add(target.name)
                    else:
                        self._unobserved_targets.discard(target.name)

                self._compare_records(
                    targets,
                    context,
                    reference_records,
                    processed_records,
                    reference_order,
                    processed_order,
                )
                self._record_result()
                execution_state.commit()
        except BaseException:
            self.result["passed"] = False
            self._abort_after_failure("process")
            raise
        finally:
            self.active_capture_count = 0
            reference_records.clear()
            processed_records.clear()
            context = None

    @torch.no_grad()
    def _process_checked_child(self, request) -> None:
        self.processor.preprocess(request)
        self.processor.process(request)
        self.processor.postprocess(request)

    def _abort_lossless_check(self) -> None:
        cleanup = getattr(self.processor, "abort_lossless_check", None)
        if callable(cleanup):
            cleanup()

    def _abort_after_failure(self, stage: str) -> None:
        try:
            self.abort_lossless_check()
        except BaseException as cleanup_error:
            get_logger().error(
                "Failed to clean up lossless check after %s failure (%s)",
                stage,
                type(cleanup_error).__name__,
            )

    def abort_lossless_check(self) -> None:
        """Mark an unfinished check failed and release its processor state."""
        incomplete = self._check_enabled and not self._check_run_complete
        self._check_run_complete = True
        try:
            self._abort_lossless_check()
        finally:
            if incomplete:
                self.result["passed"] = False
                self._record_result()

    def post_run(self) -> None:
        try:
            self._post_run()
            self._abort_lossless_check()
        except BaseException:
            self.result["passed"] = False
            self._abort_after_failure("post_run")
            raise
        self._check_run_complete = True

    def _post_run(self) -> None:
        if not self._check_enabled:
            self.processor.post_run()
            self._record_result()
            return

        selected_expected = {
            target_name for target_name in self._expected_targets if self._target_selected(target_name)
        }
        self._matched_targets.update(selected_expected)
        unmatched_include = sorted(key for key in self.include.unmatched_keys() if key != "*")
        unmatched_exclude = sorted(self.exclude.unmatched_keys())
        self.result["matched_targets"] = sorted(self._matched_targets)
        self.result["unmatched_include"] = unmatched_include
        self.result["unmatched_exclude"] = unmatched_exclude
        if unmatched_include:
            get_logger().warning(
                "Lossless check include patterns did not match any target: %s",
                unmatched_include,
            )
        if unmatched_exclude:
            get_logger().warning(
                "Lossless check exclude patterns did not match any target: %s",
                unmatched_exclude,
            )
        pending_targets = selected_expected - self._covered_targets
        known_structures = {name for name, _module in self.model.named_modules()}
        missing_targets = {target for target in pending_targets if target not in known_structures}
        skipped_targets = (pending_targets - missing_targets) | self._unobserved_targets
        self.result["missing_targets"] = sorted(missing_targets)
        self.result["skipped_targets"] = sorted(skipped_targets)
        if skipped_targets:
            get_logger().warning(
                "Lossless check skipped targets not observed in adapter-generated requests: %s",
                sorted(skipped_targets),
            )
        if missing_targets:
            self._fail(
                {
                    "request": "post_run",
                    "processor": self.processor_name,
                    "decoder": "post_run",
                    "structure": min(missing_targets),
                    "structure_type": "unknown",
                    "side": "coverage",
                    "invocation": None,
                    "path": "root",
                    "reason": "structure_not_found",
                    "missing_targets": sorted(missing_targets),
                }
            )
        self.processor.post_run()
        if skipped_targets and self.result["passed"]:
            self.result.update(
                {
                    "passed": None,
                    "skipped": True,
                    "skip_reason": "target_not_observed",
                }
            )
        self.result["checked_targets"] = sorted(self._executed_targets)
        self.result["total_checks"] = len(self.result["checks"])
        if self.result["passed"] is True and self._metric_checks_executed == 0:
            self.result.update(
                {
                    "passed": None,
                    "skipped": True,
                    "skip_reason": "target_not_observed",
                }
            )
        self._record_result()

    @staticmethod
    def _unique_targets(
        candidates: Sequence[LosslessCheckTarget],
    ) -> List[LosslessCheckTarget]:
        targets = []
        seen = set()
        for target in candidates:
            key = (
                target.name,
                target.path,
                target.check_input,
                target.check_output,
            )
            if key not in seen:
                targets.append(target)
                seen.add(key)
        return targets

    def _validate_plan_contract(self, plan) -> None:
        if not isinstance(plan, LosslessCheckPlan):
            self._fail_protocol_contract("prepare_lossless_check must return LosslessCheckPlan")
        if not isinstance(plan.expected_targets, tuple):
            self._fail_protocol_contract("LosslessCheckPlan.expected_targets must be a tuple")
        if not all(isinstance(target, str) and target for target in plan.expected_targets):
            self._fail_protocol_contract("LosslessCheckPlan.expected_targets must contain non-empty strings")

    def _validate_context_contract(self, context) -> None:
        if not isinstance(context, LosslessCheckContext):
            self._fail_protocol_contract("create_lossless_check_context must return LosslessCheckContext")
        if not isinstance(context.targets, tuple):
            self._fail_protocol_contract("LosslessCheckContext.targets must be a tuple")
        for target in context.targets:
            if not isinstance(target, LosslessCheckTarget):
                self._fail_protocol_contract("LosslessCheckContext.targets must contain LosslessCheckTarget values")
        if not isinstance(context.covered_targets, tuple):
            self._fail_protocol_contract("LosslessCheckContext.covered_targets must be a tuple")
        if not all(isinstance(target, str) and target for target in context.covered_targets):
            self._fail_protocol_contract("LosslessCheckContext.covered_targets must contain non-empty strings")
        if not isinstance(context.unresolved_targets, tuple):
            self._fail_protocol_contract("LosslessCheckContext.unresolved_targets must be a tuple")
        if not all(isinstance(target, str) and target for target in context.unresolved_targets):
            self._fail_protocol_contract("LosslessCheckContext.unresolved_targets must contain non-empty strings")

    def _fail_protocol_contract(self, message: str) -> None:
        self._fail(
            {
                "processor": self.processor_name,
                "decoder": "protocol",
                "structure": self.processor_name,
                "structure_type": "processor",
                "side": "protocol",
                "invocation": None,
                "path": "root",
                "reason": "protocol_contract_invalid",
                "error": message,
            }
        )

    def _fail_target_resolution(
        self,
        target: LosslessCheckTarget,
        request_name: str,
    ) -> None:
        self._fail(
            {
                "processor": self.processor_name,
                "decoder": target.decoder or request_name,
                "request": request_name,
                "structure": target.name,
                "structure_type": target.structure_type,
                "side": "coverage",
                "invocation": None,
                "path": target.path,
                "reason": "structure_not_found",
            }
        )

    def _target_selected(self, target_name: str) -> bool:
        included = target_name in self.include
        excluded = target_name in self.exclude
        return included and not excluded

    @staticmethod
    def _run_forward(request: BatchProcessRequest) -> None:
        outputs = []
        with torch.no_grad():
            for args, kwargs in request.datas or []:
                if args or kwargs:
                    outputs.append(request.module(*args, **kwargs))
        request.outputs = outputs

    @staticmethod
    def _validate_metric_result_contract(metric, metric_result) -> None:
        if not isinstance(metric_result, MetricResult):
            raise TypeError(
                "metric {!r} returned {!r}, expected MetricResult".format(
                    metric.name,
                    type(metric_result).__name__,
                )
            )
        if metric_result.name != metric.name:
            raise ValueError(
                "metric {!r} returned result name {!r}".format(
                    metric.name,
                    metric_result.name,
                )
            )
        if not isinstance(metric_result.passed, bool):
            raise TypeError(
                "metric {!r} returned passed with type {!r}, expected bool".format(
                    metric.name,
                    type(metric_result.passed).__name__,
                )
            )
        if not isinstance(metric_result.observations, dict):
            raise TypeError(
                "metric {!r} returned observations with type {!r}, expected dict".format(
                    metric.name,
                    type(metric_result.observations).__name__,
                )
            )
        reserved_keys = [key for key in metric_result.observations if key in _METRIC_RESULT_RESERVED_OBSERVATION_KEYS]
        if reserved_keys:
            raise ValueError(
                "metric {!r} returned observations with reserved key(s) {!r}".format(
                    metric.name,
                    reserved_keys,
                )
            )
        if not isinstance(metric_result.violations, tuple):
            raise TypeError(
                "metric {!r} returned violations with type {!r}, expected tuple".format(
                    metric.name,
                    type(metric_result.violations).__name__,
                )
            )
        for index, violation in enumerate(metric_result.violations):
            if not isinstance(violation, MetricViolation):
                raise TypeError(
                    "metric {!r} returned violations[{}] with type {!r}, expected MetricViolation".format(
                        metric.name,
                        index,
                        type(violation).__name__,
                    )
                )

    def _compare_records(
        self,
        targets,
        context,
        reference_records,
        processed_records,
        reference_order,
        processed_order,
    ):
        if reference_order != processed_order:
            self._fail(
                {
                    "processor": self.processor_name,
                    "decoder": reference_order[0][0] if reference_order else "unknown",
                    "request": reference_order[0][0] if reference_order else "",
                    "structure": "request_invocation_order",
                    "structure_type": "request",
                    "side": "structure",
                    "invocation": None,
                    "path": "root",
                    "reason": "invocation_order_mismatch",
                    "reference_order": list(reference_order),
                    "processed_order": list(processed_order),
                }
            )

        targets_by_name = {target.name: target for target in targets}
        for target in targets:
            references = reference_records.get(target.name, [])
            processed = processed_records.get(target.name, [])
            if not references and not processed:
                continue
            if len(references) != len(processed):
                request_name = references[0].request_name if references else processed[0].request_name
                self._fail(
                    {
                        "processor": self.processor_name,
                        "decoder": target.decoder or request_name,
                        "request": request_name,
                        "structure": target.name,
                        "structure_type": target.structure_type,
                        "side": "structure",
                        "invocation": None,
                        "path": target.path,
                        "reason": "invocation_count_mismatch",
                        "reference_count": len(references),
                        "processed_count": len(processed),
                    }
                )

        for target_name, invocation in reference_order:
            target = targets_by_name[target_name]
            self._executed_targets.add(target.name)
            reference = reference_records[target_name][invocation]
            processed_record = processed_records[target_name][invocation]
            try:
                aligned = self.processor.align_lossless_check_io(
                    context,
                    target,
                    reference,
                    processed_record,
                )
            except Exception as error:
                self._fail(
                    {
                        "processor": self.processor_name,
                        "decoder": target.decoder or reference.request_name,
                        "request": reference.request_name,
                        "structure": target.name,
                        "structure_type": target.structure_type,
                        "side": "structure",
                        "invocation": reference.invocation,
                        "path": "root",
                        "reason": "alignment_error",
                        "error": str(error),
                    }
                )
            if (
                not isinstance(aligned, tuple)
                or len(aligned) != 2
                or not all(isinstance(record, StructureIORecord) for record in aligned)
            ):
                self._fail_protocol_contract("align_lossless_check_io must return two StructureIORecord values")
            reference, processed_record = aligned
            if target.check_input:
                self._compare_tree(
                    (reference.args, reference.kwargs),
                    (processed_record.args, processed_record.kwargs),
                    request_name=reference.request_name,
                    decoder=target.decoder or reference.request_name,
                    structure_name=target.name,
                    structure_type=target.structure_type,
                    side="input",
                    invocation=reference.invocation,
                )
            if target.check_output:
                self._compare_tree(
                    reference.output,
                    processed_record.output,
                    request_name=reference.request_name,
                    decoder=target.decoder or reference.request_name,
                    structure_name=target.name,
                    structure_type=target.structure_type,
                    side="output",
                    invocation=reference.invocation,
                )

    def _compare_tree(
        self,
        reference,
        processed,
        *,
        request_name,
        decoder=None,
        structure_name,
        structure_type="unknown",
        side,
        invocation,
    ):
        reference_leaves = dict(_flatten_tree(reference))
        processed_leaves = dict(_flatten_tree(processed))
        if reference_leaves.keys() != processed_leaves.keys():
            self._fail(
                {
                    "processor": self.processor_name,
                    "decoder": decoder or request_name,
                    "request": request_name,
                    "structure": structure_name,
                    "structure_type": structure_type,
                    "side": side,
                    "invocation": invocation,
                    "path": "root",
                    "reason": "structure_mismatch",
                }
            )
        for path, reference_value in reference_leaves.items():
            processed_value = processed_leaves[path]
            reference_is_tensor = isinstance(reference_value, torch.Tensor)
            processed_is_tensor = isinstance(processed_value, torch.Tensor)
            if not reference_is_tensor and not processed_is_tensor:
                continue
            common = {
                "processor": self.processor_name,
                "decoder": decoder or request_name,
                "request": request_name,
                "structure": structure_name,
                "structure_type": structure_type,
                "side": side,
                "invocation": invocation,
                "path": path,
            }
            if reference_is_tensor != processed_is_tensor:
                self._fail(
                    {
                        **common,
                        "reason": "tensor_type_mismatch",
                        "reference_type": type(reference_value).__name__,
                        "processed_type": type(processed_value).__name__,
                    }
                )
            if reference_value.shape != processed_value.shape:
                self._fail(
                    {
                        **common,
                        "reason": "shape_mismatch",
                        "reference_shape": list(reference_value.shape),
                        "processed_shape": list(processed_value.shape),
                    }
                )
            if reference_value.device != processed_value.device:
                self._fail(
                    {
                        **common,
                        "reason": "device_mismatch",
                        "reference_device": str(reference_value.device),
                        "processed_device": str(processed_value.device),
                    }
                )
            if reference_value.dtype != processed_value.dtype:
                self._fail(
                    {
                        **common,
                        "reason": "dtype_mismatch",
                        "reference_dtype": str(reference_value.dtype),
                        "processed_dtype": str(processed_value.dtype),
                    }
                )
            reference_is_float = reference_value.is_floating_point()
            if not reference_is_float:
                continue
            reference_float = reference_value.float()
            processed_float = processed_value.float()
            reference_finite = bool(torch.isfinite(reference_float).all())
            processed_finite = bool(torch.isfinite(processed_float).all())
            if not reference_finite or not processed_finite:
                self._fail(
                    {
                        "request": request_name,
                        "processor": self.processor_name,
                        "decoder": decoder or request_name,
                        "structure": structure_name,
                        "structure_type": structure_type,
                        "side": side,
                        "invocation": invocation,
                        "path": path,
                        "reason": "non_finite_value",
                        "reference_finite": reference_finite,
                        "processed_finite": processed_finite,
                        "shape": list(reference_value.shape),
                        "reference_dtype": str(reference_value.dtype),
                        "processed_dtype": str(processed_value.dtype),
                    }
                )
            context = MetricContext(
                processor=self.processor_name,
                decoder=decoder or request_name,
                structure=structure_name,
                structure_type=structure_type,
                side=side,
                invocation=invocation,
                path=path,
            )
            for metric in self._resolved_metrics:
                try:
                    metric_result = metric.evaluate(
                        reference_value,
                        processed_value,
                        context,
                    )
                    self._validate_metric_result_contract(metric, metric_result)
                except Exception as error:
                    self._fail(
                        {
                            **common,
                            "metric_name": metric.name,
                            "reason": "metric_execution_error",
                            "error": str(error),
                        }
                    )
                failed_metrics = [
                    {
                        "name": violation.name,
                        "value": violation.value,
                        "threshold": violation.threshold,
                        "comparison": violation.comparison,
                    }
                    for violation in metric_result.violations
                ]
                if not metric_result.passed and not failed_metrics:
                    failed_metrics.append(
                        {
                            "name": metric_result.name,
                            "value": metric_result.value,
                            "threshold": metric_result.threshold,
                            "comparison": None,
                        }
                    )
                check = {
                    **common,
                    "metric_name": metric_result.name,
                    "metric_value": metric_result.value,
                    "threshold": metric_result.threshold,
                    "shape": list(reference_value.shape),
                    "dtype": str(reference_value.dtype),
                    "reference_dtype": str(reference_value.dtype),
                    "processed_dtype": str(processed_value.dtype),
                    "passed": bool(metric_result.passed),
                    **metric_result.observations,
                    "failed_metrics": failed_metrics,
                    "key_metrics": dict(metric_result.observations),
                }
                self.result["checks"].append(check)
                self._metric_checks_executed += 1
                if metric_result.passed:
                    get_logger().info(
                        "Lossless check passed processor=%s decoder=%s "
                        "structure=%s side=%s invocation=%s path=%s "
                        "metric=%s value=%s threshold=%s",
                        self.processor_name,
                        decoder or request_name,
                        structure_name,
                        side,
                        invocation,
                        path,
                        metric_result.name,
                        metric_result.value,
                        metric_result.threshold,
                    )
                if not metric_result.passed:
                    self._fail(
                        {
                            **check,
                            "reason": "metric_threshold_exceeded",
                        }
                    )

    def _new_result(self) -> dict:
        return new_lossless_check_result(self.config.include, self.config.exclude)

    @staticmethod
    def _format_failed_metrics(anomaly: dict) -> str:
        formatted = []
        for item in anomaly.get("failed_metrics", []):
            if item.get("comparison"):
                expectation = f"expected{item['comparison']}{item.get('threshold')}"
            else:
                expectation = f"threshold={item.get('threshold')}"
            formatted.append(f"{item.get('name')}(value={item.get('value')}, {expectation})")
        return ", ".join(formatted) if formatted else "none"

    @staticmethod
    def _format_details(details: dict) -> str:
        if not details:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in details.items())

    @staticmethod
    def _failure_details(anomaly: dict) -> dict:
        context_fields = {
            "processor",
            "decoder",
            "request",
            "structure",
            "structure_type",
            "side",
            "invocation",
            "path",
            "reason",
        }
        return {key: value for key, value in anomaly.items() if key not in context_fields}

    def _record_failure(self, anomaly: dict) -> None:
        anomaly.setdefault("processor", self.processor_name)
        self.result["anomalies"].append(anomaly)
        self.result["passed"] = False
        self._record_result()
        reason = anomaly.get("reason")
        if reason == "metric_threshold_exceeded":
            failed_metrics = anomaly.get("failed_metrics", [])
            get_logger().error(
                "Lossless check failed: metric=%s did not meet %s threshold(s)",
                anomaly.get("metric_name"),
                len(failed_metrics) or 1,
            )
            get_logger().error(
                "Lossless check location: reason=%s processor=%s decoder=%s"
                " structure=%s structure_type=%s side=%s invocation=%s path=%s",
                reason,
                anomaly.get("processor"),
                anomaly.get("decoder"),
                anomaly.get("structure"),
                anomaly.get("structure_type"),
                anomaly.get("side"),
                anomaly.get("invocation"),
                anomaly.get("path"),
            )
            get_logger().error(
                "Lossless check failed_metrics: %s",
                self._format_failed_metrics(anomaly),
            )
            get_logger().error(
                "Lossless check key_metrics: %s",
                self._format_details(anomaly.get("key_metrics", {})),
            )
            return
        get_logger().error(
            "Lossless check failed: reason=%s",
            reason,
        )
        get_logger().error(
            "Lossless check location: processor=%s decoder=%s structure=%s"
            " structure_type=%s side=%s invocation=%s path=%s",
            anomaly.get("processor"),
            anomaly.get("decoder"),
            anomaly.get("structure"),
            anomaly.get("structure_type"),
            anomaly.get("side"),
            anomaly.get("invocation"),
            anomaly.get("path"),
        )
        get_logger().error(
            "Lossless check details: %s",
            self._format_details(self._failure_details(anomaly)),
        )

    def _fail(self, anomaly: dict) -> None:
        self._record_failure(anomaly)
        metric_context = ""
        if anomaly.get("metric_name"):
            failed_metric_names = ",".join(item.get("name", "unknown") for item in anomaly.get("failed_metrics", []))
            metric_context = (
                f" metric={anomaly.get('metric_name')}"
                f" failed_metrics={failed_metric_names or anomaly.get('metric_name')}"
            )
        message = (
            "Lossless check failed at "
            f"{anomaly.get('structure')} side={anomaly.get('side')} "
            f"reason={anomaly.get('reason')}{metric_context}"
        )
        raise UnexpectedError(message)

    def _record_result(self) -> None:
        from msmodelslim.core.context import get_current_context

        ctx = get_current_context()
        if ctx is None:
            return
        namespace = operator.getitem(ctx, "lossless_check")
        namespace.state["result"] = self.result
