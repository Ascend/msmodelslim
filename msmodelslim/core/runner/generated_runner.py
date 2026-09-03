#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import gc
from typing import Any, Generator, List, Optional

import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from msmodelslim.core.base.protocol import BatchProcessRequest, DataUnit, ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.base import BaseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.processor import AutoProcessorConfig
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.utils.cache import to_device
from msmodelslim.utils.cache.memory import load_cached
from msmodelslim.utils.exception import (
    InvalidDatasetError,
    ToDoError,
    UnexpectedError,
    UnsupportedError,
)
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.memory import (
    format_memory_size,
    get_device_allocated_memory,
    get_device_reserved_memory,
)

KEY_DATA_LOADER = "data_loader"


def init_pipeline_generators(
    pipeline_interface: PipelineInterface,
    model: nn.Module,
    calib_data: Optional[List[Any]],
    use_forward_pipeline: bool,
) -> List[Generator[ProcessRequest, Any, None]]:
    """Create one compatible generator per sample for the selected pipeline."""
    if use_forward_pipeline:
        if calib_data is None:
            raise InvalidDatasetError("Calib data is needed because pipeline contains non-data-free processors")
        dataloader = get_input_datas(pipeline_interface, calib_data)
        device = next(model.parameters()).device
        return [pipeline_interface.generate_model_forward(model, to_device(data, device)) for data in dataloader]
    return [pipeline_interface.generate_model_visit(model)]


def _assert_same_module(requests: List[ProcessRequest]) -> None:
    expected_name = requests[0].name
    for index, request in enumerate(requests[1:], start=1):
        if request.name != expected_name:
            raise UnexpectedError(
                f"Pipeline generators ended out of sync at sample {index}: "
                f"expected module name {expected_name!r}, got {request.name!r}; "
                "Ensure every calibration sample yields the same model pipeline.",
                action="Ensure every calibration sample yields the same model pipeline.",
            )


def _advance_generators(
    generators: List[Generator[ProcessRequest, Any, None]],
    started: List[bool],
    feedback: Optional[List[Any]],
) -> tuple[Optional[List[ProcessRequest]], bool]:
    """Advance every generator by one step; return requests or completion."""
    requests: List[ProcessRequest] = []
    completed_indices: List[int] = []
    for index, generator in enumerate(generators):
        try:
            if not started[index]:
                request = next(generator)
                started[index] = True
            else:
                send_value = None if feedback is None else feedback[index]
                request = generator.send(send_value)
        except StopIteration:
            completed_indices.append(index)
            continue
        if not isinstance(request, ProcessRequest):
            raise UnexpectedError(
                f"sample-{index} expected ProcessRequest, got {type(request).__name__}",
            )
        requests.append(request)

    if completed_indices:
        if len(completed_indices) == len(generators):
            return None, True
        ended = [f"sample-{index}" for index in completed_indices]
        active = [f"sample-{index}" for index in range(len(generators)) if index not in completed_indices]
        raise UnexpectedError(
            f"Pipeline generators ended out of sync: ended={ended} active={active}",
            action="Ensure every calibration sample yields the same model pipeline.",
        )
    return requests, False


def _close_generators(generators: List[Generator[ProcessRequest, Any, None]]) -> None:
    for generator in generators:
        generator.close()


class GeneratedProcessUnit:
    """Bind one existing processor to GeneratedRunner's shared target batches."""

    def __init__(
        self,
        model: nn.Module,
        processor: AutoSessionProcessor,
        data_recorder: Optional[DataUnit],
    ):
        self.model = model
        self.processor = processor
        self.data_recorder = data_recorder

    def __repr__(self):
        return self.processor.__repr__()

    def pre_run(self):
        self.processor.pre_run()

    def post_run(self):
        self.processor.post_run()

    def run_step(self, batch_request: BatchProcessRequest) -> None:
        get_logger().info(
            'Run processor %s for "%s"',
            self.processor,
            batch_request.name,
        )
        self.processor.preprocess(batch_request)
        self.processor.process(batch_request)
        self.processor.postprocess(batch_request)
        if batch_request.outputs is not None:
            self.data_recorder.output = batch_request.outputs
        if hasattr(torch, "npu"):
            gc.collect()
            torch.npu.empty_cache()
            get_logger().debug(
                "After run step for %s: allocated=%s, reserved=%s",
                self.processor,
                format_memory_size(get_device_allocated_memory()),
                format_memory_size(get_device_reserved_memory()),
            )
        elif hasattr(torch, "cuda"):
            gc.collect()
            torch.cuda.empty_cache()
            get_logger().debug(
                "After run step for %s: allocated=%s, reserved=%s",
                self.processor,
                format_memory_size(get_device_allocated_memory()),
                format_memory_size(get_device_reserved_memory()),
            )


class GeneratedRunner(BaseRunner):
    """Schedule processors over pipeline-level generators with inline sample barriers."""

    def __init__(self, adapter: PipelineInterface, propagate: bool = True):
        super().__init__()
        self.process_config_list: List[AutoProcessorConfig] = []
        self.adapter = adapter
        self.propagate = propagate

    def preprocess_processor(
        self,
        processor_list: List[AutoProcessorConfig],
        model: nn.Module,
        device: DeviceType = DeviceType.NPU,
    ):
        pass

    def add_processor(
        self,
        processor_cfg: AutoProcessorConfig,
        append: bool = True,
    ):
        if append:
            self.process_config_list.append(processor_cfg)
        else:
            self.process_config_list.insert(0, processor_cfg)

    def build_process_unit(
        self,
        config_list: List[AutoProcessorConfig],
        model: nn.Module,
        adapter: PipelineInterface,
        data_recorder: DataUnit,
        calib_data: Optional[List[Any]] = None,
    ) -> List[GeneratedProcessUnit]:
        _ = calib_data
        processors = [
            AutoSessionProcessor.from_config(model, processor_config, adapter) for processor_config in config_list
        ]
        enable_kv_cache(model, adapter, processors)
        return [
            GeneratedProcessUnit(
                model=model,
                processor=processor,
                data_recorder=data_recorder,
            )
            for processor in processors
        ]

    def run(
        self,
        model: nn.Module = None,
        calib_data: Optional[List[Any]] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ):
        _ = device_indices
        _ = get_input_datas(self.adapter, calib_data, device)
        if model is None:
            get_logger().info("Start to init model")
            model = self.adapter.init_model(device=device)
            get_logger().info("Init model success")
        processor_list = self.process_config_list.copy()
        self.preprocess_processor(processor_list, model, device)
        data_recorder = DataUnit(None, None)
        process_units = self.build_process_unit(
            processor_list,
            model=model,
            adapter=self.adapter,
            calib_data=calib_data,
            data_recorder=data_recorder,
        )
        self.generated_schedule(process_units, model, calib_data)

    @torch.no_grad()
    def generated_schedule(
        self,
        process_units: List[GeneratedProcessUnit],
        model: nn.Module,
        calib_data: Optional[List[Any]] = None,
    ):
        """Run processors over strict same-target sample barriers."""
        get_logger().info("Scheduler %d unit: %s", len(process_units), process_units)
        get_logger().info("Runner propagate=%s", self.propagate)
        use_forward_pipeline = any(not unit.processor.is_data_free() for unit in process_units)
        get_logger().info(
            "Pipeline generator mode: %s",
            "forward (calibration)" if use_forward_pipeline else "visit (data-free)",
        )
        for unit in process_units:
            unit.pre_run()

        generators = init_pipeline_generators(
            self.adapter,
            model,
            calib_data,
            use_forward_pipeline,
        )
        started = [False] * len(generators)
        completed = False
        feedback: Optional[List[Any]] = None
        try:
            while True:
                requests, done = _advance_generators(generators, started, feedback)
                if done:
                    completed = True
                    break
                _assert_same_module(requests)
                batch_request = BatchProcessRequest(
                    name=requests[0].name,
                    module=requests[0].module,
                    datas=[(request.args, dict(request.kwargs)) for request in requests],
                    outputs=None,
                )
                for unit in process_units:
                    unit.run_step(batch_request)
                feedback = batch_request.outputs if self.propagate else None
                if feedback is not None and len(feedback) != len(requests):
                    raise UnexpectedError(
                        f"Processor output count expected={len(requests)} actual={len(feedback)}",
                    )
        except UnexpectedError as error:
            if "ended out of sync" in str(error) and "shared processors" not in str(error):
                raise UnexpectedError(
                    f"{error}; shared processors={[repr(unit.processor) for unit in process_units]}.",
                    action="Ensure every calibration sample yields the same model pipeline.",
                ) from error
            raise
        finally:
            _close_generators(generators)

        if completed:
            for unit in process_units:
                unit.post_run()


def enable_kv_cache(
    model: nn.Module,
    adapter: PipelineInterface,
    processors: List[AutoSessionProcessor],
):
    need_kv_cache = any(processor.need_kv_cache() for processor in processors)
    get_logger().info("KV cache requirement: %s", need_kv_cache)
    try:
        adapter.enable_kv_cache(model, need_kv_cache)
    except (AttributeError, NotImplementedError, ToDoError) as error:
        if need_kv_cache:
            raise UnsupportedError("Some processors need enable kv cache, but failed to enable kv cache") from error
        get_logger().warning("Failed to disable kv cache, this may cause more memory usage")


def get_input_datas(
    model_adapter: PipelineInterface,
    calib_data: Optional[List[Any]] = None,
    dev_type: DeviceType = DeviceType.NPU,
):
    return load_cached(
        key=KEY_DATA_LOADER,
        init_func=_get_input_datas,
        args=(model_adapter, calib_data, dev_type),
    )


def _get_input_datas(
    model_adapter: PipelineInterface,
    calib_data: Optional[List[Any]] = None,
    dev_type: DeviceType = DeviceType.NPU,
) -> DataLoader:
    get_logger().info("Start to handle dataset")
    input_datas = model_adapter.handle_dataset(calib_data, dev_type)
    data_loader = _create_dataloader(input_datas, 0, 1, 1)
    get_logger().info("Handle dataset success")
    return data_loader


def _create_dataloader(dataset, rank, world_size, batch_size) -> DataLoader:
    _ = (rank, world_size, batch_size)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return dataset
    sampler = DistributedSampler(dataset, shuffle=False)
    return DataLoader(dataset, sampler=sampler, batch_size=None)
