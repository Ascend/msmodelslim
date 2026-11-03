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
from typing import List, Optional, Any, Generator

import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from msmodelslim.core.base.protocol import ProcessRequest, BatchProcessRequest, DataUnit
from msmodelslim.core.const import DeviceType
from msmodelslim.core.runner.base import BaseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.processor import AutoProcessorConfig
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.utils.cache import to_device
from msmodelslim.utils.cache.memory import load_cached
from msmodelslim.utils.exception import ToDoError, UnsupportedError, InvalidDatasetError, SecurityError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.memory import get_device_allocated_memory, get_device_reserved_memory, format_memory_size

KEY_DATA_LOADER = "data_loader"


class GeneratedProcessUnit:
    def __init__(
        self,
        model: nn.Module,
        processor: AutoSessionProcessor,
        pipeline_interface: PipelineInterface,
        calib_data: Optional[List[Any]],
        data_recorder: Optional[DataUnit],
    ):
        self.model = model
        self.processor = processor
        self.pipeline_interface = pipeline_interface
        self.calib_data = calib_data
        self.data_recorder = data_recorder

        self.generators: Optional[List[Generator[ProcessRequest, Any, None]]] = None

    def __repr__(self):
        return self.processor.__repr__()

    def pre_run(self):
        self.processor.pre_run()

    def post_run(self):
        self.processor.post_run()

    def init_generators(self):
        if self.generators is not None:
            return

        if not self.processor.is_data_free():
            if self.calib_data is None:
                raise InvalidDatasetError(f"Calib data is needed because {self.processor} is not data-free")
            dataloader = get_input_datas(self.pipeline_interface, self.calib_data)
            self.generators = [
                self.pipeline_interface.generate_model_forward(
                    self.model, to_device(data, next(self.model.parameters()).device)
                )
                for data in dataloader
            ]
        else:
            self.generators = [self.pipeline_interface.generate_model_visit(self.model)]

    def make_progress(self) -> bool:
        requests = []
        inputs = self.data_recorder.input if self.data_recorder.input else [None for _ in self.generators]
        for gen, response in zip(self.generators, inputs):
            try:
                request = gen.send(response)
                requests.append(request)
            except StopIteration:
                return False

        if not requests:
            raise SecurityError('requests cannot be empty')

        batch_request = BatchProcessRequest(
            name=requests[0].name,
            module=requests[0].module,
            datas=[
                (
                    request.args,
                    request.kwargs,
                )
                for request in requests
            ],
            outputs=None,
        )

        get_logger().info('Run processor %s for "%s"', self.processor, batch_request.name)

        self.processor.preprocess(batch_request)
        self.processor.process(batch_request)
        self.processor.postprocess(batch_request)

        if batch_request.outputs is not None:
            self.data_recorder.output = batch_request.outputs

        if hasattr(torch, 'npu'):
            gc.collect()
            torch.npu.empty_cache()
            get_logger().debug(
                "After make progress for %s: allocated=%s, reserved=%s",
                self.processor,
                format_memory_size(get_device_allocated_memory()),
                format_memory_size(get_device_reserved_memory()),
            )
        elif hasattr(torch, 'cuda'):
            gc.collect()
            torch.cuda.empty_cache()
            get_logger().debug(
                "After make progress for %s: allocated=%s, reserved=%s",
                self.processor,
                format_memory_size(get_device_allocated_memory()),
                format_memory_size(get_device_reserved_memory()),
            )

        return True


class GeneratedRunner(BaseRunner):
    def __init__(
        self,
        adapter: PipelineInterface,
    ):
        super().__init__()
        self.process_config_list: List[AutoProcessorConfig] = []
        self.adapter = adapter

    def preprocess_processor(
        self, processor_list: List[AutoProcessorConfig], model: nn.Module, device: DeviceType = DeviceType.NPU
    ):
        pass

    def add_processor(self, processor_cfg: AutoProcessorConfig, append: bool = True):
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
        processors: List[AutoSessionProcessor] = []
        for processor_config in config_list:
            processor = AutoSessionProcessor.from_config(model, processor_config, adapter)
            processors.append(processor)

        enable_kv_cache(model, adapter, processors)

        process_unit: List[GeneratedProcessUnit] = []
        for processor in processors:
            process_unit.append(
                GeneratedProcessUnit(
                    model=model,
                    processor=processor,
                    pipeline_interface=self.adapter,
                    calib_data=calib_data,
                    data_recorder=data_recorder,
                )
            )
        return process_unit

    def run(
        self,
        model: nn.Module = None,
        calib_data: Optional[List[Any]] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ):
        # to avoid oom
        _ = get_input_datas(self.adapter, calib_data, device)

        if model is None:
            get_logger().info('Start to init model')
            model = self.adapter.init_model(device=device)
            get_logger().info('Init model success')

        processor_list = self.process_config_list.copy()
        self.preprocess_processor(processor_list, model, device)

        data_recorder = DataUnit(None, None)
        process_unit = self.build_process_unit(
            processor_list, model=model, adapter=self.adapter, calib_data=calib_data, data_recorder=data_recorder
        )

        self.generated_schedule(process_unit, data_recorder)

    @torch.no_grad()
    def generated_schedule(self, process_unit: List[GeneratedProcessUnit], data_recorder: DataUnit):
        """
        使用生成式前向函数运行模型。

        该函数从处理单元列表中提取输入数据，每个处理单元包含一个处理器和可选的输入数据。
        函数会交错调度各个处理单元，即先调度unit1的第一步，然后调度unit2的第一步，以此类推。
        当某个unit完成时，会将其从调度队列中删除。

        参数:
            process_unit: 处理单元列表，每个元素包含一个处理器和可选的输入数据
        """

        try:
            get_logger().info("Scheduler %s unit: %s", len(process_unit), process_unit)

            unit_list = list(process_unit)

            _ = [unit.pre_run() for unit in process_unit]
            _ = [unit.init_generators() for unit in process_unit]

            while unit_list:
                remove_unit = []

                for unit in unit_list:
                    if not unit.make_progress():
                        remove_unit.append(unit)

                _ = [unit_list.remove(unit) for unit in remove_unit]

                data_recorder.input = data_recorder.output
                data_recorder.output = None

            _ = [unit.post_run() for unit in process_unit]
        except BaseException:
            for unit in process_unit:
                cleanup = getattr(unit.processor, "abort_lossless_check", None)
                if not callable(cleanup):
                    continue
                try:
                    cleanup()
                except BaseException as cleanup_error:
                    get_logger().error(
                        "Failed to abort processor %s after generated schedule error (%s)",
                        unit.processor,
                        type(cleanup_error).__name__,
                    )
            raise


def enable_kv_cache(model: nn.Module, adapter: PipelineInterface, processors: List[AutoSessionProcessor]):
    need_kv_cache = any((processor.need_kv_cache() for processor in processors))
    get_logger().info("KV cache requirement: %s", need_kv_cache)
    try:
        adapter.enable_kv_cache(model, need_kv_cache)
    except (AttributeError, NotImplementedError, ToDoError) as e:
        if need_kv_cache:
            raise UnsupportedError("Some processors need enable kv cache, but failed to enable kv cache") from e
        else:
            get_logger().warning("Failed to disable kv cache, this may cause more memory usage")


def get_input_datas(
    model_adapter: PipelineInterface,
    calib_data: Optional[List[Any]] = None,
    dev_type: DeviceType = DeviceType.NPU,
):
    return load_cached(key=KEY_DATA_LOADER, init_func=_get_input_datas, args=(model_adapter, calib_data, dev_type))


def _get_input_datas(
    model_adapter: PipelineInterface,
    calib_data: Optional[List[Any]] = None,
    dev_type: DeviceType = DeviceType.NPU,
) -> DataLoader:
    get_logger().info('Start to handle dataset')
    input_datas = model_adapter.handle_dataset(calib_data, dev_type)
    data_loader = _create_dataloader(input_datas, 0, 1, 1)
    get_logger().info('Handle dataset success')
    return data_loader


def _create_dataloader(dataset, rank, world_size, batch_size) -> DataLoader:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return dataset
    sampler = DistributedSampler(dataset, shuffle=False)
    return DataLoader(dataset, sampler=sampler, batch_size=None)
