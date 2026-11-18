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
import argparse
import os
import statistics
from time import time
from abc import ABC, abstractmethod
import onnx
import numpy as np
from ascend_utils.common import acl_inference
from msmodelslim.onnx.squant_ptq.aok.utils.utilities import onnx2om, generate_model_inputs_for_onnxruntime


class Runner(ABC):

    @abstractmethod
    def run(self, model_path: str) -> float:
        pass

    @abstractmethod
    def supports_dynamic_batch_size(self) -> bool:
        pass


class OrtRunner(Runner):

    def __init__(self, logger, **kwargs) -> None:
        self._logger = logger
        self._batch_size = kwargs.get('batch_size', 1)
        self._iterations = kwargs.get('iterations', 1)
        self._runs = kwargs.get('runs', 1)

    def run(self, model_path: str) -> float:
        model = onnx.load(model_path)
        ort_inputs = generate_model_inputs_for_onnxruntime(model)

        try:
            import onnxruntime
        except ImportError as e:
            self._logger.error(e)
            raise
        try:
            ort_session = onnxruntime.InferenceSession(model_path)
        except Exception as e:
            self._logger.error(e)
            raise

        stats = []
        for _ in range(self._runs):
            time_consumed = 0
            for _ in range(self._iterations):
                start_time = time()
                try:
                    _outs = ort_session.run(None, ort_inputs)
                except Exception as e:
                    self._logger.error(e)
                    return np.inf
                end_time = time()
                time_consumed += (end_time - start_time)
            stats.append(time_consumed / self._iterations)
        return round(statistics.median(stats), 4)

    def supports_dynamic_batch_size(self) -> bool:
        return False


def acl_inference_mean(om_path: str, device_id: int, iterations: int, runs: int):
    latency_list = []
    mm = acl_inference.AclInference(om_path, device_id=device_id)
    inputs = [np.random.uniform(size=ii.shape).astype('float32') for ii in mm.get_inputs()]
    for _ in range(iterations * runs):
        _ = mm(inputs)
        latency = mm.get_execute_time()
        latency_list.append(latency)

    return float(np.mean(latency_list))


class AscendRunner(Runner):

    def __init__(self, logger, **kwargs) -> None:
        self._logger = logger
        self._batch_size = kwargs.get('batch_size', 1)
        self._iterations = kwargs.get('iterations', 1)
        self._runs = kwargs.get('runs', 1)
        self._device_id = kwargs.get('device_id', 0)
        self._soc_version = kwargs.get('soc_version', 1)
        self._om_method = kwargs.get('om_method', 'aoe')


    def run(self, model_path: str) -> float:
        original_model_path = model_path
        folder = '/'.join(original_model_path.split('/')[:-1]) + '/'
        if original_model_path.endswith('onnx'):
            original_model_name = os.path.splitext(original_model_path.split('/')[-1])[0]
            output_path = f'{folder}{original_model_name}.om'
            if os.path.exists(output_path):
                self._logger.info(f'Found file {output_path}')
            else:
                onnx2om(
                    original_model_path,
                    self._batch_size,
                    self._soc_version,
                    self._device_id,
                    self._om_method
                )
        else:
            raise ValueError('Only .onnx extensions are supported, '
                             f'but got model_path = {model_path}')

        return acl_inference_mean(output_path, self._device_id, self._iterations, self._runs)

    def supports_dynamic_batch_size(self) -> bool:
        return True


def create_runner(args: argparse.Namespace,
                  batch_size: int,
                  logger) -> Runner:
    if 'Ascend' not in args.soc_version:
        logger.info('Warning: ONNX runtime will be used for measuring inference time, because '
                     f'soc version must be specified to run on Ascend, but got soc_version={args.soc_version}')
        runner = OrtRunner(logger,
                           batch_size=batch_size,
                           iterations=args.iterations,
                           runs=args.runs)
    else:
        runner = AscendRunner(logger,
                              batch_size=batch_size,
                              iterations=args.iterations,
                              runs=args.runs,
                              device_id=args.device_id,
                              soc_version=args.soc_version,
                              om_method=args.om_method)
    return runner
