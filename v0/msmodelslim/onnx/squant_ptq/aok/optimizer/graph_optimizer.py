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
import os
from enum import Enum
from typing import Optional, Any
from copy import copy
from typing import List, Tuple
import numpy as np
import onnx
from onnx import GraphProto, ModelProto, shape_inference
# G.IMP.02 is deliberately violated because we don't want to enumerate all the many architectures here.
# We accept that names of architectures classes must not be preceded with '_'
from ascend_utils.common.security import get_valid_read_path, get_valid_write_path, SafeWriteUmask
from msmodelslim.onnx.squant_ptq.aok.optimizer import architectures
from msmodelslim.onnx.squant_ptq.aok.optimizer.optimizations import AbstractOptimization
from msmodelslim.onnx.squant_ptq.aok.utils.utilities import rename_nodes, check_topology_sorting, simplify_model


class _InferenceRunInfo:

    def __init__(self,
                 model_name: str,
                 latency: float,
                 output: np.array) -> None:
        self.model_name = model_name
        self.latency = latency
        self.output = output


class _CheckOptStatus(Enum):
    NotApplicable = 0
    Applicable = 1
    BeatsBaseline = 2


class GraphOptimizer:

    def __init__(self, logger):
        self._logger = logger
        self._opset_version = None
        self._ir_version = None
        self._soc_version = None
        self._simplify = False
        self._check_model = False
        self._check_output_threshold = None
        self._arch = None
        self._debug = None
        self._auto_quant_enabled = False

        # runner parameters
        self._runner = None


    @staticmethod
    def delete_model(folder_path: str, model_name: str, delete_onnx: bool = True) -> None:
        if delete_onnx:
            onnx_model_path = os.path.join(folder_path, f'{model_name}.onnx')
            onnx_model_path = get_valid_read_path(onnx_model_path, is_dir=False)
            try:
                os.unlink(onnx_model_path)
            except FileNotFoundError:
                pass
        om_model_path = os.path.join(folder_path, f'{model_name}.om')
        om_model_path = get_valid_read_path(om_model_path, is_dir=False)
        try:
            os.unlink(om_model_path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _get_architectures() -> dict:
        archs = {}
        arch_classes = [
            architectures.DefaultArchitecture,
            architectures.DummyArchitecture,
            architectures.BERTArchitecture,
            architectures.RoBERTaArchitecture,
            architectures.EfficientNetArchitecture,
            architectures.MobileNetV2Architecture,
            architectures.MobileNetV3Architecture,
            architectures.ShuffleNetV2Architecture,
            architectures.SWINArchitecture,
            architectures.SeResnetArchitecture,
            architectures.DenseNetArchitecture,
            architectures.U2NetArchitecture,
            architectures.YoloV5Architecture,
            architectures.YoloV5s1Architecture,
            architectures.YoloV7Architecture,
        ]
        for arch_class in arch_classes:
            arch = arch_class()
            archs[arch.get_name()] = arch
        return archs

    def delete_model_and_its_variations(self,
                                        folder_path: str,
                                        model_name: str,
                                        delete_onnx: bool = True) -> None:
        GraphOptimizer.delete_model(folder_path, model_name, delete_onnx)
        if self._auto_quant_enabled:
            model_name = model_name + 'q'
            GraphOptimizer.delete_model(folder_path, model_name, delete_onnx)

    def set_opset_version(self, op_version: int) -> 'GraphOptimizer':
        self._opset_version = op_version
        return self

    def set_ir_version(self, ir_version: int) -> 'GraphOptimizer':
        self._ir_version = ir_version
        return self

    def set_soc_version(self, soc_version: int) -> 'GraphOptimizer':
        self._soc_version = soc_version
        return self

    def set_simplify(self, simplify: bool) -> 'GraphOptimizer':
        self._simplify = simplify
        return self

    def set_check_model(self, check: bool) -> 'GraphOptimizer':
        self._check_model = check
        return self

    def set_check_output_threshold(self, threshold: float) -> 'GraphOptimizer':
        self._check_output_threshold = threshold
        return self

    def set_architecture(self, arch: str) -> 'GraphOptimizer':
        self._arch = arch
        return self

    def set_debug(self, debug: str) -> 'GraphOptimizer':
        self._debug = debug
        return self

    def set_runner(self, runner: Any) -> 'GraphOptimizer':
        self._runner = runner
        return self

    def set_auto_quantization(self, is_enabled: bool) -> 'GraphOptimizer':
        self._auto_quant_enabled = is_enabled
        return self


    def assemble_model(self,
                       graph: GraphProto,
                       graph_name: str) -> ModelProto:
        opt_graph = onnx.helper.make_graph(
            graph.node,
            graph_name,
            graph.input,
            graph.output,
            graph.initializer
        )
        opt_model = onnx.helper.make_model(opt_graph, producer_name='opt_model')
        opt_model = rename_nodes(opt_model,
                                 op_version=self._opset_version,
                                 ir_version=self._ir_version,
                                 logger=self._logger)
        return opt_model

    def apply_all_optimizations(self,
                                model_path: str,
                                optimizations: [AbstractOptimization],
                                postfix: str = 'o',
                                debug: bool = False,
                                aok_model_path: str = None) -> Tuple[List[Any], str]:
        opt_list = []

        folder_path, model_name_ext = os.path.split(model_path)
        model_name, ext = os.path.splitext(model_name_ext)
        if ext in ['', '.']:
            ext = '.onnx'
            model_path = os.path.join(folder_path, model_name + ext)
            model_path = get_valid_read_path(model_path, is_dir=False)
        model = onnx.load(model_path)
        model = shape_inference.infer_shapes(model)
        graph = model.graph

        for opt in optimizations:
            is_applicable = opt.apply(
                model_graph=graph,
                op_version=self._opset_version,
                debug=debug
            )
            if is_applicable:
                opt_list.append(opt)

        optimized_model_name = f'{model_name}_{postfix}'
        if aok_model_path:
            optimized_model_path = aok_model_path
        else:
            optimized_model_path = os.path.join(folder_path, f'{optimized_model_name}.onnx')
        optimized_model_path = get_valid_write_path(optimized_model_path)
        with SafeWriteUmask():
            self.__save_optimized_model(graph, f'{graph.name}_{postfix}', optimized_model_path)
        if self._debug:
            check_topology_sorting(graph, self._logger)
        return opt_list, optimized_model_path

    def collect_inference_run_info(self, model_path: str) -> _InferenceRunInfo:
        folder_path, model_name_ext = os.path.split(model_path)
        model_name, ext = os.path.splitext(model_name_ext)
        if ext in ['', '.']:
            ext = '.onnx'
            model_path = os.path.join(folder_path, model_name + ext)
            model_path = get_valid_read_path(model_path, is_dir=False)

        if self._debug:
            self._logger.debug(f'Measuring inference time {model_path}')
        try:
            latency = self._runner.run(model_path=model_path)
            if self._debug:
                self._logger.debug(f'Inference time of {model_path} is {latency}')
        except Exception as e:
            if self._debug:
                self._logger.debug(f'\nException occurred when trying to measure inference time of {model_path}: {e}')
            latency = np.inf
        if latency == np.inf:
            self._logger.error(f'\nError: failed to measure inference time of the {model_path} model')
            if not self._debug:
                raise RuntimeError()
        try:
            baseline_output = None if self._check_output_threshold is None \
                else self.__check_output(model_path)
        except Exception as e:
            if self._debug:
                self._logger.debug(f'\nException occurred when trying to check output of {model_path}: {e}')
            baseline_output = None

        baseline = _InferenceRunInfo(model_name, latency, baseline_output)
        return baseline

    def search_optimizations(self,
                             baseline_model_path: str,
                             opt_filter_mask: int,
                             shut_down_structures: list,
                             baseline: _InferenceRunInfo) -> Tuple[List[Any], List[Any], Optional[_InferenceRunInfo]]:
        folder_path, _ = os.path.split(baseline_model_path)
        all_optimizations = self.find_optimizations(opt_filter_mask)
        all_optimizations_cls_name = [i.__class__.__name__ for i in all_optimizations]
        for structure in shut_down_structures:
            if structure not in all_optimizations_cls_name:
                raise ValueError(f'{structure} not in all_optimizations: {all_optimizations_cls_name}')
        optimizations = list(set(all_optimizations) - set(shut_down_structures))
        if self._runner is not None:
            if self._auto_quant_enabled and 'Ascend' not in self._soc_version:
                self._logger.info("Current platform not support Quantization")
                baseline.latency = -1.0
            return self.__search_optimizations_from_list(
                folder_path,
                baseline,
                optimizations
            )
        else:
            self._logger.info("Search can't be performed because Runner is not defined")
            try:
                opts, _ = self.apply_all_optimizations(
                    baseline_model_path,
                    optimizations,
                )
            except Exception as e:
                raise Exception("Error from apply_all_optimizations function.", e) from e
            ret_val = ([], opts, None)
            return ret_val

    def find_optimizations(self, opt_filter_mask: int) -> [AbstractOptimization]:
        archs = GraphOptimizer._get_architectures()
        arch = archs.get(self._arch, archs.get('default'))
        return arch.get_optimizations(opt_filter_mask, logger=self._logger)

    def __search_optimizations_from_list(self,
                                         folder_path: str,
                                         baseline: _InferenceRunInfo,
                                         optimizations: [AbstractOptimization]
                                         ) -> Tuple[List[Any], List[Any], Optional[_InferenceRunInfo]]:
        optimizations_cur = copy(optimizations)
        best_optimization = None
        best_model_name = None
        best_model = None
        best_opt_list, app_opt_list = [], []

        opt_idx = 0
        if self._debug:
            self._logger.debug(f'checking {len(optimizations_cur)} optimizations of {baseline.model_name}')

        def _delete_old_best_model_if(new_best_model_name: str) -> None:
            if best_model_name is not None:
                self.delete_model_and_its_variations(folder_path, best_model_name)
                if self._debug:
                    self._logger.debug(f'Deleted {best_model_name} because {new_best_model_name} is better')

        while opt_idx < len(optimizations_cur):
            opt = optimizations_cur[opt_idx]
            status, new_baseline = self.__check_optimization(opt, folder_path, baseline)
            if status == _CheckOptStatus.NotApplicable:
                optimizations_cur.pop(opt_idx)
            elif status == _CheckOptStatus.Applicable:
                app_opt_list.append(opt)
                opt_idx += 1
            elif status == _CheckOptStatus.BeatsBaseline:
                app_opt_list.append(opt)
                _delete_old_best_model_if(new_baseline.model_name)
                best_optimization = opt
                best_model_name = new_baseline.model_name
                baseline.latency = new_baseline.latency
                opt_idx += 1
            else:
                raise ValueError(f'Unexpected optimization status: {status}')

        if best_optimization is None:
            if self._debug:
                self._logger.debug(f'No effective optimization was found, best model is still {baseline.model_name}')
        else:
            best_opt_list.append(best_optimization)
            optimizations.remove(best_optimization)
            new_baseline = _InferenceRunInfo(best_model_name, baseline.latency, baseline.output)
            best_opt_list_next, app_opt_list_next, best_subsequent = self.__search_optimizations_from_list(
                folder_path, new_baseline, optimizations
            )
            best_opt_list.extend(best_opt_list_next)
            app_opt_list.extend(app_opt_list_next)
            if best_subsequent is not None:
                self.delete_model_and_its_variations(folder_path, best_model_name)
                if self._debug:
                    self._logger.debug(f'Deleted {best_model_name} because {best_subsequent.model_name} is better')
                best_model = best_subsequent

        app_opt_list = list(set(app_opt_list))
        best_model = best_model if best_model is not None else baseline
        return best_opt_list, app_opt_list, best_model

    def __check_optimization(self,
                             opt: AbstractOptimization,
                             folder_path: str,
                             baseline: _InferenceRunInfo) -> (_CheckOptStatus, _InferenceRunInfo):
        baseline_model_path = os.path.join(folder_path, f'{baseline.model_name}.onnx')
        baseline_latency = baseline.latency
        optimized_model_name = f'{baseline.model_name}-{opt.get_simple_name()}'
        model = onnx.load(baseline_model_path)
        model = shape_inference.infer_shapes(model)
        graph = model.graph
        optimized_graph_name = f'{graph.name}_{opt.get_simple_name()}'
        if self._debug:
            self._logger.debug(f'Checking {optimized_model_name}')
        is_applicable = opt.apply(
            model_graph=graph,
            op_version=self._opset_version,
            debug=self._debug
        )
        if not is_applicable:
            if self._debug:
                self._logger.debug(f'{optimized_model_name} is not applicable')
            return _CheckOptStatus.NotApplicable, baseline

        optimized_model_path = os.path.join(folder_path, f'{optimized_model_name}.onnx')
        optimized_model_path = get_valid_write_path(optimized_model_path)
        with SafeWriteUmask():
            self.__save_optimized_model(graph, optimized_graph_name, optimized_model_path)
        if self._debug:
            self._logger.debug(f'Optimized and saved as {optimized_model_path}')
            check_topology_sorting(graph, self._logger)
        if self._check_output_threshold is not None and baseline.output is not None:
            optimized_model_path = get_valid_read_path(optimized_model_path)
            mse = self.__calculate_output_mse(optimized_model_path, baseline)
            if mse > self._check_output_threshold:
                self._logger.error(f'{optimized_model_path} does not pass output test. MSE error: {mse}')
                self.delete_model_and_its_variations(folder_path, optimized_model_name)
                # Keep it for future, maybe it will pass the test in conjunction with other optimizations
                return _CheckOptStatus.Applicable, baseline
        optimized_model_path_inf = optimized_model_path
        inference_info = self.collect_inference_run_info(model_path=optimized_model_path_inf)
        latency = inference_info.latency
        self._logger.info(f'Inference time of {optimized_model_name} is {latency}')
        if latency < baseline_latency:
            self.delete_model_and_its_variations(folder_path, optimized_model_name, False)
            if self._debug:
                self._logger.debug(f'Best optimization is now {optimized_model_name}')
            new_baseline = _InferenceRunInfo(optimized_model_name, latency, baseline.output)
            return _CheckOptStatus.BeatsBaseline, new_baseline

        self.delete_model_and_its_variations(
            folder_path, optimized_model_name, latency != np.inf or not self._debug
        )
        if self._debug:
            self._logger.debug(f'Deleted {optimized_model_path} because it is not the best one')
        return _CheckOptStatus.Applicable, baseline

    def __calculate_output_mse(self, optimized_model_path: str, baseline: _InferenceRunInfo) -> float:
        try:
            output = self.__check_output(optimized_model_path)
        except Exception:
            output = None
        mse = np.inf if output is None else np.sum(np.square(output[0] - baseline.output[0]))
        return mse

    def __save_optimized_model(self,
                               graph: GraphProto,
                               graph_name: str,
                               output_path: str) -> None:
        opt_model = self.assemble_model(graph, graph_name)
        if self._simplify:
            opt_model = simplify_model(opt_model,
                                       op_version=self._opset_version,
                                       ir_version=self._ir_version,
                                       logger=self._logger)
        if self._check_model:
            onnx.checker.check_model(opt_model)
        opt_model = shape_inference.infer_shapes(opt_model)
        onnx.save_model(opt_model, output_path)

    def __check_output(self, model_path: str) -> np.array:
        import onnxruntime as ort
        try:
            inf_session = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        except Exception as e:
            self._logger.error(e)
            raise
        input_name = inf_session.get_inputs()
        output_name = inf_session.get_outputs()
        inputs = {}
        for inp in input_name:
            dtype = inp.type[7:-1]
            dtype = 'float32' if dtype == 'float' else dtype
            inputs[inp.name] = np.ones(inp.shape, dtype=dtype)
        try:
            pred_onnx = inf_session.run([output_name[0].name], inputs)
        except Exception as e:
            self._logger.error(e)
            raise
        return pred_onnx[0]
