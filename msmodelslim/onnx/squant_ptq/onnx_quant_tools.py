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
import time

import onnx
import onnxruntime
import numpy as np

from ascend_utils.common.security import (
    get_valid_read_path,
    get_valid_write_path,
    SafeWriteUmask,
    safe_delete_path_if_exists,
    check_type,
)
from ascend_utils.common import acl_inference
from msmodelslim.onnx.post_training_quant.util import check_input_data
from msmodelslim.onnx.squant_ptq.quant_deploy import (
    quantize_model_deploy,
    QuantParamsDict,
)
from msmodelslim.onnx.squant_ptq import QuantConfig

from msmodelslim.onnx.squant_ptq.onnx_ptq_kia.quant_funcs_onnx import (  # pylint: disable=no-name-in-module
    merge_nodes,
    onnx_label_free_calib,
    onnx_data_free_calib,
    onnx_observer,
    om_observer,
    get_np_datatype,
    onnx_amp,
    disable_first_layer,
    disable_last_layer,
)
from msmodelslim.onnx.squant_ptq.aok.tool_main import aok_export
from msmodelslim import logger


class OnnxCalibrator:
    """OnnxCalibrator for post-training quantization."""

    def __init__(self, input_model, cfg: QuantConfig, calib_data=None):
        check_type(cfg, QuantConfig, param_name="cfg")
        self.cfg = cfg
        self.logger = logger
        self.input_path = get_valid_read_path(input_model, extensions=".onnx")
        self.device_id = cfg.device_id
        self.use_onnx = cfg.use_onnx
        self.num_input = cfg.num_input

        if not self.use_onnx or cfg.graph_optimize_level > 0:
            acl_inference.init_acl(device_id=self.device_id)

        self.aok_configuration(cfg)

        if not self.use_onnx and self.num_input == 0:
            raise ValueError("Input number should not be zero")

        with SafeWriteUmask():
            self._load_model()

        self.graph = self.model.graph
        self.input_shapes = []
        self.input_types = []
        self._get_input_of_model(cfg.input_shape, cfg.is_dynamic_shape)
        self.quant_param_ops = cfg.quant_param_ops
        self.atc_input_shape = cfg.atc_input_shape

        if calib_data is None:
            calib_data = []
        self.calib_data = self._get_calib_data(calib_data, cfg)
        if not isinstance(calib_data, list):
            raise ValueError("calib_data should be a list of tensor data")
        if self.use_onnx:
            self.graph_nodes = onnx_observer(self.model, self.calib_data, self.quant_param_ops)
        else:
            timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            temp_onnx_path = "temp_model_" + timestamp
            temp_om_model = os.path.splitext(os.path.basename(self.input_path))[0] + timestamp
            self.temp_onnx_path = get_valid_write_path(temp_onnx_path, extensions=None)
            self.temp_om_model = get_valid_write_path(temp_om_model, extensions=None)
            os.makedirs(name=self.temp_onnx_path, mode=0o750, exist_ok=True)
            self.graph_nodes = om_observer(
                self.model,
                self.calib_data,
                self.quant_param_ops,
                self.atc_input_shape,
                (self.temp_onnx_path, self.temp_om_model),
            )
        self._set_quant()
        if cfg.disable_first_layer:
            disable_first_layer(self.graph, self.graph_nodes, self.logger)
        if cfg.disable_last_layer:
            disable_last_layer(self.graph, self.graph_nodes, self.logger)

    def __del__(self):
        if not self.use_onnx or self.graph_optimize_level > 0:
            acl_inference.release_acl(self.device_id)

    def aok_configuration(self, cfg):
        self.initial_input_path = self.input_path
        self.graph_optimize_level = cfg.graph_optimize_level
        if self.graph_optimize_level > 0:
            with SafeWriteUmask():
                timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
                temp_onnx_model = os.path.splitext(os.path.basename(self.input_path))[0] + timestamp + ".onnx"
                folder_path, _ = os.path.split(self.input_path)
                self.float_aok_output = os.path.join(folder_path, temp_onnx_model)
                self.float_aok_output = get_valid_write_path(self.float_aok_output)
                self.input_path = aok_export(self.input_path, cfg, self.float_aok_output)

    def run(self):
        try:
            self._run()
        except Exception as e:
            raise RuntimeError("Please check your config, model and input!", e) from e

    def export_quant_onnx(self, save_path, fuse_add=True, use_external=False):
        """Export quantized onnx for deployment"""
        output_model_path = get_valid_write_path(save_path, extensions=".onnx")
        check_type(fuse_add, bool, param_name="fuse_add")
        check_type(use_external, bool, param_name="use_external")
        if use_external:
            external_path = os.path.dirname(output_model_path)
        quant_params = self._get_quant_params()

        quantized_weight_namd = []
        for item in self.graph.node:
            if item.op_type in ("Conv", "Gemm", "MatMul"):
                weight_name = item.input[1]
                if weight_name in quant_params.weight_scale and quant_params.weight_scale[weight_name] is not None:
                    quantized_weight_namd.append(weight_name)

        quantize_model_deploy(self.graph, quantized_weight_namd, quant_params, fuse_add)
        merge_nodes(self.graph)

        quant_model = onnx.helper.make_model(self.model.graph, opset_imports=self.ori_opset)
        with SafeWriteUmask():
            if use_external:
                onnx.save(
                    quant_model,
                    output_model_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=False,
                    location=external_path,
                    size_threshold=1024,
                    convert_attribute=True,
                )
            else:
                onnx.save(quant_model, output_model_path)
            logger.info("Quantification ended and onnx is stored in %r ", output_model_path)

            if self.graph_optimize_level > 1:
                with SafeWriteUmask():
                    aok_export(output_model_path, self.cfg, output_model_path)

        if not self.use_onnx:
            safe_delete_path_if_exists(self.temp_om_model + ".om")

        if self.graph_optimize_level > 0:
            safe_delete_path_if_exists(self.initial_input_path.replace(".onnx", ".om"))
            safe_delete_path_if_exists(self.input_path)
            safe_delete_path_if_exists(self.input_path.replace(".onnx", ".om"))
            safe_delete_path_if_exists(output_model_path.replace(".onnx", ".om"))

    def _run(self):
        """Calibration"""
        self.logger.info("Calibration start!")
        for idx in self.graph_nodes:
            node = self.graph_nodes[idx]
            if "features" in node.input_tensors:
                weight = node.input_tensors["weight"]
                features = node.input_tensors["features"]
                if self.cfg.quant_mode == 0 and self.cfg.act_method == 0:
                    # Data-free calibration
                    onnx_data_free_calib(node, weight, self.cfg, logger=self.logger)
                    onnx_data_free_calib(node, features[0], self.cfg, True, self.logger)
                elif self.cfg.quant_mode == 1:
                    # Label-free calibration, min-max ema for activation
                    onnx_label_free_calib(node, weight, self.cfg, logger=self.logger)
                    onnx_label_free_calib(node, features, self.cfg, True, logger=self.logger)
                else:
                    raise ValueError("Unsupported quant_mode")
                if node.input_scale == 0:
                    self.logger.info("automatic disable this node %r", node.name)
                    node.is_quant = False

        self.logger.info("Calibration end!")

        if self.cfg.amp_num > 0:
            self.logger.info("AMP start! AMP num: %d", self.cfg.amp_num)
            self._amp()
            self.logger.info("AMP end!")

    def _load_model(self):
        self.model = onnx.load(self.input_path)

        self.ori_opset = self.model.opset_import
        opt_model_path = self.input_path + ".optimized.onnx"
        if self.use_onnx:
            sess_option = onnxruntime.SessionOptions()
            sess_option.optimized_model_filepath = opt_model_path
            sess_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC

        if self.cfg.is_optimize_graph and self.use_onnx:
            try:
                _ = onnxruntime.InferenceSession(self.input_path, sess_option, providers=["CPUExecutionProvider"])
            except Exception as e:
                self.logger.error("ONNX Runtime Model Optimization Failed, Use Original Model: %s", e)
                self.model = onnx.load(self.input_path)
            else:
                self.model = onnx.load(opt_model_path)
                if os.path.exists(opt_model_path):
                    os.remove(opt_model_path)

        ir_version = 4
        if self.model.ir_version < ir_version:
            self.logger.warning("Model with ir_version below 4 requires to include initializer in graph input")
            return

        inputs = self.model.graph.input
        name_to_input = {}
        for graph_input in inputs:
            name_to_input[graph_input.name] = graph_input

        for initializer in self.model.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])

    def _get_input_of_model(self, cfg_input_shape, cfg_is_dynamic_shape):
        if self.use_onnx:
            null_session = onnxruntime.InferenceSession(self.model.SerializeToString())
            num_input = len(null_session.get_inputs())
        else:
            num_input = self.num_input
        if cfg_input_shape:
            if num_input != len(cfg_input_shape):
                raise ValueError(f"Invalid input shape: {cfg_input_shape}, please check it. ")
        for i in range(num_input):
            temp_shape = self.graph.input[i].type.tensor_type.shape.dim
            temp_type = str(self.graph.input[i].type.tensor_type.elem_type)
            input_shape = [x.dim_value for x in temp_shape]
            if cfg_is_dynamic_shape:
                if cfg_input_shape[i]:
                    self.input_shapes.append(cfg_input_shape[i])
                else:
                    raise ValueError(
                        "For model with dynamic shape, please specify the shape of inputto construct calib data"
                    )
            else:
                if input_shape[0] in [-1, 0] or isinstance(input_shape[0], str):
                    input_shape[0] = 1
                self.input_shapes.append(input_shape)
            self.input_types.append(temp_type)

    def _set_quant(self):
        # first/last node disable, disable nodes in cfg, etc.
        for idx in self.graph_nodes:
            node = self.graph_nodes[idx]
            if node.name in self.cfg.disable_names:
                node.is_quant = False

    def _check_calib_data(self, calib_data, model_inputs, quant_cfg=None):
        num_input = len(model_inputs)
        if not isinstance(calib_data, list):
            raise ValueError("calib_data should be list of tensors")
        # looping of samples
        for index, data_list in enumerate(calib_data):
            if len(data_list) != num_input:
                logger.warning(
                    "The number of %r data records in the calib_data is not equal to the input of the model.",
                    index,
                )
                continue
            # looping of inputs in single sample
            for input_data, input_x in zip(data_list, model_inputs):
                if not check_input_data(input_x, input_data, quant_cfg):
                    raise ValueError(f"The {index} data records in calib_data is not valid")

    def _get_calib_data(self, calib_data, quant_cfg=None):
        if self.use_onnx:
            null_session = onnxruntime.InferenceSession(self.model.SerializeToString())
            model_inputs = null_session.get_inputs()
            num_input = len(model_inputs)
        else:
            num_input = self.num_input

        if calib_data:
            if self.use_onnx:
                self._check_calib_data(calib_data, model_inputs, quant_cfg)
            return calib_data
        else:
            if len(self.input_shapes) != len(self.input_types):
                raise ValueError("len of input_shapes not equal len of input_types")
            calib_data = []
            for i in range(num_input):
                random_data = np.array(
                    np.random.random(self.input_shapes[i]),
                    dtype=get_np_datatype()[self.input_types[i]],
                )
                calib_data.append(random_data)
            return [calib_data]

    def _amp(self):
        dequant_index = onnx_amp(self.graph_nodes, self.cfg.amp_num)
        for value in dequant_index:
            self.logger.info("disable this node: %r", self.graph_nodes[value].name)
            self.graph_nodes[value].is_quant = False

    def _get_quant_params(self) -> QuantParamsDict:
        input_scale = {}
        input_offset = {}
        weight_scale = {}
        weight_offset = {}
        quant_weight = {}
        bias_name = {}
        node_name = {}
        for _, node in self.graph_nodes.items():
            if not node.is_quant:
                continue
            input_scale[node.inputs[1]] = np.array(node.input_scale)
            input_offset[node.inputs[1]] = np.array(node.input_offset)
            weight_scale[node.inputs[1]] = np.array(node.weight_scale)
            weight_offset[node.inputs[1]] = np.array(node.weight_offset)
            quant_weight[node.inputs[1]] = np.array(node.int_weight)

            if len(node.inputs) > 2:
                bias_name[node.inputs[1]] = node.inputs[2]
            else:
                bias_name[node.inputs[1]] = None
            node_name[node.inputs[1]] = node.name
        return QuantParamsDict(
            input_scale,
            input_offset,
            weight_scale,
            weight_offset,
            quant_weight,
            bias_name,
            node_name,
        )
