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
from ascend_utils.common.security import get_valid_read_path, check_type, check_int, check_element_type

QUANT_MODE_LIST = [0, 1]  # 1: label-free, 0: data-free
CALIB_METHOD_LIST = [0, 1, 2]  # 0: min-max, 1: percentile, 2: entropy


class QuantConfig:
    """
    Quantized configuration for label-free and data-free quantization.
    Args:
        quant_mode: quantized mode, 0: data-free, 1: label-free.
        is_signed_quant: symbol quantization or not for activations, True: Int8, False: UInt8.
        is_per_channel: per_channel or per_tensor for weights. True: per_channel, False: per_tensor.
        calib_data: [[numpy.ndarray, numpy.ndarray, ...], ...]
        calib_method: calibrate method for activations, 0: min-max, 1: percentile, 2: entropy
        quantize_nodes: list of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        exclude_nodes: list of nodes names to exclude. The nodes in this list will be excluded from quantization
        when it is not None.
        amp_num: number of nodes to be rolled back.
        is_optimize_graph: Optimize graph or not.
    """
    def __init__(self, quant_mode=1,
                 is_signed_quant=True,
                 is_per_channel=True,
                 calib_data=None,
                 calib_method=0,
                 quantize_nodes=None,
                 exclude_nodes=None,
                 amp_num=0,
                 is_optimize_graph=True,
                 is_quant_depthwise_conv=True,
                 input_shape=None,
                 is_dynamic_shape=False):
        self.quant_mode = quant_mode
        self.is_signed_quant = is_signed_quant
        self.is_per_channel = is_per_channel
        self.calib_data = calib_data or []
        self.calib_method = calib_method
        self.quantize_nodes = quantize_nodes or []
        self.exclude_nodes = exclude_nodes or []
        self.amp_num = amp_num
        self.is_optimize_graph = is_optimize_graph
        self.is_quant_depthwise_conv = is_quant_depthwise_conv
        self.input_shape = input_shape
        if self.input_shape is None:
            self.input_shape = []
        self.is_dynamic_shape = is_dynamic_shape
        self._check_params()

    def _check_params(self):
        if self.quant_mode not in QUANT_MODE_LIST:
            raise ValueError("quant_mode is invalid, please check it.")
        check_type(self.is_signed_quant, bool, param_name="is_signed_quant")
        check_type(self.is_per_channel, bool, param_name="is_per_channel")
        check_type(self.calib_data, list, param_name="calib_data")
        if self.calib_method not in CALIB_METHOD_LIST:
            raise ValueError("calib_method is invalid, please check it.")
        check_int(self.amp_num, min_value=0, param_name="amp_num")
        check_element_type(self.quantize_nodes, str, value_type=list, param_name="quantize_nodes")
        check_element_type(self.exclude_nodes, str, value_type=list, param_name="exclude_nodes")
        check_type(self.is_optimize_graph, bool, param_name="is_optimize_graph")
        check_type(self.is_quant_depthwise_conv, bool, param_name="is_quant_depthwise_conv")
        check_element_type(self.input_shape, list, value_type=list, param_name="input_shape")
        for item in self.input_shape:
            check_element_type(item, int, value_type=list, param_name="input_shape_item")
        check_type(self.is_dynamic_shape, bool, param_name="is_dynamic_shape")