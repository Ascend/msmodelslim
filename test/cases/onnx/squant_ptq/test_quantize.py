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
# pylint: disable=modified-iterating-list

import os
import stat

import numpy as np
import onnx
import pytest
import torch
from resources.sample_net_torch import TestAscendQuantModel
from resources.sample_net_torch import TestOnnxQuantModel

from ascend_utils.common import acl_inference
from msmodelslim.onnx.squant_ptq import QuantConfig, OnnxCalibrator
from msmodelslim.onnx.squant_ptq.aok.optimizer.graph_optimizer import GraphOptimizer


class _InferenceRunInfo:
    def __init__(self, model_name: str, latency: float, output: np.array) -> None:
        self.model_name = model_name
        self.latency = latency
        self.output = output


setattr(acl_inference, 'init_acl', lambda *args, **kargs: None)
setattr(acl_inference, 'release_acl', lambda *args, **kargs: None)
setattr(GraphOptimizer, 'collect_inference_run_info', lambda *args, **kargs: _InferenceRunInfo('test', 1.8, None))

CALIBS = []
ONNX_MODEL_PATH = "test.onnx"
ASCEND_QUANT_MODEL_PATH = "ascend_quant_model.onnx"


@pytest.fixture(scope="module", autouse=True)
def init_testing_models():
    model = TestOnnxQuantModel()
    input_x = torch.randn((1, 3, 32, 32))
    torch.onnx.export(model, input_x, ONNX_MODEL_PATH, input_names=["input"], output_names=["output"])
    os.chmod(ONNX_MODEL_PATH, stat.S_IRUSR | stat.S_IWUSR)

    quant_model = TestAscendQuantModel()
    input_quant = torch.randn((1, 1, 32, 32))
    torch.onnx.export(quant_model, input_quant, ASCEND_QUANT_MODEL_PATH, input_names=["input"], output_names=["output"])
    os.chmod(ASCEND_QUANT_MODEL_PATH, stat.S_IRUSR | stat.S_IWUSR)

    yield

    if os.path.exists(ONNX_MODEL_PATH):
        os.remove(ONNX_MODEL_PATH)
    if os.path.exists(ASCEND_QUANT_MODEL_PATH):
        os.remove(ASCEND_QUANT_MODEL_PATH)
    for calib in CALIBS:
        del calib


def test_reduce_redundant_quant_node_given_onnx_when_data_free_then_pass():
    quant_model_path = "./test_ascend_squant_model.onnx"
    quant_config = QuantConfig(quant_mode=0, disable_first_layer=False, disable_last_layer=False)
    calib = OnnxCalibrator(ASCEND_QUANT_MODEL_PATH, quant_config)
    calib.run()
    calib.export_quant_onnx(quant_model_path)

    model = onnx.load(quant_model_path)
    nodes = model.graph.node
    ascend_quant_num = 0
    ascend_dequant_num = 0
    for node in nodes:
        if node.op_type == "AscendQuant":
            ascend_quant_num += 1
        elif node.op_type == "AscendDequant":
            ascend_dequant_num += 1
    assert ascend_dequant_num - ascend_quant_num == 1
    if os.path.exists(quant_model_path):
        os.remove(quant_model_path)
    CALIBS.append(calib)


def test_quant_node_given_onnx_when_amp_num_then_pass():
    quant_model_path = "./test_ascend_squant_model.onnx"
    quant_config = QuantConfig(quant_mode=0, amp_num=2, disable_first_layer=False, disable_last_layer=False)
    calib = OnnxCalibrator(ASCEND_QUANT_MODEL_PATH, quant_config)
    calib.run()
    calib.export_quant_onnx(quant_model_path)

    model = onnx.load(quant_model_path)
    nodes = model.graph.node
    ascend_quant_num = 0
    ascend_dequant_num = 0
    for node in nodes:
        if node.op_type == "AscendQuant":
            ascend_quant_num += 1
        elif node.op_type == "AscendDequant":
            ascend_dequant_num += 1

    assert ascend_dequant_num == 1
    assert ascend_quant_num == 1
    if os.path.exists(quant_model_path):
        os.remove(quant_model_path)
    CALIBS.append(calib)


def test_run_quantize_given_onnx_when_label_free_then_pass():
    quant_model_path = "./test_squant_lf.onnx"
    quant_config = QuantConfig(quant_mode=1, act_method=1)
    calib_data = [np.random.random((1, 3, 32, 32)).astype("float32")]

    calib = OnnxCalibrator(ONNX_MODEL_PATH, quant_config, calib_data=[calib_data])
    calib.run()
    calib.export_quant_onnx(quant_model_path)

    assert os.path.exists(quant_model_path)
    os.remove(quant_model_path)
    CALIBS.append(calib)


@pytest.mark.skip()
def test_run_quantize_given_onnx_when_graph_optimize_level_is_2_then_pass():
    quant_model_path = "./test_squant_df.onnx"
    quant_config = QuantConfig(graph_optimize_level=2)
    calib = OnnxCalibrator(ONNX_MODEL_PATH, quant_config)
    calib.run()
    calib.export_quant_onnx(quant_model_path)
    assert os.path.exists(quant_model_path)
    os.remove(quant_model_path)
    CALIBS.append(calib)


@pytest.mark.skip()
def test_run_quantize_given_onnx_when_graph_optimize_level_is_2_and_om_method_is_atc_then_pass():
    quant_model_path = "./test_squant_df.onnx"
    quant_config = QuantConfig(graph_optimize_level=2, om_method="atc")
    calib = OnnxCalibrator(ONNX_MODEL_PATH, quant_config)
    calib.run()
    calib.export_quant_onnx(quant_model_path)
    assert os.path.exists(quant_model_path)
    os.remove(quant_model_path)
    CALIBS.append(calib)


def test_valid_input_shape():
    """测试合法的 input_shape"""
    config = QuantConfig(input_shape=[[1, 2], [3, 4]])
    assert config.input_shape == [[1, 2], [3, 4]]


def test_subelements_contain_non_integer():
    """测试子列表包含非整数的情况"""
    with pytest.raises(ValueError) as context:
        QuantConfig(input_shape=[[1, "a"], [3, 4]])
    assert "Element in input_shape_item is invalid. Should be all int." in str(context.value)
