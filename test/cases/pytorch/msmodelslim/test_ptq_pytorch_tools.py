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
# pylint: disable=consider-using-from-import,redefined-outer-name,super-with-arguments,ungrouped-imports

import os

import onnx
import torch
import pytest
import numpy as np
import torch.nn as nn

from ascend_utils.common.security.path import set_file_stat
from msmodelslim.pytorch.quant.ptq_tools import Calibrator, QuantConfig

ONNX_MODEL = "fake_test_ptq_pytorch_tools"
ONNX_QUANT_MODEL_PATH = "fake_test_ptq_pytorch_tools_quant.onnx"
CONV2_NAME = "conv2"
LINEAR_NAME = "linear1"


class DummyConv(nn.Module):
    def __init__(self):
        super(DummyConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv2 = nn.Conv2d(4, 4, 3)
        self.conv3 = nn.Conv2d(4, 4, 3)
        self.linear1 = nn.Linear(in_features=26, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=10)
        self.batch_norm = nn.BatchNorm2d(4)

    def forward(self, x):
        x = self.conv1(x)  # [1, 4, 30, 30]
        x = self.conv2(x)  # [1, 4, 28, 28]
        x = self.batch_norm(x)
        x = self.conv3(x)  # [1, 4, 26, 26]
        x = self.linear1(x)  # [1, 4, 26, 128]
        x = self.linear2(x)  # [1, 4, 26, 10]
        return x


@pytest.fixture(scope="function")
def generate_model():
    model = DummyConv()
    yield model


def get_onnx_params():
    onnx_model = onnx.load(ONNX_QUANT_MODEL_PATH)
    graph = onnx_model.graph

    offset_list = []
    scale_list = []

    for node in graph.node:
        if node.op_type == 'AscendQuant':
            for attr in node.attribute:
                if attr.name == 'offset':
                    current_offset = attr.f
                    offset_list.append(current_offset)
                elif attr.name == 'scale':
                    current_scale = attr.f
                    scale_list.append(current_scale)
    quant_weight_dict = {}
    conv_weight_names = ['conv2.weight_0', 'conv3.weight_1', 'onnx::MatMul_20_2']
    for initializer in onnx_model.graph.initializer:
        for name in conv_weight_names:
            if initializer.name == name:
                weight_tensor = initializer
                weight_value = np.frombuffer(weight_tensor.raw_data, dtype=np.int8)
                weight_value = weight_value.reshape(weight_tensor.dims)
                quant_weight_dict[name] = weight_value
    return offset_list, scale_list, quant_weight_dict


def test_pytorch_ptq_given_model_then_pass(generate_model):
    config = QuantConfig(disable_names=[], input_shape=[1, 3, 32, 32], amp_num=0)
    calib = Calibrator(generate_model, config)
    calib.run()
    input_scale, input_offset, _, _, quant_weight = calib.get_quant_params()
    calib.export_quant_onnx(ONNX_MODEL, "./", ["input.1"])
    set_file_stat(ONNX_QUANT_MODEL_PATH, stat_mode="440")
    offset_list, scale_list, quant_weight_dict = get_onnx_params()
    if scale_list[0] == 0 or scale_list[2] == 0:
        raise ValueError("scale can not be zero, please check.")
    # conv
    assert input_offset.get(CONV2_NAME).item() == offset_list[0]
    assert round(input_scale.get(CONV2_NAME).item(), 5) == round(1 / scale_list[0], 5)
    assert torch.equal(
        quant_weight.get(CONV2_NAME).int(), torch.from_numpy(quant_weight_dict.get('conv2.weight_0')).int()
    )
    # linear
    assert input_offset.get(LINEAR_NAME).item() == offset_list[2]
    assert round(input_scale.get(LINEAR_NAME).item(), 5) == round(1 / scale_list[2], 5)
    assert torch.equal(
        quant_weight.get(LINEAR_NAME).int().T, torch.from_numpy(quant_weight_dict.get('onnx::MatMul_20_2')).int()
    )
    if os.path.exists(ONNX_MODEL):
        os.remove(ONNX_MODEL)

    if os.path.exists(ONNX_QUANT_MODEL_PATH):
        os.remove(ONNX_QUANT_MODEL_PATH)


def test_pytorch_ptq_config():
    with pytest.raises(TypeError):
        config = QuantConfig(w_bit=4)
    with pytest.raises(TypeError):
        config = QuantConfig(a_bit=4)
    with pytest.raises(TypeError, match="disable_names must be list"):
        config = QuantConfig(disable_names="linear")
    with pytest.raises(ValueError):
        config = QuantConfig(input_shape=[1, 3, 224, 224], act_quant=-1)
    with pytest.raises(ValueError):
        config = QuantConfig(input_shape=[1, 3, 224, 224], a_signed=-1)
    with pytest.raises(ValueError):
        config = QuantConfig(input_shape=[1, 3, 224, 224], w_sym=-1)
    with pytest.raises(ValueError):
        config = QuantConfig(input_shape=[1, 3, 224, 224], a_sym=-1)
    with pytest.raises(ValueError):
        config = QuantConfig(input_shape=[1, 3, 224, 224], sigma=-1)
    with pytest.raises(ValueError):
        config = QuantConfig(input_shape=[1, 3, 224, 224], sigma=101)
    with pytest.raises(ValueError):
        config = QuantConfig(input_shape=[1, 3, 224, 224], amp_num=-1)
    with pytest.raises(ValueError, match="act_method is invalid"):
        config = QuantConfig(input_shape=[1, 3, 224, 224], act_method=-1)
    with pytest.raises(ValueError, match="quant_mode is invalid"):
        config = QuantConfig(input_shape=[1, 3, 224, 224], quant_mode=-1)  # noqa: F841


def test_pytorch_ptq_config_input_shape():
    with pytest.raises(TypeError, match="input_shape must be list"):
        config = QuantConfig(input_shape="1,2,3,4")
    with pytest.raises(ValueError):
        config = QuantConfig(input_shape=[224, 224])  # noqa: F841


def test_pytorch_ptq_config_keep_acc():
    with pytest.raises(ValueError, match="admm should be in keep_accuracy"):
        config = QuantConfig(keep_acc={'easy_quant': [False, 1000], 'round_opt': False})
    with pytest.raises(ValueError, match="round_opt should be in keep_accuracy"):
        config = QuantConfig(keep_acc={'admm': [False, 1000], 'easy_quant': [False, 1000]})
    with pytest.raises(ValueError, match="easy_quant should be in keep_accuracy"):
        config = QuantConfig(keep_acc={'admm': [False, 1000], 'round_opt': False})  # noqa: F841


def test_pytorch_ptq_dataset_type(generate_model):
    config = QuantConfig(disable_names=[], input_shape=[1, 3, 32, 32], amp_num=0)
    with pytest.raises(ValueError, match="calib_data should be list of tensors"):
        calib = Calibrator(generate_model, config, calib_data=np.ones((1)))  # noqa: F841


def test_pytorch_ptq_dataset_inputshape(generate_model):
    config = QuantConfig(disable_names=[], input_shape=[0, 0, -1, -1], amp_num=0)
    with pytest.raises(RuntimeError):
        calib = Calibrator(generate_model, config, calib_data=None)  # noqa: F841


def test_pytorch_ptq_param_act_quant(generate_model):
    config = QuantConfig(disable_names=[], act_quant=False, input_shape=[1, 3, 32, 32], amp_num=0)
    calib = Calibrator(generate_model, config, calib_data=None)
    calib.run()


def test_pytorch_ptq_param_disable_names(generate_model):
    config = QuantConfig(disable_names=['conv1'], act_quant=False, input_shape=[1, 3, 32, 32], amp_num=0)
    calib = Calibrator(generate_model, config, calib_data=None)
    calib.run()


def test_pytorch_ptq_param_amp_num(generate_model):
    config = QuantConfig(disable_names=[], input_shape=[1, 3, 32, 32], amp_num=1)
    calib = Calibrator(generate_model, config)
    calib.run()


def test_pytorch_ptq_func_export_param(generate_model):
    config = QuantConfig(disable_names=[], input_shape=[1, 3, 32, 32], amp_num=1)
    calib = Calibrator(generate_model, config)
    calib.run()
    import datetime
    import pytz
    import shutil

    timestamp = datetime.datetime.now(pytz.utc).strftime("%Y%m%d_%H%M%S")
    folder_name = f"test_ut_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    calib.export_param(folder_name)
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)


def test_pytorch_ptq_001(generate_model):
    config = QuantConfig(disable_names=[], input_shape=[3, 32, 32], amp_num=1)
    calib = Calibrator(generate_model, config)
    calib.run()


def test_pytorch_ptq_002(generate_model):
    config = QuantConfig(disable_names=[], input_shape=[3, 32, 32], amp_num=0)
    calib = Calibrator(generate_model, config)
    calib.run()


def test_pytorch_ptq_003(generate_model):
    config = QuantConfig(disable_names=[], act_quant=False, input_shape=[1, 3, 32, 32], amp_num=100)
    calib = Calibrator(generate_model, config, calib_data=None)
    calib.run()
