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
# pylint: disable=redefined-outer-name

import os
import shutil
import stat
import sys
from collections import namedtuple

import numpy as np
import onnx
import pytest
import torch
from resources.sample_net_torch import TestAscendQuantModel
from resources.sample_net_torch import TestOnnxQuantModel

from msmodelslim.onnx.post_training_quant import QuantConfig, run_quantize
from msmodelslim.onnx.post_training_quant.label_free.preprocess_func import (
    preprocess_func_imagenet,
    preprocess_func_coco,
)

fake_acl = namedtuple('acl', ['get_soc_name'])(lambda: "Ascend310P3")
sys.modules['acl'] = fake_acl


@pytest.fixture(autouse=True)
def set_random_seed():
    """为所有测试设置随机数种子以确保结果可重复"""
    torch.manual_seed(0)
    np.random.seed(0)
    try:
        import torch_npu

        torch_npu.npu.manual_seed(0)
        torch_npu.npu.manual_seed_all(0)
    except Exception:
        pass
    yield


@pytest.fixture()
def onnx_model():
    model = TestOnnxQuantModel()
    onnx_model_path = "./test.onnx"
    input_x = torch.randn((1, 3, 32, 32))
    torch.onnx.export(model, input_x, onnx_model_path, input_names=['input'], output_names=['output'])
    os.chmod(onnx_model_path, stat.S_IRUSR | stat.S_IWUSR)
    yield onnx_model_path


@pytest.fixture()
def ascend_quant_model():
    model = TestAscendQuantModel()
    ascend_quant_model_path = "./ascend_quant_model.onnx"
    input_x = torch.randn((1, 1, 32, 32))
    torch.onnx.export(model, input_x, ascend_quant_model_path, input_names=['input'], output_names=['output'])
    os.chmod(ascend_quant_model_path, stat.S_IRUSR | stat.S_IWUSR)
    yield ascend_quant_model_path


@pytest.fixture()
def onnx_model_dynamic():
    model = TestOnnxQuantModel()
    onnx_model_path = "./test_dynamic.onnx"
    input_x = torch.randn((1, 3, 224, 224))
    dynamic_axes = {'input': {0: 'batch', 2: 'h', 3: 'w'}, 'output': {0: 'batch'}}
    torch.onnx.export(
        model, input_x, onnx_model_path, input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes
    )
    os.chmod(onnx_model_path, stat.S_IRUSR | stat.S_IWUSR)
    yield onnx_model_path
    os.remove(onnx_model_path)


def test_reduce_redundant_quant_node_given_onnx_when_label_free_then_pass(ascend_quant_model):
    quant_config = QuantConfig(calib_data=[np.random.randn(1, 1, 32, 32).astype(np.float32)])
    quant_model_path = "./test_ascend_quant_model.onnx"
    run_quantize(ascend_quant_model, quant_model_path, quant_config)
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


def test_run_quantize_given_onnx_when_label_free_then_pass(onnx_model):
    quant_config = QuantConfig(calib_data=[[np.random.randn(1, 3, 32, 32).astype(np.float32)]], amp_num=1)
    quant_model_path = "./test_quant_lf.onnx"
    run_quantize(onnx_model, quant_model_path, quant_config)
    assert os.path.exists(quant_model_path)
    os.remove(quant_model_path)


def test_run_quantize_given_onnx_with_dynamic_shape_label_free_amp_0(onnx_model_dynamic):
    calib_data = [np.random.random((1, 3, 640, 640)).astype('float32')]
    quant_config = QuantConfig(
        quant_mode=1, calib_data=[calib_data], amp_num=0, input_shape=[[1, 3, 640, 640]], is_dynamic_shape=True
    )
    quant_model_path = "./test_squant_dym.onnx"

    run_quantize(onnx_model_dynamic, quant_model_path, quant_config)
    assert os.path.exists(quant_model_path)
    os.remove(quant_model_path)


def test_run_quantize_given_onnx_with_dynamic_shape_label_free_amp_5(onnx_model_dynamic):
    calib_data = [np.random.random((1, 3, 640, 640)).astype('float32')]
    quant_config = QuantConfig(
        quant_mode=1, calib_data=[calib_data], amp_num=5, input_shape=[[1, 3, 640, 640]], is_dynamic_shape=True
    )
    quant_model_path = "./test_squant_dym.onnx"

    run_quantize(onnx_model_dynamic, quant_model_path, quant_config)
    assert os.path.exists(quant_model_path)
    os.remove(quant_model_path)


def test_label_free_preprocess_func():
    TEST_SAVE_PATH = "imagenet_or_coco_dataset"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True, mode=0o750)
    from PIL import Image, ImageDraw

    # 创建一个新的空白图像
    image = Image.new("RGB", (500, 500), (255, 255, 255))
    # 在图像上绘制一些内容
    draw = ImageDraw.Draw(image)
    draw.text((100, 100), "Hello, World!", fill=(0, 0, 0))
    # 保存图像为.jpg 文件
    image.save(f"{TEST_SAVE_PATH}/example.jpg", "JPEG")
    os.chmod(f"{TEST_SAVE_PATH}/example.jpg", 0o750)

    calib_data = preprocess_func_imagenet(TEST_SAVE_PATH)
    quant_config = QuantConfig(calib_data=calib_data, amp_num=5)

    calib_data = preprocess_func_coco(TEST_SAVE_PATH)
    quant_config = QuantConfig(calib_data=calib_data, amp_num=5)  # noqa: F841

    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)
