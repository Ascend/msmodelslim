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

import os
import json
import shutil
from resources.sample_net_torch import TwoLinearTorchModel, GroupLinearTorchModel
import torch
from safetensors.torch import load_file

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import FakeQuantizeCalibrator


class Config:
    def __init__(self, torch_dtype):
        self.torch_dtype = torch_dtype


def test_llm_ptq_w8a8_when_dynamic_is_False():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path_w8a8"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)
    model_fp = TwoLinearTorchModel()

    dataset_calib = [[torch.randn(8, 8)]]

    quant_config = QuantConfig(w_bit=8, dev_type='cpu', act_method=3, pr=0.5, mm_tensor=False)
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])

    # 使用load_file()函数读取safetensor格式文件并将其解析为字典
    safetensor_dic = load_file(f"{TEST_SAVE_PATH}/quant_model_weight_w8a8.safetensors")
    # 使用json.load()函数读取文件并将其解析为字典
    with open(f"{TEST_SAVE_PATH}/quant_model_description_w8a8.json", 'r', encoding='utf-8') as file:
        description_dic = json.load(file)
    fakecalibrator = FakeQuantizeCalibrator(model_fp, None, "cpu", description_dic, safetensor_dic)
    model = fakecalibrator.model
    model.forward(dataset_calib[0][0].cpu())

    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_w4a16_per_group():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path_w4a16"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = GroupLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)
    model_fp = GroupLinearTorchModel()

    w_sym = True
    disable_names = []
    quant_config = QuantConfig(
        a_bit=16,
        w_bit=4,
        disable_names=disable_names,
        dev_type='cpu',
        w_sym=w_sym,
        mm_tensor=False,
        is_lowbit=True,
        open_outlier=False,
        group_size=64,
        w_method='MinMax',
    )
    calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])

    # 使用load_file()函数读取safetensor格式文件并将其解析为字典
    safetensor_dic = load_file(f"{TEST_SAVE_PATH}/quant_model_weight_w4a16.safetensors")
    # 使用json.load()函数读取文件并将其解析为字典
    with open(f"{TEST_SAVE_PATH}/quant_model_description_w4a16.json", 'r', encoding='utf-8') as file:
        description_dic = json.load(file)
    fakecalibrator = FakeQuantizeCalibrator(model_fp, None, "cpu", description_dic, safetensor_dic)
    model = fakecalibrator.model
    model.forward(torch.randn(256, 256).cpu())

    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_w8a8_when_dynamic_is_True():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path_w8a8_dynamic"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)
    model_fp = TwoLinearTorchModel()

    dataset_calib = [[torch.randn(8, 8, 8)]]

    quant_config = QuantConfig(w_bit=8, dev_type='cpu', act_method=3, pr=0.5, mm_tensor=False, is_dynamic=True)
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])

    # 使用load_file()函数读取safetensor格式文件并将其解析为字典
    safetensor_dic = load_file(f"{TEST_SAVE_PATH}/quant_model_weight_w8a8_dynamic.safetensors")
    # 使用json.load()函数读取文件并将其解析为字典
    with open(f"{TEST_SAVE_PATH}/quant_model_description_w8a8_dynamic.json", 'r', encoding='utf-8') as file:
        description_dic = json.load(file)
    fakecalibrator = FakeQuantizeCalibrator(model_fp, None, "cpu", description_dic, safetensor_dic)
    model = fakecalibrator.model
    model.forward(dataset_calib[0][0].cpu())

    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)
