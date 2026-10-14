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
import shutil
from resources.sample_net_torch import ThreeLinearTorchModel_for_Sparse
import torch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator as SparseQuantCalibrator
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig as SparseQuantConfig


class Config:
    def __init__(self, torch_dtype):
        self.torch_dtype = torch_dtype


def test_sparse_quant():
    TEST_SAVE_PATH = "automl_sparse_quant_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(1, 256, 256)]]

    w_bit = 4
    fraction = 0.011
    powerquant = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        disable_names=[],
        dev_type='cpu',
        act_method=3,
        pr=2.0,
        fraction=fraction,
        nonuniform=powerquant,
        mm_tensor=mm_tensor,
        co_sparse=True,
    )

    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_is_lowbit():
    TEST_SAVE_PATH = "automl_sparse_quant_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(1, 256, 256)]]

    w_bit = 4
    fraction = 0.011
    powerquant = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        disable_names=[],
        dev_type='cpu',
        act_method=1,
        fraction=fraction,
        nonuniform=powerquant,
        mm_tensor=mm_tensor,
        is_lowbit=True,
    )

    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L1')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_bias():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(1, 256, 256)]]

    w_bit = 4
    fraction = 0.011
    powerquant = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        disable_names=[],
        dev_type='cpu',
        act_method=3,
        pr=2.0,
        fraction=fraction,
        nonuniform=powerquant,
        mm_tensor=mm_tensor,
        co_sparse=True,
    )

    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_a_bit():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(1, 256, 256)]]

    w_bit = 4
    a_bit = 8
    fraction = 0.011
    powerquant = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        disable_names=[],
        dev_type='cpu',
        act_method=3,
        pr=2.0,
        fraction=fraction,
        nonuniform=powerquant,
        mm_tensor=mm_tensor,
        co_sparse=True,
    )

    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_act_method():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(1, 256, 256)]]

    w_bit = 4
    fraction = 0.011
    powerquant = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        disable_names=[],
        dev_type='cpu',
        act_method=2,
        pr=2.0,
        fraction=fraction,
        nonuniform=powerquant,
        mm_tensor=mm_tensor,
        co_sparse=True,
    )

    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_fraction():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(1, 256, 256)]]

    w_bit = 4
    fraction = 0.02
    powerquant = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        disable_names=[],
        dev_type='cpu',
        act_method=3,
        pr=2.0,
        fraction=fraction,
        nonuniform=powerquant,
        mm_tensor=mm_tensor,
        co_sparse=True,
    )

    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_mm_tensor():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(1, 256, 256)]]

    w_bit = 4
    fraction = 0.011
    powerquant = False
    mm_tensor = True

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        disable_names=[],
        dev_type='cpu',
        act_method=3,
        pr=2.0,
        fraction=fraction,
        nonuniform=powerquant,
        mm_tensor=mm_tensor,
        co_sparse=True,
    )

    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=False)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_sparse_quant_change_int_infer():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(1, 256, 256)]]

    w_bit = 4
    fraction = 0.011
    powerquant = False
    mm_tensor = False

    quant_config = SparseQuantConfig(
        w_bit=w_bit,
        disable_names=[],
        dev_type='cpu',
        act_method=3,
        pr=2.0,
        fraction=fraction,
        nonuniform=powerquant,
        mm_tensor=mm_tensor,
        co_sparse=True,
    )

    calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run(int_infer=True)
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)
