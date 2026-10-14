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
import pytest
import torch
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.simulate_tp import ParallelLinearCol
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig, Calibrator


class Fake_dag:
    def get_allreduce_linear(self):
        return ['l1']

    def get_kv_linears(self):
        return [], 0

    def get_llm_network_pattern_auto(self):
        return [], [], [], [], []


def fake_extract_dag(self, model):
    fake_dag = Fake_dag()
    return fake_dag


setattr(Calibrator, 'extract_dag', fake_extract_dag)


class Model_Config:
    def __init__(self, torch_dtype):
        self.torch_dtype = torch_dtype


class OneLinearTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = None
        self.dtype = torch.float16
        self.device = torch.device('cpu')
        self.l1 = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        x = self.l1(x)

        return x


class Config:
    def __init__(self, tp_size=4, enable_communication_quant=False, enable_per_device_quant=False):
        self.tp_size = tp_size
        self.enable_communication_quant = enable_communication_quant
        self.enable_per_device_quant = enable_per_device_quant


class TestParallelLinear:
    def test_tp_size(self):
        with torch.no_grad():
            linear = torch.nn.Linear(in_features=15, out_features=15, bias=False)
            parallel_linear = ParallelLinearCol()
            cfg = Config()
            with pytest.raises(ValueError):
                parallel_linear.set_param(linear, 'test_linear', cfg)

    def test_tp_size1(self):
        with torch.no_grad():
            linear = torch.nn.Linear(in_features=16, out_features=16, bias=False)
            parallel_linear = ParallelLinearCol()
            cfg = Config(tp_size=1)
            parallel_linear.set_param(linear, 'linear_tp1', cfg=cfg)

            hidden_states = torch.randn(1, 16)
            linear_out = linear(hidden_states)
            parallel_linear_out = parallel_linear(hidden_states)
            cos_sim = torch.cosine_similarity(linear_out, parallel_linear_out)
            assert cos_sim > 0.9

    def test_tp_size2(self):
        with torch.no_grad():
            linear = torch.nn.Linear(in_features=16, out_features=16, bias=False)
            parallel_linear = ParallelLinearCol()
            cfg = Config(tp_size=2)
            parallel_linear.set_param(linear, 'linear_tp2', cfg=cfg)

            hidden_states = torch.randn(1, 16)
            linear_out = linear(hidden_states)
            parallel_linear_out = parallel_linear(hidden_states)
            cos_sim = torch.cosine_similarity(linear_out, parallel_linear_out)
            assert cos_sim > 0.9

    def test_linear_nobias(self):
        with torch.no_grad():
            linear = torch.nn.Linear(in_features=16, out_features=32, bias=False)
            parallel_linear = ParallelLinearCol()
            cfg = Config()
            parallel_linear.set_param(linear, 'linear_nobias', cfg=cfg)

            hidden_states = torch.randn(1, 16)
            linear_out = linear(hidden_states)
            parallel_linear_out = parallel_linear(hidden_states)
            cos_sim = torch.cosine_similarity(linear_out, parallel_linear_out)
            assert cos_sim > 0.9

    def test_linear_bias(self):
        with torch.no_grad():
            linear = torch.nn.Linear(in_features=16, out_features=32, bias=True)
            parallel_linear = ParallelLinearCol()
            cfg = Config()
            parallel_linear.set_param(linear, 'linear_bias', cfg=cfg)

            hidden_states = torch.randn(1, 16)
            linear_out = linear(hidden_states)
            parallel_linear_out = parallel_linear(hidden_states)
            cos_sim = torch.cosine_similarity(linear_out, parallel_linear_out)
            assert cos_sim > 0.9

    def test_communication_quant(self):
        with torch.no_grad():
            linear = torch.nn.Linear(in_features=16, out_features=32, bias=True)
            parallel_linear = ParallelLinearCol()
            cfg = Config(enable_communication_quant=True)
            parallel_linear.set_param(linear, 'linear_bias', cfg=cfg)

            hidden_states = torch.randn(1, 16)
            linear_out = linear(hidden_states)
            parallel_linear_out = parallel_linear(hidden_states)
            cos_sim = torch.cosine_similarity(linear_out, parallel_linear_out)
            assert cos_sim > 0.9

    def test_per_device_quant(self):
        with torch.no_grad():
            linear = torch.nn.Linear(in_features=16, out_features=32, bias=True)
            parallel_linear = ParallelLinearCol()
            cfg = Config(enable_communication_quant=True, enable_per_device_quant=True)
            parallel_linear.set_param(linear, 'linear_bias', cfg=cfg)

            hidden_states = torch.randn(1, 16)
            linear_out = linear(hidden_states)
            parallel_linear_out = parallel_linear(hidden_states)
            cos_sim = torch.cosine_similarity(linear_out, parallel_linear_out)
            assert cos_sim > 0.9

    def test_quantconfig_enable_communication_quant(self):
        with pytest.raises(ValueError):
            q = QuantConfig()
            q.simulate_tp(tp_size=1, enable_communication_quant=True)

        with pytest.raises(ValueError):
            q = QuantConfig()
            q.simulate_tp(tp_size=1, enable_per_device_quant=True)

        with pytest.raises(ValueError):
            q = QuantConfig()
            q.simulate_tp(tp_size=1.5)

    def test_fp_model_allreduce_quant(self):
        TEST_SAVE_PATH = "simulate_tp_save_path"
        q = QuantConfig(w_bit=8, a_bit=8, disable_names=['l1'])
        q.simulate_tp(tp_size=4, enable_per_device_quant=True, enable_communication_quant=True)
        model = OneLinearTorchModel()
        model.config = Model_Config(torch_dtype=torch.float16)
        calibrator = Calibrator(model, q, calib_data=[[torch.Tensor(1, 8)]])
        calibrator.run()
        calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
        if os.path.exists(TEST_SAVE_PATH):
            shutil.rmtree(TEST_SAVE_PATH)

    def test_w8a8_model_allreduce_quant(self):
        TEST_SAVE_PATH = "simulate_tp_save_path"
        q = QuantConfig(w_bit=8, a_bit=8, disable_names=[])
        q.simulate_tp(tp_size=4, enable_per_device_quant=False, enable_communication_quant=True)
        model = OneLinearTorchModel()
        model.config = Model_Config(torch_dtype=torch.float16)
        calibrator = Calibrator(model, q, calib_data=[[torch.Tensor(1, 8)]])
        calibrator.run()
        calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
        if os.path.exists(TEST_SAVE_PATH):
            shutil.rmtree(TEST_SAVE_PATH)
