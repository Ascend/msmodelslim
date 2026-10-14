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
from resources.sample_net_torch import (
    TwoLinearTorchModel,
    GroupLinearTorchModel,
    AttentionTorchModel,
    MOEModel,
    ThreeLinearTorchModel_for_Sparse,
    AttentionTorchSophonModel,
)
import torch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig

setattr(AntiOutlier, 'init_dag', lambda *args, **kargs: None)
setattr(AntiOutlier, '_process', lambda *args, **kargs: None)


class Config:
    def __init__(self, torch_dtype):
        self.torch_dtype = torch_dtype


def test_llm_ptq():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(8, 8)]]

    quant_config = QuantConfig(w_bit=8, dev_type='cpu', act_method=3, pr=0.5, mm_tensor=False)
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_w8a16_nsym():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(8, 8)]]

    quant_config = QuantConfig(
        a_bit=16, w_bit=8, disable_names=[], dev_type='cpu', act_method=3, pr=1.0, w_sym=False, mm_tensor=False
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_w8a16_sym():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(8, 8)]]

    quant_config = QuantConfig(
        a_bit=16, w_bit=8, disable_names=[], dev_type='cpu', act_method=3, pr=1.0, w_sym=True, mm_tensor=False
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_gptq():
    TEST_SAVE_PATH = "automl_llm_ptq_gptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(8, 8)]]

    w_sym = False
    quant_config = QuantConfig(a_bit=16, w_bit=8, dev_type='cpu', w_sym=w_sym, mm_tensor=False, w_method='GPTQ')
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_hqq():
    TEST_SAVE_PATH = "automl_llm_ptq_hqq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    quant_config = QuantConfig(
        a_bit=16,
        w_bit=8,
        disable_names=[],
        dev_type='cpu',
        act_method=3,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
        w_method='HQQ',
    )
    calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_awq():
    TEST_SAVE_PATH = "automl_llm_ptq_awq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(8, 8)]]

    w_sym = False
    anti_config = AntiOutlierConfig(a_bit=16, w_bit=8, anti_method="m3", dev_type="cpu", w_sym=w_sym)
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()

    quant_config = QuantConfig(
        a_bit=16, w_bit=8, disable_names=[], dev_type='cpu', w_sym=w_sym, mm_tensor=False, w_method='MinMax'
    )
    calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_minmax():
    TEST_SAVE_PATH = "automl_llm_ptq_minmax_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    w_sym = False
    quant_config = QuantConfig(
        a_bit=16,
        w_bit=8,
        disable_names=[],
        dev_type='cpu',
        act_method=3,
        pr=1.0,
        w_sym=w_sym,
        mm_tensor=False,
        w_method='MinMax',
    )
    calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_lowbit_when_smooth_is_False():
    TEST_SAVE_PATH = "automl_llm_ptq_lowbit_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(8, 8)]]
    quant_config = QuantConfig(
        disable_names=[],
        do_smooth=False,
        is_lowbit=True,
        use_sigma=False,
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib)
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_w8a8_per_token():
    TEST_SAVE_PATH = "automl_llm_ptq_w8a8_per_token_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(8, 8)]]

    anti_config = AntiOutlierConfig(anti_method="m1", dev_type="cpu")
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        disable_names=[],
        dev_type='cpu',
        act_method=1,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=True,
    )
    calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_auto_optimize_w4a8():
    TEST_SAVE_PATH = "automl_llm_ptq_auto_optimize_w4a8_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = ThreeLinearTorchModel_for_Sparse()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(1, 256, 256)]]

    quant_config = QuantConfig(
        w_bit=4,
        a_bit=8,
        disable_names=[],
        dev_type='cpu',
        act_method=2,
        mm_tensor=False,
        do_smooth=False,
        is_lowbit=True,
        open_outlier=True,
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_kvcache():
    TEST_SAVE_PATH = "automl_llm_ptq_kvcache_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = AttentionTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.FloatTensor(32, 32)]]

    quant_config = QuantConfig(
        a_bit=8, w_bit=8, disable_names=[], act_method=1, pr=1.0, w_sym=True, mm_tensor=False, use_kvcache_quant=True
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_kvcache_assym():
    TEST_SAVE_PATH = "automl_llm_ptq_kvcache_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = AttentionTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.FloatTensor(32, 32)]]

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        disable_names=[],
        act_method=1,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
    ).kv_quant(kv_sym=False)
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_w8a16_kvcache():
    TEST_SAVE_PATH = "automl_llm_ptq_kvcache_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = AttentionTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.FloatTensor(32, 32)]]

    quant_config = QuantConfig(
        a_bit=16,
        w_bit=8,
        disable_names=[],
        act_method=1,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
    ).kv_quant(kv_sym=False)
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_fp16_kvcache():
    TEST_SAVE_PATH = "automl_llm_ptq_kvcache_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = AttentionTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.FloatTensor(32, 32)]]

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        disable_names=[],
        act_method=1,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
    ).kv_quant(kv_sym=False)
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L1000')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_per_group_gptq_w4a16():
    TEST_SAVE_PATH = "automl_llm_ptq_per_group_gptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = GroupLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(256, 256)]]

    w_sym = False
    quant_config = QuantConfig(
        a_bit=16,
        w_bit=4,
        disable_names=[],
        dev_type='cpu',
        w_sym=w_sym,
        mm_tensor=False,
        is_lowbit=True,
        open_outlier=False,
        w_method='GPTQ',
        group_size=128,
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_per_group_hqq_w4a16():
    TEST_SAVE_PATH = "automl_llm_ptq_per_group_hqq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = GroupLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    w_sym = True
    quant_config = QuantConfig(
        a_bit=16,
        w_bit=4,
        disable_names=[],
        dev_type='cpu',
        w_sym=w_sym,
        mm_tensor=False,
        is_lowbit=True,
        open_outlier=False,
        w_method='HQQ',
        group_size=128,
    )
    calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_per_group_gptq_w8a16():
    TEST_SAVE_PATH = "automl_llm_ptq_per_group_gptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = GroupLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(256, 256)]]

    w_sym = False
    quant_config = QuantConfig(
        a_bit=16,
        w_bit=8,
        disable_names=[],
        dev_type='cpu',
        w_sym=w_sym,
        mm_tensor=False,
        is_lowbit=True,
        open_outlier=False,
        w_method='GPTQ',
        group_size=128,
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_per_group_hqq_w8a16():
    TEST_SAVE_PATH = "automl_llm_ptq_per_group_hqq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = GroupLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    w_sym = True
    quant_config = QuantConfig(
        a_bit=16,
        w_bit=8,
        disable_names=[],
        dev_type='cpu',
        w_sym=w_sym,
        mm_tensor=False,
        is_lowbit=True,
        open_outlier=False,
        w_method='HQQ',
        group_size=128,
    )
    calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_per_group_minmax_w8a16():
    TEST_SAVE_PATH = "automl_llm_ptq_per_group_hqq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = GroupLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    w_sym = True
    quant_config = QuantConfig(
        a_bit=16,
        w_bit=8,
        disable_names=[],
        dev_type='cpu',
        w_sym=w_sym,
        mm_tensor=False,
        is_lowbit=True,
        open_outlier=False,
        w_method='MinMax',
        group_size=128,
    )
    calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_per_group_awq_w8a16():
    TEST_SAVE_PATH = "automl_llm_ptq_per_group_hqq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = GroupLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(256, 256)]]

    w_sym = True
    anti_config = AntiOutlierConfig(a_bit=16, w_bit=8, anti_method="m3", dev_type="cpu", w_sym=w_sym)
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()

    quant_config = QuantConfig(
        a_bit=16,
        w_bit=8,
        disable_names=[],
        dev_type='cpu',
        w_sym=w_sym,
        mm_tensor=False,
        is_lowbit=True,
        open_outlier=False,
        w_method='MinMax',
        group_size=128,
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_cbq():
    TEST_SAVE_PATH = "automl_llm_ptq_cbq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = TwoLinearTorchModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(8, 8)]]

    anti_config = AntiOutlierConfig(anti_method="m5", dev_type='cpu')
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config, norm_class_name="RMSNorm")
    anti_outlier.process()

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        disable_names=[],
        dev_type='cpu',  # dev_type="npu", dev_id=0  如果需要使用npu进行量化
        act_method=1,
        pr=1.0,
        mm_tensor=False,
    )

    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_W8A16_GPTQ_MOE():
    TEST_SAVE_PATH = "automl_llm_ptq_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = MOEModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(8, 32)]]

    quant_config = QuantConfig(
        a_bit=16,
        w_bit=8,
        disable_names=[],
        dev_type='cpu',
        pr=1.0,
        w_method="GPTQ",
        w_sym=False,
        mm_tensor=False,
        disable_last_linear=False,
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)
    assert model.expert2.w1.quant_weight.weight_scale is not None
    assert model.expert2.w2.quant_weight.weight_scale is not None
    assert model.expert2.w3.quant_weight.weight_scale is not None


def test_llm_ptq_pangu7b_w8a8_anti():
    TEST_SAVE_PATH = "automl_llm_ptq_pangu7b_w8a8_anti_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = AttentionTorchSophonModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(256, 256)]]

    anti_config = AntiOutlierConfig(anti_method="m5", dev_type='cpu')
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config, norm_class_name="SophonRMSNorm")
    anti_outlier.process()

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        disable_names=[],
        co_sparse=False,
        do_smooth=False,
        is_lowbit=False,
        dev_type='cpu',
        act_method=2,
        pr=1.0,
        mm_tensor=False,
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_pangu7b_w4a8_lowbit_anti():
    TEST_SAVE_PATH = "automl_llm_ptq_pangu7b_w4a8_lowbit_anti_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = AttentionTorchSophonModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(256, 256)]]

    anti_config = AntiOutlierConfig(anti_method="m5", dev_type='cpu')
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config, norm_class_name="SophonRMSNorm")
    anti_outlier.process()

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=4,
        disable_names=[],
        co_sparse=False,
        do_smooth=False,
        is_lowbit=True,
        dev_type='cpu',
        act_method=2,
        pr=1.0,
        mm_tensor=False,
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_llm_ptq_pangu7b_w4a8_lowbit_smooth():
    TEST_SAVE_PATH = "automl_llm_ptq_pangu7b_w4a8_lowbit_smooth_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    model = AttentionTorchSophonModel()
    model.config = Config(torch_dtype=torch.float16)

    dataset_calib = [[torch.randn(256, 256)]]

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=4,
        disable_names=[],
        co_sparse=False,
        do_smooth=True,
        is_lowbit=True,
        dev_type='cpu',
        act_method=2,
        pr=1.0,
        mm_tensor=False,
    )
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
    calibrator.run()  # 执行PTQ量化校准
    calibrator.save(TEST_SAVE_PATH, save_type=["numpy", "safe_tensor"])
    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)
