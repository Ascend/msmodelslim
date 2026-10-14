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
# pylint: disable=no-name-in-module,unexpected-keyword-arg

import os
import logging

import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig

# 定义输出序列的最大长度
SEQ_LEN_OUT = 32

# 从环境变量中获取模型文件的路径
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/llama2_7b/"

calib_dataset = []

# 加载模型配置
config = AutoConfig.from_pretrained(LOAD_PATH, trust_remote_code=True, local_files_only=True)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH, trust_remote_code=True, local_files_only=True
)
# 加载模型，并指定数据类型为float32，然后移动到CPU
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=LOAD_PATH, torch_dtype=torch.float32, trust_remote_code=True, local_files_only=True
).cpu()

# 设置分词器的填充侧和填充符号
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# 定义校准列表，用于量化校准的数据集
calib_list = [
    "Where is the capital of China?",
    "Please make a poem:",
    "I want to learn python, how should I learn it?",
    "Please help me write a job report on large model inference optimization:",
    "What are the most worth visiting scenic spots in China?",
]

# 定义函数，用于创建校准数据集
for data in calib_list:
    # 对每个校准数据进行编码，并转换为PyTorch张量
    inputs = tokenizer([data], return_tensors='pt').to('cpu')
    logging.info("Encoded input: %s", inputs)  # 使用logging记录输入信息
    calib_dataset.append([inputs.data['input_ids'], inputs.data['attention_mask']])

# 创建校准数据集
dataset_calib = calib_dataset

# 手动设置回退所有的down层
disable_names = []
num_layers = config.num_hidden_layers
# 遍历所有层，禁用特定的层
disable_idx_lst = list(range(num_layers))
for layer_index in disable_idx_lst:
    down_proj_name = "model.layers.{}.mlp.down_proj".format(layer_index)
    disable_names.append(down_proj_name)

# 启动flex smooth功能，配置抗异常值处理
anti_config = AntiOutlierConfig(anti_method="m6", dev_type='cpu')
anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

# 配置量化参数
quant_config = QuantConfig(
    a_bit=8,  # 激活函数量化位宽
    w_bit=8,  # 权重量化位宽
    disable_names=disable_names,  # 回退的层
    disable_last_linear=True,  # 禁用最后的线性层
    dev_type='cpu',  # 设备类型
    dev_id=model.device.index,  # 设备ID
    act_method=1,  # 激活函数量化方法
    pr=1.0,  # 量化比例
    mm_tensor=False,  # 每通道量化
    is_dynamic=True,  # 每个token量化
    w_method="KMeans",  # 启动kmeans和per tiling量化
    lut_len=16,  # 聚类类数
)

# 创建校准器，并运行校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level={"threshold": 1.0})
calibrator.run()

# 保存量化后的模型权重
calibrator.save(
    f"{os.environ['PROJECT_PATH']}/output/llm_ptq_ZS_3", save_type=["numpy", "safe_tensor"], part_file_size=1
)
# 打印保存量化权重成功的信息
logging.info('Save quant weight success!')
