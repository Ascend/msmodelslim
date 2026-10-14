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
# pylint: disable=no-name-in-module,redefined-outer-name,unspecified-encoding

import os
import json
import random
from PIL import Image  # noqa: F401
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode  # noqa: F401
import torchvision.transforms as T  # noqa: F401
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from example.InternVL2.internvl2_utils import load_image
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig


def get_textvqa_calibration(textvqa_path, calib_num=30, get_all_calib=False):
    val_json = 'textvqa_val.jsonl'
    calibration_dataset = []

    with open(os.path.join(textvqa_path, val_json), "r") as file:
        for line in file:
            line_dict = json.loads(line.strip())
            line_dict['text'] = line_dict['question']
            line_dict['image_url'] = line_dict['image']
            calibration_dataset.append(line_dict)

    if not get_all_calib:
        calibration_dataset = random.sample(calibration_dataset, calib_num)

    return calibration_dataset


def get_tokenized_data(tokenizer, inputs, dtype=torch.float16):
    tokenization_data = []
    for _, input_item in tqdm(enumerate(inputs), total=len(inputs), desc="Tokenizing data"):
        question = input_item.get('text')
        query = (
            '<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|>'
            + '<|im_start|>user\n<image>\n'
            + question
            + '<|im_end|><|im_start|>assistant\n'
        )
        image_url = input_item['image_url']
        pixel_values = load_image(image_url, max_num=12).to('npu').to(dtype)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        tokenization_data.append([tokenizer, pixel_values, query, generation_config])
    return tokenization_data


CPU = 'cpu'
NPU = 'npu'
model_path = f"{os.environ['PROJECT_PATH']}/resource/mllm/InternVL2-8B"
calib_images = f"{os.environ['PROJECT_PATH']}/resource/mllm/textvqa_val"
calib_num = 1
save_directory = f"{os.environ['PROJECT_PATH']}/output/mllm_ptq_internvl2_8b"
part_file_size = None
w_bit = 8
a_bit = 8
device_type = NPU

# 1.加载模型
device_map = CPU if device_type == CPU else "auto"
config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
dtype = config.torch_dtype
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=dtype,
    local_files_only=True,
    low_cpu_mem_usage=True,
    device_map=device_map,
    use_safetensors=True,
    trust_remote_code=True,
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, use_fast=False)

# 2.调用chat接口
model.forward = model.chat

# 3.设置回退层
disable_names = []
vision_name = []

llm_name = ["language_model.output", "mlp1.1", "mlp1.3"]
for i in range(config.vision_config.num_hidden_layers):
    vision_name.extend([f"vision_model.encoder.layers.{i}.mlp.fc2"])
for i in range(config.llm_config.num_hidden_layers):
    llm_name.extend([f"language_model.model.layers.{i}.feed_forward.w2"])

disable_names.extend(vision_name)
disable_names.extend(llm_name)

# 4.配置校准集
calibration_dataset = get_textvqa_calibration(calib_images, calib_num)
calib_data = get_tokenized_data(tokenizer, calibration_dataset, dtype=dtype)

# 5.异常值抑制
anti_config = AntiOutlierConfig(
    w_bit=w_bit, a_bit=a_bit, anti_method="m2", dev_type=device_type, dev_id=model.device.index
)

anti_outlier = AntiOutlier(model, calib_data=calib_data[:1], cfg=anti_config)
anti_outlier.process()

# 6.模型量化
quant_config = QuantConfig(
    w_bit=w_bit,
    a_bit=a_bit,
    disable_names=disable_names,
    dev_type=device_type,
    dev_id=model.device.index,
    act_method=1,
    mm_tensor=False,
)

calibrator = Calibrator(model, quant_config, calib_data=calib_data[:1], disable_level='L0')
calibrator.run()

# 7.保存权重
calibrator.save(save_directory, save_type=["safe_tensor"], part_file_size=part_file_size)
