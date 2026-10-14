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
# pylint: disable=no-name-in-module

import os
import argparse
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

CPU = 'cpu'
NPU = 'npu'
model_path = f"{os.environ['PROJECT_PATH']}/resource/mllm/Qwen2.5-VL-7B-Instruct"
calib_images = f"{os.environ['PROJECT_PATH']}/resource/mllm/coco_pic"
part_file_size = None
w_bit = 8
a_bit = 8
device_type = NPU

parser = argparse.ArgumentParser()
parser.add_argument("--anti_method", type=str, default="m2", choices=["m2", "m4"])
parser.add_argument("--save_directory", type=str, default=f"{os.environ['PROJECT_PATH']}/output/mllm_ptq_qwen2_5vl_m2")
args = parser.parse_args()

# 1.加载模型
device_map = CPU if device_type == CPU else "auto"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, device_map=device_map, trust_remote_code=True, torch_dtype="auto", local_files_only=True
).eval()
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

# 2.加载处理器
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

# 3.设置回退层
disable_names = []
vision_name = ['visual.merger.mlp.0', 'visual.merger.mlp.2']
llm_name = []
for i in range(config.vision_config.depth):
    vision_name.extend([f'visual.blocks.{i}.mlp.down_proj'])
for i in range(config.num_hidden_layers):
    llm_name.extend([f'model.layers.{i}.mlp.down_proj'])

disable_names.extend(vision_name)
disable_names.extend(llm_name)

# 4.加载校准集
images_list = os.listdir(calib_images)[:1]
calib_data = []
messageList = []
for i in images_list:
    image_path = os.path.join(calib_images, i)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "Please describe this picture in detail."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt').to(
        device_type
    )

    calib_data.append(
        [
            inputs['input_ids'],
            inputs['attention_mask'],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            inputs['pixel_values'],
            None,
            inputs['image_grid_thw'],
            None,
        ]
    )

# 5.异常值抑制
anti_config = AntiOutlierConfig(
    w_bit=w_bit, a_bit=a_bit, anti_method=args.anti_method, dev_type=device_type, dev_id=model.device.index
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
    act_method=2,
    mm_tensor=False,
)

calibrator = Calibrator(model, quant_config, calib_data=calib_data[:1], disable_level='L0')
calibrator.run()

# 7.保存权重
calibrator.save(args.save_directory, save_type=["safe_tensor"], part_file_size=part_file_size)
