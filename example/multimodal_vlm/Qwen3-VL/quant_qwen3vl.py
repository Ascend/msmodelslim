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
# pylint: disable=no-member,no-name-in-module

import os
import argparse
import sys
import torch
import torch_npu  # noqa: F401
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoConfig

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(parent_directory)

from example.common.utils import cmd_bool  # noqa: E402
from example.common.security.path import get_valid_read_path, get_write_directory  # noqa: E402
from example.common.vlm_utils import VlmSafeGenerator, ModifyConfigParams, CopyTokenizerParams  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier  # noqa: E402
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402

CPU = "cpu"
NPU = "npu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--calib_images', type=str, default='../calibImages')
    parser.add_argument('--save_directory', type=str, default='')
    parser.add_argument('--part_file_size', type=int, default=None)
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--device_type', type=str, choices=[CPU, NPU], default=CPU)
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=False)
    parser.add_argument('--anti_method', type=str, choices=['m2'], default='m2')
    parser.add_argument('--act_method', type=int, default=2)
    parser.add_argument('--open_outlier', type=cmd_bool, default=True)
    parser.add_argument('--is_dynamic', type=cmd_bool, default=False)
    parser.add_argument('--is_lowbit', type=cmd_bool, default=False)
    parser.add_argument('--co_sparse', type=cmd_bool, default=False)
    parser.add_argument('--fraction', type=float, default=0.01)
    parser.add_argument('--use_sigma', type=cmd_bool, default=False)
    parser.add_argument('--sigma_factor', type=float, default=3.0)
    parser.add_argument('--torch_dtype', type=str, choices=['bf16', 'fp16'], default='bf16')
    args = parser.parse_args()

    args.model_path = get_valid_read_path(args.model_path, is_dir=True, check_user_stat=True)
    args.calib_images = get_valid_read_path(args.calib_images, is_dir=True, check_user_stat=True)
    args.save_directory = get_write_directory(args.save_directory, write_mode=0o750)

    # 1.加载模型
    device_map = CPU if args.device_type == CPU else "auto"
    torch_dtype = torch.bfloat16 if args.torch_dtype == "bf16" else torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(  # nosec B615
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="float16",
        local_files_only=True,
        device_map=device_map,
        attn_implementation="eager",
    ).eval()

    config = AutoConfig.from_pretrained(  # nosec B615
        args.model_path, trust_remote_code=args.trust_remote_code, local_files_only=True
    )

    # 2.加载处理器
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)  # nosec B615

    # 3.设置回退层
    disable_names = []
    merger_name = ['model.visual.merger.linear_fc1', 'model.visual.merger.linear_fc2']
    disable_names.extend(merger_name)
    for i in range(len(config.vision_config.deepstack_visual_indexes)):
        disable_names.append(f'model.visual.deepstack_merger_list.{i}.linear_fc1')
        disable_names.append(f'model.visual.deepstack_merger_list.{i}.linear_fc2')

    for i in range(config.vision_config.depth):
        disable_names.append(f"model.visual.blocks.{i}.attn.qkv")
        disable_names.append(f"model.visual.blocks.{i}.attn.proj")
        disable_names.append(f"model.visual.blocks.{i}.mlp.linear_fc1")
        disable_names.append(f"model.visual.blocks.{i}.mlp.linear_fc2")

    for i in range(config.text_config.num_hidden_layers):
        disable_names.append(f"model.language_model.layers.{i}.mlp.down_proj")

    # 4.加载校准集
    images_list = os.listdir(args.calib_images)
    messageList = []
    calib_data = []
    for i in images_list:
        image_path = os.path.join(args.calib_images, i)
        image_path = get_valid_read_path(image_path)
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
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(args.device_type)

        calib_data.append(
            [
                inputs["input_ids"],
                inputs["attention_mask"],
                None,
                None,
                None,
                None,
                inputs["pixel_values"],
                None,
                inputs["image_grid_thw"],
                None,
                None,
                0,
            ]
        )

    # 5.异常值抑制
    anti_config = AntiOutlierConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        anti_method=args.anti_method,
        dev_type=args.device_type,
        dev_id=model.device.index,
    )
    anti_outlier = AntiOutlier(model, calib_data=calib_data, cfg=anti_config)
    with torch.autocast(device_type=args.device_type, dtype=torch.float16):
        anti_outlier.process()

    # 6.量化校准
    quant_config = QuantConfig(
        a_bit=args.a_bit,
        w_bit=args.w_bit,
        disable_names=disable_names,
        dev_type=args.device_type,
        dev_id=model.device.index,
        act_method=args.act_method,
        mm_tensor=False,
        is_dynamic=args.is_dynamic,
        is_lowbit=args.is_lowbit,
        co_sparse=args.co_sparse,
        fraction=args.fraction,
        use_sigma=args.use_sigma,
        sigma_factor=args.sigma_factor,
    )

    with torch.autocast(device_type=args.device_type, dtype=torch.float16):
        calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level="L0")
        calibrator.run()

    # 7.保存权重
    calibrator.save(args.save_directory, save_type=["ascendV1"], part_file_size=args.part_file_size)

    quant_type = quant_config.model_quant_type.lower()
    checker = VlmSafeGenerator()
    auto_config = checker.get_config_from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    # 使用dataclass参数
    modify_params = ModifyConfigParams(
        model_dir=args.model_path,
        dest_dir=args.save_directory,
        torch_dtype="torch.bfloat16" if torch_dtype == torch.bfloat16 else "torch.float16",
        quantize_type=quant_type,
        args=args,
        quantize_config_parts=['text_config', 'vision_config'],
    )
    checker.modify_config(modify_params)

    copy_params = CopyTokenizerParams(model_dir=args.model_path, dest_dir=args.save_directory)
    checker.copy_tokenizer_files(copy_params)
