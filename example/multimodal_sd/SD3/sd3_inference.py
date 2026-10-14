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
# pylint: disable=redefined-outer-name,unspecified-encoding

import argparse
import os
import sys

import torch
from diffusers import StableDiffusion3Pipeline
from torch import nn
from tqdm import tqdm

cur_file_dir = os.path.dirname(os.path.abspath(__file__))
example_base_dir = os.path.abspath(os.path.join(cur_file_dir, "..", "..", ".."))
sys.path.append(example_base_dir)

from example.common.security.pytorch import safe_torch_load  # noqa: E402
from example.common.security.path import get_valid_read_path, get_write_directory  # noqa: E402
from msmodelslim.quant import quant_model, SessionConfig  # noqa: E402
from msmodelslim.quant import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig  # noqa: E402


def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="SD3 inference script")

    parser.add_argument("--sd3_model_path", type=str, required=True, help='Ckpt path of sd3 model')
    parser.add_argument("--prompt_path", type=str, default="./calib_prompts.txt", help="input prompt text path")
    parser.add_argument("--width", type=int, default=1024, help='Image size width')
    parser.add_argument("--height", type=int, default=1024, help='Image size height')
    parser.add_argument("--infer_steps", type=int, default=28, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="A seed for all the prompts")
    parser.add_argument("--device", type=str, choices=["npu"], default="npu", help="model running device")
    parser.add_argument("--save_path", type=str, default="./results", help="path to save image output")

    parser.add_argument("--do_quant", action="store_true")
    parser.add_argument(
        "--quant_type",
        choices=["w8a8"],
        default="w8a8",
    )
    parser.add_argument("--quant_weight_save_folder", type=str)
    parser.add_argument("--quant_dump_calib_folder", type=str)
    parser.add_argument("--do_save_img", action="store_true", help="whether to save image output")

    args = parser.parse_args(namespace=namespace)

    args.sd3_model_path = get_valid_read_path(args.sd3_model_path, is_dir=True)
    args.prompt_path = get_valid_read_path(args.prompt_path, is_dir=False)
    args.save_path = get_write_directory(args.save_path)
    args.quant_weight_save_folder = get_write_directory(args.quant_weight_save_folder)
    args.quant_dump_calib_folder = get_write_directory(args.quant_dump_calib_folder)
    return args


def load_prompt(path):
    if not path.endswith('txt'):
        raise ValueError("prompt path must end with txt")

    with open(path, 'r') as file:
        text_prompt = file.readlines()
        prompts = [line.strip() for line in text_prompt]

    return prompts


def inference(args):
    pipe = StableDiffusion3Pipeline.from_pretrained(args.sd3_model_path)
    pipe.to(args.device)

    model = pipe.transformer
    model.eval()

    def inference_func(save_path, desc=''):
        save_path = get_write_directory(save_path)

        prompts = load_prompt(args.prompt_path)
        for cnt, prompt in enumerate(tqdm(prompts, desc=desc)):
            torch.manual_seed(args.seed)
            torch.npu.manual_seed(args.seed)
            torch.npu.manual_seed_all(args.seed)

            images = pipe(
                prompt=[prompt],
                negative_prompt=[""],
                width=args.width,
                height=args.height,
                num_inference_steps=args.infer_steps,
                guidance_scale=7.0,
            ).images
            for i, img in enumerate(images):
                img.save(os.path.join(save_path, f"{cnt}_{i}.png"))

    # quantization
    if args.do_quant:
        # do quant
        do_multimodal_quant(
            args,
            model,
            infer_func=inference_func,
            infer_args=[],
            infer_kwargs=dict(
                save_path=os.path.join(args.save_path, 'calib_fp'), desc='Dump calib data by float model inference'
            ),
        )

        if args.do_save_img:
            # run fake quant
            inference_func(
                save_path=os.path.join(args.save_path, 'calib_quant'), desc='Run fake quant using calib data'
            )

    else:
        raise ValueError("Please --do_quant to True")


def do_multimodal_quant(args, model, infer_func, infer_args, infer_kwargs):
    from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file  # noqa: F401

    dump_calib_folder = args.quant_dump_calib_folder  # 用于存放校准数据的文件夹
    safe_tensor_folder = args.quant_weight_save_folder  # 用于存放量化模型的文件夹

    rank = get_rank()
    is_distributed = rank >= 0  # 标记是否为分布式环境

    dump_data_path = os.path.join(
        dump_calib_folder,
        get_rank_suffix_file(base_name="calib_data", ext="pth", is_distributed=is_distributed, rank=rank),
    )

    # ***************************** 加载模型 *****************************
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a nn.Module")

    # ***************************** dump 校准数据 *****************************
    if not os.path.exists(dump_data_path):  # 检查校准数据是否已存在，不存在则dump
        os.makedirs(os.path.dirname(dump_data_path), exist_ok=True)

        # 添加forward hook用于dump model的forward输入
        dumper_manager = DumperManager(model, capture_mode='args')

        # 执行浮点模型推理
        infer_func(*infer_args, **infer_kwargs)

        # 保存校准数据
        dumper_manager.save(dump_data_path)

    # ***************************** 启动量化 *****************************
    # 加载校准数据
    calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')

    def get_w8a8_cfg():
        safetensors_name = get_rank_suffix_file(
            base_name='quant_model_weight_w8a8', ext='safetensors', is_distributed=is_distributed, rank=rank
        )
        json_name = get_rank_suffix_file(
            base_name='quant_model_description_w8a8', ext='json', is_distributed=is_distributed, rank=rank
        )
        _cfg = SessionConfig(
            processor_cfg_map={
                "w8a8": W8A8ProcessorConfig(
                    cfg=W8A8QuantConfig(act_method='minmax'), disable_names=['context_embedder']
                ),
                "save": SaveProcessorConfig(
                    output_path=safe_tensor_folder,
                    safetensors_name=safetensors_name,
                    json_name=json_name,
                    save_type=['safe_tensor'],
                    part_file_size=None,
                ),
            },
            calib_data=calib_dataset,
            device='npu',
        )
        return _cfg

    if args.quant_type == 'w8a8':
        session_cfg = get_w8a8_cfg()
    else:
        raise ValueError("quant_type must be w8a8")

    # pydantic库自带的数据类型校验
    session_cfg.model_validate(session_cfg)

    # 量化模型
    quant_model(model, session_cfg)


if __name__ == "__main__":
    args = parse_args()
    inference(args)
