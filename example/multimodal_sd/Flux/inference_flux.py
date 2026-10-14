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
# pylint: disable=reimported,useless-return

import os
import sys
import argparse
import csv
import json  # noqa: F401
import time

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from torch import nn

from tqdm import tqdm

from mindiesd import CacheAgent, CacheConfig
from FLUX1dev import FluxPipeline
from FLUX1dev import get_local_rank, get_world_size, initialize_torch_distributed
from FLUX1dev.utils import check_prompts_valid, check_param_valid, check_dir_safety, check_file_safety

cur_file_dir = os.path.dirname(os.path.abspath(__file__))
example_base_dir = os.path.abspath(os.path.join(cur_file_dir, "..", "..", ".."))
sys.path.append(example_base_dir)

from example.common.security.pytorch import safe_torch_load  # noqa: E402
from example.common.security.path import get_write_directory, json_safe_load, json_safe_dump  # noqa: E402
from example.common.security.path import get_write_directory, get_valid_write_path, json_safe_load, json_safe_dump  # noqa: E402, F401, F811
from msmodelslim.quant import (  # noqa: E402
    quant_model,
    SessionConfig,
    FA3ProcessorConfig,
    W8A8DynamicQuantConfig,
    W8A8DynamicProcessorConfig,
    M3ProcessorConfig,
    M4ProcessorConfig,
    M6ProcessorConfig,
    M6Config,
)
from msmodelslim.quant import W8A8TimeStepProcessorConfig, W8A8TimeStepQuantConfig, SaveProcessorConfig  # noqa: E402
from msmodelslim import logger  # noqa: E402

torch_npu.npu.set_compile_mode(jit_compile=False)

cache_dict = {}
cache_dict['cache_start_block'] = 5
cache_dict['num_cache_layer_block'] = 13
cache_dict['cache_start_single_block'] = 1
cache_dict['num_cache_layer_single_block'] = 23
cache_dict['cache_start_steps'] = 18
cache_dict['cache_interval'] = 2


class PromptLoader:
    def __init__(
        self,
        prompt_file: str,
        prompt_file_type: str,
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        max_num_prompts: int = 0,
    ):
        self.check_input_isvalid(batch_size, num_images_per_prompt, max_num_prompts)
        self.prompts = []
        self.catagories = ['Not_specified']
        self.batch_size = batch_size
        self.num_images_per_prompt = num_images_per_prompt

        if prompt_file_type == 'plain':
            self.load_prompts_plain(prompt_file, max_num_prompts)
        elif prompt_file_type == 'parti':
            self.load_prompts_parti(prompt_file, max_num_prompts)
        elif prompt_file_type == 'hpsv2':
            self.load_prompts_hpsv2(prompt_file, max_num_prompts)
        else:
            raise ValueError(f"Prompt file type {prompt_file_type} not supported")

        self.current_id = 0
        self.inner_id = 0

    def __len__(self):
        return len(self.prompts) * self.num_images_per_prompt

    def __iter__(self):
        self.current_id = 0
        self.inner_id = 0
        return self

    def __next__(self):
        if self.current_id == len(self.prompts):
            raise StopIteration

        ret = {
            'prompts': [],
            'catagories': [],
            'save_names': [],
            'n_prompts': self.batch_size,
        }
        for _ in range(self.batch_size):
            if self.current_id == len(self.prompts):
                ret['prompts'].append('')
                ret['save_names'].append('')
                ret['catagories'].append('')
                ret['n_prompts'] -= 1

            else:
                prompt, catagory_id = self.prompts[self.current_id]
                ret['prompts'].append(prompt)
                ret['catagories'].append(self.catagories[catagory_id])
                ret['save_names'].append(f'{self.current_id}_{self.inner_id}')

                self.inner_id += 1
                if self.inner_id == self.num_images_per_prompt:
                    self.inner_id = 0
                    self.current_id += 1

        return ret

    @staticmethod
    def check_input_isvalid(batch_size, num_images_per_prompt, max_num_prompts):
        if batch_size <= 0:
            raise ValueError(f"Param batch_size invalid, expected positive value, but get {batch_size}")
        if batch_size > 100:
            raise ValueError(f"Param batch_size invalid, expected less than or equal to 100, but get {batch_size}")
        if num_images_per_prompt <= 0:
            raise ValueError(
                f"Param num_images_per_prompt invalid, expected positive value, but get {num_images_per_prompt}"
            )
        if max_num_prompts < 0:
            raise ValueError(
                f"Param max_num_prompts invalid, expected greater than or equal to 0, \
                                     but get {max_num_prompts}"
            )

    def load_prompts_plain(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            for i, line in enumerate(f):
                if max_num_prompts and i == max_num_prompts:
                    break

                prompt = line.strip()
                self.prompts.append((prompt, 0))

    def load_prompts_parti(self, file_path: str, max_num_prompts: int):
        with os.fdopen(os.open(file_path, os.O_RDONLY), "r") as f:
            # Skip the first line
            next(f)
            tsv_file = csv.reader(f, delimiter="\t")
            for i, line in enumerate(tsv_file):
                if max_num_prompts and i == max_num_prompts:
                    break

                if len(line) < 2:
                    raise ValueError(
                        f"Invalid line {line}. The line must have at least 2 fields, "
                        f"but only {len(line)} field(s) found."
                    )

                prompt = line[0]
                catagory = line[1]
                if catagory not in self.catagories:
                    self.catagories.append(catagory)

                catagory_id = self.catagories.index(catagory)
                self.prompts.append((prompt, catagory_id))

    def load_prompts_hpsv2(self, file_path, max_num_prompts: int):
        all_prompts = json_safe_load(file_path)
        count = 0
        for style, prompts in all_prompts.items():
            for prompt in prompts:
                if max_num_prompts and count >= max_num_prompts:
                    break
                count += 1

                if style not in self.catagories:
                    self.catagories.append(style)

                catagory_id = self.catagories.index(style)
                self.prompts.append((prompt, catagory_id))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the flux model directory")
    parser.add_argument("--save_path", type=str, default="./res", help="ouput image path")
    parser.add_argument("--device_id", type=int, default=0, help="NPU device id")
    parser.add_argument("--device", choices=["npu", "cpu"], default="npu", help="NPU")
    parser.add_argument("--prompt_path", type=str, default="./calib_prompts.txt", help="input prompt text path")
    parser.add_argument(
        "--prompt_type", choices=["plain", "parti", "hpsv2"], default="plain", help="specify infer prompt type"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="specify image number every prompt generate"
    )
    parser.add_argument("--max_num_prompt", type=int, default=0, help="limit the prompt number[0 indicates no limit]")
    parser.add_argument("--info_file_save_path", type=str, default="./image_info.json", help="path to save image info")
    parser.add_argument("--width", type=int, default=1024, help='Image size width')
    parser.add_argument("--height", type=int, default=1024, help='Image size height')
    parser.add_argument("--infer_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="A seed for all the prompts")
    parser.add_argument("--use_cache", action="store_true", help="turn on dit cache or not")
    parser.add_argument("--batch_size", type=int, default=1, help="prompt batch size")
    parser.add_argument(
        "--device_type",
        choices=["A2-32g-single", "A2-32g-dual", "A2-64g"],
        default="A2-64g",
        help="specify device type",
    )

    parser.add_argument("--do_quant", action="store_true")
    parser.add_argument(
        "--quant_type", choices=["w8a8_timestep", "w8a8_dynamic_fa3", "w8a8_dynamic"], default="w8a8_timestep"
    )
    parser.add_argument("--anti_method", choices=["m3", "m4", "m6"], default=None)
    parser.add_argument("--quant_weight_save_folder", type=str)
    parser.add_argument("--quant_dump_calib_folder", type=str)
    parser.add_argument("--data_split_num", type=int, default=1)
    parser.add_argument("--data_split_id", type=int, default=0)
    parser.add_argument("--do_save_img", action="store_true", help="whether to save image output")

    args = parser.parse_args()

    # check args
    args.save_path = get_write_directory(args.save_path)
    args.quant_weight_save_folder = get_write_directory(args.quant_weight_save_folder)
    args.quant_dump_calib_folder = get_write_directory(args.quant_dump_calib_folder)
    return args


def infer(args):
    if args.device_type == "A2-32g-dual":
        from FLUX1dev import replace_tp_from_pretrain, replace_tp_extract_init_dict

        FluxPipeline.from_pretrained = classmethod(replace_tp_from_pretrain)
        FluxPipeline.extract_init_dict = classmethod(replace_tp_extract_init_dict)

    check_dir_safety(args.path)
    pipe = FluxPipeline.from_pretrained(args.path, torch_dtype=torch.bfloat16, local_files_only=True)

    if args.device_type == "A2-32g-single":
        torch.npu.set_device(args.device_id)
        pipe.enable_model_cpu_offload()
    elif args.device_type == "A2-64g":
        torch.npu.set_device(args.device_id)
        pipe.to(f"npu:{args.device_id}")
    else:
        local_rank = get_local_rank()
        world_size = get_world_size()
        initialize_torch_distributed(local_rank, world_size)
        pipe.to(f"npu:{local_rank}")

    if args.use_cache:
        d_stream_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=19,
            steps_count=args.infer_steps,
            step_start=18,
            step_interval=2,
            block_start=5,
            block_end=13,
        )
        d_stream_agent = CacheAgent(d_stream_config)
        pipe.transformer.d_stream_agent = d_stream_agent
        s_stream_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=38,
            steps_count=args.infer_steps,
            step_start=18,
            step_interval=2,
            block_start=1,
            block_end=23,
        )
        s_stream_agent = CacheAgent(s_stream_config)
        pipe.transformer.s_stream_agent = s_stream_agent
    else:
        d_stream_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=19,
            steps_count=args.infer_steps,
        )
        d_stream_agent = CacheAgent(d_stream_config)
        pipe.transformer.d_stream_agent = d_stream_agent
        s_stream_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=38,
            steps_count=args.infer_steps,
        )
        s_stream_agent = CacheAgent(s_stream_config)
        pipe.transformer.s_stream_agent = s_stream_agent

    def sample(save_path, desc='', **kwargs):
        if save_path is None:
            save_path = args.save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path, mode=0o750)

        infer_num = 0
        time_consume = 0
        current_prompt = None
        image_info = []
        check_file_safety(args.prompt_path)
        prompt_loader = PromptLoader(
            args.prompt_path, args.prompt_type, args.batch_size, args.num_images_per_prompt, args.max_num_prompt
        )
        check_param_valid(args.height, args.width, args.infer_steps)
        prompt_loader = list(prompt_loader)
        prompt_loader = prompt_loader[args.data_split_id :: args.data_split_num]
        for idx, input_info in enumerate(tqdm(prompt_loader, desc=desc)):
            torch.manual_seed(args.seed)
            torch.npu.manual_seed(args.seed)
            torch.npu.manual_seed_all(args.seed)

            prompts = input_info['prompts']
            save_names = input_info['save_names']
            catagories = input_info['catagories']
            save_names = input_info['save_names']
            n_prompts = input_info['n_prompts']

            check_prompts_valid(prompts)

            logger.info("[%d/%d]: %r", infer_num + n_prompts, len(prompt_loader), prompts)
            infer_num += args.batch_size
            if infer_num > 3:
                start_time = time.time()

            if idx == 0:
                # warm
                image = pipe(
                    prompts,
                    height=args.width,
                    width=args.height,
                    guidance_scale=3.5,
                    num_inference_steps=2,
                    max_sequence_length=512,
                    use_cache=args.use_cache,
                    cache_dict=cache_dict,
                )
            image = pipe(
                prompts,
                height=args.width,
                width=args.height,
                guidance_scale=3.5,
                num_inference_steps=args.infer_steps,
                max_sequence_length=512,
                use_cache=args.use_cache,
                cache_dict=cache_dict,
            )

            if infer_num > 3:
                end_time = time.time() - start_time
                time_consume += end_time

            for j in range(n_prompts):
                image_save_path = os.path.join(save_path, f"{save_names[j]}.png")
                image[0][j].save(image_save_path)

                if current_prompt != prompts[j]:
                    current_prompt = prompts[j]
                    image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

                image_info[-1]['images'].append(image_save_path)

        if os.path.exists(args.info_file_save_path):
            os.remove(args.info_file_save_path)

        json_safe_dump(image_info, args.info_file_save_path)
        image_time_count = len(prompt_loader) - 3  # noqa: F841

    if args.do_quant:
        # do quant
        do_multimodal_quant(
            args,
            pipe.transformer,
            infer_func=sample,
            infer_args=[],
            infer_kwargs=dict(
                save_path=os.path.join(args.save_path, 'calib_fp'), desc='Dump calib data by float model inference'
            ),
        )
        if args.do_save_img:
            # run fake quant
            sample(save_path=os.path.join(args.save_path, 'calib_quant'), desc='Run fake quant using calib data')
    else:
        raise ValueError("Please --do_quant to True")

    return


def do_multimodal_quant(args, model, infer_func, infer_args, infer_kwargs):
    from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager, get_rank_suffix_file

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
        capture_mode = 'timestep' if 'timestep' in args.quant_type else 'args'
        dumper_manager = DumperManager(model, capture_mode=capture_mode)

        # 执行浮点模型推理
        infer_func(*infer_args, **infer_kwargs)

        # 保存校准数据
        dumper_manager.save(dump_data_path)

    # ***************************** 启动量化 *****************************
    # 加载校准数据
    calib_dataset = safe_torch_load(dump_data_path, map_location=f'npu:{rank if is_distributed else 0}')

    # 量化配置
    def get_timestep_cfg():
        safetensors_name = get_rank_suffix_file(
            base_name='quant_model_weight_w8a8_timestep', ext='safetensors', is_distributed=is_distributed, rank=rank
        )
        json_name = get_rank_suffix_file(
            base_name='quant_model_description_w8a8_timestep', ext='json', is_distributed=is_distributed, rank=rank
        )
        _cfg = SessionConfig(
            processor_cfg_map={
                "w8a8_timestep": W8A8TimeStepProcessorConfig(
                    cfg=W8A8TimeStepQuantConfig(act_method='minmax'),
                    disable_names=get_disable_layer_names(
                        model,
                        layer_include='*',
                        layer_exclude='*net.2*',
                    ),
                    timestep_sep=25,
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

    def get_fa3_cfg():
        safetensors_name = get_rank_suffix_file(
            base_name='quant_model_weight_w8a8_dynamic', ext='safetensors', is_distributed=is_distributed, rank=rank
        )
        json_name = get_rank_suffix_file(
            base_name='quant_model_description_w8a8_dynamic', ext='json', is_distributed=is_distributed, rank=rank
        )
        _cfg = SessionConfig(
            processor_cfg_map={
                "fa3": FA3ProcessorConfig(),
                "w8a8_dynamic": W8A8DynamicProcessorConfig(
                    cfg=W8A8DynamicQuantConfig(act_method='minmax'),
                    disable_names=[],
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

    def get_w8a8_dynamic_cfg():
        safetensors_name = get_rank_suffix_file(
            base_name='quant_model_weight_w8a8_dynamic', ext='safetensors', is_distributed=is_distributed, rank=rank
        )
        json_name = get_rank_suffix_file(
            base_name='quant_model_description_w8a8_dynamic', ext='json', is_distributed=is_distributed, rank=rank
        )
        processor_cfg_map = {
            "w8a8_dynamic": W8A8DynamicProcessorConfig(
                cfg=W8A8DynamicQuantConfig(act_method='minmax'),
                disable_names=get_disable_layer_names(
                    model,
                    layer_include='*',
                    layer_exclude='*net.2*',
                ),
            ),
            "save": SaveProcessorConfig(
                output_path=safe_tensor_folder,
                safetensors_name=safetensors_name,
                json_name=json_name,
                save_type=['safe_tensor'],
                part_file_size=None,
            ),
        }
        if args.anti_method == 'm3':
            processor_cfg_map['m3'] = M3ProcessorConfig()
        elif args.anti_method == 'm4':
            processor_cfg_map['m4'] = M4ProcessorConfig()
        elif args.anti_method == 'm6':
            processor_cfg_map['m6'] = M6ProcessorConfig(cfg=M6Config(alpha=0.8, beta=0.2))

        _cfg = SessionConfig(processor_cfg_map=processor_cfg_map, calib_data=calib_dataset, device='npu')
        return _cfg

    if 'timestep' in args.quant_type:
        session_cfg = get_timestep_cfg()
    elif 'fa3' in args.quant_type:
        session_cfg = get_fa3_cfg()
    elif args.quant_type == 'w8a8_dynamic':
        model.config.model_type = 'flux'
        session_cfg = get_w8a8_dynamic_cfg()
    else:
        raise ValueError("quant_type must be timestep or fa3 or w8a8_dynamic")

    # pydantic库自带的数据类型校验
    session_cfg.model_validate(session_cfg)

    # 量化模型
    quant_model(model, session_cfg)


if __name__ == "__main__":
    inference_args = parse_arguments()
    infer(inference_args)
