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
# pylint: disable=unnecessary-list-index-lookup,unspecified-encoding,useless-return

import os
import sys
import argparse
from pathlib import Path

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401
from torch import nn
from tqdm import tqdm

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import (
    sanity_check_args,
    add_device_args,
    add_network_args,
    add_extra_models_args,
    add_denoise_schedule_args,
    add_inference_args,
    add_parallel_args,
    add_ditcache_args,
    add_attentioncache_args,
)
from hyvideo.inference import HunyuanVideoSampler
from mindiesd import CacheConfig, CacheAgent

cur_file_dir = os.path.dirname(os.path.abspath(__file__))
example_base_dir = os.path.abspath(os.path.join(cur_file_dir, "..", "..", ".."))
sys.path.append(example_base_dir)

from example.common.security.pytorch import safe_torch_load  # noqa: E402
from example.common.security.path import get_write_directory, get_valid_read_path  # noqa: E402
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
torch.npu.config.allow_internal_format = False


def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

    parser = add_device_args(parser)
    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_inference_args(parser)
    parser = add_parallel_args(parser)
    parser = add_ditcache_args(parser)
    parser = add_attentioncache_args(parser)

    parser.add_argument("--do_quant", action="store_true")
    parser.add_argument(
        "--quant_type",
        choices=["w8a8_timestep", "w8a8_dynamic_fa3", "w8a8_dynamic"],
        default="w8a8_timestep",
    )
    parser.add_argument("--anti_method", choices=["m3", "m4", "m6"], default=None)
    parser.add_argument("--quant_weight_save_folder", type=str)
    parser.add_argument("--quant_dump_calib_folder", type=str)
    parser.add_argument("--do_save_video", action="store_true", help="whether to save video output")

    args = parser.parse_args(namespace=namespace)
    args = sanity_check_args(args)

    # check args
    args.quant_weight_save_folder = get_write_directory(args.quant_weight_save_folder)
    args.quant_dump_calib_folder = get_write_directory(args.quant_dump_calib_folder)
    return args


def main():
    args = parse_args()

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    save_path = get_write_directory(save_path)
    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    transformer = hunyuan_video_sampler.pipeline.transformer

    # Get the updated args
    args = hunyuan_video_sampler.args
    if args.prompt.endswith('txt'):
        args.prompt = get_valid_read_path(args.prompt)
        with open(args.prompt, 'r') as file:
            text_prompt = file.readlines()
            prompts = [line.strip() for line in text_prompt]
    else:
        prompts = [args.prompt]

    if args.use_cache:
        # single
        config_single = CacheConfig(
            method="dit_block_cache",
            blocks_count=len(transformer.single_blocks),
            steps_count=args.infer_steps,
            step_start=args.cache_start_steps,
            step_interval=args.cache_interval,
            step_end=args.infer_steps - 1,
            block_start=args.single_block_start,
            block_end=args.single_block_end,
        )
        cache_single = CacheAgent(config_single)
        hunyuan_video_sampler.pipeline.transformer.cache_single = cache_single
    if args.use_cache_double:
        # double
        config_double = CacheConfig(
            method="dit_block_cache",
            blocks_count=len(transformer.double_blocks),
            steps_count=args.infer_steps,
            step_start=args.cache_start_steps,
            step_interval=args.cache_interval,
            step_end=args.infer_steps - 1,
            block_start=args.double_block_start,
            block_end=args.double_block_end,
        )
        cache_dual = CacheAgent(config_double)
        hunyuan_video_sampler.pipeline.transformer.cache_dual = cache_dual

    if args.use_attentioncache:
        config_double = CacheConfig(
            method="attention_cache",
            blocks_count=len(transformer.double_blocks),
            steps_count=args.infer_steps,
            step_start=args.start_step,
            step_interval=args.attentioncache_interval,
            step_end=args.end_step,
        )
        config_single = CacheConfig(
            method="attention_cache",
            blocks_count=len(transformer.single_blocks),
            steps_count=args.infer_steps,
            step_start=args.start_step,
            step_interval=args.attentioncache_interval,
            step_end=args.end_step,
        )
    else:
        config_double = CacheConfig(
            method="attention_cache", blocks_count=len(transformer.double_blocks), steps_count=args.infer_steps
        )
        config_single = CacheConfig(
            method="attention_cache", blocks_count=len(transformer.single_blocks), steps_count=args.infer_steps
        )
    cache_double = CacheAgent(config_double)
    cache_single = CacheAgent(config_single)
    for block in transformer.double_blocks:
        block.cache = cache_double
    for block in transformer.single_blocks:
        block.cache = cache_single

    def sample(save_path, desc='', **kwargs):
        # Start sampling
        for idx in tqdm(range(len(prompts)), desc=desc):
            if idx == 0:
                # warm
                outputs = hunyuan_video_sampler.predict(
                    prompt=prompts[0],
                    height=args.video_size[0],
                    width=args.video_size[1],
                    video_length=args.video_length,
                    seed=args.seed,
                    negative_prompt=args.neg_prompt,
                    infer_steps=2,
                    guidance_scale=args.cfg_scale,
                    num_videos_per_prompt=args.num_videos,
                    flow_shift=args.flow_shift,
                    batch_size=args.batch_size,
                    embedded_guidance_scale=args.embedded_cfg_scale,
                )

            outputs = hunyuan_video_sampler.predict(
                prompt=prompts[idx],
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
                seed=args.seed,
                negative_prompt=args.neg_prompt,
                infer_steps=args.infer_steps,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=args.num_videos,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale,
            )
            samples = outputs['samples']

            # Save samples
            if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
                for i, sample in enumerate(samples):
                    sample = samples[i].unsqueeze(0)
                    video_path = f"{save_path}/sample_{idx}.mp4"
                    save_videos_grid(sample, video_path, fps=24)
                    logger.info('Sample save to: %r', video_path)

    # quantization
    if args.do_quant:
        # do quant
        do_multimodal_quant(
            args,
            transformer,
            infer_func=sample,
            infer_args=[],
            infer_kwargs=dict(
                save_path=os.path.join(args.save_path, 'calib_fp'), desc='Dump calib data by float model inference'
            ),
        )

        if args.do_save_video:
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
                        layer_include=('*double_blocks*', '*single_blocks*'),
                        layer_exclude=('*img_mod*', '*modulation*', '*fc2*'),
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
                    disable_names=get_disable_layer_names(
                        model,
                        layer_include=('*double_blocks*', '*single_blocks*'),
                        layer_exclude=('*img_mod*', '*modulation*'),
                    ),
                ),
                "save": SaveProcessorConfig(
                    output_path=safe_tensor_folder,
                    safetensors_name=safetensors_name,
                    json_name=json_name,
                    save_type=["safe_tensor"],
                    part_file_size=None,
                ),
            },
            calib_data=calib_dataset[:20],
            device="npu",
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
                    layer_include=('*double_blocks*', '*single_blocks*'),
                    layer_exclude=('*img_mod*', '*modulation*', '*fc2*'),
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
        model.config.model_type = 'hunyuan_video'
        session_cfg = get_w8a8_dynamic_cfg()
    else:
        raise ValueError("quant_type must be timestep or fa3")

    # pydantic库自带的数据类型校验
    session_cfg.model_validate(session_cfg)

    # 量化模型
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        quant_model(model, session_cfg)


if __name__ == "__main__":
    main()
