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
# pylint: disable=invalid-envvar-default,no-name-in-module,unspecified-encoding
# Note: This file is copied and modified from Open-Sora-Plan repo v1.2: opensora.sample.sample_t2v_sp

import os
import argparse

import torch
import torch.distributed as dist
import torch_npu
from transformers import T5Tokenizer, MT5EncoderModel
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DEISMultistepScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler

from opensora.models.causalvideovae import ae_stride_config, CausalVAEModelWrapper
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
from opensora.npu_config import npu_config
from opensora.acceleration.parallel_states import initialize_sequence_parallel_state
from opensora.sample.pipeline_opensora_sp import OpenSoraPipeline

from example.common.security.path import get_valid_read_path, get_write_directory
from example.osp1_2.model.model_open_sora_plan1_2_sp import OpenSoraPipelineV1x2
from msmodelslim.utils.logging import logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default=None, choices=[None, '65x512x512', '65x256x256', '17x256x256'])
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--model_type', type=str, default="dit", choices=['dit', 'udit', 'latte'])
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save_memory', action='store_true')

    # sample_optimization
    parser.add_argument(
        "--save_dir", type=str, default=None, help='The path to save file about the optimized schedule timestep'
    )
    parser.add_argument("--videos_path", type=str, default=None, help='The path of calibration videos')
    parser.add_argument("--neighbour_type", type=str, required=False, default="uniform")
    parser.add_argument("--monte_carlo_iters", type=int, required=False, default=5)

    # dit cache
    parser.add_argument("--search_type", choices=['restep', 'dit_cache'], default='restep')
    parser.add_argument("--cache_ratio", type=float, default=1.3)
    parser.add_argument("--cache_save_path", type=str, default=None)

    args = parser.parse_args()
    return args


def build_pipeline(args):
    def load_t2v_checkpoint(model_path):
        logger.info('load_t2v_checkpoint, %r', model_path)
        if args.model_type == 'dit':
            transformer_model = OpenSoraT2V.from_pretrained(
                model_path,
                local_files_only=True,
                cache_dir=args.cache_dir,
                low_cpu_mem_usage=False,
                device_map=None,
                torch_dtype=weight_dtype,
            )
        else:
            raise ValueError('--model_type is not supported')

        # set eval mode
        transformer_model.eval()

        pipeline = OpenSoraPipelineV1x2(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer_model
        ).to(device)

        if args.compile:
            pipeline.transformer = torch.compile(pipeline.transformer)

        return pipeline

    if torch_npu is not None:
        npu_config.print_msg(args)

    # 初始化分布式环境
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    if torch_npu is not None and npu_config.on_npu:
        torch_npu.npu.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)
    weight_dtype = torch.bfloat16
    device = f"npu:{torch.cuda.current_device()}"
    args.ae_path = get_valid_read_path(args.ae_path, is_dir=True)
    vae = CausalVAEModelWrapper(args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        vae.vae.tile_sample_min_size = 512
        vae.vae.tile_latent_min_size = 64
        vae.vae.tile_sample_min_size_t = 29
        vae.vae.tile_latent_min_size_t = 8
        if args.save_memory:
            vae.vae.tile_sample_min_size = 256
            vae.vae.tile_latent_min_size = 32
            vae.vae.tile_sample_min_size_t = 29
            vae.vae.tile_latent_min_size_t = 8

    vae.vae_scale_factor = ae_stride_config[args.ae]

    args.cache_dir = get_write_directory(args.cache_dir)
    args.text_encoder_name = get_valid_read_path(args.text_encoder_name, is_dir=True)
    text_encoder = MT5EncoderModel.from_pretrained(
        args.text_encoder_name,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
        local_files_only=True,
    ).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, local_files_only=True)

    # set eval mode
    vae.eval()
    text_encoder.bfloat16().eval()

    if args.sample_method == 'DDIM':
        scheduler = DDIMScheduler(clip_sample=False)
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':
        scheduler = DDPMScheduler(clip_sample=False)
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler()
    elif args.sample_method == 'HeunDiscrete':
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == 'KDPM2AncestralDiscrete':
        scheduler = KDPM2AncestralDiscreteScheduler()
    else:
        raise ValueError(f'args.sample_method: {args.sample_method} not supported.')

    # read text_prompt
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        args.text_prompt[0] = get_valid_read_path(args.text_prompt[0])
        with open(args.text_prompt[0], 'r') as f:
            text_prompt = f.readlines()
        args.text_prompt = [i.strip() for i in text_prompt]

    full_path = get_valid_read_path(args.model_path, is_dir=True)

    # get original pipeline
    pipeline: OpenSoraPipeline = load_t2v_checkpoint(full_path)
    # save osp1.2 config args to the pipeline obj
    pipeline.args = args

    return pipeline
