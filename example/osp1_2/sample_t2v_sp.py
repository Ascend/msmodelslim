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
# pylint: disable=invalid-envvar-default,no-name-in-module,possibly-used-before-assignment,redefined-outer-name,undefined-variable,ungrouped-imports,unspecified-encoding
# Note: This file is copied and modified from Open-Sora-Plan repo v1.2: opensora.sample.sample_t2v_sp

import os
import math
import argparse
import gc

import torch
import torch.distributed as dist
from torchvision.utils import save_image
import torch_npu

import imageio
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

from transformers import T5Tokenizer, MT5EncoderModel

from opensora.acceleration.parallel_states import initialize_sequence_parallel_state, hccl_info
from opensora.models.causalvideovae import ae_stride_config, CausalVAEModelWrapper
from opensora.models.diffusion.udit.modeling_udit import UDiTT2V
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
from opensora.utils.utils import save_video_grid
from opensora.npu_config import npu_config

from example.common.security.path import get_write_directory, get_valid_write_path, get_valid_read_path, json_safe_load
from example.osp1_2.model.model_open_sora_plan1_2_sp import OpenSoraPipelineV1x2
from msmodelslim.utils.logging import logger


def load_t2v_checkpoint(model_path):
    logger.info('load_t2v_checkpoint, %r', model_path)
    if args.model_type == 'udit':
        transformer_model = UDiTT2V.from_pretrained(
            model_path,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=False,
            device_map=None,
            torch_dtype=weight_dtype,
            local_files_only=True,
        )
    elif args.model_type == 'dit':
        transformer_model = OpenSoraT2V.from_pretrained(
            model_path,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=False,
            device_map=None,
            torch_dtype=weight_dtype,
            local_files_only=True,
        )
    else:
        transformer_model = LatteT2V.from_pretrained(  # noqa: F821
            model_path,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=False,
            device_map=None,
            torch_dtype=weight_dtype,
            local_files_only=True,
        )
    # set eval mode
    transformer_model.eval()

    pipeline = OpenSoraPipelineV1x2(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer_model
    ).to(device)

    if args.compile:
        pipeline.transformer = torch.compile(pipeline.transformer)

    return pipeline


def run_model_and_save_images(pipeline, model_path):
    video_grids = []
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        args.text_prompt[0] = get_valid_read_path(args.text_prompt[0])
        with open(args.text_prompt[0], 'r') as txt_file:
            text_prompt = txt_file.readlines()
        args.text_prompt = [i.strip() for i in text_prompt]

    checkpoint_name = "final"
    positive_prompt = """
    (masterpiece), (best quality), (ultra-detailed), (unwatermarked),
    {}.
    emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo,
    sharp focus, high budget, cinemascope, moody, epic, gorgeous
    """

    negative_prompt = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality,
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """

    seed = int(os.environ.get('RANDOM_SEED', 42))
    for index, prompt in enumerate(args.text_prompt):
        logger.info('set all seed: %r', seed)
        npu_config.seed_everything(seed)
        videos = pipeline(
            positive_prompt.format(prompt),
            negative_prompt=negative_prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            mask_feature=True,
            device=f"npu:{torch.cuda.current_device()}",
            max_sequence_length=args.max_sequence_length,
            timesteps=timesteps_set,
        ).images

        os.umask(0o037)
        vid_name = (
            f'{args.sample_method}_{index}_{checkpoint_name}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'
        )
        if hccl_info.rank <= 0:
            if args.num_frames == 1:
                videos = videos[:, 0].permute(0, 3, 1, 2)  # b t h w c -> b c h w
                save_path = os.path.join(args.save_img_path, vid_name)
                save_path = get_valid_write_path(save_path, is_dir=False)
                save_image(videos / 255.0, save_path, nrow=1, normalize=True, value_range=(0, 1))  # t c h w

            else:
                save_path = os.path.join(args.save_img_path, vid_name)
                save_path = get_valid_write_path(save_path, is_dir=False)
                imageio.mimwrite(
                    os.path.join(save_path),
                    videos[0],
                    fps=args.fps,
                    quality=6,
                    codec='libx264',
                    output_params=['-threads', '20'],
                )  # highest quality is 10, lowest is 0
            video_grids.append(videos)
    if hccl_info.rank <= 0:
        video_grids = torch.cat(video_grids, dim=0)

        def get_file_name():
            save_path = os.path.join(
                args.save_img_path,
                f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}_{checkpoint_name}.{ext}',
            )
            save_path = get_valid_write_path(save_path, is_dir=False)
            return save_path

        output_path = get_file_name()

        if args.num_frames == 1:
            save_image(
                video_grids / 255.0,
                output_path,
                nrow=math.ceil(math.sqrt(len(video_grids))),
                normalize=True,
                value_range=(0, 1),
            )
        else:
            video_grids = save_video_grid(video_grids)
            imageio.mimwrite(output_path, video_grids, fps=args.fps, quality=6)

        logger.info('concat video file saved at: %r', output_path)


if __name__ == "__main__":
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
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
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

    # set searched timestep
    parser.add_argument("--schedule_timestep", type=str, required=False, default=None)

    # set searched dit-cache config
    parser.add_argument("--dit_cache_config", type=str, required=False, default=None)

    args = parser.parse_args()

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
    text_encoder = MT5EncoderModel.from_pretrained(  # nosec B615
        args.text_encoder_name,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
        local_files_only=True,
    ).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, local_files_only=True)  # nosec B615

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

    if args.schedule_timestep is not None:
        from example.osp1_2.model.scheduler import EulerAncestralDiscreteSchedulerExample

        scheduler = EulerAncestralDiscreteSchedulerExample()
        args.schedule_timestep = get_valid_read_path(args.schedule_timestep)
        timesteps = json_safe_load(args.schedule_timestep, extensions='txt')

        timesteps_set = [x * 1000 for x in timesteps][::-1]
        logger.info('set timesteps_set to %r', timesteps_set)
    else:
        timesteps_set = None

    if args.dit_cache_config is not None:
        args.dit_cache_config = get_valid_read_path(args.dit_cache_config)
        cache_config = json_safe_load(args.dit_cache_config)
    else:
        cache_config = None

    args.save_img_path = get_write_directory(args.save_img_path)

    if args.num_frames == 1:
        video_length = 1
        ext = 'jpg'
    else:
        ext = 'mp4'

    # read text_prompt
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        args.text_prompt[0] = get_valid_read_path(args.text_prompt[0])
        with open(args.text_prompt[0], 'r') as f:
            text_prompt = f.readlines()
        args.text_prompt = [i.strip() for i in text_prompt]

    args.save_img_path = get_write_directory(args.save_img_path)
    full_path = get_valid_read_path(args.model_path, is_dir=True)

    pipeline = load_t2v_checkpoint(full_path)
    logger.info('load model')

    gc.collect()
    torch.cuda.empty_cache()
    torch.npu.empty_cache()

    if cache_config is not None:
        from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheAdaptor, DitCacheSearchConfig

        block_num = len(getattr(pipeline.transformer, "transformer_blocks"))
        search_config = DitCacheSearchConfig(num_sampling_steps=args.num_sampling_steps, dit_block_num=block_num)
        # add adaptor to add cache func to the dit blocks
        adaptor = DitCacheAdaptor(pipeline, search_config)
        adaptor.set_timestep_idx(0)
        adaptor.update_cache_config(**cache_config)

        logger.info('using cache config: %r', cache_config)

    run_model_and_save_images(pipeline, full_path)
