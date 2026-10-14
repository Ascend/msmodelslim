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
# pylint: disable=consider-using-with,invalid-envvar-default,no-name-in-module,possibly-used-before-assignment,redefined-outer-name,ungrouped-imports,unspecified-encoding

import argparse
import logging
import os
import time
import sys

import imageio
import torch
from torch import nn
import torch_npu  # noqa: F401
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from transformers import T5Tokenizer, MT5EncoderModel

from opensora.models.causalvideovae import ae_stride_config, CausalVAEModelWrapper
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
from opensora.sample.pipeline_opensora_sp import OpenSoraPipeline
from utils.parallel_mgr import ParallelConfig, init_parallel_env, finalize_parallel_env, get_sequence_parallel_rank
from opensora.models.causalvideovae.model.causal_vae.parallel_layers import (
    register_vae_decode,
    parallel_full_model_warp,
)
from utils.file_utils import standardize_path

from msmodelslim.quant import quant_model, SessionConfig
from msmodelslim.quant import W8A8ProcessorConfig, W8A8QuantConfig, SaveProcessorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_t2v_checkpoint(model_path):
    logger.info('load_t2v_checkpoint, %r', model_path)
    transformer_model = OpenSoraT2V.from_pretrained(
        model_path,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=False,
        device_map=None,
        torch_dtype=weight_dtype,
        local_files_only=True,
    ).to("npu")
    transformer_model.eval()
    pipeline = OpenSoraPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer_model
    ).to("npu")
    if args.algorithm == "dit_cache":
        from opensora.models.diffusion.opensora.cache_mgr import CacheManager, DitCacheConfig

        config = DitCacheConfig(step_start=20, step_interval=2, block_start=7, num_blocks=21)
        cache = CacheManager(config)
        pipeline.transformer.cache = cache
    return pipeline


def run_model_and_save_images(pipeline, args, save_path):
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

    if not isinstance(args.text_prompt, list):
        args.text_prompt = [positive_prompt.format(args.text_prompt)]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()  # noqa: SIM115
        args.text_prompt = [positive_prompt.format(i.strip()) for i in text_prompt]

    if args.batch_size > 1:
        prompt_list = []
        group = len(args.text_prompt) // args.batch_size
        tail = len(args.text_prompt) % args.batch_size
        for index in range(group):
            prompt_list.append(args.text_prompt[index * args.batch_size : (index + 1) * args.batch_size])
        if tail > 0:
            prompt_list.append(args.text_prompt[:-tail])
    else:
        prompt_list = args.text_prompt

    kwargs = {}
    if args.algorithm == "sampling_optimize":
        kwargs["sampling_optimize"] = True

    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o750)

    if not args.test_time:
        for index, prompt in enumerate(prompt_list):
            videos = pipeline(
                prompt,
                negative_prompt=negative_prompt,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                mask_feature=True,
                max_sequence_length=args.max_sequence_length,
                seed=args.seed,
                **kwargs,
            ).images
            logger.info(videos.shape)

            if get_sequence_parallel_rank() <= 0:
                for i in range(len(prompt) if args.batch_size > 1 else 1):
                    imageio.mimwrite(
                        os.path.join(
                            save_path,
                            f'EulerAncestralDiscrete_{index * args.batch_size + i}'
                            + f'_final__gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4',
                        ),
                        videos[i],
                        fps=args.fps,
                        quality=6,
                        codec='libx264',
                        output_params=['-threads', '20'],
                    )  # highest quality is 10, lowest is 0

    else:
        for _ in range(2):
            start_time = time.time()
            videos = pipeline(
                prompt_list[0],
                negative_prompt=negative_prompt,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                mask_feature=True,
                max_sequence_length=args.max_sequence_length,
                **kwargs,
            ).images
            torch.npu.synchronize()
            use_time = time.time() - start_time
            logger.info("=========  use time %s", str(use_time))
        logger.info(videos.shape)


def do_multimodal_quant(args, model, infer_func, infer_args, infer_kwargs):
    from example.multimodal_sd.utils import get_disable_layer_names, get_rank, DumperManager

    dump_calib_folder = args.quant_dump_calib_folder  # 用于存放校准数据的文件夹
    safe_tensor_folder = args.quant_weight_save_folder  # 用于存放量化模型的文件夹

    dump_data_path = os.path.join(dump_calib_folder, f'calib_data_{get_rank()}.pth')
    safe_tensor_path = os.path.join(safe_tensor_folder, f'rank_{get_rank()}.safetensors')

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
    calib_dataset = torch.load(dump_data_path, map_location=f'npu:{get_rank()}')

    def get_w8a8_cfg():
        _cfg = SessionConfig(
            processor_cfg_map={
                "w8a8": W8A8ProcessorConfig(
                    cfg=W8A8QuantConfig(act_method='minmax'),
                    disable_names=get_disable_layer_names(
                        model, layer_include=None, layer_exclude=('*net.2*', '*adaln_single*')
                    ),
                ),
                "save": SaveProcessorConfig(
                    output_path=os.path.dirname(safe_tensor_path),
                    safetensors_name=os.path.basename(safe_tensor_path),
                    json_name=None,
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
    cur_file_dir = os.path.dirname(os.path.abspath(__file__))
    example_base_dir = os.path.abspath(os.path.join(cur_file_dir, "..", "..", ".."))
    sys.path.append(example_base_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default=None, choices=[None, '65x512x512', '65x256x256', '17x256x256'])
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument('--dtype', type=str, default='bf16', help='Data type used in inference')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='google/mt5-xxl')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--algorithm", type=str, default=None, choices=[None, 'dit_cache', 'sampling_optimize'])
    parser.add_argument("--use_cfg_parallel", action='store_true')
    parser.add_argument("--test_time", action='store_true')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--vae_parallel", action='store_true')
    parser.add_argument("--do_quant", action="store_true")
    parser.add_argument(
        "--quant_type",
        choices=["w8a8"],
        default="w8a8",
    )
    parser.add_argument("--quant_weight_save_folder", type=str)
    parser.add_argument("--quant_dump_calib_folder", type=str)
    parser.add_argument("--save_image_flag", action="store_true")

    args = parser.parse_args()

    if args.dtype not in ['bf16', 'fp16']:
        logger.error("Not supported.")
    weight_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    torch.npu.config.allow_internal_format = False

    world_size = int(os.getenv('WORLD_SIZE', 1))
    if world_size > 1:
        sp_degree = world_size // 2 if args.use_cfg_parallel else world_size
        parallel_config = ParallelConfig(
            sp_degree=sp_degree, use_cfg_parallel=args.use_cfg_parallel, world_size=world_size
        )
        init_parallel_env(parallel_config)

    args.ae_path = standardize_path(args.ae_path)
    args.text_encoder_name = standardize_path(args.text_encoder_name)
    args.model_path = standardize_path(args.model_path)

    vae = CausalVAEModelWrapper(args.ae_path, dtype=torch.float16, local_files_only=True).to("npu")
    vae.vae.enable_tiling()
    vae.vae.tile_overlap_factor = args.tile_overlap_factor
    vae.vae.tile_sample_min_size = 256
    vae.vae.tile_latent_min_size = 32
    vae.vae.tile_sample_min_size_t = 29
    vae.vae.tile_latent_min_size_t = 8
    vae.vae_scale_factor = ae_stride_config[args.ae]
    vae.eval()
    VAE_PARALLEL = args.vae_parallel

    if VAE_PARALLEL:
        parallel_dim = -1
        parallel_overlap = True
        parallel_full_model_warp(vae.vae, parallel_dim)
        vae = register_vae_decode(vae, parallel_dim, parallel_overlap)

    text_encoder = MT5EncoderModel.from_pretrained(
        args.text_encoder_name,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
        local_files_only=True,
    ).to("npu")
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, local_files_only=True)
    text_encoder.eval()

    scheduler = EulerAncestralDiscreteScheduler()

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    pipeline = load_t2v_checkpoint(args.model_path)
    logger.info('load model')

    # quantization
    if args.do_quant:
        # do quant
        do_multimodal_quant(
            args,
            pipeline.transformer,
            infer_func=run_model_and_save_images,
            infer_args=[
                pipeline,
                args,
            ],
            infer_kwargs=dict(
                save_path=os.path.join(args.save_img_path, 'calib_fp'),
            ),
        )

        if args.save_image_flag:
            # run fake quant
            run_model_and_save_images(pipeline, args, save_path=os.path.join(args.save_img_path, 'calib_quant'))

    else:
        if args.save_image_flag:
            # run float inference
            run_model_and_save_images(pipeline, args, save_path=args.save_img_path)

    if world_size > 0:
        finalize_parallel_env()
