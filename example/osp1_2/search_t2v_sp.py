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
# pylint: disable=not-callable,redefined-outer-name
# Note: This file is copied and modified from Open-Sora-Plan repo v1.2: opensora.sample.sample_t2v_sp

import os
import gc
from typing import List

import torch
import torch.distributed as dist

from opensora.npu_config import npu_config

from example.common.security.path import json_safe_dump, get_write_directory, safe_delete_path_if_exists
from example.osp1_2.model.scheduler import EulerAncestralDiscreteSchedulerExample
from example.osp1_2.build_pipeline import get_args, build_pipeline
from msmodelslim.utils.logging import logger


def search_restep(pipeline, args):
    if args.sample_method == 'EulerAncestralDiscrete':
        pipeline.scheduler = EulerAncestralDiscreteSchedulerExample()
    else:
        raise ValueError(f'args.sample_method: {args.sample_method} not supported.')

    if args.num_frames not in {29}:
        raise ValueError(f'args.num_frames: {args.sample_method} not supported. Only support 29 frames.')

    args.save_dir = get_write_directory(args.save_dir)
    args.videos_path = get_write_directory(args.videos_path)

    # set restep search config
    from msmodelslim.pytorch.multi_modal.sampling_optimization import ReStepSearchConfig, ReStepAdaptor

    config = ReStepSearchConfig(
        videos_path=args.videos_path,
        save_dir=args.save_dir,
        neighbour_type=args.neighbour_type,
        monte_carlo_iters=args.monte_carlo_iters,
        num_sampling_steps=args.num_sampling_steps,
    )

    # create ReStepAdaptor
    restep_adaptor = ReStepAdaptor(pipeline, config)

    # do the scheduler timestep search
    scheduler_timestep = restep_adaptor.search()

    if dist.get_rank() == 0:
        logger.info("Searched scheduler timestep: %r", scheduler_timestep)


def search_dit_cache(pipeline, args):
    from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheAdaptor, DitCacheConfig, DitCacheSearchConfig

    def run_pipeline_and_save_videos(pipeline) -> List[torch.Tensor]:
        """
        Args:
            pipeline:

        Returns:
            List[torch.Tensor]:  B, num_frames, h, w, c
        """

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

        all_videos = []
        for _, prompt in enumerate(args.text_prompt):
            npu_config.seed_everything(42)
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
            ).images

            if dist.get_rank() > 0:
                continue
            all_videos.append(videos[0])

        gc.collect()
        torch.cuda.empty_cache()
        torch.npu.empty_cache()
        return all_videos

    block_num = len(getattr(pipeline.transformer, "transformer_blocks"))
    config = DitCacheSearchConfig(
        num_sampling_steps=args.num_sampling_steps, cache_ratio=args.cache_ratio, dit_block_num=block_num
    )

    if args.cache_save_path is None:
        raise ValueError('You should set --cache_save_path to save the searched config to file.')

    if dist.get_rank() == 0:
        get_write_directory(os.path.dirname(args.cache_save_path))

    cache_adaptor = DitCacheAdaptor(pipeline, config)
    cache_adaptor.set_timestep_idx(0)
    searched_config: DitCacheConfig = cache_adaptor.search(
        run_pipeline_and_save_videos=run_pipeline_and_save_videos, prompts_num=len(args.text_prompt)
    )

    if dist.get_rank() == 0:
        safe_delete_path_if_exists(args.cache_save_path)
        json_safe_dump(dict(searched_config), args.cache_save_path)
        logger.info("Searched cache config saved at %r", args.cache_save_path)


if __name__ == "__main__":
    args = get_args()
    pipeline = build_pipeline(args)

    if args.search_type == 'dit_cache':
        search_dit_cache(pipeline, args)

    elif args.search_type == 'restep':
        search_restep(pipeline, args)
