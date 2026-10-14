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
# pylint: disable=reimported,ungrouped-imports

import os
import sys
import time  # noqa: F401
from pprint import pformat

import colossalai
from tqdm import tqdm
from thop import clever_format, profile  # noqa: F401
from safetensors import safe_open  # noqa: F401
import torch.distributed as dist  # noqa: F401
import torch
from colossalai.utils import get_current_device

sys.path.append(f"{os.environ['PROJECT_PATH']}/resource/multi_modal/opensora_project")

from opensora.datasets import save_sample  # noqa: F401
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing  # noqa: F401
from opensora.acceleration.parallel_mgr import set_parallel_manager
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import create_logger, to_torch_dtype
from mmengine.runner import set_random_seed  # noqa: F401
from opensora.utils.inference_utils import (
    add_watermark,  # noqa: F401
    append_generated,  # noqa: F401
    append_score_to_prompts,  # noqa: F401
    apply_mask_strategy,  # noqa: F401
    collect_references_batch,  # noqa: F401
    dframe_to_frame,  # noqa: F401
    extract_json_from_prompts,  # noqa: F401
    extract_prompts_loop,  # noqa: F401
    get_save_path_name,  # noqa: F401
    load_prompts,  # noqa: F401
    merge_prompt,  # noqa: F401
    prepare_multi_resolution_info,  # noqa: F401
    refine_prompts_by_openai,  # noqa: F401
    split_prompt,  # noqa: F401
)  # noqa: F401
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype  # noqa: F401, F811


from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator


@torch.no_grad()
def main():
    # 1.cfg and init distrubuted env
    cfg = parse_configs(False)
    print(cfg)

    # 2.Initialize Distributed Training
    colossalai.launch_from_torch({}, seed=cfg.seed)
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    # 3.Initialize Process Group
    sp_size = 1
    dp_size = 1
    enable_sequence_parallelism = False
    set_parallel_manager(sp_size, dp_size, dp_axis=0, sp_axis=1)

    # 4.runtime variables
    torch.set_grad_enabled(False)
    device = "npu" if torch.npu.is_available() else "cpu"  # noqa: F841
    device_0, device_1 = torch.device("npu:0"), torch.device("npu:0")
    dtype = to_torch_dtype(cfg.dtype)
    prompts = cfg.prompt  # noqa: F841

    # init logger
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 2)  # noqa: F841
    progress_wrap = tqdm  # noqa: F841

    # 5.build mode & load weights
    logger.info("Building model...")

    model_cfg = cfg.model
    vae_cfg = cfg.vae
    text_encoder_cfg = cfg.text_encoder

    model_cfg['from_pretrained'] = (
        f"{os.environ['PROJECT_PATH']}/resource/multi_modal/" + "opensora_project/open-sora/transformer/"
    )
    vae_cfg['from_pretrained'] = f"{os.environ['PROJECT_PATH']}/resource/multi_modal/opensora_project/open-sora/vae/"
    text_encoder_cfg['from_pretrained'] = (
        f"{os.environ['PROJECT_PATH']}/resource/multi_modal/" + "opensora_project/open-sora/text_encoder/"
    )

    text_encoder = build_module(text_encoder_cfg, MODELS, device=device_1)
    text_encoder.t5.model.to(dtype).eval()
    vae = build_module(vae_cfg, MODELS).to(device_1, dtype).eval()
    image_size = cfg.get("images_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert resolution is not None and aspect_ratio is not None, (
            "resolution and aspect_ratio must be provided if images_size is not provided"
        )
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            model_cfg,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device_0, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder

    # buiild scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)  # noqa: F841

    ori_fp_weight = {}
    for key, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            continue
        ori_fp_weight[key] = value

    calib_dataset = torch.load(
        f"{os.environ['PROJECT_PATH']}/resource/multi_modal/opensora_project/" + "opensora_calib_v2.pth",
        map_location="npu",
    )

    # quantization config
    q_config = QuantConfig(
        w_bit=8,
        a_bit=8,
        w_signed=True,
        a_signed=True,
        w_sym=True,
        a_sym=False,
        act_quant=True,
        act_method=1,
        quant_mode=1,
        disable_names=None,
        amp_num=0,
        keep_acc=None,
        sigma=25,
        device="npu",
    )

    calibrator = Calibrator(model, q_config, calib_data=calib_dataset[:1])
    calibrator.run()
    calibrator.export_quant_safetensor(f"{os.environ['PROJECT_PATH']}/output/ptq-tools/quant_opensora_1_2")


if __name__ == "__main__":
    main()
