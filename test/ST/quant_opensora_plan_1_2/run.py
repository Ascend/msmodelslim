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
# pylint: disable=consider-using-with,invalid-envvar-default,no-name-in-module,pointless-string-statement,possibly-used-before-assignment,redefined-outer-name,undefined-variable,ungrouped-imports,unspecified-encoding,used-before-assignment,using-constant-test

"""
导入相关依赖
"""
import os  # noqa: E402
import gc  # noqa: E402
import sys  # noqa: E402
import math  # noqa: E402
import argparse  # noqa: E402
import imageio  # noqa: E402
from tqdm import tqdm  # noqa: E402
from safetensors import safe_open  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402
import torch.distributed as dist  # noqa: E402

from diffusers.schedulers import (  # noqa: E402
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
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler  # noqa: E402
from torchvision.utils import save_image  # noqa: E402
from transformers import T5Tokenizer, MT5EncoderModel  # noqa: E402

sys.path.append(f"{os.environ['PROJECT_PATH']}/resource/multi_modal/opensoraplan_project")
from opensora.models.causalvideovae import ae_stride_config, CausalVAEModelWrapper  # noqa: E402
from opensora.models.diffusion.udit.modeling_udit import UDiTT2V  # noqa: E402
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V  # noqa: E402
from opensora.utils.utils import save_video_grid  # noqa: E402
from opensora.sample.pipeline_opensora_sp_without_text_encoder import TextEncoderWrapper, OpenSoraPipeline  # noqa: E402
from opensora.npu_config import npu_config  # noqa: E402
from opensora.acceleration.parallel_states import initialize_sequence_parallel_state, hccl_info  # noqa: E402

from msmodelslim.pytorch.quant.ptq_tools import Calibrator, QuantConfig  # noqa: E402
from msmodelslim.pytorch.quant.ptq_tools.quant_modules import TensorQuantizer  # noqa: E402

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
导入相关模型
"""


def load_t2v_checkpoint(model_path):
    if args.model_type == 'udit':
        transformer_model = UDiTT2V.from_pretrained(
            model_path, cache_dir=args.cache_dir, low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype
        )
    elif args.model_type == 'dit':
        transformer_model = OpenSoraT2V.from_pretrained(
            model_path, cache_dir=args.cache_dir, low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype
        )
    else:
        transformer_model = LatteT2V.from_pretrained(  # noqa: F821
            model_path, cache_dir=args.cache_dir, low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype
        )

    # set eval mode
    transformer_model.eval()
    pipeline = OpenSoraPipeline(vae=vae, scheduler=scheduler, transformer=transformer_model).to(device)

    if args.compile:
        pipeline.transformer = torch.compile(pipeline.transformer)
    return pipeline


def get_latest_path():
    # Get the most recent checkpoint
    dirs = os.listdir(args.model_path)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None

    return path


def encode_prompts(text_encoder, tokenizer):
    text_encoder_wrapper = TextEncoderWrapper(text_encoder, tokenizer)
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()  # noqa: SIM115
        args.text_prompt = [i.strip() for i in text_prompt]

    positive_prompt = """
    (masterpiece), (best quality), (ultra-detailed), (unwatermarked),
    {}.
    emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo,
    sharp focus, high budget, cinemascope, moody, epic, gorgeous
    """

    negative_prompt = """nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit,
    fewer digits, cropped, worst quality,
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """
    text_encoder_res_list = []
    for prompt in tqdm(args.text_prompt):
        npu_config.seed_everything(42)
        text_encoder_res = text_encoder_wrapper(
            positive_prompt.format(prompt),
            negative_prompt=negative_prompt,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            clean_caption=True,
            max_sequence_length=args.max_sequence_length,
        )
        text_encoder_res_list.append(text_encoder_res)
    text_encoder_wrapper.text_encoder = None
    del text_encoder_wrapper.text_encoder
    torch.cuda.empty_cache()
    return text_encoder_res_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='Open-Sora-Plan-v1.2.0/93x720p/')
    parser.add_argument("--version", type=str, default=None, choices=[None, '65x512x512', '65x256x256', '17x256x256'])
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--device", type=str, default='')
    parser.add_argument("--cache_dir", type=str, default='../cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_D4_4x8x8')
    parser.add_argument("--ae_path", type=str, default='Open-Sora-Plan-v1.2.0/vae/')
    parser.add_argument("--text_encoder_name", type=str, default='mt5-xxl/')
    parser.add_argument("--save_img_path", type=str, default="sample_videos/")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="EulerAncestralDiscrete")
    parser.add_argument("--num_sampling_steps", type=int, default=1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--text_prompt", nargs='+', default="prompt_list_0.txt")
    parser.add_argument('--tile_overlap_factor', type=float, default=0.125)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--model_type', type=str, default="dit", choices=['dit', 'udit', 'latte'])
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save_memory', action='store_true')
    args = parser.parse_args()

    if torch_npu is not None:
        npu_config.print_msg(args)

    # 初始化分布式环境
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    print('world_size', world_size)
    if torch_npu is not None and npu_config.on_npu:
        torch_npu.npu.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    initialize_sequence_parallel_state(world_size)

    npu_config.seed_everything(42)
    weight_dtype = torch.bfloat16
    device = torch.cuda.current_device()

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
    vae.eval()

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

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    if args.num_frames == 1:
        video_length = 1
        ext = 'jpg'
    else:
        ext = 'mp4'

    latest_path = None
    save_img_path = args.save_img_path

    if True:
        full_path = f"{args.model_path}"
        pipeline = load_t2v_checkpoint(full_path)
        print('load model')
        text_encoder = MT5EncoderModel.from_pretrained(
            args.text_encoder_name, cache_dir=args.cache_dir, low_cpu_mem_usage=True, torch_dtype=weight_dtype
        ).to(device)
        tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
        text_encoder.cuda().bfloat16().eval()

        if npu_config is not None and npu_config.on_npu and npu_config.profiling:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            )
            profile_output_path = "/path_to/image_data/npu_profiling_t2v"
            os.makedirs(profile_output_path, exist_ok=True)

            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(wait=10000, warmup=0, active=1, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"{profile_output_path}/"),
            ) as prof:
                prof.step()
        else:
            print('not using profiling!')
            video_grids = []
            if not isinstance(args.text_prompt, list):
                args.text_prompt = [args.text_prompt]
            if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
                text_prompt = open(args.text_prompt[0], 'r').readlines()  # noqa: SIM115
                args.text_prompt = [i.strip() for i in text_prompt]

            try:
                checkpoint_name = f"{os.path.basename(args.model_path)}"
            except:  # noqa: E722
                checkpoint_name = "final"
            positive_prompt = """
            (masterpiece), (best quality), (ultra-detailed), (unwatermarked),
            {}.
            emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo,
            sharp focus, high budget, cinemascope, moody, epic, gorgeous
            """

            negative_prompt = """nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit,
            fewer digits, cropped, worst quality,
            low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
            """

            text_encoder_res_list = encode_prompts(text_encoder, tokenizer)
            torch.cuda.synchronize()
            referrers = gc.get_referrers(text_encoder)
            for ref in referrers:
                try:
                    print(str(ref)[:100])
                except Exception as e:
                    print(f"Could not print referrer: {e}")

            for ref in referrers:
                if isinstance(ref, dict):
                    keys_to_delete = [key for key, value in ref.items() if value is text_encoder]
                    for key in keys_to_delete:
                        del ref[key]
                elif isinstance(ref, list):
                    while text_encoder in ref:
                        ref.remove(text_encoder)

            text_encoder = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            calib_dataset = torch.load(
                f"{os.environ['PROJECT_PATH']}/resource/multi_modal/opensoraplan_project/"
                + "calib_data_osp_93x720_subset.pth",
                map_location="cpu",
            )

            # quantization initialization
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
            calibrator = Calibrator(pipeline.transformer, q_config, calib_dataset[:1])
            calibrator.run()

            calibrator.export_quant_safetensor(f"{os.environ['PROJECT_PATH']}/output/ptq-tools/quant_opensora_plan_1_2")

            state_dict = {}
            with safe_open(
                f"{os.environ['PROJECT_PATH']}/output/ptq-tools/quant_opensora_plan_1_2/"
                + "quant_model_weight_w8a8.safetensors",
                framework="pt",
            ) as f:
                # 获取所有张量的名称
                for key in f.keys():
                    # 将张量加载到 state_dict 中
                    state_dict[key] = f.get_tensor(key)

            # 将加载好的 state_dict 应用到模型
            for name, module in pipeline.transformer.named_modules():
                if isinstance(module, TensorQuantizer):
                    module.stop_calibration()
            for name, module in pipeline.transformer.named_modules():
                if isinstance(module, TensorQuantizer):
                    if module.is_input and module.input_offset is None:
                        print(name)
                    if module.is_input and module.input_offset is not None:
                        fp_name = name.rsplit(".", 1)[0]
                        module.input_offset = state_dict[fp_name + ".input_offset"]
                        module.input_scale = state_dict[fp_name + ".input_scale"]
                    module.stop_calibration()

            quant_params = {}
            for name, module in pipeline.transformer.named_modules():
                if isinstance(module, TensorQuantizer):
                    quant_params[name + ".q_weights"] = {
                        "weight_scale": module.weight_scale,
                        "weight_offset": module.weight_offset,
                    }
                    quant_params[name + ".q_acts"] = {
                        "input_scale": module.input_scale,
                        "input_offset": module.input_offset,
                    }

            torch.save(
                quant_params,
                f"{os.environ['PROJECT_PATH']}/output/ptq-tools/" + "quant_opensora_plan_1_2/osp_quant_params.pth",
            )

            q_params = torch.load(
                f"{os.environ['PROJECT_PATH']}/output/ptq-tools/quant_opensora_plan_1_2/" + "osp_quant_params.pth",
                map_location=pipeline.transformer.device,
            )
            for name, module in pipeline.transformer.named_modules():
                if isinstance(module, TensorQuantizer):
                    module.weight_scale = q_params[name + ".q_weights"]["weight_scale"]
                    module.weight_offset = q_params[name + ".q_weights"]["weight_offset"]
                    module.input_offset = q_params[name + ".q_acts"]["input_offset"]
                    module.input_scale = q_params[name + ".q_acts"]["input_scale"]
                    module.stop_calibration()

            for index, prompt in enumerate(args.text_prompt):
                npu_config.seed_everything(42)
                videos = pipeline(
                    text_encoder_res_list[index],
                    positive_prompt.format(prompt),
                    negative_prompt=negative_prompt,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_sampling_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    mask_feature=True,
                    device=args.device,
                    max_sequence_length=args.max_sequence_length,
                ).images
                print(videos.shape)

                if hccl_info.rank <= 0:
                    try:
                        if args.num_frames == 1:
                            videos = videos[:, 0].permute(0, 3, 1, 2)  # b t h w c -> b c h w
                            save_image(
                                videos / 255.0,
                                os.path.join(
                                    args.save_img_path,
                                    f'{args.sample_method}_{index}_{checkpoint_name}_gs{args.guidance_scale}'
                                    + '_s{args.num_sampling_steps}.{ext}',
                                ),
                                nrow=1,
                                normalize=True,
                                value_range=(0, 1),
                            )  # t c h w

                        else:
                            imageio.mimwrite(
                                os.path.join(
                                    args.save_img_path,
                                    f'{args.sample_method}_{index}_{checkpoint_name}'
                                    + '__gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}',
                                ),
                                videos[0],
                                fps=args.fps,
                                quality=6,
                                codec='libx264',
                                output_params=['-threads', '20'],
                            )  # highest quality is 10, lowest is 0
                    except:  # noqa: E722
                        print('Error when saving {}'.format(prompt))
                    video_grids.append(videos)
            if hccl_info.rank <= 0:
                video_grids = torch.cat(video_grids, dim=0)

                def get_file_name():
                    return os.path.join(
                        args.save_img_path,
                        f'{args.sample_method}_gs{args.guidance_scale}\
                                            _s{args.num_sampling_steps}_{checkpoint_name}.{ext}',
                    )

                if args.num_frames == 1:
                    save_image(
                        video_grids / 255.0,
                        get_file_name(),
                        nrow=math.ceil(math.sqrt(len(video_grids))),
                        normalize=True,
                        value_range=(0, 1),
                    )
                else:
                    video_grids = save_video_grid(video_grids)
                    imageio.mimwrite(get_file_name(), video_grids, fps=args.fps, quality=6)

                print('save path {}'.format(args.save_img_path))
