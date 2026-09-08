#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.
Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

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

# pylint: disable=logging-fstring-interpolation,too-many-ancestors,consider-merging-isinstance,arguments-renamed,consider-using-from-import,attribute-defined-outside-init,invalid-envvar-default,logging-not-lazy,too-many-lines,unnecessary-comprehension,simplifiable-if-expression

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Generator, List, Callable
from importlib import import_module

import torch
from torch import nn, distributed as dist
from tqdm import tqdm

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func_with_keyword,
)
from msmodelslim.utils.cache import to_device
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.processor.quant.fa3.interface import FA3QuantAdapterInterface, FA3QuantPlaceHolder
from ..interface_hub import (
    ModelInfoInterface,
    LegacyMultimodalPipelineInterface,
    OnlineQuaRotInterface,
    IterSmoothInterface,
    OASQInterface,
)

MAX_RECURSION_DEPTH = 20

EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.\
                The fluffy-furred feline gazes directly at the camera with a relaxed expression.\
                Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills,\
                and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture,\
                as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's\
                intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}

SUPPORTED_TASKS = ['i2v-A14B', 't2v-A14B', 'ti2v-5B']
TASK_CONFIGS = {
    't2v-A14B': 't2v',
    'i2v-A14B': 'i2v',
    'ti2v-5B': 'ti2v',
}


@logger_setter()
class Wan2Point2Adapter(
    BaseModelAdapter,
    ModelInfoInterface,
    LegacyMultimodalPipelineInterface,
    FA3QuantAdapterInterface,
    OnlineQuaRotInterface,
    IterSmoothInterface,
    OASQInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.model_args = None
        self.low_noise_model = None
        self.high_noise_model = None

        self._get_default_model_args()

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'wan2_2'

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> Generator[Any, None, None]:
        return dataset

    def init_model(self, device: DeviceType = DeviceType.NPU) -> Dict[str, nn.Module]:
        if "ti2v" in self.model_args.task:
            return {'': self.transformer}
        else:
            return {'low_noise_model': self.low_noise_model, 'high_noise_model': self.high_noise_model}

    def generate_model_forward(
        self,
        model: torch.nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if "attentionblock" in module.__class__.__name__.lower()
        ]

        first_block_input = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (
                hook_args,
                hook_kwargs,
            )
            raise TransformersForwardBreak()

        hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True)]

        try:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                model(*inputs)
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        except Exception as e:
            raise e
        finally:
            for hook in hooks:
                hook.remove()

        if first_block_input is None:
            raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

        first_block_input = to_device(first_block_input, 'cpu')
        current_inputs = first_block_input

        if dist.is_initialized():
            dist.barrier()

        for name, block in transformer_blocks:
            args, kwargs = current_inputs
            outputs = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = outputs
            current_inputs = ((hidden_states,), current_inputs[1])

    def generate_model_visit(
        self,
        model: torch.nn.Module,
        transformer_blocks: Optional[List[Tuple[str, torch.nn.Module]]] = None,
    ) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func_with_keyword(model, keyword="attentionblock")

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass

    def run_calib_inference(self):
        from wan.configs import SIZE_CONFIGS, MAX_AREA_CONFIGS
        from PIL import Image

        stream = torch.npu.Stream()
        args = self.model_args

        img = None
        for _ in tqdm(range(1), desc='Dump calib data by float model inference'):
            torch.manual_seed(args.base_seed)
            torch.npu.manual_seed(args.base_seed)
            torch.npu.manual_seed_all(args.base_seed)

            begin = time.time()
            if "t2v" in args.task:
                _ = self.wan_t2v.generate(
                    args.prompt,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model,
                )
            elif "ti2v" in args.task:
                if args.image is not None:
                    img = Image.open(args.image).convert("RGB")
                    logging.info("Input image: %r" % args.image)
                _ = self.wan_ti2v.generate(
                    args.prompt,
                    img,
                    size=SIZE_CONFIGS[args.size],
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model,
                )
            elif "i2v" in args.task:
                if args.image is not None:
                    img = Image.open(args.image).convert("RGB")
                    logging.info("Input image: %r" % args.image)
                _ = self.wan_i2v.generate(
                    args.prompt,
                    img,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model,
                )
            stream.synchronize()
            end = time.time()
            logging.info(f"Generating video used time {end - begin: .4f}s")

    def apply_quantization(self, process_model_func):
        from contextlib import contextmanager
        import torch.cuda.amp as amp

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync', noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync', noop_no_sync)

        # 遍历所有子模块，将非blocks部分移至npu
        for name, module in self.transformer.named_modules():
            # 最大的一层，不处理
            if not name:
                continue
            # 处理非blocks模块：确保在npu上
            if not name.startswith('blocks'):
                module.to('npu')
            # 处理blocks模块：确保在cpu上
            else:
                module.to('cpu')

        with (
            amp.autocast(dtype=self.model_args.param_dtype),
            torch.no_grad(),
            no_sync_low_noise(),
            no_sync_high_noise(),
        ):
            process_model_func()

    def load_pipeline(self):
        self._load_pipeline()

    def set_model_args(self, override_model_config: object):
        """
        将 override_model_config 的属性更新到 model_args
        :param override_model_config: 来自 YAML 的配置对象
        """
        self.model_args.ckpt_dir = self.model_path

        missing_attrs = []
        for key in override_model_config.keys():
            if not hasattr(self.model_args, key):
                missing_attrs.append(key)

        if missing_attrs:
            available = [a for a in dir(self.model_args)]
            raise SchemaValidateError(
                f"illegal config attributes: {missing_attrs}. \nsupported config attributes: {available}"
            )

        for key in override_model_config.keys():
            setattr(self.model_args, key, override_model_config[key])

        parser = self._get_parser()
        argv = []
        for key, val in vars(self.model_args).items():
            if val is None:
                continue
            if key == "offload_model":
                argv.extend(["--offload_model", str(val).lower()])
            elif isinstance(val, bool):
                if val:
                    argv.append(f"--{key}")
            else:
                argv.extend([f"--{key}", str(val)])

        self.model_args = parser.parse_args(argv)

        self._validate_args(self.model_args)

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """
        Get adapter config for subgraph-based anti-outlier processing (iter_smooth).

        Defines the subgraph structure for non-fusion.
        Based on wan2_2.py implementation but adapted for Wan2.2 model.

        Includes both vision encoder and text decoder layers.
        """
        adapter_config = []

        # Text decoder layers
        for layer_idx in range(self.transformer.num_layers):
            # Non-fusion: non-fusion
            self_attn_qkv_mapping_config = MappingConfig(
                targets=[
                    f"blocks.{layer_idx}.self_attn.q",
                    f"blocks.{layer_idx}.self_attn.k",
                    f"blocks.{layer_idx}.self_attn.v",
                ]
            )

            # Cross-Attention Q 与 K/V 来源不同，分开统计其smooth scale
            cross_attn_q_mapping_config = MappingConfig(targets=[f"blocks.{layer_idx}.cross_attn.q"])
            cross_attn_kv_mapping_config = MappingConfig(
                targets=[f"blocks.{layer_idx}.cross_attn.k", f"blocks.{layer_idx}.cross_attn.v"]
            )

            # Norm-Linear mappings
            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=self_attn_qkv_mapping_config),
                    AdapterConfig(subgraph_type="norm-linear", mapping=cross_attn_q_mapping_config),
                    AdapterConfig(subgraph_type="norm-linear", mapping=cross_attn_kv_mapping_config),
                ]
            )

            # O层 非融合
            o_self_mapping_config = MappingConfig(targets=[f"blocks.{layer_idx}.self_attn.o"])
            o_cross_mapping_config = MappingConfig(targets=[f"blocks.{layer_idx}.cross_attn.o"])
            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=o_self_mapping_config),
                    AdapterConfig(subgraph_type="norm-linear", mapping=o_cross_mapping_config),
                ]
            )

            up_mapping_config = MappingConfig(targets=[f"blocks.{layer_idx}.ffn.0"])
            down_mapping_config = MappingConfig(targets=[f"blocks.{layer_idx}.ffn.2"])
            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=up_mapping_config),
                    AdapterConfig(subgraph_type="norm-linear", mapping=down_mapping_config),
                ]
            )

        return adapter_config

    def _add_attentioncache_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Attention Cache args")

        group.add_argument("--use_attentioncache", action='store_true')
        group.add_argument("--attentioncache_ratio", type=float, default=1.2)
        group.add_argument("--attentioncache_interval", type=int, default=4)
        group.add_argument("--start_step", type=int, default=12)
        group.add_argument("--end_step", type=int, default=37)

        return parser

    def _add_rainfusion_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Rainfusion args")

        group.add_argument("--use_rainfusion", action='store_true', help="Whether to use sparse fa")
        group.add_argument(
            "--sparsity", type=float, default=0.64, help="Sparsity of flash attention, greater means more speed"
        )
        group.add_argument("--sparse_start_step", type=int, default=15)

        return parser

    def _check_import_dependency(self):
        import importlib

        try:
            for mod in (
                "PIL",
                "wan",
                "wan.configs",
                "wan.distributed.util",
                "wan.utils.prompt_extend",
                "wan.utils.utils",
                "wan.distributed.parallel_mgr",
                "wan.distributed.tp_applicator",
                "mindiesd",
            ):
                importlib.import_module(mod)
        except ImportError as e:
            # Concise import error message
            raise ImportError(
                "Failed to import required components from wan. "
                "Please install the Wan2.2 from Modelers, "
                "make sure you can run the original floating-point inference successfully, "
                "and add the Wan2.2 repository to the Python search path environment variable PYTHONPATH. "
                "e.g. export PYTHONPATH=/path/to/Wan2.2:$PYTHONPATH"
            ) from e

    def _validate_args(self, args):
        """Get default parameter configuration, integrating wan config parameters"""
        self._check_import_dependency()
        from wan.configs import SUPPORTED_SIZES, WAN_CONFIGS

        # Basic check
        if args.ckpt_dir is None:
            raise InvalidModelError("Please specify the checkpoint directory.")
        if not isinstance(args.task, str):
            raise SchemaValidateError(f"task must be a str, but got {type(args.task)}")
        if args.task not in SUPPORTED_TASKS:
            raise UnsupportedError("Unsupported task: %r. Supported tasks are: %s" % (args.task, SUPPORTED_TASKS))
        if args.task not in EXAMPLE_PROMPT:
            raise UnsupportedError("Unsupported task: %r. Supported tasks are: %s" % (args.task, EXAMPLE_PROMPT))
        if "prompt" in args and args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
            args.image = EXAMPLE_PROMPT[args.task]["image"]

        if args.task == "i2v-A14B":
            if args.image is None:
                raise SchemaValidateError("Please specify the image path for i2v.")

        cfg = WAN_CONFIGS[args.task]

        args.task_config = TASK_CONFIGS[args.task]
        args.param_dtype = cfg.param_dtype

        if args.sample_steps is None:
            args.sample_steps = cfg.sample_steps

        if args.sample_shift is None:
            args.sample_shift = cfg.sample_shift

        if args.sample_guide_scale is None:
            args.sample_guide_scale = cfg.sample_guide_scale

        if args.frame_num is None:
            args.frame_num = cfg.frame_num

        args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)  # nosec B311

        if not isinstance(args.sample_steps, int):
            raise SchemaValidateError(f"sample_steps must be an integer, got {type(args.sample_steps).__name__}")
        if args.sample_steps <= 0:
            raise SchemaValidateError("sample_steps must be greater than 0")

        if not isinstance(args.frame_num, int):
            raise SchemaValidateError(f"frame_num must be an integer, got {type(args.frame_num).__name__}")
        if args.frame_num <= 0:
            raise SchemaValidateError("frame_num must be greater than 0")

        # Size check
        if args.size not in SUPPORTED_SIZES[args.task]:
            raise UnsupportedError(
                "Unsupported size %r for task %r, supported sizes are: %r"
                % (args.size, args.task, SUPPORTED_SIZES[args.task])
            )

        # Validate prompt
        prompt = getattr(args, "prompt", None)
        if prompt is None:
            raise SchemaValidateError("Missing required parameter: prompt")
        if not isinstance(args.prompt, str):
            raise SchemaValidateError(f"prompt must be a string, got {type(args.prompt).__name__}")
        if not args.prompt.strip():
            raise SchemaValidateError("prompt cannot be an empty string")

        # Validate offload_model
        if "offload_model" in args and args.offload_model and not isinstance(args.offload_model, bool):
            raise SchemaValidateError(
                f"offload_model must be a boolean (True/False), got {type(args.offload_model).__name__}"
            )

    def _get_parser(self) -> Dict[str, Any]:
        """Get default parameter configuration, integrating wan config parameters"""
        self._check_import_dependency()
        from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
        from wan.utils.utils import str2bool

        # Create argument parser and add all necessary configurations
        parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
        parser.add_argument(
            "--task", type=str, default="t2v-A14B", choices=list(WAN_CONFIGS.keys()), help="The task to run."
        )
        parser.add_argument(
            "--size",
            type=str,
            default="1280*720",
            choices=list(SIZE_CONFIGS.keys()),
            help="The area (width*height) of the generated video. For the I2V task,\
                the aspect ratio of the output video will follow that of the input image.",
        )
        parser.add_argument(
            "--frame_num",
            type=int,
            default=None,
            help="How many frames of video are generated. The number should be 4n+1",
        )
        parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory.")
        parser.add_argument(
            "--offload_model",
            type=str2bool,
            default=None,
            help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
        )
        parser.add_argument("--cfg_size", type=int, default=1, help="The size of the cfg parallelism in DiT.")
        parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
        parser.add_argument(
            "--ring_size", type=int, default=1, help="The size of the ring attention parallelism in DiT."
        )
        parser.add_argument("--tp_size", type=int, default=1, help="The size of the tensor parallelism in DiT.")
        parser.add_argument(
            "--vae_parallel", action="store_true", default=False, help="Whether to use parallel for vae."
        )
        parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
        parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
        parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
        parser.add_argument("--save_file", type=str, default=None, help="The file to save the generated video to.")
        parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the video from.")
        parser.add_argument(
            "--use_prompt_extend", action="store_true", default=False, help="Whether to use prompt extend."
        )
        parser.add_argument(
            "--prompt_extend_method",
            type=str,
            default="local_qwen",
            choices=["dashscope", "local_qwen"],
            help="The prompt extend method to use.",
        )
        parser.add_argument("--prompt_extend_model", type=str, default=None, help="The prompt extend model to use.")
        parser.add_argument(
            "--prompt_extend_target_lang",
            type=str,
            default="zh",
            choices=["zh", "en"],
            help="The target language of prompt extend.",
        )
        parser.add_argument("--base_seed", type=int, default=-1, help="The seed to use for generating the video.")
        parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")
        parser.add_argument(
            "--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="The solver used to sample."
        )
        parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
        parser.add_argument(
            "--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers."
        )
        parser.add_argument("--sample_guide_scale", type=float, default=None, help="Classifier free guidance scale.")
        parser.add_argument(
            "--convert_model_dtype",
            action="store_true",
            default=False,
            help="Whether to convert model paramerters dtype.",
        )
        parser.add_argument(
            "--quant_dit_path", type=str, default=None, help="Path for calibration data or weight export."
        )
        parser = self._add_attentioncache_args(parser)
        parser = self._add_rainfusion_args(parser)
        return parser

    def _get_default_model_args(self):
        parser = self._get_parser()
        args = parser.parse_args([])
        self.model_args = args

    def _init_logging(self, rank):
        if rank == 0:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                handlers=[logging.StreamHandler(stream=sys.stdout)],
            )
        else:
            logging.basicConfig(level=logging.ERROR)

    def _load_pipeline(self):
        self._check_import_dependency()

        from PIL import Image
        import wan
        from wan.configs import WAN_CONFIGS
        from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
        from mindiesd import CacheConfig, CacheAgent

        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        self._init_logging(rank)

        args = self.model_args

        if args.offload_model is None:
            args.offload_model = False if world_size > 1 else True
            logging.info(f"offload_model is not specified, set to {args.offload_model}.")

        # 不支持并行
        if args.t5_fsdp or args.dit_fsdp:
            raise SchemaValidateError("t5_fsdp and dit_fsdp are not supported in non-distributed environments.")

        if args.cfg_size > 1 or args.ulysses_size > 1 or args.ring_size > 1:
            raise SchemaValidateError("context parallel are not supported in non-distributed environments.")

        if args.vae_parallel:
            raise SchemaValidateError("vae parallel are not supported in non-distributed environments.")

        if args.tp_size > 1:
            raise UnsupportedError("Tensor Parallel is not supported now")
        if "ti2v" not in args.task and args.use_attentioncache:
            raise UnsupportedError("%r unsupported attentioncache now" % args.task)

        if args.use_prompt_extend:
            if args.prompt_extend_method == "dashscope":
                prompt_expander = DashScopePromptExpander(
                    model_name=args.prompt_extend_model, task=args.task, is_vl=args.image is not None
                )
            elif args.prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=args.prompt_extend_model, task=args.task, is_vl=args.image is not None, device=rank
                )
            else:
                raise UnsupportedError("Unsupported prompt_extend_method: %r" % args.prompt_extend_method)

        cfg = WAN_CONFIGS[args.task]

        logging.info("Generation job args: %r", args)
        logging.info("Generation model config: %r", cfg)

        logging.info("Input prompt: %r" % args.prompt)
        img = None
        if args.image is not None:
            img = Image.open(args.image).convert("RGB")
            logging.info("Input image: %r" % args.image)

        # prompt extend
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt, image=img, tar_lang=args.prompt_extend_target_lang, seed=args.base_seed
                )
                if not prompt_output.status:
                    logging.info("Extending prompt failed: %r" % prompt_output.message)
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info("Extended prompt: %r" % args.prompt)

        rainfusion_config = {
            "sparsity": args.sparsity,
            "skip_timesteps": args.sparse_start_step,
            "grid_size": None,
            "atten_mask_all": None,
        }

        if "t2v" in args.task:
            logging.info("Creating WanT2V pipeline.")
            self.wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                quant_dit_path=args.quant_dit_path,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_sp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
                convert_model_dtype=args.convert_model_dtype,
                use_vae_parallel=args.vae_parallel,
            )

            transformer_low = self.wan_t2v.low_noise_model
            transformer_high = self.wan_t2v.high_noise_model

            if args.use_rainfusion:
                if args.dit_fsdp:
                    transformer_low._fsdp_wrapped_module.rainfusion_config = rainfusion_config
                    transformer_high._fsdp_wrapped_module.rainfusion_config = rainfusion_config
                else:
                    transformer_low.rainfusion_config = rainfusion_config
                    transformer_high.rainfusion_config = rainfusion_config

            if args.use_attentioncache:
                config_high = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(transformer_high.blocks),
                    steps_count=args.sample_steps,
                    step_start=args.start_step,
                    step_interval=args.attentioncache_interval,
                    step_end=args.end_step,
                )
            else:
                config_high = CacheConfig(
                    method="attention_cache", blocks_count=len(transformer_high.blocks), steps_count=args.sample_steps
                )
            config_low = CacheConfig(
                method="attention_cache", blocks_count=len(transformer_low.blocks), steps_count=args.sample_steps
            )
            cache_high = CacheAgent(config_high)
            cache_low = CacheAgent(config_low)

            if args.dit_fsdp:
                for block in transformer_high._fsdp_wrapped_module.blocks:
                    block._fsdp_wrapped_module.cache = cache_high
                    block._fsdp_wrapped_module.args = args
                for block in transformer_low._fsdp_wrapped_module.blocks:
                    block._fsdp_wrapped_module.cache = cache_low
                    block._fsdp_wrapped_module.args = args
            else:
                for block in transformer_high.blocks:
                    block.cache = cache_high
                    block.args = args
                for block in transformer_low.blocks:
                    block.cache = cache_low
                    block.args = args

            self.low_noise_model = self.wan_t2v.low_noise_model
            self.high_noise_model = self.wan_t2v.high_noise_model

        elif "ti2v" in args.task:
            logging.info("Creating WanTI2V pipeline.")
            self.wan_ti2v = wan.WanTI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                quant_dit_path=args.quant_dit_path,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_sp=(args.ulysses_size > 1),
                t5_cpu=args.t5_cpu,
                convert_model_dtype=args.convert_model_dtype,
                use_vae_parallel=args.vae_parallel,
            )

            transformer = self.wan_ti2v.model

            if args.use_rainfusion:
                if args.dit_fsdp:
                    transformer._fsdp_wrapped_module.rainfusion_config = rainfusion_config
                else:
                    transformer.rainfusion_config = rainfusion_config

            if args.use_attentioncache:
                config = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(transformer.blocks),
                    steps_count=args.sample_steps,
                    step_start=args.start_step,
                    step_interval=args.attentioncache_interval,
                    step_end=args.end_step,
                )
            else:
                config = CacheConfig(
                    method="attention_cache", blocks_count=len(transformer.blocks), steps_count=args.sample_steps
                )
            cache = CacheAgent(config)
            if args.dit_fsdp:
                for block in transformer._fsdp_wrapped_module.blocks:
                    block._fsdp_wrapped_module.cache = cache
                    block._fsdp_wrapped_module.args = args
            else:
                for block in transformer.blocks:
                    block.cache = cache
                    block.args = args

            self.transformer = transformer

        else:
            logging.info("Creating WanI2V pipeline.")
            self.wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                quant_dit_path=args.quant_dit_path,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_sp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
                convert_model_dtype=args.convert_model_dtype,
                use_vae_parallel=args.vae_parallel,
            )

            transformer_low = self.wan_i2v.low_noise_model
            transformer_high = self.wan_i2v.high_noise_model

            if args.use_rainfusion:
                if args.dit_fsdp:
                    transformer_low._fsdp_wrapped_module.rainfusion_config = rainfusion_config
                    transformer_high._fsdp_wrapped_module.rainfusion_config = rainfusion_config
                else:
                    transformer_low.rainfusion_config = rainfusion_config
                    transformer_high.rainfusion_config = rainfusion_config

            if args.use_attentioncache:
                config_low = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(transformer_low.blocks),
                    steps_count=args.sample_steps,
                    step_start=args.start_step,
                    step_interval=args.attentioncache_interval,
                    step_end=args.end_step,
                )
            else:
                config_low = CacheConfig(
                    method="attention_cache", blocks_count=len(transformer_low.blocks), steps_count=args.sample_steps
                )
            config_high = CacheConfig(
                method="attention_cache", blocks_count=len(transformer_high.blocks), steps_count=args.sample_steps
            )
            cache_low = CacheAgent(config_low)
            cache_high = CacheAgent(config_high)

            if args.dit_fsdp:
                for block in transformer_high._fsdp_wrapped_module.blocks:
                    block._fsdp_wrapped_module.cache = cache_high
                    block._fsdp_wrapped_module.args = args
                for block in transformer_low._fsdp_wrapped_module.blocks:
                    block._fsdp_wrapped_module.cache = cache_low
                    block._fsdp_wrapped_module.args = args
            else:
                for block in transformer_high.blocks:
                    block.cache = cache_high
                    block.args = args
                for block in transformer_low.blocks:
                    block.cache = cache_low
                    block.args = args

            self.low_noise_model = self.wan_i2v.low_noise_model
            self.high_noise_model = self.wan_i2v.high_noise_model

    # ===== OnlineQuaRotInterface =====
    def get_online_rotation_configs(self, model: Optional[nn.Module] = None):
        """
        返回在线旋转配置，配置 q_rot 和 k_rot 为旋转矩阵替换。

        如果提供了 model，会在此方法中直接给 WanSelfAttention 和 WanCrossAttention 挂载 q_rot 和 k_rot Identity 模块。

        Args:
            model: 可选的模型实例，如果提供，会在此方法中挂载 Identity 模块

        Returns:
            Dict[str, RotationConfig]: 模块名到旋转配置的映射
        """
        configs = {}

        # 如果提供了 model，直接挂载 Identity 模块
        if model is not None:
            for name, module in model.named_modules():
                module_type = module.__class__.__name__

                # 只处理目标模块类型
                if module_type not in ["WanSelfAttention", "WanCrossAttention"]:
                    continue

                try:
                    # 创建并挂载 q_rot 和 k_rot Identity 模块
                    if not hasattr(module, 'q_rot'):
                        module.register_module('q_rot', nn.Identity())
                    if not hasattr(module, 'k_rot'):
                        module.register_module('k_rot', nn.Identity())
                    get_logger().debug(f"Registered q_rot and k_rot Identity modules for {name}")
                except Exception as e:
                    get_logger().warning(f"Failed to register rotation modules for {name}: {str(e)}")

        # 配置旋转，q_rot 和 k_rot 使用相同的随机数种子，确保生成相同的旋转矩阵
        shared_seed = 1234  # q_rot 和 k_rot 共享的随机数种子

        # 遍历模型找到所有目标模块并配置旋转
        target_model = model if model is not None else getattr(self, 'transformer', None)
        if target_model is None:
            # 尝试从其他可能的模型获取
            if hasattr(self, 'low_noise_model') and self.low_noise_model is not None:
                target_model = self.low_noise_model
            elif hasattr(self, 'high_noise_model') and self.high_noise_model is not None:
                target_model = self.high_noise_model

        if target_model is None:
            get_logger().warning("No model provided and transformer not available, returning empty rotation configs")
            return configs

        # 获取 head_dim - 从第一个 attention 模块获取
        head_dim = None
        for name, module in target_model.named_modules():
            if module.__class__.__name__ in ["WanSelfAttention", "WanCrossAttention"]:
                if hasattr(module, 'head_dim'):
                    head_dim = module.head_dim
                    break

        if head_dim is None:
            # 尝试从 transformer 配置获取
            if hasattr(target_model, 'dim') and hasattr(target_model, 'num_heads'):
                head_dim = target_model.dim // target_model.num_heads
            else:
                get_logger().warning("Could not determine head_dim, returning empty rotation configs")
                return configs

        # 使用全局 head_dim 为所有目标模块配置旋转
        for name, module in target_model.named_modules():
            module_type = module.__class__.__name__

            # 只处理目标模块类型
            if module_type not in ["WanSelfAttention", "WanCrossAttention"]:
                continue

            # 配置 q_rot
            q_rot_path = f"{name}.q_rot" if name else "q_rot"
            configs[q_rot_path] = OnlineQuaRotInterface.RotationConfig(
                rotation_type="replace",
                rotation_size=head_dim,
                rotation_mode=OnlineQuaRotInterface.QuaRotMode.HADAMARD,
                block_size=-1,
                seed=shared_seed,
                dtype=torch.bfloat16,
            )

            # 配置 k_rot（使用相同的种子，确保与 q_rot 使用相同的旋转矩阵）
            k_rot_path = f"{name}.k_rot" if name else "k_rot"
            configs[k_rot_path] = OnlineQuaRotInterface.RotationConfig(
                rotation_type="replace",
                rotation_size=head_dim,
                rotation_mode=OnlineQuaRotInterface.QuaRotMode.HADAMARD,
                block_size=-1,
                seed=shared_seed,
                dtype=torch.bfloat16,
            )

        return configs

    # ===== FA3QuantAdapterInterface =====
    def inject_fa3_placeholders(
        self, root_name: str, root_module: nn.Module, should_inject: Callable[[str], bool]
    ) -> None:
        """为 Wan 模型的 WanSelfAttention 和 WanCrossAttention 安装 FA3 占位，并包裹 forward 调用这些占位。

        - 在每个目标模块下注入子模块：fa3_q, fa3_k, fa3_v
        - 包裹其 forward 方法，在计算 Q、K、V 后，依次调用占位：
            q = self.fa3_q(q)
            k = self.fa3_k(k)
            v = self.fa3_v(v)
        """

        def _wrap_self_attention_forward(module: nn.Module):
            """包裹 WanSelfAttention 的 forward 方法"""
            original_forward = module.forward

            # 动态导入必要的函数
            # rope_apply 在 wan.modules.model 中定义
            wan_model_module = import_module(original_forward.__module__)
            rope_apply = getattr(wan_model_module, 'rope_apply', None)
            if rope_apply is None:
                raise ImportError(f"Could not find rope_apply in {original_forward.__module__}")

            # attention 从 wan.modules.attention 导入（相对导入 .attention）
            module_parts = original_forward.__module__.rsplit('.', 1)
            if len(module_parts) == 2:
                base_module_path = module_parts[0]
                attention_module_path = base_module_path + '.attention'
                try:
                    wan_attention_module = import_module(attention_module_path)
                    attention = getattr(wan_attention_module, 'attention', None)
                    if attention is None:
                        raise AttributeError(f"attention not found in {attention_module_path}")
                except (ImportError, AttributeError) as e:
                    raise ImportError(f"Could not import attention from {attention_module_path}: {e}")
            else:
                raise ImportError(f"Could not determine attention module path from {original_forward.__module__}")

            def new_forward(
                self,
                x,
                seq_lens,
                grid_sizes,
                freqs,
                args=None,
                rainfusion_config=None,
                t_idx=None,
            ):
                b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

                # query, key, value function
                def qkv_fn(x):
                    q = self.norm_q(self.q(x)).view(b, s, n, d)
                    k = self.norm_k(self.k(x)).view(b, s, n, d)
                    v = self.v(x).view(b, s, n, d)
                    return q, k, v

                q, k, v = qkv_fn(x)

                # ===== 应用在线旋转（在 FA3 量化之前）=====
                if hasattr(self, 'q_rot'):
                    q = self.q_rot(q)
                if hasattr(self, 'k_rot'):
                    k = self.k_rot(k)
                # ==========================================

                # ===== 插入 FA3 占位 =====
                if hasattr(self, 'fa3_q'):
                    q = self.fa3_q(q)
                if hasattr(self, 'fa3_k'):
                    k = self.fa3_k(k)
                if hasattr(self, 'fa3_v'):
                    v = self.fa3_v(v)
                # ========================

                x = attention(
                    q=rope_apply(q, grid_sizes, freqs),
                    k=rope_apply(k, grid_sizes, freqs),
                    v=v,
                    k_lens=seq_lens,
                    window_size=self.window_size,
                    rainfusion_config=rainfusion_config,
                    t_idx=t_idx,
                )

                # output
                x = x.flatten(2)
                x = self.o(x)
                return x

            # pylint: disable=no-value-for-parameter
            module.forward = new_forward.__get__(module, module.__class__)

        def _wrap_cross_attention_forward(module: nn.Module):
            """包裹 WanCrossAttention 的 forward 方法"""
            original_forward = module.forward

            # 动态导入必要的函数
            # attention 从 wan.modules.attention 导入（相对导入 .attention）
            module_parts = original_forward.__module__.rsplit('.', 1)
            if len(module_parts) == 2:
                base_module_path = module_parts[0]
                attention_module_path = base_module_path + '.attention'
                try:
                    wan_attention_module = import_module(attention_module_path)
                    attention = getattr(wan_attention_module, 'attention', None)
                    if attention is None:
                        raise AttributeError(f"attention not found in {attention_module_path}")
                except (ImportError, AttributeError) as e:
                    raise ImportError(f"Could not import attention from {attention_module_path}: {e}")
            else:
                raise ImportError(f"Could not determine attention module path from {original_forward.__module__}")

            def new_forward(
                self,
                x,
                context,
                context_lens,
            ):
                b, n, d = x.size(0), self.num_heads, self.head_dim

                # compute query, key, value
                q = self.norm_q(self.q(x)).view(b, -1, n, d)
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)

                # ===== 应用在线旋转（在 FA3 量化之前）=====
                if hasattr(self, 'q_rot'):
                    q = self.q_rot(q)
                if hasattr(self, 'k_rot'):
                    k = self.k_rot(k)
                # ==========================================

                # ===== 插入 FA3 占位 =====
                if hasattr(self, 'fa3_q'):
                    q = self.fa3_q(q)
                if hasattr(self, 'fa3_k'):
                    k = self.fa3_k(k)
                if hasattr(self, 'fa3_v'):
                    v = self.fa3_v(v)
                # ========================

                # compute attention
                x = attention(q, k, v, k_lens=context_lens)

                # output
                x = x.flatten(2)
                x = self.o(x)
                return x

            # pylint: disable=no-value-for-parameter
            module.forward = new_forward.__get__(module, module.__class__)

        # 遍历并注入占位符
        for name, module in root_module.named_modules():
            module_type = module.__class__.__name__

            # 检查是否是目标模块类型
            if module_type not in ["WanSelfAttention", "WanCrossAttention"]:
                continue

            full_name = f"{root_name}.{name}" if root_name else name
            if not should_inject(full_name):
                continue

            if name == "":
                prefix = ""
            else:
                prefix = f"{name}."

            # 为该模块注入占位符
            root_module.set_submodule(f"{prefix}fa3_q", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{prefix}fa3_k", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{prefix}fa3_v", FA3QuantPlaceHolder(ratio=1.0))

            # 包裹对应的 forward 方法
            if module_type == "WanSelfAttention":
                _wrap_self_attention_forward(module)
            elif module_type == "WanCrossAttention":
                _wrap_cross_attention_forward(module)

            get_logger().info(f"Injected FA3 placeholders for {full_name}")
