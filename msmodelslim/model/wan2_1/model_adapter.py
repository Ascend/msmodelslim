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

# pylint: disable=logging-fstring-interpolation,too-many-ancestors,consider-merging-isinstance,arguments-renamed,consider-using-from-import,attribute-defined-outside-init,invalid-envvar-default,unnecessary-comprehension

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Generator, List

import torch
from torch import nn, distributed as dist
from tqdm import tqdm

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func_with_keyword,
)
from msmodelslim.utils.cache import to_device
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import logger_setter
from ..interface_hub import ModelInfoInterface, LegacyMultimodalPipelineInterface

MAX_RECURSION_DEPTH = 20

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted "
        "stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted "
        "stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred "
        "feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the "
        "background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. "
        "The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up "
        "shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
}

SUPPORTED_TASKS = ['t2v-14B', 't2v-1.3B']
TASK_CONFIGS = {
    't2v-1.3B': 't2v-1.3B',
    't2v-14B': 't2v',
}


@logger_setter()
class Wan2Point1Adapter(
    BaseModelAdapter,
    ModelInfoInterface,
    LegacyMultimodalPipelineInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.model_args = None

        self._get_default_model_args()

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'wan2_1'

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> Generator[Any, None, None]:
        return dataset

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return {'': self.transformer}

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

        # 存储第一个transformer block的输入
        first_block_input = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (
                hook_args,
                hook_kwargs,
            )
            raise TransformersForwardBreak()

        hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True)]

        # 执行一次前向传播以获取输入
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

        # 循环处理每个transformer block
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
        """运行校准推理"""
        from wan.configs import SIZE_CONFIGS

        stream = torch.npu.Stream()
        args = self.model_args

        self.wan_t2v.model.to('npu')
        # Start sampling
        for _ in tqdm(range(1), desc='Dump calib data by float model inference'):
            # set seed
            torch.manual_seed(args.base_seed)
            torch.npu.manual_seed(args.base_seed)
            torch.npu.manual_seed_all(args.base_seed)

            begin = time.time()
            _ = self.wan_t2v.generate(
                self.model_args.prompt,
                size=SIZE_CONFIGS[args.size],
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

        no_sync = getattr(self, 'no_sync', noop_no_sync)

        # 遍历所有子模块，将非blocks部分移至npu
        for name, module in self.transformer.named_modules():
            # 处理非blocks模块：确保在npu上
            if not name.startswith('blocks'):
                module.to('npu')
            # 处理blocks模块：确保在cpu上
            else:
                module.to('cpu')
        with amp.autocast(dtype=torch.bfloat16), torch.no_grad(), no_sync():
            process_model_func()

    def load_pipeline(self):
        self._load_pipeline()
        self._setup_cache()

    def set_model_args(self, override_model_config: object):
        """
        将 override_model_config 的属性更新到 model_args
        :param override_model_config: 来自 YAML 的配置对象
        """
        # 模型路径
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

        # 执行更新
        for key in override_model_config.keys():
            setattr(self.model_args, key, override_model_config[key])

        # 验证参数配置
        self._validate_args(self.model_args)

        parser = self._get_parser()
        # 1. 把 Namespace 还原为等价的命令行参数
        argv = []
        for key, val in vars(self.model_args).items():
            if val is None:  # 允许 None 的参数直接跳过
                continue
            # flag 型（bool/存根）特殊处理
            if key == "offload_model":  # 专门处理 offload_model，让它带值
                argv.extend(["--offload_model", str(val).lower()])
            elif isinstance(val, bool):
                if val:  # True -> --flag
                    argv.append(f"--{key}")
                # False 时忽略
            else:
                argv.extend([f"--{key}", str(val)])

        # 2. 重新解析，得到经过校验/类型转换的新 Namespace
        self.model_args = parser.parse_args(argv)
        self.model_args.task_config = TASK_CONFIGS[self.model_args.task]

    def _add_attentioncache_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Attention Cache args")

        group.add_argument("--use_attentioncache", action='store_true')
        group.add_argument("--attentioncache_ratio", type=float, default=1.2)
        group.add_argument("--attentioncache_interval", type=int, default=4)
        group.add_argument("--start_step", type=int, default=12)
        group.add_argument("--end_step", type=int, default=37)

        return parser

    def _setup_cache(self):
        """设置Cache机制"""
        try:
            from mindiesd import CacheConfig, CacheAgent
        except ImportError as e:
            # Concise import error message
            raise ImportError("Failed to import required components from mindiesd. ") from e

        if self.model_args.use_attentioncache:
            config = CacheConfig(
                method="attention_cache",
                blocks_count=len(self.transformer.blocks),
                steps_count=self.model_args.sample_steps,
                step_start=self.model_args.start_step,
                step_interval=self.model_args.attentioncache_interval,
                step_end=self.model_args.end_step,
            )
        else:
            config = CacheConfig(
                method="attention_cache",
                blocks_count=len(self.transformer.blocks),
                steps_count=self.model_args.sample_steps,
            )
        cache = CacheAgent(config)
        if self.model_args.dit_fsdp:
            for block in self.transformer._fsdp_wrapped_module.blocks:
                block._fsdp_wrapped_module.cache = cache
                block._fsdp_wrapped_module.args = self.model_args
        else:
            for block in self.transformer.blocks:
                block.cache = cache
                block.args = self.model_args

    def _check_import_dependency(self):
        import importlib

        try:
            for mod in (
                "wan",
                "wan.configs",
                "wan.utils.prompt_extend",
                "wan.utils.utils",
                "wan.distributed.parallel_mgr",
                "wan.distributed.tp_applicator",
            ):
                importlib.import_module(mod)
        except ImportError as e:
            # Concise import error message
            raise ImportError(
                "Failed to import required components from wan. "
                "Please install the Wan2.1 from Modelers, "
                "make sure you can run the original floating-point inference successfully, "
                "and add the Wan2.1 repository to the Python search path environment variable PYTHONPATH. "
                "e.g. export PYTHONPATH=/path/to/Wan2.1:$PYTHONPATH"
            ) from e

    def _validate_args(self, args):
        """Get default parameter configuration, integrating wan config parameters"""
        self._check_import_dependency()
        from wan.configs import SUPPORTED_SIZES

        # Basic check
        if args.ckpt_dir is None:
            raise InvalidModelError("Please specify the checkpoint directory.")
        if not isinstance(args.task, str):
            raise SchemaValidateError(f"task must be a str, but got {type(args.task)}")
        if args.task not in SUPPORTED_TASKS:
            raise UnsupportedError("Unsupported task: %r. Supported tasks are: %s" % (args.task, SUPPORTED_TASKS))

        # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
        if args.sample_steps is None:
            args.sample_steps = 40 if "i2v" in args.task else 50
        if not isinstance(args.sample_steps, int):
            raise SchemaValidateError(f"sample_steps must be an integer, got {type(args.sample_steps).__name__}")
        if args.sample_steps <= 0:
            raise SchemaValidateError("sample_steps must be greater than 0")

        if args.sample_shift is None:
            args.sample_shift = 5.0
            if "i2v" in args.task and args.size in ["832*480", "480*832"]:
                args.sample_shift = 3.0

        # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
        if args.frame_num is None:
            args.frame_num = 1 if "t2i" in args.task else 81
        if not isinstance(args.frame_num, int):
            raise SchemaValidateError(f"frame_num must be an integer, got {type(args.frame_num).__name__}")
        if args.frame_num <= 0:
            raise SchemaValidateError("frame_num must be greater than 0")

        # T2I frame_num check
        if "t2i" in args.task and args.frame_num != 1:
            raise UnsupportedError("Unsupported frame_num %r for task %r" % (args.frame_num, args.task))

        args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)  # nosec B311
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
            "--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run."
        )
        parser.add_argument(
            "--size",
            type=str,
            default="1280*720",
            choices=list(SIZE_CONFIGS.keys()),
            help="The area (width*height) of the generated video. For the I2V task,"
            "the aspect ratio of the output video will follow that of the input image.",
        )
        parser.add_argument(
            "--frame_num",
            type=int,
            default=None,
            help="How many frames to sample from a image or video. The number should be 4n+1",
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
        parser.add_argument(
            "--save_file", type=str, default=None, help="The file to save the generated image or video to."
        )
        parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the image or video from.")
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
        parser.add_argument(
            "--base_seed", type=int, default=-1, help="The seed to use for generating the image or video."
        )
        parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")
        parser.add_argument(
            "--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="The solver used to sample."
        )
        parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
        parser.add_argument(
            "--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers."
        )
        parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="Classifier free guidance scale.")
        parser = self._add_attentioncache_args(parser)
        return parser

    def _get_default_model_args(self):
        parser = self._get_parser()
        args = parser.parse_args([])
        self.model_args = args

    def _init_logging(self, rank):
        # logging
        if rank == 0:
            # set format
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                handlers=[logging.StreamHandler(stream=sys.stdout)],
            )
        else:
            logging.basicConfig(level=logging.ERROR)

    def _load_pipeline(self):
        # 动态导入外部依赖
        self._check_import_dependency()

        import wan
        from wan.configs import WAN_CONFIGS

        rank = int(os.getenv("RANK", 0))
        _ = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        self._init_logging(rank)

        args = self.model_args

        # 不支持并行
        if args.t5_fsdp or args.dit_fsdp:
            raise SchemaValidateError("t5_fsdp and dit_fsdp are not supported in non-distributed environments.")

        if args.cfg_size > 1 or args.ulysses_size > 1 or args.ring_size > 1:
            raise SchemaValidateError("context parallel are not supported in non-distributed environments.")

        if args.vae_parallel:
            raise SchemaValidateError("vae parallel are not supported in non-distributed environments.")

        cfg = WAN_CONFIGS[args.task]
        if args.ulysses_size > 1:
            if cfg.num_heads % args.ulysses_size != 0:
                raise SchemaValidateError("`num_heads` must be divisible by `ulysses_size`.")
        logging.info("Generation job args: %r", args)
        logging.info("Generation model config: %r", cfg)

        logging.info("Creating WanT2V pipeline.")
        self.wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            use_vae_parallel=args.vae_parallel,
        )

        self.transformer = self.wan_t2v.model
