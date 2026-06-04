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

# pylint: disable=logging-fstring-interpolation,too-many-ancestors,unnecessary-comprehension,no-else-continue,consider-merging-isinstance,arguments-renamed,consider-using-from-import,attribute-defined-outside-init

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, Generator, Tuple, Optional

from tqdm import tqdm
import torch
from torch import nn, distributed as dist

from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func_with_keyword,
)
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path
from msmodelslim.utils.cache import to_device
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError
from msmodelslim.processor.quant.fa3.interface import FA3QuantAdapterInterface, FA3QuantPlaceHolder
from ..interface_hub import ModelInfoInterface, LegacyMultimodalPipelineInterface, OnlineQuaRotInterface


@logger_setter()
class FLUX1ModelAdapter(
    BaseModelAdapter,
    ModelInfoInterface,
    LegacyMultimodalPipelineInterface,
    FA3QuantAdapterInterface,
    OnlineQuaRotInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.model_args = None
        self.transformer_blocks_layers = 19
        self.single_transformer_blocks_layers = 38

        self._get_default_model_args()

    def set_model_args(self, override_model_config: object):
        """
        update model_args with override_model_config from yaml
        """
        self.model_args.model_path = self.model_path

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
            elif isinstance(val, bool):
                if val:
                    argv.append(f"--{key}")
            else:
                argv.extend([f"--{key}", str(val)])
        self.model_args = parser.parse_args(argv)
        self._validate_args(self.model_args)

    def init_model(self, device: DeviceType = DeviceType.CPU) -> nn.Module:
        """
        Initialize model on CPU
        """
        return {'': self.transformer}

    def get_model_pedigree(self):
        return "flux1"

    def get_model_type(self) -> str:
        return self.model_type

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> Generator[Any, None, None]:
        return dataset

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """
        Based on diffusers 0.33.0/0.33.1 source code:
        transformer_blocks          module: input hidden_states=hidden_states,encoder_hidden_states=encoder_hidden_states,...
                                            output (encoder_hidden_states, hidden_states)
        single_transformer_blocks   module: input hidden_states=hidden_states,...
                                            output hidden_states
        Need to handle conversion between the two structures
        concatenate the final layer's output from transformer_blocks into a single hidden_states tensor for input to the single block.
        """
        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if "transformerblock" in module.__class__.__name__.lower()
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

        blocks_length = len(transformer_blocks)
        for idx in range(blocks_length):
            name, block = transformer_blocks[idx]
            block_args, block_kwargs = current_inputs
            outputs = yield ProcessRequest(name, block, block_args, block_kwargs)

            if idx < self.transformer_blocks_layers - 1:
                encoder_hidden_states, hidden_states = outputs
                current_inputs[1]["encoder_hidden_states"] = encoder_hidden_states
            elif idx == self.transformer_blocks_layers - 1:
                hidden_states = torch.cat([outputs[0], outputs[1]], dim=1)
                del current_inputs[1]["encoder_hidden_states"]
            else:
                hidden_states = outputs
            current_inputs[1]["hidden_states"] = hidden_states

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func_with_keyword(model, keyword="transformerblock")

    def run_calib_inference(self):
        device = self.model.device
        stream = torch.npu.Stream()
        args = self.model_args
        for _ in tqdm(range(1), desc='Dump calib data by float model inference'):
            torch.manual_seed(args.seed)
            torch.npu.manual_seed(args.seed)
            torch.npu.manual_seed_all(args.seed)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            begin = time.time()
            _ = self.model(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                generator=generator,
            ).images[0]
            stream.synchronize()
            end = time.time()
            get_logger().info(f"It took {end - begin: .4f}s to generate images for calibration")

    def apply_quantization(self, process_model_func):
        from contextlib import contextmanager
        import torch.cuda.amp as amp

        @contextmanager
        def noop_no_sync():
            yield

        # 延迟梯度同步
        no_sync_all_block = getattr(self, 'no_sync', noop_no_sync)
        with amp.autocast(dtype=torch.bfloat16), torch.no_grad(), no_sync_all_block():
            process_model_func()

    def load_pipeline(self):
        self._load_pipeline()

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass

    def get_online_rotation_configs(self, model: Optional[nn.Module] = None):
        configs = {}
        if model is not None:
            for name, module in model.named_modules():
                # 只处理目标模块类型
                if module.__class__.__name__ not in "FluxAttention":
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
        target_model = model if model is not None else getattr(self, 'transformer', None)
        head_dim = model.attention_head_dim
        for name, module in target_model.named_modules():
            module_type = module.__class__.__name__

            # 只处理目标模块类型
            if module_type not in "FluxAttention":
                continue
            # 配置 q_rot
            q_rot_path = f"{name}.q_rot" if name else "q_rot"
            configs[q_rot_path] = OnlineQuaRotInterface.RotationConfig(
                rotation_type="replace",
                rotation_size=head_dim,
                rotation_mode=OnlineQuaRotInterface.QuaRotMode.HADAMARD,
                block_size=-1,
                seed=shared_seed,
                dtype=model.dtype,
            )

            # 配置 k_rot（使用相同的种子，确保与 q_rot 使用相同的旋转矩阵）
            k_rot_path = f"{name}.k_rot" if name else "k_rot"
            configs[k_rot_path] = OnlineQuaRotInterface.RotationConfig(
                rotation_type="replace",
                rotation_size=head_dim,
                rotation_mode=OnlineQuaRotInterface.QuaRotMode.HADAMARD,
                block_size=-1,
                seed=shared_seed,
                dtype=model.dtype,
            )
        return configs

    def inject_fa3_placeholders(self, root_name: str, root_module: nn.Module, should_inject) -> None:
        """目前仅支持fa动态量化"""
        for name, module in root_module.named_modules():
            if 'Attention' in module.__class__.__name__ and should_inject(f"{root_name}.{name}" if root_name else name):
                if name == "":
                    prefix = ""
                else:
                    prefix = f"{name}."
                root_module.set_submodule(f"{prefix}fa3_q", FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule(f"{prefix}fa3_k", FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule(f"{prefix}fa3_v", FA3QuantPlaceHolder(ratio=1.0))

    def _load_pipeline(self):
        try:
            from diffusers import FluxPipeline
        except ImportError as e:
            raise InvalidModelError(
                "Failed to import FluxPipeline. Please install diffusers with Flux.1-dev support.",
                "Currently, only version 0.33.0 and 0.33.1 are supported",
                action="pip install diffusers==0.33.0 or pip install diffusers==0.33.1",
            ) from e

        # Validate model path
        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)
        get_logger().info("Loading flux model")
        self.model = FluxPipeline.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code, local_files_only=True, torch_dtype=torch.bfloat16
        )  # Not supported directly on the CPU
        self.model.enable_model_cpu_offload()
        self.transformer = self.model.transformer
        self.transformer_blocks_layers = self.model.transformer.config.num_layers
        self.single_transformer_blocks_layers = self.model.transformer.config.num_single_layers

    def _get_default_model_args(self):
        parser = self._get_parser()
        args = parser.parse_args([])
        self.model_args = args

    def _get_parser(self) -> argparse.ArgumentParser:
        """Get default parameter configuration"""
        parser = argparse.ArgumentParser(description="Flux.1-dev inference script")
        parser = self.__add_inference_args(parser)
        return parser

    def __add_inference_args(self, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="Inference args")
        group.add_argument("--model_path", type=str, default="", help="model path of flux model directory")
        group.add_argument("--batch_size", type=int, default=1, help="Batch size for inference and evaluation.")
        group.add_argument(
            "--num_inference_steps", type=int, default=50, help="Number of denoising steps for inference."
        )
        group.add_argument("--save_path", type=str, default="./results", help="Path to save the generated samples.")
        group.add_argument(
            "--save_path_suffix", type=str, default="", help="Suffix for the directory of saved samples."
        )
        group.add_argument("--prompt", type=str, default=None, help="Prompt for sampling during evaluation.")
        group.add_argument(
            "--guidance_scale",
            type=float,
            default=3.5,
            help="Higher `guidance_scale` encourages a model to generate images more aligned with `prompt` at the expense of lower image quality",
        )
        parser.add_argument(
            "--num_images_per_prompt", type=int, default=1, help="The number of images to generate per prompt."
        )
        group.add_argument("--height", type=int, default=1024, help="height of image")
        group.add_argument("--width", type=int, default=1024, help="width of image")
        group.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
        return parser

    def _validate_args(self, args):
        args.task_config = 'FLUX.1-dev'
        save_path = args.save_path if not args.save_path_suffix else f'{args.save_path}_{args.save_path_suffix}'
        os.makedirs(save_path, exist_ok=True)

        if args.num_inference_steps is None:
            args.num_inference_steps = 50

        if args.batch_size is None:
            args.batch_size = 1

        if args.seed is None:
            args.seed = 42

        if args.num_inference_steps <= 0:
            raise SchemaValidateError("num_inference_steps must be greater than 0")

        prompt = getattr(args, "prompt", None)
        if prompt is None:
            raise SchemaValidateError("Missing required parameter: prompt")
        if not isinstance(args.prompt, str):
            raise SchemaValidateError(f"prompt must be a string, got {type(args.prompt).__name__}")
        if not args.prompt.strip():
            raise SchemaValidateError("prompt cannot be an empty string")
