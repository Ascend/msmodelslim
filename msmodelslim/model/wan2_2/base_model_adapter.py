#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

# pylint: disable=logging-fstring-interpolation,too-many-ancestors,consider-merging-isinstance,consider-using-from-import,attribute-defined-outside-init,invalid-envvar-default,simplifiable-if-expression,logging-not-lazy,too-many-lines,arguments-differ

import logging
import os
import sys
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Generator, List, Callable, ClassVar, Union
from importlib import import_module

import torch
from torch import nn, distributed as dist
from pydantic import BaseModel
from tqdm import tqdm

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func_with_keyword,
)
from msmodelslim.utils.cache import load_cached_data_for_models, to_device
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.processor.quant.fa3.interface import FA3QuantAdapterInterface, FA3QuantPlaceHolder
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from ..interface_hub import (
    ModelInfoInterface,
    MultimodalPipelineInterface,
    OnlineQuaRotInterface,
    IterSmoothInterface,
)
from .constants import DEFAULT_SIZE, DUAL_EXPERT_SCENE_TASKS, EXAMPLE_PROMPT, TASK_TYPES
from .expert_sub_adapter import (
    Wan2_2ExpertSubAdapter,
    Wan2_2HighNoiseSubAdapter,
    Wan2_2LowNoiseSubAdapter,
)


@logger_setter()
class Wan2_2BaseModelAdapter(
    BaseModelAdapter,
    ModelInfoInterface,
    MultimodalPipelineInterface,
    FA3QuantAdapterInterface,
    OnlineQuaRotInterface,
    IterSmoothInterface,
):
    """
    1.公共流水线接口
    2.公共运行时配置
    3.公共校准执行
    4.基类运行时通用辅助
    5.私有专家子适配器装配
    6.私有参数桥接（配置与解析）
    7.私有运行时与缓存装配
    8.量化扩展接口
    """

    # 由子类固定，对应 Wan generate.py 的 --task；与 config.ini 的 model_type 一一绑定，不由 YAML 切换。
    scene_task: ClassVar[str] = ""
    # generate._parse_args 合法字段名（进程内懒加载一次，避免 configure_runtime 双次 parse）
    _GENERATE_CONFIG_KEYS: ClassVar[Optional[frozenset[str]]] = None

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        if not self.scene_task:
            raise SchemaValidateError(
                "Wan2_2BaseModelAdapter must be used via a concrete task subclass "
                "(Wan2_2T2VModelAdapter / Wan2_2I2VModelAdapter / Wan2_2TI2VModelAdapter).",
            )
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.model_args = None
        self.low_noise_model = None
        self.high_noise_model = None
        self._expert_adapters: Dict[str, Wan2_2ExpertSubAdapter] = {}

        self._check_import_dependency()

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'wan2_2'

    # ===== 分区 1：公共流水线接口 =====
    def validate_calib_samples(self, samples: List[VlmCalibSample]) -> List[VlmCalibSample]:
        """子类覆盖：如 T2V 禁止带图、I2V 强制 image、TI2V 可选 image（默认 T2V）。"""
        return samples

    def handle_dataset(
        self,
        dataset: Any,
        device: DeviceType = DeviceType.NPU,
    ) -> List[Any]:
        """
        dump 前仅做场景校验，不做模型 forward。
        支持两种输入：
        - List[VlmCalibSample]：原始校准样本，走 validate_calib_samples 校验
        - List[List[...]]：prepare_calib_data 已处理的 tensor 数据列表，透传
        """
        _ = device
        if dataset is None:
            return []
        if isinstance(dataset, VlmCalibSample):
            return self.validate_calib_samples([dataset])
        if isinstance(dataset, list) and dataset and isinstance(dataset[0], VlmCalibSample):
            return self.validate_calib_samples(dataset)
        if not isinstance(dataset, list):
            raise SchemaValidateError("handle_dataset expects dataset to be a list, got %s" % type(dataset).__name__)
        return dataset

    def init_model(self, device: DeviceType = DeviceType.NPU) -> Dict[str, nn.Module]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement init_model() for its Wan2.2 task.",
        )

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

    def get_expert_adapter(self, expert_name: str):
        """
        按专家名返回 LayerWiseRunner 使用的子适配器。

        双专家 T2V/I2V：必须已通过 _bind_expert_sub_adapters 绑定，未找到则报错（避免误用父适配器的
        quantization_context）。TI2V 单专家：expert_name 为 '' 且未绑定时可回退 self。
        """
        adapter = self._expert_adapters.get(expert_name)
        if adapter is not None:
            return adapter
        if self.scene_task not in DUAL_EXPERT_SCENE_TASKS and expert_name == "":
            return self
        known = sorted(self._expert_adapters.keys())
        if self.scene_task in DUAL_EXPERT_SCENE_TASKS:
            action = (
                "Ensure init_model() calls _bind_expert_sub_adapters with keys "
                "'low_noise_model' and 'high_noise_model' matching QuantService expert names."
            )
        elif expert_name == "":
            action = "Ensure init_model() binds expert '' for TI2V single DiT."
        else:
            action = f"Use a valid expert name. Bound experts: {known!r}."
        raise InvalidModelError(
            f"Expert sub-adapter not found for {expert_name!r} (scene_task={self.scene_task!r}).",
            action=action,
        )

    def get_inference_config_class(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_inference_config_class() for its Wan2.2 task.",
        )

    # ===== 分区 2：公共运行时配置 =====
    def configure_runtime(self, inference_config: Any) -> None:
        """
        将 InferenceConfig 落到 model_args。
        仅调用一次 generate._parse_args（避免 __init__ 预解析 + 配置期双次解析）。

        argv 顺序：场景最小 CLI → YAML 覆盖 → 量化固定 flag → 强制 task/ckpt_dir（后者覆盖前者）。
        """
        from wan.configs import WAN_CONFIGS

        override = inference_config.model_dump(exclude_none=True)

        allowed_attrs = self._allowed_generate_config_keys()
        unknown_attrs = [key for key in override.keys() if key not in allowed_attrs]
        if unknown_attrs:
            raise SchemaValidateError(
                f"illegal config attributes: {unknown_attrs}. \nsupported config attributes: {sorted(allowed_attrs)}"
            )

        quant_overrides = {
            "cfg_size": 1,
            "ulysses_size": 1,
            "ring_size": 1,
            "tp_size": 1,
            "vae_parallel": False,
            "t5_fsdp": False,
            "dit_fsdp": False,
            "use_attentioncache": False,
            "use_rainfusion": False,
        }
        argv = self._build_default_generate_cli()
        argv.extend(self._namespace_to_argv(override))
        argv.extend(self._namespace_to_argv(quant_overrides))
        argv.extend(["--task", self.scene_task, "--ckpt_dir", str(self.model_path)])

        self.model_args = self._parse_args_from_generate(argv)
        self.model_args.task_config = TASK_TYPES[self.scene_task]
        self.model_args.param_dtype = WAN_CONFIGS[self.scene_task].param_dtype

    # ===== 分区 3：公共校准执行 =====
    def _calib_data_when_dump_disabled(self, models: Dict[str, nn.Module]) -> Dict[str, Any]:
        """enable_dump=False 时跳过 pth 加载/浮点 dump，与 Legacy 量化路径行为一致。"""
        get_logger().info("enable_dump=False in config, skipping calibration data load/dump")
        return {expert_name: None for expert_name in models}

    def prepare_calib_data(
        self,
        models: Dict[str, nn.Module],
        dump_config: Any,
        save_path: Path,
        dataset: List[VlmCalibSample],
        inference_config: Any,
    ) -> Dict[str, Any]:
        if not dump_config.enable_dump:
            return self._calib_data_when_dump_disabled(models)

        config_dump_data_dir = dump_config.dump_data_dir
        base_dir = config_dump_data_dir if config_dump_data_dir else save_path
        pth_file_path_list: Dict[str, str] = {}
        for expert_name in models:
            pth_file_path_list[expert_name] = os.path.join(
                str(base_dir),
                f"calib_data_{self.model_args.task_config}_{expert_name}.pth",
            )
        calib_data = load_cached_data_for_models(
            pth_file_path_list=pth_file_path_list,
            generate_func=lambda: self.inference_dump_calib_data(
                dataset=dataset,
                inference_config=inference_config,
            ),
            models=models,
            dump_config=dump_config,
        )
        get_logger().info("prepare calib_data from %s success", base_dir)
        return calib_data

    def inference_dump_calib_data(self, dataset=None, inference_config: Any = None):
        stream = torch.npu.Stream()

        for sample in tqdm(dataset, desc="Dump calib data by float model inference"):
            seed = self._runtime_value(inference_config, "base_seed")
            torch.manual_seed(seed)
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
            begin = time.time()
            prompt = sample.text
            image_path = sample.image
            self._generate_video(prompt, image_path, inference_config)
            stream.synchronize()
            end = time.time()
            logging.info(f"Generating video used time {end - begin: .4f}s")

    def quantization_context(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement quantization_context() for its Wan2.2 task.",
        )

    # ===== 分区 4：基类运行时通用辅助 =====
    def _runtime_value(
        self,
        inference_config: Optional[Union[BaseModel, Dict[str, Any]]],
        name: str,
    ) -> Any:
        """统一取值入口：优先 inference_config，未配置时回退到 model_args。"""
        if inference_config is not None:
            if isinstance(inference_config, dict):
                val = inference_config.get(name)
            else:
                val = getattr(inference_config, name, None)
            if val is not None:
                return val
        return getattr(self.model_args, name, None)

    @contextmanager
    def _quantization_context_with_no_sync(self, *dit_models: nn.Module):
        """
        Wan2.2 量化运行时上下文（子类 quantization_context 直接 return 本生成器即可）。

        包含三层包装：
        1. amp.autocast(param_dtype) — 与 configure_runtime 写入的推理 dtype 一致；
        2. torch.no_grad — 校准量化不做反向；
        3. 各 DiT 的 no_sync() — 分布式训练时推迟梯度同步；单卡或未实现时用 nullcontext 等价跳过。

        为何用 ExitStack：T2V/I2V 需同时进入 low_noise、high_noise 两个 no_sync，
        数量随场景变化，单层 ``with A, B, C`` 不便写死在基类里。

        子类用法示例::
            def quantization_context(self):
                return self._quantization_context_with_no_sync(
                    self.low_noise_model, self.high_noise_model)
        """

        import torch.cuda.amp as amp

        # 遍历所有子模块，将非blocks部分移至npu
        for m in dit_models:
            if m is None:
                continue
            for name, module in m.named_modules():
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
            ExitStack() as stack,
        ):
            for m in dit_models:
                if m is None:
                    continue
                # 有 no_sync 则进入（如 DDP）；否则 nullcontext() 无操作
                stack.enter_context(getattr(m, "no_sync", nullcontext)())
            yield

    # ===== 分区 5：私有专家子适配器装配 =====
    def _bind_expert_sub_adapters(self, expert_modules: Dict[str, nn.Module]) -> None:
        """
        绑定内部 expert 子适配器。

        默认会根据 expert_name 创建同构子适配器：
        - low_noise_model -> Wan2_2LowNoiseSubAdapter
        - high_noise_model -> Wan2_2HighNoiseSubAdapter
        - 其它 -> Wan2_2ExpertSubAdapter

        如需 low/high 差异化（forward/visit/context/process），
        子类覆盖 _create_expert_sub_adapter 即可，无需改 quant_service。
        """
        adapters: Dict[str, Wan2_2ExpertSubAdapter] = {}
        for expert_name, module in expert_modules.items():
            sub = self._create_expert_sub_adapter(expert_name)
            sub.bind_module(module)
            adapters[expert_name] = sub
        self._expert_adapters = adapters

    def _create_expert_sub_adapter(self, expert_name: str) -> Wan2_2ExpertSubAdapter:
        """
        工厂方法：按 expert_name 返回子适配器实例。
        子类可覆盖，返回自定义的 low/high 子适配器实现。
        """
        if expert_name == "low_noise_model":
            return Wan2_2LowNoiseSubAdapter(self, expert_name)
        if expert_name == "high_noise_model":
            return Wan2_2HighNoiseSubAdapter(self, expert_name)
        return Wan2_2ExpertSubAdapter(self, expert_name)

    # ===== 分区 6：私有参数桥接（配置与解析） =====
    def _allowed_generate_config_keys(self) -> frozenset[str]:
        """generate 支持的 inference_config 键（进程内探测一次，不随每次 quant 重复 parse）。"""
        cls = type(self)
        if cls._GENERATE_CONFIG_KEYS is None:
            probe = self._parse_args_from_generate(self._build_default_generate_cli())
            cls._GENERATE_CONFIG_KEYS = frozenset(vars(probe).keys())
        return cls._GENERATE_CONFIG_KEYS

    def _build_default_generate_cli(self) -> List[str]:
        """
        构造传给 generate._parse_args() 的最小 argv（仅用于 configure_runtime 合并 YAML）。

        须满足 generate._validate_args 对当前 scene_task 的约束（如 ti2v 的 size）。
        """
        cli_args = [
            "--task",
            self.scene_task,
            "--ckpt_dir",
            str(self.model_path),
            "--prompt",
            EXAMPLE_PROMPT[self.scene_task]["prompt"],
            "--size",
            DEFAULT_SIZE[self.scene_task],
        ]
        if "image" in EXAMPLE_PROMPT[self.scene_task]:
            cli_args.extend(["--image", EXAMPLE_PROMPT[self.scene_task]["image"]])
        return cli_args

    @staticmethod
    def _namespace_to_argv(namespace_dict: Dict[str, Any]) -> List[str]:
        """
        将 Namespace 风格的 dict 转回 CLI 列表（供 _parse_args_from_generate 使用）。

        与 Wan2.2 generate.py 保持一致：None 不传；store_true 类 bool 仅在 True 时加 flag；
        offload_model 为 str2bool，须显式 true/false；tuple/list/dict 不传（如 T2V 双专家
        sample_guide_scale 默认由 _validate_args 从 WAN_CONFIGS 回填；YAML 仅可配置 CLI
        支持的标量 float，见各场景 InferenceConfig）。
        """
        argv: List[str] = []
        for key, val in namespace_dict.items():
            if val is None:
                continue
            if isinstance(val, (tuple, list, dict)):
                continue
            if key == "offload_model":
                argv.extend(["--offload_model", str(val).lower()])
                continue
            if isinstance(val, bool):
                if val:
                    argv.append(f"--{key}")
                continue
            argv.extend([f"--{key}", str(val)])
        return argv

    def _parse_args_from_generate(self, cli_args: List[str]):
        """
        调用 Wan2.2 仓库根目录 generate.py 的 _parse_args()（含 parser 与推理侧 _validate_args）。

        通过临时改写 sys.argv 模拟命令行；必须在 finally 中恢复，避免污染其它逻辑。
        """
        try:
            generate = import_module("generate")
        except ImportError as e:
            raise ImportError(
                "Failed to import Wan2.2 inference entry 'generate.py'. "
                "Please export PYTHONPATH=/path/to/Wan2.2:$PYTHONPATH"
            ) from e

        if not hasattr(generate, "_parse_args"):
            raise ImportError("Cannot find _parse_args() in Wan2.2 generate.py")

        original_argv = sys.argv
        try:
            sys.argv = ["generate.py", *cli_args]
            return generate._parse_args()
        finally:
            sys.argv = original_argv

    # ===== 分区 7：私有运行时与缓存装配 =====
    def _check_import_dependency(self):
        import importlib

        try:
            for mod in (
                "PIL",
                "wan",
                "wan.configs",
                "wan.utils.prompt_extend",
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

    def _init_logging(self, rank):
        if rank == 0:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                handlers=[logging.StreamHandler(stream=sys.stdout)],
            )
        else:
            logging.basicConfig(level=logging.ERROR)

    def _build_attention_cache_config(self, args, blocks_count: int, *, tuned: bool = False):
        """构造 attention_cache 配置：flag 为 False 时用默认参数（与 generate.py else 分支一致）。"""
        from mindiesd import CacheConfig

        if tuned and args.use_attentioncache:
            return CacheConfig(
                method="attention_cache",
                blocks_count=blocks_count,
                steps_count=args.sample_steps,
                step_start=args.start_step,
                step_interval=args.attentioncache_interval,
                step_end=args.end_step,
            )
        return CacheConfig(
            method="attention_cache",
            blocks_count=blocks_count,
            steps_count=args.sample_steps,
        )

    @staticmethod
    def _attach_attention_cache_to_blocks(transformer: nn.Module, cache, args) -> None:
        if args.dit_fsdp:
            for block in transformer._fsdp_wrapped_module.blocks:
                block._fsdp_wrapped_module.cache = cache
                block._fsdp_wrapped_module.args = args
        else:
            for block in transformer.blocks:
                block.cache = cache
                block.args = args

    def _setup_dual_expert_attention_cache(
        self,
        args,
        transformer_low: nn.Module,
        transformer_high: nn.Module,
        *,
        i2v: bool = False,
    ) -> None:
        """与 legacy model_adapter / generate.py 一致：始终注入 attention_cache agent。"""
        from mindiesd import CacheAgent

        if i2v:
            config_low = self._build_attention_cache_config(
                args,
                len(transformer_low.blocks),
                tuned=True,
            )
            config_high = self._build_attention_cache_config(
                args,
                len(transformer_high.blocks),
                tuned=False,
            )
        else:
            config_high = self._build_attention_cache_config(
                args,
                len(transformer_high.blocks),
                tuned=True,
            )
            config_low = self._build_attention_cache_config(
                args,
                len(transformer_low.blocks),
                tuned=False,
            )
        cache_low = CacheAgent(config_low)
        cache_high = CacheAgent(config_high)
        self._attach_attention_cache_to_blocks(transformer_low, cache_low, args)
        self._attach_attention_cache_to_blocks(transformer_high, cache_high, args)

    def _setup_single_transformer_attention_cache(self, args, transformer: nn.Module) -> None:
        from mindiesd import CacheAgent

        config = self._build_attention_cache_config(args, len(transformer.blocks), tuned=True)
        cache = CacheAgent(config)
        self._attach_attention_cache_to_blocks(transformer, cache, args)

    def _setup_wan_dit_runtime(self, args, *transformers: nn.Module, dual_i2v: bool = False) -> None:
        """
        为 DiT 注入 MindIE attention_cache（与 generate.py 一致：始终挂载 agent）。

        rainfusion 无默认注入路径，量化侧在 configure_runtime 固定 use_rainfusion=False，故不在此处理。
        """
        if len(transformers) == 2:
            transformer_low, transformer_high = transformers
            self._setup_dual_expert_attention_cache(
                args,
                transformer_low,
                transformer_high,
                i2v=dual_i2v,
            )
        elif len(transformers) == 1:
            self._setup_single_transformer_attention_cache(args, transformers[0])
        else:
            raise SchemaValidateError(
                f"expected 1 or 2 DiT transformers, got {len(transformers)}",
            )

    def _generate_video(
        self,
        prompt: str,
        image_path: Optional[str],
        inference_config: Any = None,
    ) -> None:
        """Wan2.2 场景子类实现具体视频生成。"""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _generate_video() for its Wan2.2 task.",
        )

    def _build_wan_pipeline(
        self,
        args,
        cfg,
        device: int,
        rank: int,
    ) -> None:
        """子类实现：按场景创建 WanT2V/WanI2V/WanTI2V pipeline。"""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_wan_pipeline() for its Wan2.2 task.",
        )

    def _load_pipeline(self):
        from PIL import Image
        from wan.configs import WAN_CONFIGS
        from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        self._init_logging(rank)

        args = self.model_args

        if args.offload_model is None:
            args.offload_model = False if world_size > 1 else True
            logging.info(f"offload_model is not specified, set to {args.offload_model}.")

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
        # 量化初始化默认不读图；I2V 与 TI2V 的校准图来自 dataset(index.jsonl)。
        # TI2V 默认与推理仓一致走 T2V（无图）；仅 prompt_extend 开启时才可能用到 args.image。
        if args.use_prompt_extend and args.image is not None:
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

        self._build_wan_pipeline(args, cfg, device, rank)

    # ===== 分区 8：量化扩展接口 =====
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

            # Wan2.2 attention 存在两个不兼容版本，需按优先级探测：
            #   - 新版（commit 38fb8eb）：attention 抽取为 WanSelfAttention 实例方法 self.attention()
            #   - 旧版：attention 为独立函数，位于 wan.modules.attention.attention()
            # 探测策略：优先使用 self.attention（新版），若不存在则回退导入旧版函数。
            use_method_attention = False
            attention_fn = None
            if hasattr(module, "attention") and callable(getattr(module, "attention")):
                use_method_attention = True
            else:
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

            # 若未能解析到 attention，跳过不处理
            if not use_method_attention and attention_fn is None:
                get_logger().warning(
                    f"Skip wrapping {module.__class__.__name__} because attention callable cannot be resolved "
                    f"(module={original_forward.__module__})"
                )
                raise ImportError(f"Could not find attention in {original_forward.__module__}")

            def new_forward(
                self,
                x,
                seq_lens,
                grid_sizes,
                freqs,
                args=None,
                rainfusion_config=None,
                t_idx=None,
                b_idx=None,
                **kwargs,
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

                if use_method_attention:
                    x = self.attention(
                        q=rope_apply(q, grid_sizes, freqs),
                        k=rope_apply(k, grid_sizes, freqs),
                        v=v,
                        k_lens=seq_lens,
                        window_size=self.window_size,
                        rainfusion_config=rainfusion_config,
                        t_idx=t_idx,
                        b_idx=b_idx,
                        **kwargs,
                    )
                else:
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

            # 解析 attention 实现：优先使用模块自身的 self.attention（新版本），否则回退到旧版函数
            use_method_attention = False
            attention_fn = None
            if hasattr(module, "attention") and callable(getattr(module, "attention")):
                use_method_attention = True
            else:
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

            # 若未能解析到 attention，跳过不处理
            if not use_method_attention and attention_fn is None:
                get_logger().warning(
                    f"Skip wrapping {module.__class__.__name__} because attention callable cannot be resolved "
                    f"(module={original_forward.__module__})"
                )
                raise ImportError(f"Could not find attention in {original_forward.__module__}")

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
                if use_method_attention:
                    x = self.attention(q, k, v, k_lens=context_lens)
                else:
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

    def get_adapter_config_for_subgraph(self, num_layers) -> List[AdapterConfig]:
        """
        Get adapter config for subgraph-based anti-outlier processing (iter_smooth).

        Defines the subgraph structure for non-fusion.
        Based on wan2_2.py implementation but adapted for Wan2.2 model.

        Includes both vision encoder and text decoder layers.
        """
        adapter_config = []

        # Text decoder layers
        # for layer_idx in range(self.transformer.num_layers):
        for layer_idx in range(num_layers):
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
