#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

# pylint: disable=logging-fstring-interpolation,too-many-ancestors,consider-merging-isinstance,consider-using-from-import,attribute-defined-outside-init

import logging
import random
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import ClassVar, Optional, Dict, Any, Tuple, Generator, List, Literal, Callable, Union

import torch
from torch import nn, distributed as dist
from tqdm import tqdm
from pydantic import BaseModel, ConfigDict

from msmodelslim.core.const import DeviceType
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func_with_keyword,
)
from msmodelslim.utils.cache import load_cached_data_for_models, to_device
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from ..interface_hub import (
    ModelInfoInterface,
    MultimodalPipelineInterface,
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
    OnlineQuaRotInterface,
)
from .constants import (
    DEFAULT_MODEL_RESOLUTION,
    DEFAULT_VIDEO_SIZE,
    DIT_WEIGHT_REL,
    HYVIDEO_CLI_LIST_FIELDS,
    PLACEHOLDER_PROMPT,
    TASK_TYPE,
    TEXT_ENCODER_2_PATH_REL,
    TEXT_ENCODER_PATH_REL,
    VAE_PATH_REL,
)


@logger_setter()
class HunyuanVideoModelAdapter(
    BaseModelAdapter,
    ModelInfoInterface,
    MultimodalPipelineInterface,
    FA3QuantAdapterInterface,
    OnlineQuaRotInterface,
):
    """
    1.公共流水线接口
    2.公共运行时配置
    3.公共校准执行
    4.运行时通用辅助
    5.私有参数桥接（配置与解析）
    6.私有运行时与缓存装配
    7.量化扩展接口
    """

    _HYVIDEO_CONFIG_KEYS: ClassVar[Optional[frozenset[str]]] = None

    class HunyuanVideoInferenceConfig(BaseModel):
        # inference_config 仅允许声明过的字段；旧 model_config 兼容由 quant_config.resolve_inference_raw 处理。
        model_config = ConfigDict(extra="forbid")

        model_resolution: Literal["540p", "720p"] | None = "720p"
        video_size: Tuple[int, int] | List[int] | None = (720, 1280)
        video_length: int | None = 129
        infer_steps: int | None = 50
        seed: int | None = None
        neg_prompt: str | None = None
        cfg_scale: float | None = 1.0
        embedded_cfg_scale: float | None = 6.0
        num_videos: int | None = 1
        flow_shift: float | None = 7.0
        batch_size: int | None = 1

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.transformer = None
        self.model_args = None

        self._check_import_dependency()

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'hunyuan_video'

    # ===== 分区 1：公共流水线接口 =====
    def validate_calib_samples(self, samples: List[VlmCalibSample]) -> List[VlmCalibSample]:
        for idx, sample in enumerate(samples):
            if not isinstance(sample.text, str) or not sample.text.strip():
                raise SchemaValidateError(
                    f"hunyuan_video sample[{idx}] requires non-empty text",
                    action="Provide text in dataset entries (index.jsonl / VlmCalibSample.text).",
                )
            if sample.image is not None:
                raise SchemaValidateError(
                    f"hunyuan_video sample[{idx}] must not include image",
                    action="HunyuanVideo T2V calibration is text-only; remove image from dataset.",
                )
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
        _ = device
        self._load_pipeline()
        # 与 sample_video.py 一致：加载后必须 _setup_cache（block 级 attention_cache，见方法内注释）。
        self._setup_cache()
        return {'': self.transformer}

    def generate_model_forward(
        self,
        model: torch.nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if "streamblock" in module.__class__.__name__.lower()
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
        return generated_decoder_layer_visit_func_with_keyword(model, keyword="streamblock")

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass

    # ===== 分区 2：公共运行时配置 =====
    def get_inference_config_class(self):
        return self.HunyuanVideoInferenceConfig

    def configure_runtime(self, inference_config: HunyuanVideoInferenceConfig) -> None:
        """
        将 InferenceConfig 落到 model_args。
        仅做一次 hyvideo.parse_args（不在 __init__ 预解析）。
        """
        override = inference_config.model_dump(exclude_none=True)
        allowed_attrs = self._allowed_hyvideo_config_keys()
        unknown_attrs = [key for key in override if key not in allowed_attrs]
        if unknown_attrs:
            raise SchemaValidateError(
                f"illegal config attributes: {unknown_attrs}. supported config attributes: {sorted(allowed_attrs)}",
            )

        argv = self._build_default_quant_cli()
        argv.extend(self._namespace_to_argv(override))
        argv.extend(self._namespace_to_argv(self._fixed_quant_runtime_overrides()))
        self.model_args = self._parse_args_from_hyvideo(argv)
        self.model_args.task_config = TASK_TYPE

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
            pth_file_path_list[expert_name] = str(
                Path(base_dir).joinpath(f"calib_data_{self.model_args.task_config}_{expert_name}.pth")
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
        video_size = tuple(self._runtime_value(inference_config, "video_size") or self.model_args.video_size)

        for sample in tqdm(dataset, desc="Dump calib data by float model inference"):
            seed = self._runtime_value(inference_config, "seed")
            if seed is not None and seed >= 0:
                random.seed(seed)
                torch.manual_seed(seed)
                torch.npu.manual_seed(seed)
                torch.npu.manual_seed_all(seed)

            begin = time.time()
            self.hunyuan_video_sampler.predict(
                prompt=sample.text,
                height=video_size[0],
                width=video_size[1],
                video_length=self._runtime_value(inference_config, "video_length"),
                seed=seed,
                negative_prompt=self._runtime_value(inference_config, "neg_prompt"),
                infer_steps=self._runtime_value(inference_config, "infer_steps"),
                guidance_scale=self._runtime_value(inference_config, "cfg_scale"),
                num_videos_per_prompt=self._runtime_value(inference_config, "num_videos"),
                flow_shift=self._runtime_value(inference_config, "flow_shift"),
                batch_size=self._runtime_value(inference_config, "batch_size"),
                embedded_guidance_scale=self._runtime_value(inference_config, "embedded_cfg_scale"),
            )
            stream.synchronize()
            end = time.time()
            logging.info(f"Generating video used time {end - begin: .4f}s")

    def quantization_context(self):
        from contextlib import contextmanager
        import torch.cuda.amp as amp

        @contextmanager
        def _ctx():
            @contextmanager
            def noop_no_sync():
                yield

            no_sync = getattr(self, 'no_sync', noop_no_sync)
            for name, module in self.transformer.named_modules():
                if 'blocks' not in name:
                    module.to('npu')
                else:
                    module.to('cpu')
            with amp.autocast(dtype=torch.bfloat16), torch.no_grad(), no_sync():
                yield

        return _ctx()

    # ===== 分区 4：运行时通用辅助 =====
    def _runtime_value(
        self,
        inference_config: Optional[Union[BaseModel, Dict[str, Any]]],
        name: str,
    ) -> Any:
        """推理执行期通用取值钩子：优先 inference_config，回退到 model_args。"""
        if inference_config is not None:
            if isinstance(inference_config, dict):
                val = inference_config.get(name)
            else:
                val = getattr(inference_config, name, None)
            if val is not None:
                return val
        return getattr(self.model_args, name, None)

    # ===== 分区 5：私有参数桥接（配置与解析） =====
    @staticmethod
    def _fixed_quant_runtime_overrides() -> Dict[str, Any]:
        """量化校准时写入 parse_args 的固定覆盖项（见 configure_runtime 注释）。"""
        return {
            "ulysses_degree": 1,
            "ring_degree": 1,
            "vae_parallel": False,
            "use_cache": False,
            "use_cache_double": False,
            "use_attentioncache": False,
        }

    def _allowed_hyvideo_config_keys(self) -> frozenset[str]:
        """hyvideo parse_args 支持的 inference_config 键（进程内探测一次）。"""
        cls = type(self)
        if cls._HYVIDEO_CONFIG_KEYS is None:
            probe = self._parse_args_from_hyvideo(self._build_default_quant_cli())
            cls._HYVIDEO_CONFIG_KEYS = frozenset(vars(probe).keys())
        return cls._HYVIDEO_CONFIG_KEYS

    def _build_default_quant_cli(self) -> List[str]:
        """configure_runtime 合并 YAML 用的最小 argv（满足 hyvideo resolution/size 约束）。"""
        model_base = str(self.model_path)
        h, w = DEFAULT_VIDEO_SIZE
        return [
            "--model-base",
            model_base,
            "--prompt",
            PLACEHOLDER_PROMPT,
            "--model-resolution",
            DEFAULT_MODEL_RESOLUTION,
            "--video-size",
            str(h),
            str(w),
            "--dit-weight",
            str(Path(model_base).joinpath(*DIT_WEIGHT_REL)),
            "--vae-path",
            str(Path(model_base).joinpath(*VAE_PATH_REL)),
            "--text-encoder-path",
            str(Path(model_base).joinpath(*TEXT_ENCODER_PATH_REL)),
            "--text-encoder-2-path",
            str(Path(model_base).joinpath(*TEXT_ENCODER_2_PATH_REL)),
        ]

    @staticmethod
    def _namespace_to_argv(namespace_dict: Dict[str, Any]) -> List[str]:
        """
        Namespace-like dict -> argv 片段（供 _parse_args_from_hyvideo 使用）。

        与 hyvideo parse_args CLI 一致：list/tuple 仅 HYVIDEO_CLI_LIST_FIELDS（nargs=\"+\"）
        会展开；dict 及未登记的复合类型跳过。
        """
        argv: List[str] = []
        for key, val in namespace_dict.items():
            if val is None:
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(val, dict):
                continue
            if isinstance(val, bool):
                if val:
                    argv.append(flag)
                continue
            if isinstance(val, (list, tuple)):
                if key in HYVIDEO_CLI_LIST_FIELDS:
                    argv.append(flag)
                    argv.extend(str(v) for v in val)
                continue
            argv.extend([flag, str(val)])
        return argv

    def _parse_args_from_hyvideo(self, cli_args: List[str]):
        """
        调用 hyvideo.config.parse_args（含 sanity_check 与 assert）。

        推理仓将 argv 列表参数命名为 namespace，但内部传给 argparse 的 namespace=
        是“预填充对象”语义，不能直接传 CLI 列表。通过 sys.argv 模拟命令行（与 Wan2.2 generate 一致）。
        """
        from hyvideo.config import parse_args

        original_argv = sys.argv
        try:
            sys.argv = ["sample_video.py", *cli_args]
            return parse_args()
        finally:
            sys.argv = original_argv

    # ===== 分区 6：私有运行时与缓存装配 =====
    def _check_import_dependency(self):
        import importlib

        try:
            for mod in (
                "hyvideo",
                "hyvideo.constants",
                "hyvideo.modules.models",
                "hyvideo.inference",
                "hyvideo.utils.file_utils",
            ):
                importlib.import_module(mod)
        except ImportError as e:
            # Concise import error message
            raise ImportError(
                "Failed to import required components from hunyuanvideo. "
                "Please install the hunyuanvideo dependencies from the official source, "
                "make sure you can run the original floating-point inference successfully, "
                "and add the hunyuanvideo repository to the Python search path environment variable PYTHONPATH. "
                "e.g. export PYTHONPATH=/path/to/hunyuanvideo:$PYTHONPATH"
            ) from e

    def _setup_cache(self):
        """
        MindIE 推理 cache（非 KV cache）。

        - use_cache / use_cache_double：DiT 双流 block cache，量化 configure_runtime 已关。
        - block.cache（attention_cache）：无论 use_attentioncache 与否都必须注入 CacheAgent
          （hyvideo DiT block.forward 走 self.cache.apply，与 sample_video.py else 分支一致）。
        """
        try:
            from mindiesd import CacheConfig, CacheAgent
        except ImportError as e:
            raise ImportError("Failed to import required components from mindiesd. ") from e
        args = self.model_args
        # DiT 级 dit_block_cache（仅显式开启时生效；量化路径 use_cache* 均为 False）
        if args.use_cache and len(self.transformer.single_blocks) > 0:
            # single
            config_single = CacheConfig(
                method="dit_block_cache",
                blocks_count=len(self.transformer.single_blocks),
                steps_count=args.infer_steps,
                step_start=args.cache_start_steps,
                step_interval=args.cache_interval,
                step_end=args.infer_steps - 1,
                block_start=args.single_block_start,
                block_end=args.single_block_end,
            )
            cache_single = CacheAgent(config_single)
            self.transformer.cache_single = cache_single
        if args.use_cache_double and len(self.transformer.double_blocks) > 0:
            # double
            config_double = CacheConfig(
                method="dit_block_cache",
                blocks_count=len(self.transformer.double_blocks),
                steps_count=args.infer_steps,
                step_start=args.cache_start_steps,
                step_interval=args.cache_interval,
                step_end=args.infer_steps - 1,
                block_start=args.double_block_start,
                block_end=args.double_block_end,
            )
            cache_dual = CacheAgent(config_double)
            self.transformer.cache_dual = cache_dual

        # block 级 attention_cache：flag 为 True 用跳步参数，否则仍挂默认 agent（必走 apply）
        if args.use_attentioncache:
            if len(self.transformer.double_blocks) > 0:
                config_double = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(self.transformer.double_blocks),
                    steps_count=args.infer_steps,
                    step_start=args.start_step,
                    step_interval=args.attentioncache_interval,
                    step_end=args.end_step,
                )
            if len(self.transformer.single_blocks) > 0:
                config_single = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(self.transformer.single_blocks),
                    steps_count=args.infer_steps,
                    step_start=args.start_step,
                    step_interval=args.attentioncache_interval,
                    step_end=args.end_step,
                )
        else:
            if len(self.transformer.double_blocks) > 0:
                config_double = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(self.transformer.double_blocks),
                    steps_count=args.infer_steps,
                )
            if len(self.transformer.single_blocks) > 0:
                config_single = CacheConfig(
                    method="attention_cache",
                    blocks_count=len(self.transformer.single_blocks),
                    steps_count=args.infer_steps,
                )
        if len(self.transformer.double_blocks) > 0:
            cache_double = CacheAgent(config_double)
            for block in self.transformer.double_blocks:
                block.cache = cache_double
        if len(self.transformer.single_blocks) > 0:
            cache_single = CacheAgent(config_single)
            for block in self.transformer.single_blocks:
                block.cache = cache_single

    def _load_pipeline(self):
        self._check_import_dependency()

        from hyvideo.inference import HunyuanVideoSampler

        args = self.model_args
        # 量化单卡路径：configure_runtime 已将 ulysses/ring=1、vae_parallel=False
        if args.ulysses_degree > 1 or args.ring_degree > 1:
            raise UnsupportedError("context parallel are not supported in non-distributed environments")
        if args.vae_parallel:
            raise UnsupportedError("vae parallel are not support in non-distributed environment")

        logging.info("load hunyuan_video models")
        models_root_path = Path(args.model_base)
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
        self.transformer = self.hunyuan_video_sampler.pipeline.transformer
        self.pipeline = self.hunyuan_video_sampler.pipeline

    # ===== 分区 7：量化扩展接口 =====
    # ===== OnlineQuaRotInterface =====
    def get_online_rotation_configs(self, model: Optional[nn.Module] = None):
        """
        返回在线旋转配置，配置 q_rot 和 k_rot 为旋转矩阵替换。

        如果提供了 model，会在此方法中直接给 MMDoubleStreamBlock 和 MMSingleStreamBlock 挂载 q_rot 和 k_rot Identity 模块。

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
                if module_type not in ["MMDoubleStreamBlock", "MMSingleStreamBlock"]:
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
            get_logger().warning("No model provided and transformer not available, returning empty rotation configs")
            return configs

        # 获取全局 head_dim - 从 transformer 直接获取
        if not hasattr(target_model, 'hidden_size') or not hasattr(target_model, 'heads_num'):
            get_logger().warning("Could not determine head_dim from transformer, returning empty rotation configs")
            return configs

        head_dim = target_model.hidden_size // target_model.heads_num

        # 使用全局 head_dim 为所有目标模块配置旋转
        for name, module in target_model.named_modules():
            module_type = module.__class__.__name__

            # 只处理目标模块类型
            if module_type not in ["MMDoubleStreamBlock", "MMSingleStreamBlock"]:
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
        """为 HunyuanVideo 模型的 MMDoubleStreamBlock 和 MMSingleStreamBlock 安装 FA3 占位，并包裹 forward 调用这些占位。

        - 在每个目标模块下注入子模块：fa3_q, fa3_k, fa3_v
        - 包裹其 forward 方法，在计算 Q、K、V 并 cat 后，依次调用占位：
            q = self.fa3_q(q)
            k = self.fa3_k(k)
            v = self.fa3_v(v)
        """

        def _wrap_double_forward(module: nn.Module):
            """包裹 MMDoubleStreamBlock 的 forward 方法"""
            original_forward = module.forward

            # 动态导入必要的函数
            hyvideo_double_module = import_module(original_forward.__module__)
            modulate = hyvideo_double_module.modulate
            apply_gate = hyvideo_double_module.apply_gate
            rearrange = hyvideo_double_module.rearrange
            apply_rotary_emb = hyvideo_double_module.apply_rotary_emb
            attention = hyvideo_double_module.attention
            parallel_attention = hyvideo_double_module.parallel_attention

            def new_forward(
                self,
                img: torch.Tensor,
                txt: torch.Tensor,
                vec: torch.Tensor,
                cu_seqlens_q: Optional[torch.Tensor] = None,
                cu_seqlens_kv: Optional[torch.Tensor] = None,
                max_seqlen_q: Optional[int] = None,
                max_seqlen_kv: Optional[int] = None,
                freqs_cis: tuple = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                # 从 vec 中提取 modulation 参数
                (
                    img_mod1_shift,
                    img_mod1_scale,
                    img_mod1_gate,
                    img_mod2_shift,
                    img_mod2_scale,
                    img_mod2_gate,
                ) = self.img_mod(vec).chunk(6, dim=-1)
                (
                    txt_mod1_shift,
                    txt_mod1_scale,
                    txt_mod1_gate,
                    txt_mod2_shift,
                    txt_mod2_scale,
                    txt_mod2_gate,
                ) = self.txt_mod(vec).chunk(6, dim=-1)
                # Prepare image for attention.
                img_modulated = self.img_norm1(img)
                img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
                img_qkv = self.img_attn_qkv(img_modulated)
                img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
                # Apply QK-Norm if needed
                img_q = self.img_attn_q_norm(img_q).to(img_v)
                img_k = self.img_attn_k_norm(img_k).to(img_v)

                # Apply RoPE if needed.
                if freqs_cis is not None:
                    img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                    if not (img_qq.shape == img_q.shape and img_kk.shape == img_k.shape):
                        raise ValueError(
                            f"Rotary embedding output shape mismatch. "
                            f"img_qq shape: {img_qq.shape}, img_q shape: {img_q.shape}, "
                            f"img_kk shape: {img_kk.shape}, img_k shape: {img_k.shape}"
                        )
                    img_q, img_k = img_qq, img_kk

                # Prepare txt for attention.
                txt_modulated = self.txt_norm1(txt)
                txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
                txt_qkv = self.txt_attn_qkv(txt_modulated)
                txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
                # Apply QK-Norm if needed.
                txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
                txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

                # Run actual attention.
                q = torch.cat((img_q, txt_q), dim=1)
                k = torch.cat((img_k, txt_k), dim=1)
                v = torch.cat((img_v, txt_v), dim=1)
                expected_cu_seqlens_q_length = 2 * img.shape[0] + 1
                if cu_seqlens_q.shape[0] != expected_cu_seqlens_q_length:
                    raise ValueError(
                        f"cu_seqlens_q shape mismatch: "
                        f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
                        f"expected first dimension length: {expected_cu_seqlens_q_length}"
                    )

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

                # attention computation start
                if not self.hybrid_seq_parallel_attn:
                    attn = attention(
                        q,
                        k,
                        v,
                        mode="torch",
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        batch_size=img_k.shape[0],
                    )
                else:
                    attn = parallel_attention(
                        self.hybrid_seq_parallel_attn,
                        q,
                        k,
                        v,
                        img_q_len=img_q.shape[1],
                        img_kv_len=img_k.shape[1],
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                    )

                # attention computation end

                img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

                # Calculate the img blocks.
                img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
                img = img + apply_gate(
                    self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
                    gate=img_mod2_gate,
                )

                # Calculate the txt blocks.
                txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
                txt = txt + apply_gate(
                    self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
                    gate=txt_mod2_gate,
                )

                return img, txt

            # pylint: disable=no-value-for-parameter
            module.forward = new_forward.__get__(module, module.__class__)

        def _wrap_single_forward(module: nn.Module):
            """包裹 MMSingleStreamBlock 的 forward 方法"""
            original_forward = module.forward

            # 动态导入必要的函数
            hyvideo_single_module = import_module(original_forward.__module__)
            modulate = hyvideo_single_module.modulate
            apply_gate = hyvideo_single_module.apply_gate
            rearrange = hyvideo_single_module.rearrange
            apply_rotary_emb = hyvideo_single_module.apply_rotary_emb
            attention = hyvideo_single_module.attention
            parallel_attention = hyvideo_single_module.parallel_attention

            def new_forward(
                self,
                x: torch.Tensor,
                vec: torch.Tensor,
                txt_len: int,
                cu_seqlens_q: Optional[torch.Tensor] = None,
                cu_seqlens_kv: Optional[torch.Tensor] = None,
                max_seqlen_q: Optional[int] = None,
                max_seqlen_kv: Optional[int] = None,
                freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
            ) -> torch.Tensor:
                mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
                x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
                qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

                q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

                # Apply QK-Norm if needed.
                q = self.q_norm(q).to(v)
                k = self.k_norm(k).to(v)

                # Apply RoPE if needed.
                if freqs_cis is not None:
                    img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
                    img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
                    img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                    if not (img_qq.shape == img_q.shape and img_kk.shape == img_k.shape):
                        raise ValueError(
                            f"Rotary embedding output shape mismatch. "
                            f"img_qq shape: {img_qq.shape}, img_q shape: {img_q.shape}, "
                            f"img_kk shape: {img_kk.shape}, img_k shape: {img_k.shape}"
                        )
                    img_q, img_k = img_qq, img_kk
                    q = torch.cat((img_q, txt_q), dim=1)
                    k = torch.cat((img_k, txt_k), dim=1)
                else:
                    # 如果 freqs_cis 为 None，需要计算 img_q_len 和 img_kv_len 用于 parallel_attention
                    img_q_len = q.shape[1] - txt_len
                    img_kv_len = k.shape[1] - txt_len

                # Compute attention.
                expected_cu_seqlens_q_length = 2 * x.shape[0] + 1
                if cu_seqlens_q.shape[0] != expected_cu_seqlens_q_length:
                    raise ValueError(
                        f"cu_seqlens_q shape mismatch. "
                        f"cu_seqlens_q.shape: {cu_seqlens_q.shape}, x.shape[0]: {x.shape[0]}, "
                        f"expected first dimension length: {expected_cu_seqlens_q_length}"
                    )

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

                # attention computation start
                if not self.hybrid_seq_parallel_attn:
                    attn = attention(
                        q,
                        k,
                        v,
                        mode="torch",
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        batch_size=x.shape[0],
                    )
                else:
                    # 如果 freqs_cis 不为 None，使用 img_q 和 img_k 的长度；否则使用计算出的长度
                    if freqs_cis is not None:
                        img_q_len_val = img_q.shape[1]
                        img_kv_len_val = img_k.shape[1]
                    else:
                        img_q_len_val = img_q_len
                        img_kv_len_val = img_kv_len
                    attn = parallel_attention(
                        self.hybrid_seq_parallel_attn,
                        q,
                        k,
                        v,
                        img_q_len=img_q_len_val,
                        img_kv_len=img_kv_len_val,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                    )
                # attention computation end

                # Compute activation in mlp stream, cat again and run second linear layer.
                output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
                return x + apply_gate(output, gate=mod_gate)

            # pylint: disable=no-value-for-parameter
            module.forward = new_forward.__get__(module, module.__class__)

        # 遍历并注入占位符
        for name, module in root_module.named_modules():
            module_type = module.__class__.__name__

            # 检查是否是目标模块类型
            if module_type not in ["MMDoubleStreamBlock", "MMSingleStreamBlock"]:
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
            if module_type == "MMDoubleStreamBlock":
                _wrap_double_forward(module)
            elif module_type == "MMSingleStreamBlock":
                _wrap_single_forward(module)

            get_logger().info(f"Injected FA3 placeholders for {full_name}")
