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
InternVL-U
Modifications Copyright (c) 2026 OpenGVLab
This file includes code from Qwen-Image and HuggingFace,
licensed under the Apache License, Version 2.0.
--------------------------------------------------------
Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
--------------------------------------------------------
"""

import argparse
import os
from types import MethodType
from pathlib import Path
from typing import Dict, Any, Generator, Tuple, Optional, Callable
from importlib import import_module

import torch
import torch.nn.functional as F
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

# QwenImageTransformer2DModel (diffusers) uses blocks with "TransformerBlock" in class name
# (e.g. BasicTransformerBlock). Match by keyword for visit/forward.
TRANSFORMER_BLOCK_KEYWORD = "transformerblock"
QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS = "QwenImageTransformerBlock"


@logger_setter()
# pylint: disable=too-many-ancestors
class QwenImageEditModelAdapter(
    BaseModelAdapter,
    ModelInfoInterface,
    LegacyMultimodalPipelineInterface,
    FA3QuantAdapterInterface,
    OnlineQuaRotInterface,
):
    def __init__(
        self,
        model_type: str,
        model_path: Path,
        trust_remote_code: bool = False,
    ):
        super().__init__(model_type, model_path, trust_remote_code)
        self.pipeline = None
        self.model = None
        self.transformer = None
        self.model_args = None
        self._get_default_model_args()

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return "qwen_image_edit"

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> Generator[Any, None, None]:
        return dataset

    def init_model(self, device: DeviceType = DeviceType.CPU) -> Dict[str, nn.Module]:
        return {"": self.transformer}

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if TRANSFORMER_BLOCK_KEYWORD in module.__class__.__name__.lower()
        ]

        if not transformer_blocks:
            raise InvalidModelError(
                f"No module with '{TRANSFORMER_BLOCK_KEYWORD}' in class name found in transformer.",
                action="Check that the model is QwenImageTransformer2DModel.",
            )

        first_block_input = None

        def break_hook(
            module: nn.Module,
            hook_args: Tuple[Any, ...],
            hook_kwargs: Dict[str, Any],
        ):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs)
            raise TransformersForwardBreak()

        hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True)]

        try:
            if isinstance(inputs, (list, tuple)):
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
            for h in hooks:
                h.remove()

        if first_block_input is None:
            raise InvalidModelError(
                "Could not capture first block input.",
                action="Check model and calibration input.",
            )

        first_block_input = to_device(first_block_input, "cpu")
        current_inputs = first_block_input

        if dist.is_initialized():
            dist.barrier()

        for name, block in transformer_blocks:
            args, kwargs = current_inputs
            outputs = yield ProcessRequest(name, block, args, kwargs)
            # QwenImageTransformerBlock 返回 (hidden_states, encoder_hidden_states)，双流
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = outputs[0]
                kwargs["encoder_hidden_states"] = outputs[1]
                current_inputs = (args, kwargs)
            elif isinstance(outputs, torch.Tensor):
                current_inputs = ((outputs,), kwargs)
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                current_inputs = ((outputs[0],), kwargs)
            else:
                current_inputs = ((outputs,), kwargs)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func_with_keyword(model, keyword=TRANSFORMER_BLOCK_KEYWORD)

    def run_calib_inference(self) -> None:
        """当前不做推理校准。若需启用校准，请在 model_config 中提供 img_paths、prompt_file，并在此方法内实现一次 pipeline 前向以 dump 校准数据。"""
        pass

    def apply_quantization(self, quant_model_func) -> None:
        from contextlib import contextmanager
        from torch.cuda import amp

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self, 'no_sync', noop_no_sync)
        with amp.autocast(dtype=torch.bfloat16), torch.no_grad(), no_sync():
            quant_model_func()

    def load_pipeline(self) -> None:
        self._load_pipeline()

    def set_model_args(self, override_model_config: object) -> None:
        self.model_args.model_path = str(self.model_path)
        missing = [k for k in override_model_config if not hasattr(self.model_args, k)]
        if missing:
            raise SchemaValidateError(
                f"Invalid config attributes: {missing}. "
                f"Supported: {[a for a in dir(self.model_args) if not a.startswith('_')]}"
            )
        for key in override_model_config:
            setattr(self.model_args, key, override_model_config[key])
        parser = self._get_parser()
        argv = []
        for key, val in vars(self.model_args).items():
            if val is None:
                continue
            if isinstance(val, bool):
                if val:
                    argv.append(f"--{key}")
            else:
                argv.extend([f"--{key}", str(val)])
        self.model_args = parser.parse_args(argv)
        self._validate_args(self.model_args)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass

    def _load_pipeline(self) -> None:
        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)
        model_path_str = str(self.model_path)
        transformer_subdir = os.path.join(model_path_str, "transformer")
        torch_dtype = torch.bfloat16
        if self.model_args and getattr(self.model_args, "torch_dtype", None) == "float32":
            torch_dtype = torch.float32

        try:
            from qwenimage_edit.transformer_qwenimage import QwenImageTransformer2DModel
            from qwenimage_edit.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
        except ImportError as e:
            raise InvalidModelError(
                "从仓库加载 qwenimage_edit 失败，请确保在 Qwen-Image-Edit-2509 根目录下或安装 diffusers。",
                action=str(e),
            ) from e
        get_logger().info("Loading Qwen-Image-Edit-2509 from repo (qwenimage_edit) at %s", model_path_str)
        self.transformer = QwenImageTransformer2DModel.from_pretrained(
            transformer_subdir,
            torch_dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        self.model = QwenImageEditPlusPipeline.from_pretrained(
            model_path_str,
            transformer=self.transformer,
            torch_dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        )

    def _get_default_model_args(self) -> None:
        parser = self._get_parser()
        self.model_args = parser.parse_args([])

    def _get_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="使用 Qwen-Image-Edit-2509 模型生成编辑图像")
        parser.add_argument("--model_path", type=str, default="/home/weight/Qwen-Image-Edit-2509/", help="模型本地路径")
        parser.add_argument(
            "--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="模型数据类型"
        )
        parser.add_argument("--device", type=str, default="npu", help="运行设备（npu/cuda/cpu）")
        parser.add_argument("--device_id", type=int, default=0, help="设备ID")
        # 输入配置（多图支持，用逗号分隔路径，校准时由 model_config 注入）
        parser.add_argument(
            "--img_paths", type=str, default=None, help="输入图像路径（多图用逗号分隔，如 'img1.png,img2.png'）"
        )
        parser.add_argument(
            "--prompt_file", type=str, default="edit_prompts.txt", help="提示词文件路径（每行一个提示词）"
        )
        parser.add_argument("--negative_prompt_file", type=str, default=None, help="负面提示词文件路径（每行一个）")
        # 推理配置
        parser.add_argument("--num_inference_steps", type=int, default=40, help="推理步数")
        parser.add_argument("--true_cfg_scale", type=float, default=4.0, help="真实CFG缩放系数")
        parser.add_argument("--guidance_scale", type=float, default=1.0, help="引导缩放系数（Qwen特有）")
        parser.add_argument("--seed", type=int, default=0, help="随机种子（确保 reproducibility）")
        parser.add_argument("--num_images_per_prompt", type=int, default=1, help="每个提示词生成的图像数量")
        # 输出配置
        parser.add_argument("--output_dir", type=str, default="output_images", help="生成图像保存目录")
        parser.add_argument(
            "--quant_desc_path",
            type=str,
            default=None,
            help="Path to quantization description file (e.g., quant_model_description_*.json). "
            "Enables quantization if provided (applies to Text Encoder and Transformer).",
        )
        return parser

    def _validate_args(self, args: Any) -> None:
        setattr(args, "task_config", "qwen_image_edit")
        output_dir = getattr(args, "output_dir", "output_images")
        os.makedirs(output_dir, exist_ok=True)
        if getattr(args, "num_inference_steps", None) is None:
            args.num_inference_steps = 40
        if getattr(args, "num_inference_steps", 0) <= 0:
            raise SchemaValidateError("num_inference_steps must be > 0")
        if getattr(args, "quant_desc_path", None):
            qp = args.quant_desc_path
            if not os.path.exists(qp):
                raise FileNotFoundError(f"Quantization description file not found: {qp}")
            if not qp.endswith(".json") or "quant_model_description" not in qp:
                raise SchemaValidateError(
                    f"Invalid quant_desc_path: {qp}. Expected format: 'quant_model_description_*.json'"
                )

    # ===== OnlineQuaRotInterface =====
    def get_online_rotation_configs(self, model: Optional[nn.Module] = None):
        """
        返回在线旋转配置，为 QwenImageTransformerBlock 的 q_rot / k_rot 配置旋转矩阵替换。
        若提供 model，会在各目标 block 上挂载 q_rot、k_rot 的 Identity 模块。
        """
        configs = {}
        if model is not None:
            for name, module in model.named_modules():
                if module.__class__.__name__ != QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS:
                    continue
                try:
                    if not hasattr(module, "q_rot"):
                        module.register_module("q_rot", nn.Identity())
                    if not hasattr(module, "k_rot"):
                        module.register_module("k_rot", nn.Identity())
                    get_logger().debug("Registered q_rot and k_rot Identity for %s", name)
                except Exception as e:
                    get_logger().warning("Failed to register rotation modules for %s: %s", name, e)

        shared_seed = 1234
        target_model = model if model is not None else getattr(self, "transformer", None)
        if target_model is None:
            get_logger().warning("No model/transformer for rotation configs, returning empty")
            return configs

        head_dim = None
        for _name, m in target_model.named_modules():
            if m.__class__.__name__ == QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS and hasattr(m, "attention_head_dim"):
                head_dim = m.attention_head_dim
                break
        if head_dim is None:
            get_logger().warning("Could not determine head_dim, returning empty rotation configs")
            return configs

        for name, module in target_model.named_modules():
            if module.__class__.__name__ != QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS:
                continue
            q_rot_path = f"{name}.q_rot" if name else "q_rot"
            configs[q_rot_path] = OnlineQuaRotInterface.RotationConfig(
                rotation_type="replace",
                rotation_size=head_dim,
                rotation_mode=OnlineQuaRotInterface.QuaRotMode.HADAMARD,
                block_size=-1,
                seed=shared_seed,
                dtype=torch.bfloat16,
            )
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
        """为 QwenImageTransformerBlock 注入 fa3_q/fa3_k/fa3_v 占位并包裹 forward，在 joint Q/K/V 上应用 q_rot、k_rot 与 FA3。"""

        def _wrap_block_forward(module: nn.Module) -> None:
            original_forward = module.forward
            mod = import_module(original_forward.__module__)
            apply_rotary_emb_qwen = getattr(mod, "apply_rotary_emb_qwen", None)
            if apply_rotary_emb_qwen is None:
                raise ImportError(f"apply_rotary_emb_qwen not found in {original_forward.__module__}")
            try:
                from mindiesd import attention_forward as _attention_forward
            except ImportError:
                _attention_forward = None

            ADALN_FUSE = getattr(mod, "ADALN_FUSE", False)

            def new_forward(
                self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                encoder_hidden_states_mask: torch.Tensor,
                temb: torch.Tensor,
                image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                txt_pad_len=None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                img_mod_params = self.img_mod(temb)
                txt_mod_params = self.txt_mod(temb)
                img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
                txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

                if not ADALN_FUSE:
                    img_normed = self.img_norm1(hidden_states)
                    img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)
                else:
                    img_modulated, img_gate1 = self.img_norm1(hidden_states, img_mod1)
                if not ADALN_FUSE:
                    txt_normed = self.txt_norm1(encoder_hidden_states)
                    txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
                else:
                    txt_modulated, txt_gate1 = self.txt_norm1(encoder_hidden_states, txt_mod1)

                attn = self.attn
                seq_txt = encoder_hidden_states.shape[1]
                img_query = attn.to_q(img_modulated).unflatten(-1, (attn.heads, -1))
                img_key = attn.to_k(img_modulated).unflatten(-1, (attn.heads, -1))
                img_value = attn.to_v(img_modulated).unflatten(-1, (attn.heads, -1))
                txt_query = attn.add_q_proj(txt_modulated).unflatten(-1, (attn.heads, -1))
                txt_key = attn.add_k_proj(txt_modulated).unflatten(-1, (attn.heads, -1))
                txt_value = attn.add_v_proj(txt_modulated).unflatten(-1, (attn.heads, -1))

                if attn.norm_q is not None:
                    img_query = attn.norm_q(img_query)
                if attn.norm_k is not None:
                    img_key = attn.norm_k(img_key)
                if getattr(attn, "norm_added_q", None) is not None:
                    txt_query = attn.norm_added_q(txt_query)
                if getattr(attn, "norm_added_k", None) is not None:
                    txt_key = attn.norm_added_k(txt_key)

                if image_rotary_emb is not None:
                    img_freqs, txt_freqs = image_rotary_emb
                    img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
                    img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
                    txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
                    txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

                joint_query = torch.cat([txt_query, img_query], dim=1)
                joint_key = torch.cat([txt_key, img_key], dim=1)
                joint_value = torch.cat([txt_value, img_value], dim=1)

                if hasattr(self, "q_rot"):
                    joint_query = self.q_rot(joint_query)
                if hasattr(self, "k_rot"):
                    joint_key = self.k_rot(joint_key)
                if hasattr(self, "fa3_q"):
                    joint_query = self.fa3_q(joint_query)
                if hasattr(self, "fa3_k"):
                    joint_key = self.fa3_k(joint_key)
                if hasattr(self, "fa3_v"):
                    joint_value = self.fa3_v(joint_value)

                if _attention_forward is not None:
                    joint_hidden_states = _attention_forward(
                        joint_query,
                        joint_key,
                        joint_value,
                        opt_mode="manual",
                        op_type="fused_attn_score",
                        layout="BNSD",
                    )
                else:
                    # fallback: BNSD -> (B, N, S, D), then SDPA
                    q, k, v = joint_query, joint_key, joint_value
                    attn_scale = getattr(attn, "scale", None)
                    if isinstance(attn_scale, torch.Tensor):
                        scale = attn_scale
                    else:
                        # If attn.scale is missing (None) fall back to sqrt(dk) rule.
                        scale = attn_scale if attn_scale is not None else (q.shape[-1] ** -0.5)
                    joint_hidden_states = F.scaled_dot_product_attention(  # pylint: disable=not-callable
                        q, k, v, scale=scale, dropout_p=0.0
                    )

                joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)
                txt_attn_output = joint_hidden_states[:, :seq_txt, :]
                img_attn_output = joint_hidden_states[:, seq_txt:, :]
                img_attn_output = attn.to_out[0](img_attn_output)
                if len(attn.to_out) > 1:
                    img_attn_output = attn.to_out[1](img_attn_output)
                txt_attn_output = attn.to_add_out(txt_attn_output)

                hidden_states = hidden_states + img_gate1 * img_attn_output
                encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

                if not ADALN_FUSE:
                    img_normed2 = self.img_norm2(hidden_states)
                    img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
                else:
                    img_modulated2, img_gate2 = self.img_norm2(hidden_states, img_mod2)
                img_mlp_output = self.img_mlp(img_modulated2)
                hidden_states = hidden_states + img_gate2 * img_mlp_output

                if not ADALN_FUSE:
                    txt_normed2 = self.txt_norm2(encoder_hidden_states)
                    txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
                else:
                    txt_modulated2, txt_gate2 = self.txt_norm2(encoder_hidden_states, txt_mod2)
                txt_mlp_output = self.txt_mlp(txt_modulated2)
                encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

                if encoder_hidden_states.dtype == torch.float16:
                    encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
                if hidden_states.dtype == torch.float16:
                    hidden_states = hidden_states.clip(-65504, 65504)
                return hidden_states, encoder_hidden_states

            module.forward = MethodType(new_forward, module)

        for name, module in root_module.named_modules():
            if module.__class__.__name__ != QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS:
                continue
            full_name = f"{root_name}.{name}" if root_name else name
            if not should_inject(full_name):
                continue
            prefix = f"{name}." if name else ""
            root_module.set_submodule(f"{prefix}fa3_q", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{prefix}fa3_k", FA3QuantPlaceHolder(ratio=0.9999))
            root_module.set_submodule(f"{prefix}fa3_v", FA3QuantPlaceHolder(ratio=1.0))
            _wrap_block_forward(module)
            get_logger().info("Injected FA3 placeholders for %s", full_name)
