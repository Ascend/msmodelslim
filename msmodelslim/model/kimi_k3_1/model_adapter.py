#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=C0302

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

# Kimi-K3.1 VLM adapter.
#
# Unlike ``kimi_k3`` this adapter never instantiates ``transformers`` model classes: the
# released Kimi-K3.1 snapshot only registers ``AutoConfig`` in ``config.json``
# (``auto_map`` has no ``AutoModel*`` entry), so the graph is built from
# ``msmodelslim.model.kimi_k3_1.modeling_kimi_k3`` / ``modeling_kimi_linear`` directly.
# That also lets the graph be
# truncated to a single decoder layer for lazy, layer-wise loading:
#
#   * ``init_model`` builds ``vision_tower`` + ``mm_projector`` + ``language_model`` with
#     ``num_hidden_layers = 1`` and loads only the non-layer tensors plus layer 0;
#   * ``generate_decoder_layer`` materialises layers 1..92 on demand, dequantizing the
#     ``mxfp4-pack-quantized`` routed experts to BF16 exactly like ``kimi_k3``.
#
# Everything that has to be *derived* from the architecture (QuaRot rotation targets,
# IterSmooth/FlexSmooth subgraphs) lives in ``quarot.py`` /
# ``modeling_kimi_linear.py`` / ``modeling_kimi_k3.py`` so the checkpoint keys stay
# byte-identical to the released snapshot.

import os
import types
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import patch

import torch
import torch.distributed as dist
from pydantic import BaseModel, Field
from safetensors import safe_open
from torch import nn
from tqdm import tqdm

from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func
from msmodelslim.model.common.utils import _get_expert_range
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.model.interface_hub import (
    ModelSlimPipelineInterfaceV1,
    LayerWiseOffloadOptionalInterface,
    AscendV1SaveInterface,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
)
from msmodelslim.model.kimi_k3.runtime_patches import apply_kimi_k3_runtime_patches
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import (
    MAX_READ_FILE_SIZE_32G,
    get_valid_read_path,
    json_safe_load,
    safe_copy_file,
)

from .convert_mxfp4_to_bf16 import dequant_subtree_mxfp4_to_bf16, get_full_weight_map
from .modeling_kimi_k3 import KimiK31ForConditionalGeneration
from .modeling_kimi_linear import KimiDecoderLayer
from .quarot import (
    attn_prefix,
    get_ln_fuse_map as build_ln_fuse_map,
    get_rotate_map as build_rotate_map,
    input_layernorm_targets,
    is_dense_mlp,
    is_kda_layer,
    layer_prefix,
    moe_prefix,
    post_attention_layernorm_targets,
    q_a_layernorm_targets,
    routed_expert_norm_targets,
)

#: Top-level roots the layer-wise pipeline walks as whole modules.
VISION_ROOT = "vision_tower"
PROJECTOR_ROOT = "mm_projector"
LAYER_ROOT = "language_model.model.layers"


class KimiConfigNode(dict):
    """``config.json`` block with both mapping and attribute access.

    The modelling code only ever reads config values (``cfg.hidden_size``,
    ``cfg.linear_attn_config["kda_layers"]``, ``getattr(cfg, "dsa_config", None)``), so a
    dict with attribute access is enough and keeps the adapter free of
    ``AutoConfig``/remote code.  Lists / nested dicts are converted recursively so
    ``cfg.text_config.dsa_config["dsa_layers"]`` stays a plain ``dict`` for
    ``serializable`` access.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def to_config_node(value: Any) -> Any:
    """Recursively wrap config dicts into :class:`KimiConfigNode`."""
    if isinstance(value, dict):
        return KimiConfigNode({key: to_config_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_config_node(item) for item in value]
    return value


@contextmanager
def default_dtype(dtype: torch.dtype):
    """Run a block with a custom default dtype (Kimi-K3.1 is a BF16 checkpoint)."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


@contextmanager
def skip_parameter_init():
    """Disable weight initialisation of ``nn.Linear`` / ``nn.Embedding`` while building.

    Every parameter of a K3.1 decoder layer is read from the checkpoint right after the
    module is constructed, so running ``reset_parameters`` would only burn time (and touch
    uninitialised memory that is then overwritten anyway).  ``nn.Embedding`` is included
    because ``embed_tokens`` alone is 163840 x 7168 elements.
    """
    with (
        patch.object(nn.Linear, "reset_parameters", lambda _self: None),
        patch.object(nn.Embedding, "reset_parameters", lambda _self: None),
    ):
        yield


class KimiK31AdapterConfig(BaseModel):
    """Control parameters declared in the practice ``spec.model_adapter``.

    Provided by ``KimiK31AdapterLoader.get_adapter_config_class`` and validated
    uniformly during the quantization pre-check; the adapter applies it via
    :meth:`KimiK31ModelAdapter.apply_adapter_config`.
    """

    add_rot_linear: bool = Field(
        default=True,
        description="是否在 mm_projector 尾部挂 QuaRot 旋转点。True（默认）挂 Identity "
        "`rot_proj`，产物里会多出全 1 权重的 `mm_projector.rot_proj.weight`；"
        "False 不执行该 patch，产物里也就没有这个线性层。",
    )


@logger_setter("msmodelslim.model.kimi_k3_1")
class KimiK31ModelAdapter(  # pylint: disable=too-many-ancestors
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    LayerWiseOffloadOptionalInterface,
    AscendV1SaveInterface,
    IterSmoothInterface,
    FlexSmoothQuantInterface,
    QuaRotInterface,
):
    """Kimi-K3.1 (MoonViT3d + 69 KDA / 24 NoPE-MLA-DSA layers + latent MoE) adapter.

    Layout walked by the pipeline: ``vision_tower`` -> ``mm_projector`` -> 93 text layers,
    which is the order the released ``KimiK31ForConditionalGeneration`` applies them.
    Unlike ``kimi_k3`` the whole graph is built from the self-contained
    ``msmodelslim.model.kimi_k3_1`` implementation, so no remote code is executed.

    Per-matrix switches of :meth:`get_rotate_map` (``rot_uv`` is intentionally absent:
    ``mla_use_output_gate`` breaks the offline V/o_proj Hadamard cancellation).

    ``add_rot_linear`` (``spec.model_adapter.add_rot_linear``, default True) is the higher-level
    switch: it controls whether the graph gets the Identity ``mm_projector.rot_proj``
    rotation point that only QuaRot consumes. With ``False`` the pipeline patch is
    skipped, so the exported ``mm_projector`` keeps its original graph.
    """

    enable_rot = True
    enable_rot_b_proj = True
    enable_rot_kv_b_proj = True
    enable_rot_latent = True

    def __init__(self, model_type: str, model_path: Any, trust_remote_code: bool = False):
        self._processor = None
        self._tokenizer = None
        #: Whether the Identity ``mm_projector.rot_proj`` linear is added
        self._add_rot_linear = True
        super().__init__(model_type, model_path, trust_remote_code)

    def apply_adapter_config(self, config: KimiK31AdapterConfig) -> None:
        """Apply the validated ``spec.model_adapter`` control parameters."""
        self._add_rot_linear = config.add_rot_linear

    # ------------------------------------------------------------------ #
    # IModel / ModelInfoInterface
    # ------------------------------------------------------------------ #
    def get_model_pedigree(self) -> str:
        return "kimi_k3_1"

    def get_model_type(self) -> str:
        return self.model_type

    def _load_config(self, trust_remote_code: bool = False) -> KimiConfigNode:
        """Read ``config.json`` directly instead of going through remote code.

        ``kimi_k31`` is not a ``transformers`` ``model_type`` and the snapshot's
        ``auto_map`` only registers ``AutoConfiguration``; parsing the JSON keeps the
        adapter usable without ``trust_remote_code`` and without executing the shipped
        ``configuration_kimi_k3.py``.
        """
        config_path = os.path.join(str(self.model_path), "config.json")
        config_data = json_safe_load(config_path, check_user_stat=True)
        if not isinstance(config_data, dict):
            raise SchemaValidateError(f"Invalid config.json at {config_path}.")
        if "text_config" not in config_data:
            raise UnsupportedError(
                f"{config_path} has no text_config, it does not look like a Kimi-K3.1 checkpoint.",
                action="Please pass the directory that contains the released Kimi-K3.1 config.json.",
            )
        return to_config_node(config_data)

    def get_global_model_torch_dtype(self) -> torch.dtype:
        """Kimi-K3.1 ships BF16; ``dtype``/``torch_dtype`` are read before falling back."""
        text_config = self._get_text_config()
        for holder in (text_config, self.config, getattr(self.config, "vision_config", None)):
            if holder is None:
                continue
            for key in ("dtype", "torch_dtype"):
                value = holder.get(key) if isinstance(holder, dict) else getattr(holder, key, None)
                if isinstance(value, torch.dtype):
                    return value
                if isinstance(value, str) and value in ("float32", "float16", "bfloat16"):
                    return getattr(torch, value)
        return torch.bfloat16

    def get_layer_wise_offload_device(self):
        """Prefer ``meta`` so finished layers leave host memory before the next one loads."""
        return "meta"

    def _get_text_config(self) -> Any:
        config = self.config
        text_config = config.get("text_config") if isinstance(config, dict) else getattr(config, "text_config", None)
        if text_config is None:
            raise UnsupportedError("Kimi-K3.1 config.json is missing text_config.")
        return text_config

    # ------------------------------------------------------------------ #
    # Calibration data
    # ------------------------------------------------------------------ #
    @staticmethod
    def _field(item: Any, name: str) -> Any:
        if isinstance(item, dict):
            return item.get(name)
        return getattr(item, name, None)

    def _load_processor(self) -> None:
        """Load the Kimi-K3.1 processor, falling back to the tokenizer for text-only data."""
        try:
            from transformers import AutoProcessor  # pylint: disable=import-outside-toplevel

            self._processor = AutoProcessor.from_pretrained(  # nosec B615 - local dir only
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,
            )
            return
        except Exception as error:  # pylint: disable=broad-exception-caught
            processor_error = error

        try:
            from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

            self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 - local dir only
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise UnsupportedError(
                f"Failed to load the Kimi-K3.1 processor/tokenizer from {self.model_path}.",
                action=(
                    "Make sure the checkpoint directory contains the processor assets "
                    "(preprocessor_config.json, kimi_k3_processor.py, tiktoken.model) and that "
                    "trust_remote_code is enabled. processor_error="
                    f"{processor_error}; tokenizer_error={error}"
                ),
            ) from error

        get_logger().warning(
            "AutoProcessor is unavailable (%s); only text-only calibration samples can be used.",
            processor_error,
        )

    def _encode_sample(self, image: Optional[str], text: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image_url", "image_url": image}, {"type": "text", "text": text}]
                    if image is not None
                    else [{"type": "text", "text": text}]
                ),
            }
        ]
        if self._processor is not None:
            return self._processor(messages=messages, return_tensors="pt")
        if image is not None:
            raise UnsupportedError(
                "Image calibration sample requires the Kimi-K3.1 AutoProcessor.",
                action="Ship the processor assets with the checkpoint or use text-only calibration data.",
            )
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._tokenizer(prompt, return_tensors="pt")

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """Tokenize/preprocess a (possibly mixed image+text) calibration set."""
        if self._processor is None and self._tokenizer is None:
            self._load_processor()

        processed_data: List[Any] = []
        for item in tqdm(dataset, desc="Processing Kimi-K3.1 calibration dataset"):
            text = self._field(item, "text")
            if text is None:
                raise UnsupportedError(
                    "Kimi-K3.1 calibration samples must carry a 'text' field.",
                    action="Please use a text (or image+text) calibration dataset.",
                )
            image = self._field(item, "image")
            image_path = get_valid_read_path(image) if image else None
            inputs = self._encode_sample(None if image_path is None else str(image_path), text)
            processed_data.append(
                self._collect_inputs_to_device(
                    inputs,
                    device,
                    keys=["input_ids", "attention_mask", "pixel_values", "grid_thws"],
                    defaults={},
                )
            )

        get_logger().info("Kimi-K3.1 dataset preprocessing finished, samples=%d", len(processed_data))
        return processed_data

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #
    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """Build vision + projector + one decoder layer; the rest load on demand."""
        del device  # the layer-wise runner moves every module itself
        text_config = self._get_text_config()
        origin_layers = int(text_config.num_hidden_layers)
        vision_layers = int(getattr(getattr(self.config, "vision_config", None), "vt_num_hidden_layers", 0))
        get_logger().info("Kimi-K3.1: %d text layers + %d vision layers", origin_layers, vision_layers)

        self.config.use_cache = False
        text_config.use_cache = False
        # The released snapshot is FP8-ish BF16 + mxfp4 experts; keep the dtype but load
        # every tensor through _get_state_dict so the compressed-tensors metadata is never
        # interpreted by transformers.
        with default_dtype(torch.bfloat16), torch.device("cpu"), skip_parameter_init():
            text_config.num_hidden_layers = 1
            try:
                model = KimiK31ForConditionalGeneration(self.config)
            finally:
                text_config.num_hidden_layers = origin_layers

        state_dict = self._get_state_dict(model)
        self._load_state_dict_compatible(model, state_dict)
        dequant_subtree_mxfp4_to_bf16(model, "", str(self.model_path))
        # KDA is literally Kimi-K3's code, so apply Kimi-K3's Ascend runtime patch: it swaps
        # the Triton ``chunk_kda`` for ``runtime_patches._run_naive_kda`` (the Triton kernel
        # times out on NPU).  Patching the module-level name covers every on-demand layer.
        apply_kimi_k3_runtime_patches(model)
        # QuaRot treats the projector output as part of the residual stream: an Identity
        # rot_proj keeps the media path in the same rotated basis as the text tokens.
        # Without QuaRot there is no rotation to apply, so the placeholder is skipped and
        # the export carries no all-ones ``mm_projector.rot_proj.weight``.
        if self._add_rot_linear:
            self._patch_mm_projector_rot_proj(model)
        else:
            get_logger().info("add_rot_linear=False: skipping the mm_projector.rot_proj patch")
        model.eval()

        get_logger().info("Kimi-K3.1 model initialised (%d layers on demand)", origin_layers)
        return model

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """Data-free walk: vision tower, projector, then every decoder layer."""
        get_logger().info("Visiting vision tower...")
        yield ProcessRequest(name=VISION_ROOT, module=model.vision_tower, args=(), kwargs={})

        get_logger().info("Visiting mm_projector...")
        yield ProcessRequest(name=PROJECTOR_ROOT, module=model.mm_projector, args=(), kwargs={})

        get_logger().info("Visiting text decoder layers...")
        yield from generated_decoder_layer_visit_func(model, transformer_blocks=self.generate_decoder_layer(model))

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        num_layers = int(self._get_text_config().num_hidden_layers)
        for layer_idx in range(num_layers):
            name = f"{LAYER_ROOT}.{layer_idx}"
            yield name, self._load_decoder_if_not_exist(model, name, layer_idx)

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """Layer-wise forward equivalent to ``KimiK31ForConditionalGeneration.forward``."""
        sample = inputs[0] if isinstance(inputs, list) else inputs
        if not isinstance(sample, dict) or sample.get("input_ids") is None:
            raise SchemaValidateError(
                f"Kimi-K3.1 calibration sample must be a dict with input_ids, got {type(sample).__name__}.",
            )

        input_ids = sample["input_ids"]
        attention_mask = sample.get("attention_mask")
        pixel_values = sample.get("pixel_values")
        grid_thws = sample.get("grid_thws")

        backbone = model.language_model.model
        inputs_embeds = backbone.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        has_media = pixel_values is not None and pixel_values.numel() > 0 and input_ids.shape[1] > 1
        if has_media:
            target_dtype = model.vision_tower.patch_embed.proj.weight.dtype
            image_features = yield ProcessRequest(
                name=VISION_ROOT,
                module=model.vision_tower,
                args=(pixel_values.to(target_dtype),),
                kwargs={"grid_thws": grid_thws} if grid_thws is not None else {},
            )
            if model.mm_projector is not None:
                image_features = yield ProcessRequest(
                    name=PROJECTOR_ROOT,
                    module=model.mm_projector,
                    args=(image_features,),
                    kwargs={},
                )
            inputs_embeds = inputs_embeds.to(image_features[0].dtype)
            inputs_embeds, attention_mask, _ = model.merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        batch, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
        # MLA / DSA layers consume the boolean ``[B, 1, S, S]`` keep-mask, KDA layers the
        # ``[B, S]`` padding mask; each layer picks the flavour it understands.
        causal_mask = backbone.build_causal_mask(seq_len, batch, attention_mask, inputs_embeds.device)
        block_residual = backbone.empty_block_residual(inputs_embeds)
        hidden_states = inputs_embeds

        for name, layer in self.generate_decoder_layer(model):
            if dist.is_initialized():
                dist.barrier()
            outputs = yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    "attention_mask": causal_mask,
                    "padding_mask": attention_mask,
                    "block_residual": block_residual,
                },
            )
            hidden_states, block_residual = self._split_layer_output(outputs, hidden_states, block_residual)

    @staticmethod
    def _split_layer_output(
        outputs: Any, hidden_states: torch.Tensor, block_residual: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``KimiDecoderLayer`` returns ``(hidden_states, block_residual)``.

        Processors may replace the layer output with a plain tensor (e.g. pure fake-quant
        wrappers), in which case the AttnRes buffer has to be carried over unchanged.
        """
        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 2 and isinstance(outputs[0], torch.Tensor):
                return outputs[0], outputs[1]
            if len(outputs) >= 1 and isinstance(outputs[0], torch.Tensor):
                return outputs[0], block_residual
            raise SchemaValidateError(
                f"Unexpected Kimi-K3.1 decoder layer output type: {type(outputs).__name__}.",
            )
        if isinstance(outputs, torch.Tensor):
            return outputs, block_residual
        raise SchemaValidateError(
            f"Unexpected Kimi-K3.1 decoder layer output type: {type(outputs).__name__}.",
            action="Please make sure the processor returns the layer activations or (activations, block_residual).",
        )

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        self.config.use_cache = need_kv_cache
        self._get_text_config().use_cache = need_kv_cache
        get_logger().info("KV cache %s", "enabled" if need_kv_cache else "disabled")

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #
    def _get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        """Load ``module``'s parameters from the safetensors shards named by ``prefix``."""
        weight_map = get_full_weight_map(str(self.model_path))
        file_groups: Dict[str, List[str]] = defaultdict(list)
        for param_name, _ in module.named_parameters():
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if full_name in weight_map:
                file_groups[weight_map[full_name]].append(param_name)

        state_dict: Dict[str, torch.Tensor] = {}
        for file_name, names in tqdm(file_groups.items(), desc=f"Loading {prefix or 'model'}", leave=False):
            file_path = get_valid_read_path(
                os.path.join(str(self.model_path), file_name),
                extensions="safetensors",
                size_max=MAX_READ_FILE_SIZE_32G,
            )
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for param_name in names:
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    state_dict[param_name] = handle.get_tensor(full_name)
        return state_dict

    def _load_state_dict_compatible(self, module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load tolerating the known checkpoint/runtime shape skew.

        ``self_attn.A_log`` is the only parameter whose released layout may disagree with
        the runtime: FLA stores one value per head (``[num_heads]``, 96) while the
        checkpoint is free to dump the per-channel form (``[head_dim]``, 128).  Any other
        size-preserving layout difference (e.g. a flattened ``dt_bias``) is reshaped
        instead of being dropped, so no parameter is silently left uninitialised.
        """
        model_state = module.state_dict()
        adapted: Dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            target = model_state.get(name)
            if target is None:
                adapted[name] = tensor
                continue
            if tuple(tensor.shape) == tuple(target.shape):
                adapted[name] = tensor
                continue
            if name.endswith("A_log") and tensor.ndim == 1 and target.ndim == 1 and tensor.numel() >= target.numel():
                adapted[name] = tensor[: target.numel()].contiguous()
                get_logger().warning(
                    "Adapted %s from ckpt %s to model %s (FLA keeps one value per head).",
                    name,
                    tuple(tensor.shape),
                    tuple(target.shape),
                )
                continue
            if tensor.numel() == target.numel():
                adapted[name] = tensor.reshape(target.shape)
                get_logger().warning(
                    "Reshaped %s from ckpt %s to model %s.", name, tuple(tensor.shape), tuple(target.shape)
                )
                continue
            get_logger().warning(
                "Skip mismatched weight %s: ckpt %s vs model %s", name, tuple(tensor.shape), tuple(target.shape)
            )

        incompatible = module.load_state_dict(adapted, strict=False)
        missing = len(incompatible.missing_keys)
        if missing:
            # mxfp4 routed experts are addressable only via weight_packed/_scale here and are
            # filled in by dequant_subtree_mxfp4_to_bf16 right after this call.
            get_logger().debug("Missing keys after compatible load: %d", missing)
        if incompatible.unexpected_keys:
            get_logger().debug("Unexpected keys after compatible load: %d", len(incompatible.unexpected_keys))

    @staticmethod
    def _is_materialized(module: Optional[nn.Module]) -> bool:
        """``False`` for a layer that was offloaded to the ``meta`` device."""
        if module is None:
            return False
        for parameter in module.parameters():
            return parameter.device.type != "meta"
        return True

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        module_list: nn.ModuleList = model.language_model.model.layers
        existing = module_list[idx] if idx < len(module_list) else None
        if self._is_materialized(existing):
            get_logger().debug("Layer %d already loaded", idx)
            return existing

        get_logger().info("Loading Kimi-K3.1 decoder layer %d...", idx)
        with skip_parameter_init(), default_dtype(torch.bfloat16), torch.device("cpu"):
            decoder = KimiDecoderLayer(self._get_text_config(), layer_idx=idx)
            state_dict = self._get_state_dict(decoder, prefix=name)
            self._load_state_dict_compatible(decoder, state_dict)
            dequant_subtree_mxfp4_to_bf16(decoder, name, str(self.model_path))
            decoder.eval()

        if idx < len(module_list):
            module_list[idx] = decoder
        else:
            module_list.append(decoder)
        get_logger().info("Decoder layer %d loaded", idx)
        return decoder

    # ------------------------------------------------------------------ #
    # QuaRot
    # ------------------------------------------------------------------ #
    def _patch_mm_projector_rot_proj(self, model: nn.Module) -> None:
        """Append an Identity ``rot_proj`` so QuaRot can left-rotate the projector output."""
        projector = getattr(model, "mm_projector", None)
        if projector is None or hasattr(projector, "rot_proj"):
            return

        hidden = int(self._get_text_config().hidden_size)
        rot_proj = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            rot_proj.weight.copy_(torch.eye(hidden, dtype=rot_proj.weight.dtype, device=rot_proj.weight.device))
        projector.add_module("rot_proj", rot_proj)

        original_forward = projector.forward

        def forward_with_rot(self, x, *args, **kwargs):
            out = original_forward(x, *args, **kwargs)
            if isinstance(out, (list, tuple)):
                return type(out)(self.rot_proj(item) for item in out)
            return self.rot_proj(out)

        projector.forward = types.MethodType(forward_with_rot, projector)
        get_logger().info("Patched mm_projector.rot_proj (Linear %d->%d, eye init)", hidden, hidden)

    def get_ln_fuse_map(self):
        return build_ln_fuse_map(self.config, num_hidden_layers=int(self._get_text_config().num_hidden_layers))

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        return build_rotate_map(
            self.config,
            block_size,
            num_hidden_layers=int(self._get_text_config().num_hidden_layers),
            enable_rot=self.enable_rot,
            enable_rot_b_proj=self.enable_rot_b_proj,
            enable_rot_kv_b_proj=self.enable_rot_kv_b_proj,
            enable_rot_latent=self.enable_rot_latent,
        )

    # ------------------------------------------------------------------ #
    # IterSmooth / FlexSmooth
    # ------------------------------------------------------------------ #
    def _mla_subgraph_configs(self, layer_idx: int) -> List[AdapterConfig]:
        """Subgraphs of a NoPE-MLA + DSA layer (24 of the 93 layers)."""
        text_cfg = self._get_text_config()
        prefix = layer_prefix(layer_idx)
        attn = attn_prefix(layer_idx)
        return [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source=f"{prefix}.input_layernorm",
                    # q_a_proj / kv_a_proj_with_mqa / g_proj and, on DSA layers, the indexer
                    # wk + weights_proj that read the same hidden state.
                    targets=input_layernorm_targets(text_cfg, layer_idx),
                ),
            ),
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source=f"{attn}.q_a_layernorm",
                    # q_b_proj and the indexer wq_b that consumes the same q residual.
                    targets=q_a_layernorm_targets(text_cfg, layer_idx),
                ),
            ),
            # kv_a_layernorm -> kv_b_proj is handled by get_ln_fuse_map (QuaRot), so
            # IterSmooth only smooths the norm-linear pairs above.
        ]

    def _kda_subgraph_configs(self, layer_idx: int) -> List[AdapterConfig]:
        """Subgraphs of a KDA linear-attention layer (69 of the 93 layers)."""
        text_cfg = self._get_text_config()
        prefix = layer_prefix(layer_idx)
        return [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source=f"{prefix}.input_layernorm",
                    targets=input_layernorm_targets(text_cfg, layer_idx),
                ),
            ),
        ]

    def _ffn_subgraph_configs(self, layer_idx: int) -> List[AdapterConfig]:
        """Dense MLP (layer 0) or latent-MoE (layers 1..92) subgraphs."""
        text_cfg = self._get_text_config()
        prefix = layer_prefix(layer_idx)
        configs: List[AdapterConfig] = [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(
                    source=f"{prefix}.post_attention_layernorm",
                    # Every Linear fed by the same norm must be listed (router, shared
                    # experts and the latent down projection), otherwise absorbing the
                    # smoothing scale leaves those branches uncompensated.
                    targets=post_attention_layernorm_targets(text_cfg, layer_idx),
                ),
            )
        ]

        if is_dense_mlp(text_cfg, layer_idx):
            configs.append(
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=MappingConfig(
                        source=f"{prefix}.mlp.up_proj",
                        targets=[f"{prefix}.mlp.down_proj"],
                    ),
                )
            )
            return configs

        moe = moe_prefix(layer_idx)
        configs.append(
            AdapterConfig(
                subgraph_type="up-down",
                mapping=MappingConfig(
                    source=f"{moe}.shared_experts.up_proj",
                    targets=[f"{moe}.shared_experts.down_proj"],
                ),
            )
        )

        routed_norm = routed_expert_norm_targets(text_cfg, layer_idx, require_latent_hidden=True)
        if routed_norm is not None:
            configs.append(
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=MappingConfig(
                        source=f"{moe}.routed_expert_norm",
                        targets=routed_norm,
                    ),
                )
            )

        # EP: only the experts owned by this rank exist in the module tree.
        expert_start, expert_end = _get_expert_range(text_cfg)
        for expert in range(expert_start, expert_end):
            configs.append(
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=MappingConfig(
                        source=f"{moe}.experts.{expert}.w3",
                        targets=[f"{moe}.experts.{expert}.w2"],
                    ),
                )
            )
        return configs

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """IterSmooth (and FlexSmooth) subgraph pairs of Kimi-K3.1.

        ``norm-linear`` targets of a shared Norm list every Linear that consumes it (the
        same completeness rule as :meth:`get_ln_fuse_map`); the AttnRes norms stay
        QuaRot-only because each of them feeds a single scalar projection.
        """
        adapter_config: List[AdapterConfig] = []
        text_cfg = self._get_text_config()
        num_layers = int(text_cfg.num_hidden_layers)
        for layer_idx in range(num_layers):
            if is_kda_layer(text_cfg, layer_idx):
                adapter_config.extend(self._kda_subgraph_configs(layer_idx))
            else:
                adapter_config.extend(self._mla_subgraph_configs(layer_idx))
            adapter_config.extend(self._ffn_subgraph_configs(layer_idx))
        get_logger().info("Built %d IterSmooth subgraph configs for %d layers", len(adapter_config), num_layers)
        return adapter_config

    # ------------------------------------------------------------------ #
    # AscendV1 save hooks
    # ------------------------------------------------------------------ #
    def _checkpoint_tensor_shape(self, full_name: str) -> Optional[Tuple[int, ...]]:
        """Shape of ``full_name`` in the checkpoint without materialising the tensor."""
        file_name = get_full_weight_map(str(self.model_path)).get(full_name)
        if file_name is None:
            return None
        file_path = get_valid_read_path(
            os.path.join(str(self.model_path), file_name),
            extensions="safetensors",
            size_max=MAX_READ_FILE_SIZE_32G,
        )
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            if full_name not in handle.keys():
                return None
            return tuple(handle.get_slice(full_name).get_shape())

    def ascendv1_save_module_preprocess(
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        """Restore the checkpoint layout of ``KDA.A_log`` before it is written out.

        FLA keeps one decay bound per head (``[num_heads]``); the released float weights may
        dump the per-channel form (``[head_dim]``) whose tail is zero.  The target length is
        taken from the checkpoint itself, so a snapshot that already stores ``[num_heads]``
        is written back untouched.
        """
        a_log = getattr(module, "A_log", None)
        if not isinstance(a_log, nn.Parameter) or a_log.ndim != 1:
            return prefix, module

        checkpoint_shape = self._checkpoint_tensor_shape(f"{prefix}.A_log")
        if not checkpoint_shape or len(checkpoint_shape) != 1 or checkpoint_shape[0] <= a_log.numel():
            return prefix, module

        padded = a_log.detach().new_zeros(int(checkpoint_shape[0]))
        padded[: a_log.numel()] = a_log.detach()
        module.A_log = nn.Parameter(padded, requires_grad=False)
        get_logger().info(
            "Padded %s.A_log from %s to %s to match the float checkpoint layout.",
            prefix,
            tuple(a_log.shape),
            checkpoint_shape,
        )
        return prefix, module

    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """Ship the tiktoken vocabulary, which ``copy_files`` skips (``.model`` suffix)."""
        tiktoken_file = os.path.join(str(self.model_path), "tiktoken.model")
        if not os.path.isfile(tiktoken_file):
            return
        dest_file = os.path.join(save_directory, "tiktoken.model")
        safe_copy_file(src_path=tiktoken_file, dest_path=dest_file)
        os.chmod(dest_file, int("600", 8))
