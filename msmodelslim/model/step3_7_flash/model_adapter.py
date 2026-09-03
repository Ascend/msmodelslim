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

from typing import List, Any, Generator, Dict, Tuple
import os
from collections import defaultdict
from pathlib import Path
from functools import lru_cache

from torch import nn
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.modelslim_v1.save.interface import AscendV1SaveInterface
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_load, MAX_READ_FILE_SIZE_512G
from ..step3_5_flash.moe_utils import convert_step35_moe_to_unpacked
from ..common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
)
from ..common.vlm_base import VLMBaseModelAdapter
from ..interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    LayerWiseOffloadOptionalInterface,
)
from .step3p7_mtp import Step3p7MTPModule


# pylint: disable=too-many-ancestors
@logger_setter()
class Step3_7FlashModelAdapter(
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    AscendV1SaveInterface,
    LayerWiseOffloadOptionalInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        self._tokenizer = None
        super().__init__(model_type, model_path, trust_remote_code)
        self.mtp_start_layer = self.config.text_config.num_hidden_layers
        self.mtp_layer_num = 3
        # Captured in init_model after from_pretrained. The Step3p7DecoderLayer
        # class is loaded dynamically via trust_remote_code, so we obtain it
        # from the live model rather than importing it directly.
        self._decoder_layer_cls = None
        # MoE layer indices resolved once from config; reused per layer visit.
        self._moe_layers_idx = self._resolve_moe_layers_idx()

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return 'step_3_7_flash'

    def get_layer_wise_offload_device(self):
        # Prefer meta so layer-wise runner can drop finished layers from host memory.
        return "meta"

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """Legacy v0 / StandingHighWithExperience entry point.

        Delegates to ``init_model`` so callers that still hit ``load_model``
        (e.g. the v0 quantisation pipeline) get the same lazy-load behaviour
        that ``init_model`` provides for v1.
        """
        return self.init_model(device)

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """Tokenise the calibration dataset and return ``list[dict]``.

        Each item is a dict with at least ``input_ids`` and ``attention_mask``,
        matching the contract expected by ``generate_model_forward`` (which
        indexes via ``sample["input_ids"]``). The framework's default
        tokeniser path (``_get_tokenized_data``) returns
        ``list[[input_ids, attn_mask]]`` instead, which triggers
        ``IndexError: too many indices for tensor of dimension 2`` in the
        forward generator — hence this override.
        """
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                str(self.model_path),
                trust_remote_code=self.trust_remote_code,
                local_files_only=True,
                use_fast=False,
            )
        device_name = "npu" if device is DeviceType.NPU else "cpu"
        processed = []
        for item in dataset:
            if isinstance(item, dict):
                text = item.get("text") or item.get("prompt") or item.get("inputs_pretokenized") or str(item)
            else:
                text = str(item)
            inputs = self._tokenizer(str(text), return_tensors="pt").to(device_name)
            processed.append(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
        return processed

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        get_logger().info("Initializing Step3_7Flash model with msmodelslim v1 framework (lazy MoE unpack)!")

        self.config.text_config.use_cache = False  # Disable cache to save device memory

        self.model_path = get_valid_read_path(str(self.model_path), is_dir=True, check_user_stat=True)

        # Snapshot the original layer count, then ask from_pretrained to build
        # only 1 decoder layer. The remaining layers are materialised lazily in
        # generate_decoder_layer so we never hold more than one layer's worth
        # of expert weights on CPU at a time.
        origin_layers = self.config.text_config.num_hidden_layers
        self.config.text_config.num_hidden_layers = 1

        model = AutoModelForCausalLM.from_pretrained(  # nosec B615: model_path is validated and local_files_only blocks Hub downloads.
            self.model_path,
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
            torch_dtype="auto",
            device_map="cpu",
            attn_implementation='eager',
        ).eval()

        # Restore the configured layer count so downstream code (MoE indices,
        # MTP offsets, generated_decoder_layer_visit_func, etc.) sees the
        # actual model shape.
        self.config.text_config.num_hidden_layers = origin_layers
        self._decoder_layer_cls = type(model.model.language_model.layers[0])

        # Step-3.7 keeps the attention-head counts only on ``text_config``.
        # Mirror them onto ``model.config`` so any downstream consumer that
        # reads ``model.config.num_attention_heads`` / ``num_key_value_heads``
        # (e.g. calibration processors) finds them. See
        # ``modeling_step3p7.py``/``Step3p5Attention.num_key_value_heads``
        # which sources its KV-head count from ``config.num_attention_groups``.
        model.config.num_attention_heads = self.config.text_config.num_attention_heads
        model.config.num_key_value_heads = self.config.text_config.num_attention_groups

        return model

    def _resolve_moe_layers_idx(self) -> List[int]:
        moe_layers_enum = getattr(self.config.text_config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            return [int(i) for i in moe_layers_enum.strip().split(',')]
        return list(range(1, self.config.text_config.num_hidden_layers))

    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        """Get weight map from model.safetensors.index.json"""
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data['weight_map']

    def _load_raw_weights_for_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Read one layer's tensors straight from safetensors.

        Returns a dict keyed by the *raw* safetensors name (``model.layers.X.*``).
        Key remapping to the live model namespace
        (``model.language_model.layers.X.*``) is the caller's responsibility.
        """
        weight_map = self._get_weight_map()
        raw_prefix = f"model.layers.{layer_idx}."

        file_groups: Dict[str, List[str]] = defaultdict(list)
        for ckpt_name, shard_file in weight_map.items():
            if ckpt_name.startswith(raw_prefix):
                file_groups[shard_file].append(ckpt_name)

        raw_weights: Dict[str, torch.Tensor] = {}
        for file_name, ckpt_names in tqdm(file_groups.items(), desc=f"Loading layer {layer_idx}", leave=False):
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(file_path, extensions='safetensors', size_max=MAX_READ_FILE_SIZE_512G)
            with safe_open(file_path, framework='pt', device='cpu') as f:
                for ckpt_name in ckpt_names:
                    raw_weights[ckpt_name] = f.get_tensor(ckpt_name)

        return raw_weights

    @staticmethod
    def _remap_to_model_keys(raw_weights: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
        """Rewrite safetensors keys ``model.layers.X.*`` → ``model.language_model.layers.X.*``."""
        raw_prefix = f"model.layers.{layer_idx}"
        model_prefix = f"model.language_model.layers.{layer_idx}"
        remapped: Dict[str, torch.Tensor] = {}
        for ckpt_name, tensor in raw_weights.items():
            if ckpt_name.startswith(raw_prefix + "."):
                remapped[model_prefix + ckpt_name[len(raw_prefix) :]] = tensor
        return remapped

    @staticmethod
    def _strip_layer_prefix(state_dict: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
        """Strip the ``model.language_model.layers.X.`` prefix so the keys match a
        freshly-constructed ``Step3p7DecoderLayer`` / ``Step3p7MTPModule``.

        ``load_state_dict`` matches keys against the module's ``named_parameters()``,
        which for a bare decoder are ``self_attn.q_proj.weight`` etc. — *without*
        any parent-module prefix. Passing a full-path state dict (as returned by
        ``_remap_to_model_keys``) under ``strict=False`` silently drops every
        tensor, leaving the decoder at its random initialisation. Always call this
        helper before ``decoder.load_state_dict(...)`` in the lazy-load paths.
        """
        prefix = f"model.language_model.layers.{layer_idx}."
        return {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()}

    def _load_decoder_if_not_exist(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Materialise one decoder layer on demand and unpack its MoE experts.

        If the layer is already loaded (e.g. layer 0 from init_model), reuse it.
        Otherwise construct a fresh ``Step3p7DecoderLayer``, load weights from
        safetensors, and — for MoE layers — swap the fused experts for the
        unpacked ``Step3p5MoEMLPWithUnpackExperts`` so the linear quantiser can
        reach every expert weight as an independent ``nn.Linear``.
        """
        name = f"model.language_model.layers.{layer_idx}"

        # Fast path: layer already loaded by from_pretrained or a prior visit.
        try:
            existing = model.get_submodule(name)
            try:
                _ = existing.input_layernorm.weight
                return existing
            except (RuntimeError, AttributeError):
                pass  # placeholder / meta tensor → fall through to real load
        except AttributeError:
            pass

        get_logger().info("Lazy-loading Step3_7Flash decoder layer %s...", layer_idx)

        decoder = self._decoder_layer_cls(model.config.text_config, layer_idx=layer_idx)
        decoder.eval()

        raw_weights = self._load_raw_weights_for_layer(layer_idx)
        state_dict = self._remap_to_model_keys(raw_weights, layer_idx)
        state_dict = self._strip_layer_prefix(state_dict, layer_idx)

        missing, unexpected = decoder.load_state_dict(state_dict, strict=False)
        if unexpected:
            get_logger().debug("Layer %s unexpected keys (first 5): %s", layer_idx, unexpected[:5])
        if missing:
            get_logger().debug("Layer %s missing keys (first 5): %s", layer_idx, missing[:5])

        module_list: nn.ModuleList = model.model.language_model.layers
        if len(module_list) <= layer_idx:
            module_list.append(decoder)
        else:
            module_list[layer_idx] = decoder

        # Per-layer MoE unpack. This is the lazy equivalent of the previous
        # _convert_moe_layers_to_unpacked sweep — it runs on this layer only,
        # so the peak CPU footprint is one fused + one unpacked MoE (~15 GB)
        # rather than the whole model at once.
        if layer_idx in self._moe_layers_idx and hasattr(decoder, "moe") and decoder.moe is not None:
            try:
                new_moe = convert_step35_moe_to_unpacked(decoder.moe, self.config.text_config)
                decoder.moe = new_moe
                get_logger().debug("Layer %s MoE unpacked", layer_idx)
            except Exception as e:
                get_logger().error("Failed to unpack MoE for layer %s: %s", layer_idx, e)
                raise

        return decoder

    def _load_mtp_layer_if_not_exist(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Materialise one MTP layer (45/46/47) on demand and load its weights."""
        name = f"model.language_model.layers.{layer_idx}"

        try:
            existing = model.get_submodule(name)
            try:
                _ = existing.eh_proj.weight
                return existing
            except (RuntimeError, AttributeError):
                pass
        except AttributeError:
            pass

        get_logger().info("Lazy-loading Step3_7Flash MTP layer %s...", layer_idx)

        mtp = Step3p7MTPModule(model.config.text_config, layer_idx=layer_idx)
        mtp.eval()

        raw_weights = self._load_raw_weights_for_layer(layer_idx)
        state_dict = self._remap_to_model_keys(raw_weights, layer_idx)
        state_dict = self._strip_layer_prefix(state_dict, layer_idx)
        mtp.load_state_dict(state_dict, strict=False)

        module_list: nn.ModuleList = model.model.language_model.layers
        if len(module_list) <= layer_idx:
            module_list.append(mtp)
        else:
            module_list[layer_idx] = mtp

        return mtp

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        """Yield (name, layer) for every layer the saver should write.

        Materialises layers on demand so only one layer's weights live on CPU
        at a time. After the framework offloads the previous layer to ``meta``,
        this generator pulls the next one from disk.
        """
        num_layers = self.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            name = f"model.language_model.layers.{layer_idx}"
            layer = self._load_decoder_if_not_exist(model, layer_idx)
            yield name, layer

        # MTP layers follow the regular decoder layers at positions
        # num_hidden_layers..num_hidden_layers+mtp_layer_num-1.
        for layer_idx in range(self.mtp_start_layer, self.mtp_start_layer + self.mtp_layer_num):
            name = f"model.language_model.layers.{layer_idx}"
            layer = self._load_mtp_layer_if_not_exist(model, layer_idx)
            yield name, layer

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(
            model,
            transformer_blocks=self.generate_decoder_layer(model),
        )

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        """Drive a forward pass through the language model so calibration
        processors can collect per-layer stats.

        Mirrors the structure of ``minimax_m3.generate_model_forward`` but
        for Step-3.7-Flash: it visits the multimodal vision encoder first if
        the calibration batch carries images, then iterates the text decoder
        layers directly via ``_load_decoder_if_not_exist`` (the same lazy
        materialiser ``generate_decoder_layer`` uses, but skipping MTP).

        The returned ``ProcessRequest`` for each decoder layer uses the
        Step3p7DecoderLayer signature (see modeling_step3p7.py:823). We pass
        ``position_ids`` rather than precomputed ``position_embeddings`` —
        Step3p7Attention recomputes cos/sin internally from ``position_ids``
        via its own rotary embedding.
        """
        sample = inputs[0] if isinstance(inputs, list) else inputs
        input_ids = sample["input_ids"]
        attention_mask = sample.get("attention_mask")

        pixel_values = sample.get("pixel_values")
        image_grid_thw = sample.get("image_grid_thw")
        has_images = pixel_values is not None and image_grid_thw is not None

        if has_images:
            # StepRoboticsVisionEncoder.forward takes pixel_values only;
            # image_grid_thw is unused at this layer (derived from H/W).
            image_embeds = yield ProcessRequest(
                name="vision_model",
                module=model.model.vision_model,
                args=(pixel_values,),
                kwargs={},
            )
        else:
            image_embeds = None

        inputs_embeds = model.model.language_model.embed_tokens(input_ids)

        if has_images and image_embeds is not None:
            if isinstance(image_embeds, (list, tuple)):
                image_embeds_cat = torch.cat(image_embeds, dim=0)
            else:
                image_embeds_cat = image_embeds
            image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask,
                image_embeds_cat.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        # 4. Build causal mask + position ids.
        # Mirrors Step3p7Model.forward (modeling_step3p7.py:972-991): a dict of
        # masks keyed by attention_type ("full_attention" / "sliding_attention"),
        # so each decoder layer can pick its own mask via
        # ``causal_mask_mapping[decoder_layer.attention_type]``.
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)

        # transformers API drift across versions:
        #   4.57.x : create_causal_mask(config, input_embeds, attention_mask,
        #                                 cache_position, past_key_values, ...)
        #   5.x     : create_causal_mask(config, inputs_embeds, attention_mask,
        #                                 past_key_values, ...)
        # The original Step-3.7 forward picks the right kw name via the
        # ``_MASK_INPUT_EMBEDS_ARG`` constant in modeling_step3p7.py:44-48.
        import inspect

        sig = inspect.signature(create_causal_mask).parameters
        if "inputs_embeds" in sig:
            mask_embeds_key = "inputs_embeds"
        else:
            mask_embeds_key = "input_embeds"
        mask_kwargs = {
            "config": model.config.text_config,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
            mask_embeds_key: inputs_embeds,
        }
        # 4.57.x requires cache_position; 5.x dropped it.
        if "cache_position" in sig:
            mask_kwargs["cache_position"] = cache_position

        # ``has_sliding_layers`` mirrors Step3p7TextModel.__init__
        # (modeling_step3p7.py:899-901): empty/missing layer_types defaults to
        # True (sliding_attention appears somewhere); otherwise check the
        # explicit list. The real Step-3.7 config has a non-empty layer_types
        # including both "sliding_attention" and "full", so this branch fires.
        layer_types = getattr(model.config.text_config, "layer_types", None) or []
        has_sliding_layers = not layer_types or "sliding_attention" in layer_types

        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if has_sliding_layers:
            sliding_sig = inspect.signature(create_sliding_window_causal_mask).parameters
            sliding_kwargs = dict(mask_kwargs)
            # The sliding-window mask factory only honours the legacy name on
            # 4.57.x; remap so both API versions are covered.
            if (
                mask_embeds_key == "input_embeds"
                and "input_embeds" not in sliding_sig
                and "inputs_embeds" in sliding_sig
            ):
                sliding_kwargs["inputs_embeds"] = sliding_kwargs.pop("input_embeds")
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**sliding_kwargs)

        # Slice to ``num_hidden_layers`` and skip MTP, mirroring
        # ``Step3p7Model.forward`` (modeling_step3p7.py:997:
        # ``self.layers[:num_hidden_layers]``). MTP slots are reserved for
        # speculative decoding — ``Step3p7MTPModule`` has no ``forward``, so
        # the framework's ``module(*args, attention_mask=...)`` call would
        # ``TypeError``. ``generate_decoder_layer`` still yields MTP for the
        # save/quant path.
        hidden_states = inputs_embeds
        num_layers = self.config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            name = f"model.language_model.layers.{layer_idx}"
            layer = self._load_decoder_if_not_exist(model, layer_idx)
            attention_type = getattr(layer, "attention_type", "full_attention")
            layer_mask = causal_mask_mapping.get(attention_type)
            if layer_mask is None:
                layer_mask = causal_mask_mapping["full_attention"]
            hidden_states = yield ProcessRequest(
                name=name,
                module=layer,
                args=(hidden_states,),
                kwargs={
                    "attention_mask": layer_mask,
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_value": None,
                    "use_cache": False,
                    "output_attentions": False,
                },
            )

        # Terminal RMSNorm lives outside the layers list, so yield it
        # explicitly so calibration processors visit it too.
        yield ProcessRequest(
            name="model.language_model.norm",
            module=model.model.language_model.norm,
            args=(hidden_states,),
            kwargs={},
        )

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)

    def ascendv1_save_module_preprocess(self, prefix: str, module: nn.Module, model: nn.Module):
        """将 PyTorch 模块层次命名转换为 HF safetensors 命名约定。

        model.language_model.layers.X  ->  model.layers.X
        model.vision_model.X           ->  vision_model.X
        model.vit_large_projector.X    ->  vit_large_projector.X
        """
        prefix = prefix.replace("model.language_model.layers.", "model.layers.")
        prefix = prefix.replace("model.vision_model.", "vision_model.")
        prefix = prefix.replace("model.vit_large_projector.", "vit_large_projector.")
        return prefix, module
