#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
Copyright (c) 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved
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

msModelSlim adapter for the Qwen3.8-Flash-Next text backbone (qwen4_exp).

The adapter is text-only and layer-wise: only a one-layer skeleton is
materialized, remaining decoder layers are loaded from safetensors on demand
and offloaded to meta after quantization, so the ~360 GB BF16 checkpoint never
lives in host memory as a whole.

Special handling:
- MTP (`mtp.*`) is rebuilt as a real module and quantized by the same linear
  processors as the backbone; only data-free recipes are supported because the
  MTP block has no calibration forward.
- PLE (`*.ple.*` / ngram_embedding) is skipped during calibration and streamed
  into the export after the layer-wise save pass. Its huge n-gram tables may be
  exported as BF16 (default), symmetric INT8, or MXFP8.
- FA3 (`fa3_quant`) support injects fa_q/fa_k/fa_v placeholders into the QSA
  attention layers (`Qwen4ExpTextAttention`) after RoPE / cache update, enabling
  per-head static activation quantization (INT8 / FP8 E4M3) on Q/K/V.
- FA3 also injects fa_q/fa_k into the QSA token selector
  (`Qwen4ExpTextQSAIndexer`) so indexer scoring can use its own quantization
  recipe (e.g. HiF4 per-block) independently from the main attention path.

Multimodal calibration and real MTP / 3D-expert inference are out of scope;
the unstacked expert forward only exists for state-dict compatibility.
"""

import gc
import glob
import itertools
import json
import os
import re
import types
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple
from unittest.mock import patch

import torch
from pydantic import BaseModel, Field
from safetensors import safe_open
from torch import nn, distributed as dist

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.core.quant_service.modelslim_v1.save.interface import (
    AscendV1SaveInterface,
)
from msmodelslim.core.quant_service.modelslim_v1.save.utils.safetensors import (
    BufferedSafetensorsWriter,
)
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func,
)
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.interface_hub import (
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
    FlexSmoothQuantInterface,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
)
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger, logger_setter, get_logger
from msmodelslim.utils.security import (
    get_valid_read_path,
    json_safe_load,
    MAX_READ_FILE_SIZE_32G,
)

# pylint: disable=too-many-lines

_PLE_QUANT_INT8 = "int8_sym"
_PLE_QUANT_MXFP8 = "mxfp8"
_PLE_QUANT_BF16 = "bf16"

# OCP MX v1.0 block size and E4M3 / E8M0 constants used by the MXFP8 export.
_PLE_MXFP8_BLOCK = 32
_E4M3_MAX_NORM = 448.0
_E4M3_EMAX = 8
_E8M0_BIAS = 127


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype):
    """Build modules under ``dtype`` and restore the global default afterwards.

    Lazily materialized decoder/MTP layers must be created with the source
    checkpoint dtype; otherwise ``nn.Linear`` falls back to the global
    ``float32`` and silently upcasts the BF16 checkpoint weights.
    """
    previous = torch.get_default_dtype()
    if dtype != previous:
        torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        if dtype != previous:
            torch.set_default_dtype(previous)


def _text_config(config: Any) -> Any:
    """Return the text sub-config; text-only checkpoints use ``config`` itself."""
    return getattr(config, "text_config", config)


def _is_vlm_model(model: nn.Module) -> bool:
    """Whether ``model`` is the multimodal ``Qwen4ExpForConditionalGeneration``."""
    return hasattr(getattr(model, "model", None), "language_model")


def _load_json(path: str) -> Optional[dict]:
    """Best-effort JSON load; returns ``None`` on any read/parse failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _dump_json(path: str, data: Any, indent: int) -> None:
    """Write UTF-8 JSON with a stable, human-readable layout."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


class Qwen4ExpFlashNextModelConfig(BaseModel):
    """Control parameters from the practice ``spec.model_adapter`` block.

    Controls only the PLE n-gram table export; all other tensors keep the
    recipe's quantization decisions.
    """

    ple_quant_type: Literal["int8_sym", "mxfp8", "bf16"] = Field(
        default="bf16",
        description="PLE n-gram 表导出格式：`bf16`（默认）不量化，原样保留源精度；"
        "`int8_sym` 写 INT8 权重 + 每组 float32 scale；"
        "`mxfp8` 写 E4M3 权重 + 每 32 个元素的 uint8 E8M0 共享指数"
        "（与 linear 层 MXFP8 落盘格式一致）。只有显式声明量化类型才会量化 PLE 表。",
    )
    ple_quant_group_size: int = Field(
        default=32,
        gt=0,
        description="仅 `int8_sym` 使用的对称量化每组元素数（沿 embedding 宽度方向）。`mxfp8` 的块长固定为 32。",
    )


def _ple_scale_key(weight_key: str, quant_type: str = _PLE_QUANT_INT8) -> str:
    """Companion scale key for a PLE n-gram table weight.

    ``int8_sym`` writes a float32 ``.scale``; ``mxfp8`` follows the AscendV1
    linear-layer convention and writes a uint8 ``.weight_scale``.
    """
    base = weight_key.removesuffix(".weight")
    if quant_type == _PLE_QUANT_MXFP8:
        return base + ".weight_scale"
    return base + ".scale"


def _is_ple_table_key(key: str) -> bool:
    """Whether ``key`` is a PLE n-gram embedding table (the huge tensors)."""
    return ".ple." in key and ".ngram_embedding." in key and key.endswith(".weight")


def _quantize_ple_int8(
    tensor: torch.Tensor,
    group_size: int,
    block_rows: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group INT8 quantization for one PLE n-gram table shard.

    Returns ``(int8_weight, float32_scale)`` where the scale has one value
    per ``group_size`` columns. Mirrors DeepSeek-V4.1-Flash's Engram table
    export (INT8 symmetric weights + group scale), but keeps the scale in
    float32 because the Qwen PLE source is plain BF16 rather than MXFP8.

    Rows are processed in chunks so a single ~95 GB BF16 n-gram table is
    never promoted to fp32 in one shot (that would need several times the
    table size in transient buffers). Chunking is element-wise, so the
    result is byte-identical to the unchunked computation.
    """
    if tensor.dim() != 2:
        raise ValueError(f"PLE n-gram table must be 2-D, got shape {tuple(tensor.shape)}")
    rows, width = tensor.shape
    if width % group_size != 0:
        raise ValueError(f"PLE n-gram table width {width} is not divisible by ple_quant_group_size={group_size}")
    groups = width // group_size
    if block_rows <= 0:
        # Budget ~512 MiB of fp32 working set per chunk; the division and
        # rounding temporaries are of the same order of magnitude.
        block_rows = max(1, (1 << 29) // (width * 4))

    quantized = torch.empty((rows, width), dtype=torch.int8)
    scale = torch.empty((rows, groups), dtype=torch.float32)
    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        blocked = tensor[start:stop].detach().float().reshape(stop - start, groups, group_size)
        block_scale = (blocked.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-12)
        quantized[start:stop] = (
            (blocked / block_scale).round().clamp_(-127, 127).to(torch.int8).reshape(stop - start, width)
        )
        scale[start:stop] = block_scale.squeeze(-1)
        del blocked, block_scale
    return quantized, scale


def _quantize_ple_mxfp8(tensor: torch.Tensor, block_rows: int = 1 << 19) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 (OCP MX v1.0) quantization for one PLE n-gram table shard.

    Returns ``(e4m3_weight, uint8_scale)``:

    - weight: ``torch.float8_e4m3fn`` shaped ``[rows, width]``;
    - scale: ``torch.uint8`` shaped ``[rows, width // 32]``; each byte is the
      E8M0 shared exponent biased by 127.

    Per 32-element block along the embedding width::

        shared_exp = clamp(floor(log2(amax)) - 8, -127, 127)
        q          = clamp(round_ties_to_even(w / 2**shared_exp), ±448)

    This matches the AscendV1 MXFP8 linear-layer on-disk contract. Rows are
    processed in chunks to bound peak memory.
    """
    if tensor.dim() != 2:
        raise ValueError(f"PLE n-gram table must be 2-D, got shape {tuple(tensor.shape)}")
    rows, width = tensor.shape
    if width % _PLE_MXFP8_BLOCK != 0:
        raise ValueError(f"PLE n-gram table width {width} is not divisible by the MXFP8 block size {_PLE_MXFP8_BLOCK}")
    groups = width // _PLE_MXFP8_BLOCK
    weight = torch.empty((rows, width), dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, groups), dtype=torch.uint8)
    tiny = torch.finfo(torch.float32).tiny

    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        blocked = tensor[start:stop].float().reshape(stop - start, groups, _PLE_MXFP8_BLOCK)
        amax = blocked.abs().amax(dim=-1)
        shared_exp = torch.floor(torch.log2(amax + tiny * (amax == 0)))
        shared_exp = (shared_exp - _E4M3_EMAX).clamp_(-_E8M0_BIAS, _E8M0_BIAS)
        scaled = torch.ldexp(blocked, (-shared_exp.to(torch.int32)).unsqueeze(-1))
        # floor(se)+... guarantees |scaled| < 512; clamp removes the residual
        # [449, 512) saturation range so the fp8 cast never produces NaN.
        scaled.clamp_(-_E4M3_MAX_NORM, _E4M3_MAX_NORM)
        weight[start:stop] = scaled.reshape(stop - start, width).to(torch.float8_e4m3fn)
        scale[start:stop] = (shared_exp + _E8M0_BIAS).round_().clamp_(0, 254).to(torch.uint8)
        del blocked, amax, shared_exp, scaled
    return weight, scale


def _generated_forward_with_blocks(
    model: nn.Module,
    inputs: Any,
    transformer_blocks: Generator[Tuple[str, nn.Module], None, None],
) -> Generator[ProcessRequest, Any, None]:
    """Layer-wise calibration forward with a lazy decoder-layer generator.

    The generic ``transformers_generated_forward_func`` auto-detects layers via
    ``model.named_modules()``, but this adapter only keeps one decoder layer in
    the in-memory skeleton; the rest are materialized on demand from
    safetensors.  This local variant accepts that generator directly so the
    calibration pass can run every layer without materializing all of them at
    once.  MTP has no forward and must not be passed here.
    """
    transformer_blocks = iter(transformer_blocks)
    try:
        first_name, first_block = next(transformer_blocks)
    except StopIteration:
        raise InvalidModelError(
            "No decoder layers discovered.",
            action="Please check the model structure and layer_types.",
        ) from None

    first_block_input = None

    def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
        nonlocal first_block_input
        first_block_input = (hook_args, hook_kwargs)
        raise TransformersForwardBreak()

    hooks = [first_block.register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)]
    try:
        if isinstance(inputs, (list, tuple)):
            model(*inputs)
        elif isinstance(inputs, dict):
            model(**inputs)
        else:
            model(inputs)
    except TransformersForwardBreak:
        pass
    finally:
        for hook in hooks:
            hook.remove()

    if first_block_input is None:
        raise InvalidModelError(
            "Can't get first block input.",
            action="Please check the model and input.",
        )

    current_inputs = first_block_input
    if dist.is_initialized():
        dist.barrier()

    for name, block in itertools.chain(((first_name, first_block),), transformer_blocks):
        args, kwargs = current_inputs  # pylint: disable=unpacking-non-sequence
        outputs = yield ProcessRequest(name, block, args, kwargs)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        current_inputs = ((hidden_states,), current_inputs[1])  # pylint: disable=unsubscriptable-object


def _unpack_fused_experts(
    state: Dict[str, torch.Tensor],
    gate_up_3d: torch.Tensor,
    down_3d: torch.Tensor,
    layer_idx: int,
) -> Dict[str, torch.Tensor]:
    """Expand fused 3D expert tensors into per-expert Linear weights.

    ``gate_up_proj`` has shape ``(E, 2D, H)`` and ``down_proj`` ``(E, H, D)``;
    each expert's ``gate_up`` slice is split into ``gate`` / ``up``.
    """
    num_experts = gate_up_3d.shape[0]
    if down_3d.shape[0] != num_experts:
        raise ValueError(
            f"Layer {layer_idx}: 3D expert count mismatch (gate_up={num_experts}, down={down_3d.shape[0]})."
        )
    if gate_up_3d.shape[1] % 2 != 0:
        raise ValueError(
            f"Layer {layer_idx}: gate_up_proj shape {tuple(gate_up_3d.shape)} not divisible by 2 along axis 1."
        )
    inter = gate_up_3d.shape[1] // 2
    hidden = gate_up_3d.shape[2]
    if down_3d.shape[1] != hidden or down_3d.shape[2] != inter:
        raise ValueError(
            f"Layer {layer_idx}: down_proj shape {tuple(down_3d.shape)} "
            f"inconsistent with gate_up_proj (inter={inter}, hidden={hidden}). "
            f"Expected (E={num_experts}, H={hidden}, D={inter})."
        )
    for i in range(num_experts):
        gate_w, up_w = gate_up_3d[i].chunk(2, dim=0)
        state[f"mlp.experts.{i}.gate_proj.weight"] = gate_w.contiguous()
        state[f"mlp.experts.{i}.up_proj.weight"] = up_w.contiguous()
        state[f"mlp.experts.{i}.down_proj.weight"] = down_3d[i].contiguous()
    return state


def _read_bf16_tensors(
    path: str,
    keys: Optional[List[str]] = None,
    prefix: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Read selected tensors from a safetensors file as CPU bf16 tensors.

    ``keys=None`` reads all keys matching ``prefix`` (all keys when
    ``prefix`` is also ``None``).
    """
    out: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as h:
        for key in keys if keys is not None else h.keys():
            if key not in h.keys():
                continue
            if prefix is not None and not key.startswith(prefix):
                continue
            tensor = h.get_tensor(key)
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)
            out[key] = tensor
    return out


@logger_setter()
class Qwen4ExpFlashNextModelAdapter(  # pylint: disable=too-many-ancestors
    TransformersModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    AscendV1SaveInterface,
    FlexSmoothQuantInterface,
    FA3QuantAdapterInterface,
):
    """Basic msModelSlim adapter for the Qwen3.8-Flash-Next (qwen4_exp) text backbone."""

    # Defaults used when the practice config does not declare ``spec.model_adapter``.
    # PLE tables stay in their source dtype unless the YAML opts into a quant type.
    _ple_quant_type = _PLE_QUANT_BF16
    _ple_quant_group_size = 32
    # Key set of the materialized MTP module, captured at build time. The
    # post-save check cannot rely on ``model.mtp.state_dict()`` because the
    # layer-wise runner offloads every visited module to ``meta`` before
    # ``ascendv1_save_postprocess`` runs.
    _mtp_expected_keys: Optional[Set[str]] = None

    def apply_adapter_config(self, config: Qwen4ExpFlashNextModelConfig) -> None:
        """Apply the validated ``spec.model_adapter`` control parameters."""
        self._ple_quant_type = config.ple_quant_type
        self._ple_quant_group_size = config.ple_quant_group_size

    # ==================== ModelInfoInterface ====================
    def get_model_pedigree(self) -> str:
        return "qwen4_exp_flash_next"

    def get_model_type(self) -> str:
        return self.model_type

    # ==================== FlexSmoothQuantInterface ====================
    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        """Subgraphs consumed by FlexAWQSSZ / FlexSmoothQuant.

        ``up-down``: every routed expert and the shared expert on MoE layers
        (``up_proj -> down_proj``). The processors see the per-expert
        ``nn.Linear`` shells created by the layer loader, not the fused 3D
        checkpoint tensors.

        ``ov``: ``self_attn.v_proj -> self_attn.o_proj`` for full-attention
        (QSA) layers only. The element-wise ``sigmoid(gate)`` applied before
        ``o_proj`` commutes with the per-channel OV scales, so standard OV
        smoothing is exact.

        ``norm-linear`` is intentionally omitted: the linear-attention norm
        stores one shared per-head vector (128 values), while ``out_proj``
        consumes all flattened heads (6144), which the generic one-scale-per-
        input-channel fusion cannot express. Hyper-connection norms are also
        excluded because their output feeds the element-wise bypass.

        MTP has no calibration forward, so it receives no smoothing.
        ``FlexSmoothQuantInterface`` is used because it preserves the gate
        argument of ``Qwen4ExpTextRMSNormGated``, unlike IterSmooth /
        SmoothQuant.
        """
        text_config = _text_config(self.config)
        layer_types = getattr(text_config, "layer_types", None) or []
        num_experts = int(getattr(text_config, "num_experts", 0) or 0)
        # VLM checkpoints write the backbone under ``model.language_model``;
        # a text-only ``Qwen4ExpForCausalLM`` writes it under ``model``.
        if hasattr(self.config, "text_config"):
            prefix = "model.language_model.layers"
        else:
            prefix = "model.layers"

        configs: List[AdapterConfig] = []
        for layer_idx, _layer_type in enumerate(layer_types):
            layer_prefix = f"{prefix}.{layer_idx}"
            mlp_prefix = f"{layer_prefix}.mlp"
            if _layer_type in ("full_attention", "qwen_sparse_attention"):
                attn_prefix = f"{layer_prefix}.self_attn"
                configs.append(
                    AdapterConfig(
                        subgraph_type="ov",
                        mapping=MappingConfig(
                            source=f"{attn_prefix}.v_proj",
                            targets=[f"{attn_prefix}.o_proj"],
                        ),
                    )
                )
            if not self._is_moe_layer(layer_idx):
                continue
            for expert in range(num_experts):
                configs.append(
                    AdapterConfig(
                        subgraph_type="up-down",
                        mapping=MappingConfig(
                            source=f"{mlp_prefix}.experts.{expert}.up_proj",
                            targets=[f"{mlp_prefix}.experts.{expert}.down_proj"],
                        ),
                    )
                )
            configs.append(
                AdapterConfig(
                    subgraph_type="up-down",
                    mapping=MappingConfig(
                        source=f"{mlp_prefix}.shared_expert.up_proj",
                        targets=[f"{mlp_prefix}.shared_expert.down_proj"],
                    ),
                )
            )
        return configs

    def get_num_attention_heads(self) -> int:
        """QSA head count from the text sub-config."""
        return int(_text_config(self.config).num_attention_heads)

    def get_num_key_value_heads(self) -> int:
        text_config = _text_config(self.config)
        return int(getattr(text_config, "num_key_value_heads", text_config.num_attention_heads))

    # ==================== FA3QuantAdapterInterface ====================
    def inject_fa3_placeholders(
        self, root_name: str, root_module: nn.Module, should_inject: Callable[[str], bool]
    ) -> None:
        """为 QSA 注意力层（Qwen4ExpTextAttention）注入 FA3 占位（fa_q/fa_k/fa_v）。

        Qwen4ExpTextDecoderLayer 每层只实例化一种注意力（linear_attention 层挂
        Qwen4ExpTextGatedDeltaNet，其余层挂 Qwen4ExpTextAttention），按类名过滤即可，
        无需 layer_types 判断。

        forward 对齐 config.ini 中 qwen4_exp_flash_next 约束的 transformers==5.19.0
        的 Qwen4ExpTextAttention：indexer 先用完整位置的 cos/sin 选 token，再切片取
        当前位置；q_proj 输出 chunk 出 query/gate；cache update 无 cache_kwargs。
        占位插在 RoPE / cache update 之后、attention kernel 之前，保持 (B, H, S, D)。

        QSA indexer（Qwen4ExpTextQSAIndexer）也有独立的 q/k 打分路径，额外注入
        fa_q/fa_k：q 在 RoPE 后注入；k 在每个 block 的 pooled key 经 RoPE 后、打分
        matmul 前注入。indexer 没有 value 分支，因此不注入 fa_v。
        """
        from importlib import import_module

        def _is_qsa_attention(module: nn.Module) -> bool:
            # 仅 QSA 注意力；GatedDeltaNet / indexer 等其他模块不匹配
            return module.__class__.__name__ == "Qwen4ExpTextAttention"

        def _is_qsa_indexer(module: nn.Module) -> bool:
            # QSA token selector，嵌在 Qwen4ExpTextAttention 内部
            return module.__class__.__name__ == "Qwen4ExpTextQSAIndexer"

        def _wrap_attention_forward(attn_mod: nn.Module):
            attn_module = import_module(attn_mod.forward.__module__)
            apply_rotary_pos_emb = attn_module.apply_rotary_pos_emb
            eager_attention_forward = attn_module.eager_attention_forward
            all_attention_functions = attn_module.ALL_ATTENTION_FUNCTIONS

            def new_forward(
                self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Any] = None,
                **kwargs,
            ):
                # QSA indexer：基于完整位置的 cos/sin 选出参与注意力的 token
                selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
                # The eager indexer represents selected positions as zero in its additive mask.
                if selected_token_mask.is_floating_point():
                    selected_token_mask = selected_token_mask == 0
                # bool 掩码（sdpa）按位与；float 掩码（eager）填 finfo.min，
                # 不能加 finfo.min（溢出 -inf 导致全掩码 query 产生 NaN）
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask & selected_token_mask
                else:
                    attention_mask = attention_mask.masked_fill(
                        ~selected_token_mask, torch.finfo(attention_mask.dtype).min
                    )

                # indexer 使用完整位置的 cos/sin，注意力部分切到最后 hidden_states 个位置
                position_embeddings = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states, gate = torch.chunk(
                    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
                )
                gate = gate.reshape(*input_shape, -1)

                query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    # transformers==5.19.0：Qwen4Exp cache update 无 cache_kwargs
                    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

                # ===== FA3 placeholders (after RoPE / cache, before attention kernel) =====
                if hasattr(self, "fa_q"):
                    query_states = self.fa_q(query_states)
                if hasattr(self, "fa_k"):
                    key_states = self.fa_k(key_states)
                if hasattr(self, "fa_v"):
                    value_states = self.fa_v(value_states)
                # ========================================================================

                attention_interface: Callable = all_attention_functions.get_interface(
                    self.config._attn_implementation, eager_attention_forward
                )
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_output * torch.sigmoid(gate)
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights

            attn_mod.forward = types.MethodType(new_forward, attn_mod)

        def _wrap_indexer_forward(indexer_mod: nn.Module):
            indexer_module = import_module(indexer_mod.forward.__module__)
            apply_rotary_pos_emb = indexer_module.apply_rotary_pos_emb

            def new_forward(
                self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple,
                attention_mask: torch.Tensor,
                past_key_values: Optional[Any],
            ) -> torch.Tensor:
                batch_size, seq_length, _ = hidden_states.shape
                hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
                full_cos, full_sin = position_embeddings
                current_cos = full_cos[:, -seq_length:, :]
                current_sin = full_sin[:, -seq_length:, :]

                qk = self.index_qk_proj(hidden_states)
                q, token_k = torch.split(
                    qk,
                    [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
                    dim=-1,
                )
                q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
                q = self.q_layernorm(q)
                q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

                # ===== FA3 q placeholder (after RoPE, before scoring) =====
                # (B, S, H, D) -> (B, H, S, D)，与主注意力 FA3 per-head 布局一致
                if hasattr(self, "fa_q"):
                    q = self.fa_q(q.transpose(1, 2)).transpose(1, 2)

                if past_key_values is not None:
                    raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

                # mask 永远非 None：indexer 只允许 eager / sdpa，且不允许 sdpa 跳过 mask
                visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0

                selected_token_indices = torch.full(
                    (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
                    -1,
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
                for batch_idx in range(batch_size):
                    for query_idx in range(seq_length):
                        local_visible_indices = torch.nonzero(
                            visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
                        ).flatten()
                        num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
                        if num_complete_blocks > 0:
                            block_token_indices = local_visible_indices[
                                : num_complete_blocks * self.compress_ratio
                            ].view(num_complete_blocks, self.compress_ratio)

                            key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
                            key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
                            pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                            pooled_keys = self.k_layernorm(pooled_keys)
                            group_starts = block_token_indices[:, 0]
                            block_key_states = apply_rotary_pos_emb(
                                pooled_keys.unsqueeze(1),
                                cos=full_cos[batch_idx].index_select(0, group_starts),
                                sin=full_sin[batch_idx].index_select(0, group_starts),
                            ).squeeze(1)

                            # ===== FA3 k placeholder (after RoPE, before scoring matmul) =====
                            # (num_blocks, 1, D) -> (num_blocks, 1, 1, D)，按 (B, H, S, D) 语义量化
                            if hasattr(self, "fa_k"):
                                block_key_states = self.fa_k(block_key_states.unsqueeze(1)).squeeze(1)

                            scores = torch.matmul(
                                q[batch_idx, query_idx].float(),
                                block_key_states.float().transpose(-1, -2),
                            ).transpose(-1, -2)
                            scores = torch.relu(scores).sum(dim=-1) / (self.index_head_dim**0.5)

                            selected_block_indices = scores.topk(
                                min(self.block_topk, num_complete_blocks), dim=0
                            ).indices
                            selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
                        else:
                            selected_tokens = torch.tensor([], device=hidden_states.device)
                        tail = local_visible_indices[num_complete_blocks * self.compress_ratio :]
                        selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
                        selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

                kv_length = attention_mask.shape[-1]
                selected_token_mask = torch.zeros(
                    (*selected_token_indices.shape[:-1], kv_length + 1),
                    device=attention_mask.device,
                    dtype=torch.bool,
                )
                scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
                selected_token_mask = selected_token_mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(
                    1
                )
                if attention_mask.is_floating_point():
                    min_dtype = torch.finfo(attention_mask.dtype).min
                    selected_token_mask = torch.where(selected_token_mask, attention_mask.new_zeros(()), min_dtype)

                return selected_token_mask

            indexer_mod.forward = types.MethodType(new_forward, indexer_mod)

        for name, module in root_module.named_modules():
            full_name = f"{root_name}.{name}" if root_name else name

            if _is_qsa_attention(module):
                if not should_inject(full_name):
                    continue
                prefix = f"{name}." if name else ""
                root_module.set_submodule(f"{prefix}fa_q", FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule(f"{prefix}fa_k", FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule(f"{prefix}fa_v", FA3QuantPlaceHolder(ratio=1.0))
                _wrap_attention_forward(module)
                get_logger().info("Injected FA3 placeholders into %s", full_name)
                continue

            if _is_qsa_indexer(module):
                if not should_inject(full_name):
                    continue
                prefix = f"{name}." if name else ""
                root_module.set_submodule(f"{prefix}fa_q", FA3QuantPlaceHolder(ratio=0.9999))
                root_module.set_submodule(f"{prefix}fa_k", FA3QuantPlaceHolder(ratio=1.0))
                _wrap_indexer_forward(module)
                get_logger().info("Injected FA3 indexer placeholders into %s", full_name)

    # ==================== ModelSlimPipelineInterfaceV1 ====================
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        # Reuse the inherited tokenization helper; calibration is text-only.
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        # Layer-wise IO: load a *skeleton* (num_hidden_layers=1) + vision/audio
        # encoder (VLM) on CPU only. Other decoder layers are lazy-loaded by
        # `_load_decoder_if_not_exist` from the source safetensors when the
        # runner visits them. See `qwen3_vl_moe` for the same pattern.
        return self._load_model(device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        # Stream layers one at a time. We MUST NOT materialize the generator
        # with ``list(...)`` — each materialization allocates the per-expert
        # Linear shells for that layer (1536 ``nn.Linear`` * 1.6M weights for
        # 512 experts), so pre-loading all 48 layers balloons CPU memory to
        # 100s of GB before any quant/save can offload them to meta. Passing
        # the generator directly lets the layer-wise scheduler pull one
        # ``ProcessRequest`` per ``make_progress`` call: the first processor
        # (``LoadProcessor(NPU)``) loads layer ``i`` from safetensors, the
        # next two quantize and save it, and the last offloads it to meta —
        # peak CPU footprint stays at "skeleton + one active layer".
        yield from generated_decoder_layer_visit_func(
            model,
            transformer_blocks=self.generate_decoder_layer(model),
        )

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        # Calibration forward pass. The skeleton only holds layer 0, so the
        # generic auto-detection (`model.named_modules()`) would stop after the
        # first decoder layer. Pass the lazy generator explicitly: the helper
        # hooks layer 0 to capture the real inputs, then pulls each layer from
        # safetensors just-in-time. MTP is included as a passthrough layer so
        # its unquantized weights (including per-expert Linears) are visited and
        # exported as FLOAT; it must stay excluded from the quant processors
        # (the W4A8 recipe excludes ``mtp*``).
        yield from _generated_forward_with_blocks(
            model,
            inputs,
            self.generate_decoder_layer(model),
        )

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        # Mirror `_enable_kv_cache` from `TransformersModel` but tolerant of the
        # Qwen4ExpConfig heterogeneity integration that explodes on `hasattr`.
        # Try the outer VLM config first (`model.model.config`), then the
        # text-only config path used by the verification random model.
        for cfg_obj in (
            getattr(getattr(model, "model", None), "config", None),
            getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
            getattr(getattr(model, "model", None), "text_config", None),
        ):
            try:
                if cfg_obj is None:
                    continue
                # `cfg_obj` may be either a Config object or a use_cache bool here.
                if isinstance(cfg_obj, bool):
                    continue
                setattr(cfg_obj, "use_cache", need_kv_cache)
            except Exception:  # nosec B112
                continue

    # ==================== Helpers ====================
    # ------------------------------------------------------------------
    # Skeleton / per-layer IO
    # ------------------------------------------------------------------
    @lru_cache(maxsize=1)
    def _get_weight_map(self) -> Dict[str, str]:
        """Read ``model.safetensors.index.json`` once and cache it.

        Returns the ``weight_map`` dict that maps every ``state_dict`` key to
        the safetensors file it lives in. Used by ``_get_state_dict`` /
        ``_load_decoder_state_dict_for_layer`` to know which file to
        ``safe_open`` for each parameter.
        """
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        index_data = json_safe_load(index_path)
        return index_data['weight_map']

    def _get_state_dict(self, module: nn.Module, prefix: str = "") -> Dict[str, torch.Tensor]:
        """Load ``module``'s full state_dict from safetensors.

        Groups every ``named_parameter`` of ``module`` by source file (using
        ``_get_weight_map``), opens each file once via ``safe_open``, and
        copies the requested tensors onto CPU in bf16. Mirrors
        ``model/qwen3_vl_moe/model_adapter.py:_get_state_dict`` so the same
        loader works for any module prefix (a decoder layer, a single MoE
        expert, the embed_tokens, etc.).
        """
        weight_map = self._get_weight_map()
        param_names = [name for name, _ in module.named_parameters()]

        file_groups: Dict[str, List[str]] = defaultdict(list)
        for param_name in param_names:
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if full_name in weight_map:
                file_name = weight_map[full_name]
                file_groups[file_name].append(param_name)

        state_dict: Dict[str, torch.Tensor] = {}
        for file_name, names in file_groups.items():
            file_path = os.path.join(self.model_path, file_name)
            file_path = get_valid_read_path(
                file_path,
                extensions='safetensors',
                size_max=MAX_READ_FILE_SIZE_32G,
            )
            with safe_open(file_path, framework='pt', device='cpu') as f:
                for param_name in names:
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    t = f.get_tensor(full_name)
                    if t.dtype != torch.bfloat16:
                        t = t.to(torch.bfloat16)
                    state_dict[param_name] = t.contiguous()
        return state_dict

    def _get_decoder_module_list(self, model: nn.Module) -> nn.ModuleList:
        """Return the ``ModuleList`` of text decoder layers.

        The path depends on whether the loaded model is a multimodal
        ``Qwen4ExpForConditionalGeneration`` (``model.model.language_model.layers``)
        or a text-only ``Qwen4ExpForCausalLM`` (``model.model.layers``).
        """
        if hasattr(getattr(model, "model", None), "language_model"):
            return model.model.language_model.layers  # type: ignore[attr-defined]
        return model.model.layers

    def _get_decoder_prefix(self, model: nn.Module) -> str:
        """Return the short dotted prefix shared by source keys and layer names.

        Processors resolve modules with ``self.model.set_submodule(prefix)``,
        starting from the top-level model. The prefix must therefore be
        ``model.language_model.layers`` (VLM) or ``model.layers`` (text-only);
        a longer ``model.model...`` prefix would fail because the inner
        ``Qwen4ExpModel`` has no ``.model`` attribute.
        """
        return "model.language_model.layers" if _is_vlm_model(model) else "model.layers"

    def _get_decoder_config(self) -> Any:
        """Return the original text config (never the shrunken skeleton copy).

        The loaded model may expose a different config instance whose
        ``num_hidden_layers`` stays frozen at the skeleton value of 1.
        Reading from ``self.config`` keeps the true layer count.
        """
        return _text_config(self.config)

    def _is_moe_layer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` should be MoE-unpacked (default: yes)."""
        text_cfg = _text_config(self.config)
        if not hasattr(text_cfg, "mlp_only_layers"):
            return True
        if layer_idx in text_cfg.mlp_only_layers:
            return False
        sparse_step = getattr(text_cfg, "decoder_sparse_step", 1)
        if sparse_step and sparse_step > 1:
            return (layer_idx + 1) % sparse_step == 0
        return True

    def generate_decoder_layer(self, model: nn.Module) -> Generator[Tuple[str, nn.Module], None, None]:
        """Yield ``(name, layer)`` for every decoder layer, lazy-loading each.

        Same shape as ``qwen3_vl_moe.generate_decoder_layer`` and the sibling
        helpers in ``qwen3_omni_moe``. Called from both
        ``generate_model_visit`` and (indirectly via ``generate_model_forward``)
        the quant scheduler, so each layer is materialized exactly once per
        pipeline pass.
        """
        text_cfg = self._get_decoder_config()
        num_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
        source_prefix = self._get_decoder_prefix(model)
        for idx in range(num_layers):
            layer_name = f"{source_prefix}.{idx}"
            yield layer_name, self._load_decoder_if_not_exist(model, layer_name, idx)

        # MTP draft block last, as a regular quantizable layer. Its linears
        # follow the same include/exclude rules as the backbone (the YAMLs
        # keep the MTP glue floating point, see lab_practice comments).
        if self._has_mtp():
            yield "mtp", self._load_mtp_if_not_loaded(model)

    def _load_decoder_state_dict_for_layer(self, idx: int, source_prefix: str) -> Dict[str, torch.Tensor]:
        """Read one decoder layer from safetensors, without PLE tensors.

        Both checkpoint layouts are accepted:
        - per-expert ``mlp.experts.{i}.{gate,up,down}_proj.weight`` keys,
        - fused 3D ``mlp.experts.gate_up_proj`` / ``down_proj`` tensors,
          which are split into per-expert Linear weights.
        """
        weight_map = self._get_weight_map()
        prefix = f"{source_prefix}.{idx}"
        # Skip ``*.ple.*`` keys: PLE (Positional Layer Embedding /
        # ``ngram_embedding``) lives only on a tiny subset of layers (e.g.
        # layer 1 in Qwen3.8-Flash-Next) but each per-layer ``ngram_embedding``
        # tensor is ~95 GB of bf16. Loading it here would blow the per-layer
        # CPU budget and the ``LoadProcessor``'s NPU ``.to()`` would OOM. We
        # re-introduce the tensors after the layer-wise save pass via
        # ``ascendv1_save_postprocess -> _patch_missing_ple_weights``, which
        # streams them from the source safetensors straight into output shards.
        layer_keys = [k for k in weight_map.keys() if k.startswith(prefix + ".") and ".ple." not in k]
        if not layer_keys:
            return {}

        # Group by source safetensors file for one open per shard.
        file_groups: Dict[str, List[str]] = defaultdict(list)
        for k in layer_keys:
            file_groups[weight_map[k]].append(k)

        # Pass 1: load raw tensors and detect 3D MoE.
        raw: Dict[str, torch.Tensor] = {}
        has_3d_moe = False
        for fname, keys in file_groups.items():
            fpath = get_valid_read_path(
                os.path.join(self.model_path, fname),
                extensions='safetensors',
                size_max=MAX_READ_FILE_SIZE_32G,
            )
            with safe_open(fpath, framework='pt', device='cpu') as f:
                for k in keys:
                    if k in f.keys():
                        t = f.get_tensor(k)
                        if t.dtype != torch.bfloat16:
                            t = t.to(torch.bfloat16)
                        raw[k] = t.contiguous()
                        if k.endswith("mlp.experts.gate_up_proj") or k.endswith("mlp.experts.down_proj"):
                            has_3d_moe = True

        if not has_3d_moe:
            # Source is already in per-expert Linear layout; just strip the
            # layer prefix so ``decoder.load_state_dict`` sees relative names.
            return {k[len(prefix) + 1 :]: v for k, v in raw.items()}

        # Pass 2: unpack 3D tensors into per-expert Linear weights.
        state: Dict[str, torch.Tensor] = {}
        gate_up_3d = None
        down_3d = None
        for k, v in raw.items():
            short = k[len(prefix) + 1 :]
            if short == "mlp.experts.gate_up_proj":
                gate_up_3d = v  # (E, 2D, H)
            elif short == "mlp.experts.down_proj":
                down_3d = v  # (E, H, D)
            else:
                state[short] = v

        if gate_up_3d is not None and down_3d is not None:
            return _unpack_fused_experts(state, gate_up_3d, down_3d, idx)

        # Mixed storage: keep the fused keys so load_state_dict reports the
        # mismatch instead of silently dropping the experts.
        if gate_up_3d is not None:
            state["mlp.experts.gate_up_proj"] = gate_up_3d
        if down_3d is not None:
            state["mlp.experts.down_proj"] = down_3d
        return state

    def _convert_single_moe_layer(
        self,
        layer: nn.Module,
        layer_idx: int,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Replace 3D-packed MoE experts with per-expert Linear modules.

        The unstacked module takes over the weights in place, then releases
        the original 3D parameters to keep peak memory flat.
        """
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is None:
            return
        if experts.__class__.__name__ != "Qwen4ExpTextExperts":
            return

        text_cfg = self._get_decoder_config()
        gate_up_3d = getattr(experts, "gate_up_proj", None)
        down_3d = getattr(experts, "down_proj", None)
        if not isinstance(gate_up_3d, nn.Parameter) or not isinstance(down_3d, nn.Parameter):
            # Already converted (or not a fused-expert module).
            return

        num_experts = int(getattr(experts, "num_experts", 0)) or int(getattr(text_cfg, "num_experts", 0))
        hidden_dim = int(getattr(experts, "hidden_dim", 0)) or int(getattr(text_cfg, "hidden_size", 0))
        intermediate_dim = int(getattr(experts, "intermediate_dim", 0)) or int(
            getattr(text_cfg, "moe_intermediate_size", 0)
        )
        if not (num_experts and hidden_dim and intermediate_dim):
            raise InvalidModelError(
                f"Layer {layer_idx}: cannot resolve MoE dims for 3D->per-expert unpack "
                f"(num_experts={num_experts}, hidden={hidden_dim}, inter={intermediate_dim})",
                action="Check the qwen4_exp experts module and text config attributes.",
            )

        # Take a config-shaped object for the unstacked module. Either the
        # real text config (preferred) or our dummy one as a fallback.
        cfg = _DummyQwen4ExpTextConfig(
            hidden_size=hidden_dim,
            moe_intermediate_size=intermediate_dim,
            num_experts=num_experts,
            hidden_act=getattr(text_cfg, "hidden_act", "silu") if text_cfg is not None else "silu",
        )
        unstacked = _UnstackedQwen4ExpTextExperts(cfg, dtype=dtype)
        unstacked._transform_weights_from_original(experts, in_place=True)
        unstacked.eval()

        layer.mlp.experts = unstacked
        # Drop the Python reference to the original experts submodule so its
        # nn.Parameters are released even if any outer variable holds them.
        del experts, gate_up_3d, down_3d
        gc.collect()

    def _text_config_without_ple(self, layer_idx: int) -> Any:
        """Return a config copy that does not build a PLE module for this layer.

        PLE tables are ~95 GB of BF16 per layer, so the decoder constructor
        must not materialize them. The checkpoint keys are filtered in
        ``_load_decoder_state_dict_for_layer`` and restored during export.
        """
        text_cfg = self._get_decoder_config()
        if not getattr(text_cfg, "ple_layer_ids", None):
            return text_cfg
        try:
            text_cfg = text_cfg.copy()
        except Exception:
            import copy as _copy

            try:
                text_cfg = _copy.copy(text_cfg)
            except Exception:  # nosec B110
                pass
        try:
            text_cfg.ple_layer_ids = []
        except Exception:
            get_logger().warning(
                "[Qwen4ExpFlashNext] Layer %d: cannot strip ple_layer_ids; PLE module may still be built and OOM.",
                layer_idx,
            )
        return text_cfg

    def _source_dtype(self) -> torch.dtype:
        """Return the dtype declared by the source text config.

        The loader materializes every source tensor in BF16; falling back to
        the global ``float32`` default would silently upcast them.
        """
        text_cfg = self._get_decoder_config()
        for attr in ("dtype", "torch_dtype"):
            value = getattr(text_cfg, attr, None)
            if isinstance(value, torch.dtype):
                return value
            if isinstance(value, str) and value != "auto":
                try:
                    return getattr(torch, value)
                except AttributeError:
                    continue
        return torch.bfloat16

    def _replace_with_unstacked_experts(
        self,
        decoder: nn.Module,
        text_cfg: Any,
        layer_idx: int,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Install the per-expert Linear shell before loading the checkpoint.

        The per-expert keys produced by the loader do not match the fused 3D
        ``Qwen4ExpTextExperts`` tree; swapping it out first makes
        ``load_state_dict`` populate the experts instead of reporting them as
        unexpected.
        """
        if not self._is_moe_layer(layer_idx):
            return
        experts = getattr(getattr(decoder, "mlp", None), "experts", None)
        if experts is None or experts.__class__.__name__ != "Qwen4ExpTextExperts":
            return
        cfg = _DummyQwen4ExpTextConfig(
            hidden_size=int(getattr(text_cfg, "hidden_size", 0)),
            moe_intermediate_size=int(getattr(text_cfg, "moe_intermediate_size", 0)),
            num_experts=int(getattr(text_cfg, "num_experts", 0)),
            hidden_act=getattr(text_cfg, "hidden_act", "silu"),
        )
        decoder.mlp.experts = _UnstackedQwen4ExpTextExperts(cfg, dtype=dtype)
        del experts
        gc.collect()

    def _load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int) -> nn.Module:
        """Materialize one decoder layer from safetensors, once per run.

        Freshly built layers are tagged ``_qwen4_lazy_loaded`` so the visit and
        forward passes can reuse them. The skeleton's random layer 0 is not
        tagged, so the first call replaces it with the checkpoint version.
        """
        module_list = self._get_decoder_module_list(model)

        if idx < len(module_list):
            existing = module_list[idx]
            if getattr(existing, "_qwen4_lazy_loaded", False):
                return existing

        get_logger().info(
            "[Qwen4ExpFlashNext] Lazy-loading decoder layer %d (%s).",
            idx,
            name,
        )

        try:
            from transformers.models.qwen4_exp.modeling_qwen4_exp import (
                Qwen4ExpTextDecoderLayer,
            )
        except Exception as e:  # pragma: no cover - best-effort
            raise RuntimeError(
                "Qwen4ExpTextDecoderLayer is not importable from transformers; cannot perform layer-wise loading."
            ) from e

        # Build without allocating random weights; the checkpoint fills them.
        # Construct under the source dtype so BF16 weights are not upcast.
        text_cfg = self._text_config_without_ple(idx)
        dtype = self._source_dtype()
        with _temporary_default_dtype(dtype):
            with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
                decoder = Qwen4ExpTextDecoderLayer(text_cfg, layer_idx=idx)

            # The loader emits per-expert keys, so replace the fused 3D experts
            # before loading.
            self._replace_with_unstacked_experts(decoder, text_cfg, idx, dtype=dtype)

        source_prefix = self._get_decoder_prefix(model)
        state = self._load_decoder_state_dict_for_layer(idx, source_prefix)
        missing, unexpected = decoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise InvalidModelError(
                f"Layer {idx} state_dict mismatch: missing={missing[:8]} unexpected={unexpected[:8]}",
                action="Check the transformers version and the checkpoint MoE layout (3D fused vs per-expert).",
            )
        decoder.eval()

        # Reuse this instance on the next visit / forward pass.
        decoder._qwen4_lazy_loaded = True  # type: ignore[attr-defined]

        if idx < len(module_list):
            module_list[idx] = decoder
        else:
            module_list.append(decoder)

        return decoder

    def _from_pretrained_skeleton(self, model_cls: Any, config: Any) -> nn.Module:
        """Load the one-layer skeleton with the common ``from_pretrained`` options."""
        return model_cls.from_pretrained(  # nosec B615
            str(self.model_path),
            config=config,
            device_map="cpu",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=self.trust_remote_code,
        )

    def _load_model(self, device: DeviceType) -> nn.Module:
        """Load a one-decoder-layer skeleton on CPU.

        The full VLM uses ``AutoModelForImageTextToText``; text-only checkpoints
        use ``AutoModelForCausalLM``. ``num_hidden_layers`` is temporarily set
        to 1 (and restored in ``finally``); the remaining layers are materialized
        by ``_load_decoder_if_not_exist``. MTP is rebuilt lazily as well.
        """
        from msmodelslim.utils.security.model import SafeGenerator
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

        # Validate the model path through the security helper (sanity + perms).
        SafeGenerator.get_config_from_pretrained(
            model_path=str(self.model_path),
            trust_remote_code=self.trust_remote_code,
        )

        config_cls_name = type(self.config).__name__
        is_outer_vlm_config = config_cls_name == "Qwen4ExpConfig"

        # Determine which sub-config to shrink and the model class to load.
        if is_outer_vlm_config:
            text_cfg = self.config.text_config
            model_cls = AutoModelForImageTextToText
        else:
            text_cfg = self.config
            model_cls = AutoModelForCausalLM

        origin_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
        if origin_layers > 1:
            text_cfg.num_hidden_layers = 1
        try:
            if is_outer_vlm_config:
                model = self._from_pretrained_skeleton(model_cls, self.config)
            else:
                # Re-read the on-disk config so AutoModelForCausalLM can
                # resolve the text-only architecture tag.
                try:
                    disk_config = AutoConfig.from_pretrained(  # nosec B615
                        str(self.model_path),
                        local_files_only=True,
                        trust_remote_code=self.trust_remote_code,
                    )
                except Exception:
                    disk_config = None
                if disk_config is not None:
                    disk_config.num_hidden_layers = 1
                model = self._from_pretrained_skeleton(
                    model_cls, disk_config if disk_config is not None else self.config
                )
        finally:
            # Always restore the original layer count so downstream code that
            # reads ``text_cfg.num_hidden_layers`` (subgraph configs, the
            # lazy-load scheduler) sees the right number.
            text_cfg.num_hidden_layers = origin_layers

        # ``BaseSmoothProcessor._apply_standard_ov_smooth`` reads the head
        # counts from ``model.config``; the outer Qwen4ExpConfig is a composite
        # config with no such attributes.  Mirror the qwen3.5 adapter and copy
        # them from ``text_config`` after loading so standard OV smoothing can
        # resolve the GQA split without touching the processor.
        model_text_cfg = getattr(getattr(model, "config", None), "text_config", None)
        if model_text_cfg is not None:
            for attr in ("num_attention_heads", "num_key_value_heads"):
                if hasattr(model_text_cfg, attr):
                    setattr(model.config, attr, getattr(model_text_cfg, attr))
            get_logger().info(
                "[Qwen4ExpFlashNext] model.config heads: num_attention_heads=%s, num_key_value_heads=%s",
                getattr(model.config, "num_attention_heads", None),
                getattr(model.config, "num_key_value_heads", None),
            )

        # MTP is rebuilt lazily and layer 0 is replaced on its first visit, so
        # the skeleton phase never holds the real 3D expert weights.
        return model

    # ------------------------------------------------------------------
    # MTP (Multi-Token Prediction) quantization
    # ------------------------------------------------------------------
    def _has_mtp(self) -> bool:
        """Whether the source checkpoint carries ``mtp.*`` tensors."""
        return any(k.startswith("mtp.") for k in self._get_weight_map())

    def _load_mtp_if_not_loaded(self, model: nn.Module) -> nn.Module:
        """Build (once) and return the real MTP module mounted at ``model.mtp``.

        A real module tree is required so the layer-wise quant pipeline can
        visit it: the previous ``_MtpPassthrough`` container only carried a
        flat state_dict and was intentionally invisible to the processors,
        which forced the saver to emit every ``mtp.*`` tensor as ``FLOAT``.
        """
        existing = getattr(model, "mtp", None)
        if existing is not None:
            return existing
        module = self._build_mtp_module(model)
        if module is None:
            raise RuntimeError("The checkpoint advertises mtp.* tensors but the MTP module could not be built.")
        try:
            model.set_submodule("mtp", module)
        except Exception:
            setattr(model, "mtp", module)
        get_logger().info(
            "[Qwen4ExpFlashNext] MTP module built: %d layer(s), quantizable "
            "attention projections + routed experts + shared expert MLP.",
            len(module.layers),
        )
        return module

    def _build_mtp_module(self, model: nn.Module) -> Optional[nn.Module]:
        """Build the MTP draft block from the source ``mtp.*`` tensors.

        Layout: ``fc_embedding`` / ``fc_hidden`` / ``pre_fc_norm_*`` /
        ``hyper_connection_mixer`` plus one full-attention decoder layer. The
        decoder loads fused 3D experts first, then converts them to per-expert
        Linears for quantization, exactly like the backbone.
        """
        mtp_tensors = self._load_mtp_tensors_from_checkpoint(str(self.model_path))
        if not mtp_tensors:
            return None

        text_cfg = self._get_decoder_config()
        num_mtp_layers = int(getattr(text_cfg, "mtp_num_hidden_layers", 1) or 1)

        # Build a text-config copy whose layer_types has one full-attention
        # entry per MTP layer and whose PLE list is empty (the draft has none).
        import copy as _copy

        mtp_cfg = _copy.deepcopy(text_cfg)
        mtp_cfg.num_hidden_layers = num_mtp_layers
        mtp_cfg.layer_types = ["full_attention"] * num_mtp_layers
        mtp_cfg.ple_layer_ids = []

        module = None
        dtype = self._source_dtype()
        with _temporary_default_dtype(dtype):
            module = _Qwen4ExpMtpContainer(mtp_cfg, num_mtp_layers)

        state = {k[len("mtp.") :]: v for k, v in mtp_tensors.items()}
        result = module.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "MTP weight layout does not match the rebuilt module: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        # Record the export key set now: the layer-wise runner offloads the
        # module to ``meta`` after the visit pass, so ``state_dict()`` is no
        # longer usable in ``ascendv1_save_postprocess``.
        del state, mtp_tensors
        gc.collect()

        # 3D fused experts -> per-expert nn.Linear (quant pipeline input).
        for idx, layer in enumerate(module.layers):
            self._convert_single_moe_layer(layer, idx, dtype=dtype)

        # Capture the keys *after* the fused-expert unpack: the saver writes
        # per-expert Linear names, not the checkpoint's 3D ``gate_up_proj`` /
        # ``down_proj`` parameters.
        self._mtp_expected_keys = {f"mtp.{k}" for k in module.state_dict()}
        module.eval()
        return module

    @staticmethod
    def _load_mtp_tensors_from_checkpoint(model_path: str) -> Dict[str, torch.Tensor]:
        """Read every ``mtp.*`` tensor from sharded or single-file safetensors."""
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            index = json_safe_load(index_path)
            weight_map = (index or {}).get("weight_map", {}) or {}
            mtp_keys = [k for k in weight_map.keys() if k.startswith("mtp.")]
            if not mtp_keys:
                return {}
            out: Dict[str, torch.Tensor] = {}
            for fname in sorted({weight_map[k] for k in mtp_keys}):
                fpath = os.path.join(model_path, fname)
                if not os.path.exists(fpath):
                    continue
                fpath = get_valid_read_path(fpath, extensions="safetensors", size_max=MAX_READ_FILE_SIZE_32G)
                out.update(_read_bf16_tensors(fpath, [k for k in mtp_keys if weight_map.get(k) == fname]))
            return out

        single = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single):
            return {}
        single = get_valid_read_path(single, extensions="safetensors", size_max=MAX_READ_FILE_SIZE_32G)
        return _read_bf16_tensors(single, prefix="mtp.")

    def _load_tokenizer(self, trust_remote_code: bool = False):
        """Wrap the inherited tokenizer so pad_token defaults to eos_token.

        Qwen3.8-Flash-Next ships a tokenizer where `pad_token` is None. The
        inherited `_get_tokenized_data` calls `tokenizer(..., padding=True)`,
        which would raise `ValueError: Asking to pad but the tokenizer does
        not have a padding token.` during Step2 (full-fallback quant).
        Falling back to `eos_token` is the standard Qwen recommendation.
        """
        tokenizer = super()._load_tokenizer(trust_remote_code=trust_remote_code)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # ------------------------------------------------------------------
    # AscendV1SaveInterface postprocess hook
    # ------------------------------------------------------------------
    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """Ensure PLE and MTP tensors are present in the AscendV1 export."""
        self._patch_missing_ple_weights(save_directory)

        mtp_module = getattr(model, "mtp", None)
        if mtp_module is not None:
            self._verify_mtp_saved(mtp_module, save_directory)

    def _verify_mtp_saved(self, mtp_module: nn.Module, save_directory: str) -> None:
        """Fail if the saver missed any materialized ``mtp.*`` tensor.

        MTP is yielded as a regular layer, so the saver owns its shards.
        Rewriting a shard in place would break the part-file-size contract and
        index metadata, so missing keys are treated as an export error.
        """
        expected = self._mtp_expected_keys
        if not expected:
            # Fallback for callers that did not go through ``_build_mtp_module``
            # (e.g. a full-model pipeline where the module still holds weights).
            expected = {
                k if k.startswith("mtp.") else f"mtp.{k}" for k, v in mtp_module.state_dict().items() if not v.is_meta
            }
        if not expected:
            return

        written = _collect_current_export_keys(save_directory)
        if written is None:
            written = _collect_quant_safetensors_keys(save_directory)
        missing = sorted(expected - written)
        if missing:
            raise InvalidModelError(
                f"MTP export is missing {len(missing)} tensors, e.g. {missing[:8]}",
                action="Check that the MTP module was visited and saved as a regular layer.",
            )

        desc_path = os.path.join(save_directory, "quant_model_description.json")
        desc_data: Dict[str, Any] = _load_json(desc_path) or {}
        missing_desc = [k for k in expected if k not in desc_data]
        if missing_desc:
            for key in missing_desc:
                desc_data[key] = "FLOAT"
            _dump_json(desc_path, desc_data, 4)

    # ------------------------------------------------------------------
    # PLE (Positional Layer Embedding / ngram_embedding) passthrough
    # ------------------------------------------------------------------
    def _patch_missing_ple_weights(self, save_directory: str) -> None:
        """Stream source ``*.ple.*`` weights into the exported bundle.

        PLE is excluded from calibration and from the saver (the n-gram tables
        are ~95 GB of BF16), so it is restored here after the layer-wise save.

        ``ple_quant_type`` controls the huge ``ngram_embedding`` tables:
        - ``bf16``: keep the source dtype;
        - ``int8_sym``: INT8 weight + float32 scale per ``ple_quant_group_size``
          columns, described as ``INT8_SYM``;
        - ``mxfp8``: E4M3 weight + uint8 E8M0 scale per 32 columns, described
          as ``W8A8_MXFP8``.

        Small PLE projections/norms keep BF16; int64 hash-layout buffers keep
        int64. New shards follow the standard
        ``quant_model_weights-NNNNN-of-NNNNN.safetensors`` naming; existing
        shards are renumbered and the index/description JSON are updated.
        One source shard is held in memory at a time.
        """
        # ---- 1. Discover PLE keys in the source checkpoint. ----
        src_index_path = os.path.join(str(self.model_path), "model.safetensors.index.json")
        if not os.path.exists(src_index_path):
            return
        src_index = json_safe_load(src_index_path)
        src_weight_map = src_index.get("weight_map", {}) or {}

        ple_keys = sorted(k for k in src_weight_map.keys() if ".ple." in k)
        if not ple_keys:
            return

        quant_mode = self._ple_quant_type
        quant_tables = quant_mode != _PLE_QUANT_BF16
        group_size = int(self._ple_quant_group_size)
        if quant_mode == _PLE_QUANT_INT8 and group_size <= 0:
            raise ValueError(f"ple_quant_group_size must be positive, got {group_size}")

        # Every source PLE key maps to one output key, or two when it is a
        # quantized n-gram table (weight + scale).
        expected_outputs: Dict[str, List[str]] = {}
        for k in ple_keys:
            if quant_tables and _is_ple_table_key(k):
                expected_outputs[k] = [k, _ple_scale_key(k, quant_mode)]
            else:
                expected_outputs[k] = [k]

        # Group by source safetensors file so we open each source shard once.
        src_files: Dict[str, List[str]] = defaultdict(list)
        for k in ple_keys:
            src_files[src_weight_map[k]].append(k)

        # ---- 2. Enumerate the current run's quant shards. ----
        # Prefer the index that the saver just rewrote: it lists exactly the
        # shards produced by this run. A directory scan is only used for the
        # single-file layout (no index). Otherwise shards left behind by an
        # earlier run in the same directory would inflate the shard count and
        # make PLE keys look already exported.
        on_disk_shards = _enumerate_quant_shard_files(save_directory)
        scratch_shards = [name for _, name in on_disk_shards if name.endswith("-of-00000.safetensors")]
        if scratch_shards:
            raise InvalidModelError(
                f"Found {len(scratch_shards)} stale saver scratch shards, e.g. {scratch_shards[:3]}",
                action="Remove the of-00000 scratch shards and rerun.",
            )
        existing_shards = _enumerate_index_shard_files(save_directory)
        if existing_shards is None:
            existing_shards = on_disk_shards
        else:
            current_names = {name for _, name in existing_shards}
            stale = [name for _, name in on_disk_shards if name not in current_names]
            if stale:
                logger.warning(
                    "[Qwen4ExpFlashNext] PLE export: ignoring %d shard(s) not referenced "
                    "by the current index, e.g. %s.",
                    len(stale),
                    stale[:3],
                )
        if not existing_shards and os.path.exists(os.path.join(save_directory, "quant_model_weights.safetensors")):
            logger.warning(
                "[Qwen4ExpFlashNext] PLE export: export uses single-file "
                "layout. PLE shards will still use the sharded naming, which "
                "is inconsistent with the main weights. Set ``part_file_size > 0`` "
                "in the YAML to keep the naming uniform."
            )

        # ---- 3. Stream missing PLE keys into new shards. ----
        # We use ``BufferedSafetensorsWriter`` directly so we inherit its
        # 4 GB shard cap, safetensors metadata, and shared-storage dedup
        # logic. We deliberately do NOT call ``writer.close()`` or
        # ``writer.save_index()`` — those would overwrite the existing
        # index JSON and rename shards with a wrong total. Instead we
        # capture ``writer.saved_keys_map`` and finish the rename + JSON
        # merge ourselves.
        #
        # The writer's intermediate naming (``{prefix}-NNNNN-of-00000``) would
        # clash with any stale AscendV1 saver scratch shards left over from
        # a prior incomplete run, so we use a dedicated prefix for PLE and
        # rename to the final ``quant_model_weights-NNNNN-of-NNNNN`` form
        # below.
        written_keys = _collect_current_export_keys(save_directory)
        if written_keys is None:
            written_keys = _collect_quant_safetensors_keys(save_directory)
        missing_set = {k for k, outputs in expected_outputs.items() if any(out not in written_keys for out in outputs)}

        # Validate every required source shard before creating any PLE output.
        for src_file, src_keys in src_files.items():
            src_missing = [k for k in src_keys if k in missing_set]
            src_path = os.path.join(str(self.model_path), src_file)
            if src_missing and not os.path.exists(src_path):
                raise InvalidModelError(
                    f"PLE export: source shard {src_file} missing on disk; cannot export PLE keys {src_missing[:3]}",
                    action="Restore the checkpoint shard and rerun.",
                )

        desc_path = os.path.join(save_directory, "quant_model_description.json")
        desc_data: Dict[str, Any] = _load_json(desc_path) or {}
        desc_changed = False

        def _set_desc(key: str, value: Any) -> None:
            nonlocal desc_changed
            if desc_data.get(key) != value:
                desc_data[key] = value
                desc_changed = True

        def _drop_desc(key: str) -> None:
            nonlocal desc_changed
            if key in desc_data:
                del desc_data[key]
                desc_changed = True

        for k in ple_keys:
            if quant_tables and _is_ple_table_key(k):
                # The bytes are quantized; the serving loader reads their dtype
                # from the tensor headers. INT8 follows the Enram-style
                # per-table ``quant_type`` marker; MXFP8 mirrors the linear
                # layers and marks both tensors as ``W8A8_MXFP8``.
                scale_key = _ple_scale_key(k, quant_mode)
                if quant_mode == _PLE_QUANT_MXFP8:
                    _drop_desc(k.removesuffix(".weight") + ".quant_type")
                    _set_desc(k, "W8A8_MXFP8")
                    _set_desc(scale_key, "W8A8_MXFP8")
                else:
                    _drop_desc(k)
                    _drop_desc(scale_key)
                    _set_desc(k.removesuffix(".weight") + ".quant_type", "INT8_SYM")
            else:
                _set_desc(k, "FLOAT")

        stale_ple_tmp = sorted(glob.glob(os.path.join(save_directory, "quant_model_weights_ple_tmp-*.safetensors")))
        for stale_path in stale_ple_tmp:
            try:
                os.remove(stale_path)
            except OSError as exc:
                raise InvalidModelError(
                    f"Failed to remove stale PLE temp shard {stale_path}: {exc}",
                    action="Clean the output directory and rerun.",
                )
        if stale_ple_tmp:
            logger.warning(
                "[Qwen4ExpFlashNext] PLE export: removed %d stale PLE temp shards.",
                len(stale_ple_tmp),
            )

        ple_tmp_filenames: Dict[str, str] = {}  # output_key -> tmp shard filename
        # Bytes this pass actually writes, in the same unit the AscendV1 saver
        # uses for ``metadata.total_size`` (tensor bytes, not file bytes). The
        # PLE shards are created after the saver finished, so this value has to
        # be folded into the index metadata; otherwise the declared model size
        # silently omits the (large) PLE tables.
        ple_written_bytes = 0
        if missing_set or desc_changed:
            writer = BufferedSafetensorsWriter(
                logger,
                max_gb_size=4,
                save_directory=save_directory,
                save_prefix="quant_model_weights_ple_tmp",
            )
            for src_file, src_keys in src_files.items():
                src_missing = [k for k in src_keys if k in missing_set]
                if not src_missing:
                    continue
                src_path = os.path.join(str(self.model_path), src_file)
                if not os.path.exists(src_path):
                    raise InvalidModelError(
                        f"PLE export: source shard {src_file} missing on disk; "
                        f"cannot export PLE keys {src_missing[:3]}",
                        action="Restore the checkpoint shard and rerun.",
                    )
                src_path = get_valid_read_path(src_path, extensions="safetensors", size_max=MAX_READ_FILE_SIZE_32G)
                with safe_open(src_path, framework="pt", device="cpu") as h:
                    available = set(h.keys())
                    for k in src_missing:
                        if k not in available:
                            logger.warning(
                                "[Qwen4ExpFlashNext] PLE export: key %s missing in %s.",
                                k,
                                src_file,
                            )
                            continue
                        tensor = h.get_tensor(k)
                        if quant_tables and _is_ple_table_key(k):
                            if quant_mode == _PLE_QUANT_MXFP8:
                                quantized, scale = _quantize_ple_mxfp8(tensor)
                            else:
                                quantized, scale = _quantize_ple_int8(tensor, group_size)
                            writer.write(k, quantized)
                            writer.write(_ple_scale_key(k, quant_mode), scale)
                            del quantized, scale
                        else:
                            # Keep the source dtype for every non-quantized
                            # tensor. The float PLE params are BF16, but the
                            # n-gram layout buffers are int64; casting them to
                            # BF16 destroys the exact hash offsets/sizes.
                            writer.write(k, tensor.contiguous())
                        del tensor
            # Flush any final buffered tensors to disk.
            writer.save_one_file()
            ple_tmp_filenames = dict(writer.saved_keys_map)
            # ``total_size`` accumulates ``numel * element_size`` for every
            # tensor handed to ``write``, i.e. exactly the PLE weight + scale
            # bytes now added to the export.
            ple_written_bytes = writer.total_size

        # Description must not declare PLE tensors that were not written.
        available_outputs = set(written_keys) | set(ple_tmp_filenames)
        for source_key, outputs in expected_outputs.items():
            if all(out in available_outputs for out in outputs):
                continue
            for out in outputs:
                _drop_desc(out)
            if quant_tables and _is_ple_table_key(source_key):
                _drop_desc(source_key.removesuffix(".weight") + ".quant_type")

        # ---- 4. Compute new shard total and rename everything. ----
        existing_count = len(existing_shards)
        next_idx = max((i for i, _ in existing_shards), default=0) + 1
        new_ple_count = len(set(ple_tmp_filenames.values()))
        new_total = existing_count + new_ple_count
        total_str = f"{new_total:05d}"

        # 4a. Existing quant shards: ensure ``of-{total_str}`` matches the new total.
        rename_existing: Dict[str, str] = {}
        for idx, old_name in existing_shards:
            if old_name.endswith(f"-of-{total_str}.safetensors"):
                continue
            new_name = f"quant_model_weights-{idx:05d}-of-{total_str}.safetensors"
            old_path = os.path.join(save_directory, old_name)
            new_path = os.path.join(save_directory, new_name)
            if os.path.exists(new_path):
                raise InvalidModelError(
                    f"PLE export: target {new_name} already exists; cannot rename {old_name}",
                    action="Remove stale shards and rerun.",
                )
            try:
                os.rename(old_path, new_path)
                rename_existing[old_name] = new_name
            except OSError as exc:
                raise InvalidModelError(
                    f"PLE export: failed to rename {old_name} -> {new_name}: {exc}",
                    action="Check the output directory and rerun.",
                )

        # 4b. PLE tmp shards: renumber after the existing shards and apply the new total.
        # The writer's tmp names look like
        # ``quant_model_weights_ple_tmp-NNNNN-of-00000.safetensors``; we
        # renumber to ``quant_model_weights-NNNNN-of-{total_str}.safetensors``.
        rename_ple: Dict[str, str] = {}  # tmp_name -> final_name
        for tmp_name in sorted(set(ple_tmp_filenames.values())):
            # Extract the writer-side 1-based index from the tmp filename.
            m = _PLE_TMP_NAME_RE.match(tmp_name)
            if not m:
                continue
            writer_idx = int(m.group(1))
            new_idx = next_idx + writer_idx - 1
            new_name = f"quant_model_weights-{new_idx:05d}-of-{total_str}.safetensors"
            tmp_path = os.path.join(save_directory, tmp_name)
            new_path = os.path.join(save_directory, new_name)
            if not os.path.exists(tmp_path):
                if not os.path.exists(new_path):
                    raise InvalidModelError(
                        f"PLE export: both temp and final shard are missing: {tmp_name}",
                        action="Remove stale shards and rerun.",
                    )
                expected_keys = [key for key, shard in ple_tmp_filenames.items() if shard == tmp_name]
                with safe_open(new_path, framework="pt", device="cpu") as h:
                    if not expected_keys or not all(k in h.keys() for k in expected_keys):
                        raise InvalidModelError(
                            f"PLE export: {new_name} exists but does not contain the "
                            f"expected PLE tensors for {tmp_name}",
                            action="Remove stale shards and rerun.",
                        )
                rename_ple[tmp_name] = new_name
                continue
            if os.path.exists(new_path):
                raise InvalidModelError(
                    f"PLE export: target {new_name} already exists; cannot rename {tmp_name}",
                    action="Remove stale shards and rerun.",
                )
            try:
                os.rename(tmp_path, new_path)
                rename_ple[tmp_name] = new_name
            except OSError as exc:
                raise InvalidModelError(
                    f"PLE export: failed to rename {tmp_name} -> {new_name}: {exc}",
                    action="Check the output directory and rerun.",
                )

        # ---- 5. Merge PLE entries into the index JSON. ----
        index_path = os.path.join(save_directory, "quant_model_weights.safetensors.index.json")
        index_data: Dict[str, Any] = _load_json(index_path) or {}
        metadata = index_data.setdefault("metadata", {})
        index_data.setdefault("weight_map", {})
        wm = index_data["weight_map"]

        # 5c. Keep ``metadata.total_size`` in sync with the tensors actually in
        # the bundle. The saver sized the file before the PLE shards existed, so
        # add the bytes written here (0 on a re-run that had nothing to add).
        if ple_written_bytes:
            metadata["total_size"] = int(metadata.get("total_size", 0)) + ple_written_bytes

        # 5a. Update existing keys whose shard was renamed.
        for k, fname in list(wm.items()):
            if fname in rename_existing:
                wm[k] = rename_existing[fname]

        # 5b. Insert/refresh every PLE output key (weight and scale) with its
        # final filename.
        for ple_key, tmp_name in ple_tmp_filenames.items():
            wm[ple_key] = rename_ple.get(tmp_name, tmp_name)

        _dump_json(index_path, index_data, 2)

        # ---- 6. Persist description JSON. ----
        if desc_changed or not os.path.exists(desc_path):
            _dump_json(desc_path, desc_data, 4)

        logger.info(
            "[Qwen4ExpFlashNext] PLE export: %d source keys -> %d output tensors "
            "in %d new shards (quant=%s, existing=%d, new_total=%d).",
            len(missing_set),
            len(ple_tmp_filenames),
            new_ple_count,
            quant_mode,
            existing_count,
            new_total,
        )


class _Qwen4ExpMtpContainer(nn.Module):
    """Real module tree for the Qwen3.8-Flash-Next MTP draft block.

    Mirrors the checkpoint layout (``mtp.fc_embedding`` / ``mtp.fc_hidden`` /
    ``mtp.pre_fc_norm_*`` / ``mtp.hyper_connection_mixer`` /
    ``mtp.layers.N``) so the source tensors load verbatim and the export keeps
    the original ``mtp.*`` names. The decoder blocks subclass
    ``Qwen4ExpTextDecoderLayer`` under a class name without "decoderlayer";
    the generic forward helpers match on that substring and must not route the
    draft through the backbone forward. The container forward is an export-only
    passthrough so calibration recipes can still visit and save the draft
    tensors as FLOAT; no real MTP inference is implemented.
    """

    def __init__(self, mtp_config, num_mtp_layers: int):
        super().__init__()
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextDecoderLayer,
            Qwen4ExpTextGatedResidual,
            Qwen4ExpTextRMSNorm,
        )

        class _MtpBlock(Qwen4ExpTextDecoderLayer):
            pass

        hidden_size = int(mtp_config.hidden_size)
        hc_count = int(mtp_config.hc_count)
        with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
            self.fc_embedding = nn.Linear(hidden_size, hidden_size, bias=False)
            self.fc_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
            self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(hidden_size, eps=mtp_config.rms_norm_eps)
            self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(hidden_size * hc_count, eps=mtp_config.rms_norm_eps)
            self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(mtp_config, use_combine=False)
            self.layers = nn.ModuleList([_MtpBlock(mtp_config, idx) for idx in range(num_mtp_layers)])

    # pylint: disable-next=keyword-arg-before-vararg
    def forward(self, hidden_states=None, *args, **kwargs):  # pragma: no cover - export only
        """Export-only passthrough.

        The calibration pipeline still has to visit ``mtp.*`` so the saver
        writes its unquantized tensors (including the unpacked per-expert
        Linears) as FLOAT.  This container is never used for real inference,
        so returning the incoming hidden states is sufficient and keeps the
        layer-wise scheduler able to advance to the end of the generator.
        """
        if hidden_states is None and args:
            hidden_states = args[0]
        return hidden_states


def _collect_quant_safetensors_keys(save_directory: str) -> set:
    """Return the union of keys across final quant safetensors shards.

    PLE scratch shards (``quant_model_weights_ple_tmp-*``) are ignored so a
    previous interrupted run cannot make PLE keys look already exported.
    """
    keys: set = set()
    for fname in sorted(glob.glob(os.path.join(save_directory, "quant_model_weights*.safetensors"))):
        if os.path.basename(fname).startswith("quant_model_weights_ple_tmp"):
            continue
        try:
            with safe_open(fname, framework="pt", device="cpu") as h:
                keys.update(h.keys())
        except Exception:  # nosec B112
            continue
    return keys


# Matches the writer's intermediate naming produced by
# ``BufferedSafetensorsWriter.save_one_file`` for PLE shards (we use a
# dedicated prefix to avoid clashing with any stale AscendV1 saver
# scratch shards left over from a prior incomplete run).
_PLE_TMP_NAME_RE = re.compile(r"^quant_model_weights_ple_tmp-(\d{5})-of-00000\.safetensors$")

# Matches the *final* sharded naming (post-rename). Used by
# ``_enumerate_quant_shard_files`` to enumerate existing quant shards so
# the PLE patcher can renumber them to the new total.
_QUANT_SHARD_RE = re.compile(r"^quant_model_weights-(\d{5})-of-(\d{5})\.safetensors$")

# Index rewritten by ``BufferedSafetensorsWriter.close -> save_index`` right
# before ``ascendv1_save_postprocess`` runs. Its ``weight_map`` is the
# authoritative list of tensors/shard files written by the current run.
_QUANT_INDEX_NAME = "quant_model_weights.safetensors.index.json"


def _collect_current_export_keys(save_directory: str) -> Optional[set]:
    """Return the current run's tensor keys from the freshly written index.

    ``None`` means the export used the single-file layout (no index), so the
    caller should fall back to scanning the output directory.
    """
    index_path = os.path.join(save_directory, _QUANT_INDEX_NAME)
    if not os.path.exists(index_path):
        return None
    data = json_safe_load(index_path) or {}
    weight_map = data.get("weight_map") or {}
    if not isinstance(weight_map, dict):
        return None
    return set(weight_map.keys())


def _enumerate_index_shard_files(save_directory: str) -> Optional[List[Tuple[int, str]]]:
    """Return the current run's ``(index, filename)`` shard pairs.

    The index is preferred over a directory scan so shards left behind by a
    previous run (same output directory, different shard total) are not
    renumbered or counted as current output.
    """
    index_path = os.path.join(save_directory, _QUANT_INDEX_NAME)
    if not os.path.exists(index_path):
        return None
    data = json_safe_load(index_path) or {}
    weight_map = data.get("weight_map") or {}
    if not isinstance(weight_map, dict):
        return None
    shards = set()
    for fname in set(weight_map.values()):
        m = _QUANT_SHARD_RE.match(fname)
        if m:
            shards.add((int(m.group(1)), fname))
    return sorted(shards)


def _enumerate_quant_shard_files(save_directory: str) -> List[Tuple[int, str]]:
    """Return ``(index, filename)`` for existing quant shards, sorted by index.

    Stale ``of-00000`` shards from an interrupted run are included; the PLE
    writer uses a distinct prefix, so its scratch files cannot collide.
    """
    out: List[Tuple[int, str]] = []
    try:
        entries = sorted(os.listdir(save_directory))
    except FileNotFoundError:
        return out
    for fname in entries:
        m = _QUANT_SHARD_RE.match(fname)
        if m:
            out.append((int(m.group(1)), fname))
    return out


class _DummyQwen4ExpTextConfig:
    """Minimal stand-in for the qwen4_exp text config used by the drop-in
    MoE replacement module.

    Only the attributes consumed by ``_UnstackedQwen4ExpTextExperts`` are
    populated. The real ``Qwen4ExpTextConfig`` requires ``Qwen4ExpConfig``
    parents and many extra fields; for the verification flow we only need
    the MoE dimensions and the activation function name.
    """

    def __init__(self, hidden_size: int, moe_intermediate_size: int, num_experts: int, hidden_act: str):
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.hidden_act = hidden_act


class _UnstackedQwen4ExpTextExperts(nn.Module):
    """Drop-in replacement for ``Qwen4ExpTextExperts`` using per-expert Linears.

    Two-stage construction (memory-friendly, matches
    ``qwen3_vl_moe.UnstackedQwen3VLMoeSparseMoeBlock``):

    1. ``__init__(cfg, ...)``: build ``num_experts`` empty per-expert
       ``nn.Linear`` shells (``gate_proj`` / ``up_proj`` / ``down_proj``).
       Weights are placeholder (zero / random) until step 2 fills them in.
    2. ``_transform_weights_from_original(original, in_place=True)``: chunk
       the 3D packed ``gate_up_proj`` (shape ``(E, 2D, H)``) along dim 0 into
       ``(D, H)`` ``gate`` / ``up`` slices and slice the 3D ``down_proj``
       (shape ``(E, H, D)``) per expert. Each per-expert ``nn.Linear.weight``
       is assigned via ``nn.Parameter(..., requires_grad=False)`` (no
       ``copy_`` -> no second peak of expert memory). With ``in_place=True``
       the original 3D ``nn.Parameter``s are then ``del``'d and ``gc.collect``
       runs, freeing them immediately.

    Public surface mirrors the original:
        forward(hidden_states, top_k_index, top_k_weights) -> Tensor
    State-dict layout mirrors what transformers' ``save_pretrained`` produces:
        gate_up_proj / down_proj  -->  experts.{i}.{gate,up,down}_proj.weight
    """

    def __init__(self, cfg, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.num_experts = int(getattr(cfg, "num_experts"))
        self.hidden_dim = int(getattr(cfg, "hidden_size"))
        self.intermediate_dim = int(getattr(cfg, "moe_intermediate_size"))

        try:
            from transformers.activations import ACT2FN

            self.act_fn = ACT2FN[getattr(cfg, "hidden_act", "silu")]
        except Exception:  # pragma: no cover - best-effort
            self.act_fn = torch.nn.functional.silu

        kwargs = {"bias": False}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype

        for i in range(self.num_experts):
            expert = nn.Module()
            expert.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, **kwargs)
            expert.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, **kwargs)
            expert.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim, **kwargs)
            self.add_module(str(i), expert)

    def _transform_weights_from_original(
        self,
        original_experts: nn.Module,
        in_place: bool = True,
    ) -> None:
        """Chunk the 3D ``gate_up_proj`` / ``down_proj`` from
        ``original_experts`` and reassign the per-expert ``nn.Linear`` weights
        via ``nn.Parameter`` so the 3D buffers can be freed immediately.

        ``original_experts`` is expected to expose:
          - ``gate_up_proj``: ``nn.Parameter`` of shape ``(E, 2D, H)``
          - ``down_proj``:    ``nn.Parameter`` of shape ``(E, H, D)``
        where ``E = num_experts``, ``D = intermediate_dim``, ``H = hidden_dim``.

        Axis convention follows ``Qwen4ExpTextExperts``:
          - ``gate_up_proj[i].chunk(2, dim=0) -> (D, H), (D, H)``
            (no transpose; the Linear expects ``(out, in) == (D, H)``).
          - ``down_proj[i]`` is already ``(H, D)``.
        """
        gate_up_3d = getattr(original_experts, "gate_up_proj", None)
        down_3d = getattr(original_experts, "down_proj", None)
        if not isinstance(gate_up_3d, nn.Parameter) or not isinstance(down_3d, nn.Parameter):
            raise TypeError(
                "_UnstackedQwen4ExpTextExperts._transform_weights_from_original "
                "expects original_experts.gate_up_proj / down_proj to be nn.Parameter; "
                f"got {type(gate_up_3d).__name__} / {type(down_3d).__name__}."
            )

        with torch.no_grad():
            # Move to CPU so per-expert assignments do not blow up NPU memory.
            full_gate_up = gate_up_3d.data.cpu().contiguous()  # (E, 2D, H)
            full_down = down_3d.data.cpu().contiguous()  # (E, H, D)

            E = full_gate_up.shape[0]
            if E != self.num_experts:
                raise ValueError(
                    f"MoE expert count mismatch: gate_up_proj has {E}, this module expects {self.num_experts}."
                )
            if full_gate_up.shape[1] != 2 * self.intermediate_dim:
                raise ValueError(
                    f"gate_up_proj axis 1 = {full_gate_up.shape[1]} "
                    f"not equal to 2 * intermediate_dim ({2 * self.intermediate_dim})."
                )
            if full_gate_up.shape[2] != self.hidden_dim:
                raise ValueError(
                    f"gate_up_proj axis 2 = {full_gate_up.shape[2]} not equal to hidden_dim ({self.hidden_dim})."
                )

            for i in range(self.num_experts):
                gate_w, up_w = full_gate_up[i].chunk(2, dim=0)  # (D, H), (D, H)
                self._modules[str(i)].gate_proj.weight = nn.Parameter(
                    gate_w.contiguous(),
                    requires_grad=False,
                )
                self._modules[str(i)].up_proj.weight = nn.Parameter(
                    up_w.contiguous(),
                    requires_grad=False,
                )
                self._modules[str(i)].down_proj.weight = nn.Parameter(
                    full_down[i].contiguous(),
                    requires_grad=False,
                )

            # Release the 3D CPU tensors before freeing the originals so
            # peak memory does not hold two copies.
            del full_gate_up, full_down

        if in_place:
            # Free the original 3D ``nn.Parameter`` attributes so the parent's
            # state_dict no longer carries ``mlp.experts.gate_up_proj`` /
            # ``mlp.experts.down_proj`` keys.
            if hasattr(original_experts, "gate_up_proj"):
                del original_experts.gate_up_proj
            if hasattr(original_experts, "down_proj"):
                del original_experts.down_proj
            gc.collect()

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Reference forward; quant calibration only inspects parameters.

        The real inference path uses the fused 3D experts. This implementation
        exists for state-dict compatibility and unit-test parity.
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t[0].item())
            if expert_idx >= self.num_experts:
                continue
            expert = self._modules[str(expert_idx)]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self.act_fn(expert.gate_proj(current_state)) * expert.up_proj(current_state)
            current_hidden_states = expert.down_proj(current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
