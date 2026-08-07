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

from typing import Any, Dict, List, Optional, Tuple

import torch

from msmodelslim.model.common.utils import _get_expert_range
from msmodelslim.model.interface_hub import QuaRotInterface


# ---------------------------------------------------------------------------
# Layer naming helpers (shared by QuaRot maps and IterSmooth subgraph configs)
# ---------------------------------------------------------------------------


def is_kda_layer(text_cfg: Any, layer_idx: int) -> bool:
    if hasattr(text_cfg, "is_kda_layer"):
        return bool(text_cfg.is_kda_layer(layer_idx))
    lac = getattr(text_cfg, "linear_attn_config", None) or {}
    kda_layers = lac.get("kda_layers") or []
    return (layer_idx + 1) in kda_layers


def is_dense_mlp(text_cfg: Any, layer_idx: int) -> bool:
    return layer_idx < getattr(text_cfg, "first_k_dense_replace", 1)


def use_attn_residuals(text_cfg: Any) -> bool:
    return getattr(text_cfg, "attn_res_block_size", None) is not None


def kda_use_full_rank_gate(text_cfg: Any) -> bool:
    lac = getattr(text_cfg, "linear_attn_config", None) or {}
    return bool(lac.get("use_full_rank_gate", False))


def has_latent_moe(text_cfg: Any) -> bool:
    return getattr(text_cfg, "routed_expert_hidden_size", None) is not None


def latent_moe_use_norm(text_cfg: Any) -> bool:
    """True when latent MoE is configured and norm before up-proj is enabled."""
    return has_latent_moe(text_cfg) and bool(getattr(text_cfg, "latent_moe_use_norm", False))


def layer_prefix(layer_idx: int) -> str:
    return f"language_model.model.layers.{layer_idx}"


def attn_prefix(layer_idx: int) -> str:
    return f"{layer_prefix(layer_idx)}.self_attn"


def moe_prefix(layer_idx: int) -> str:
    return f"{layer_prefix(layer_idx)}.block_sparse_moe"


def input_layernorm_targets(text_cfg: Any, layer_idx: int) -> List[str]:
    attn = attn_prefix(layer_idx)
    if is_kda_layer(text_cfg, layer_idx):
        targets = [
            f"{attn}.q_proj",
            f"{attn}.k_proj",
            f"{attn}.v_proj",
            f"{attn}.f_a_proj",
            f"{attn}.b_proj",
        ]
        if kda_use_full_rank_gate(text_cfg):
            targets.append(f"{attn}.g_proj")
        else:
            targets.append(f"{attn}.g_a_proj")
        return targets
    targets = [
        f"{attn}.q_a_proj",
        f"{attn}.kv_a_proj_with_mqa",
    ]
    if getattr(text_cfg, "mla_use_output_gate", False):
        targets.append(f"{attn}.g_proj")
    return targets


def q_a_layernorm_targets(text_cfg: Any, layer_idx: int) -> Optional[List[str]]:
    if is_kda_layer(text_cfg, layer_idx):
        return None
    return [f"{attn_prefix(layer_idx)}.q_b_proj"]


def kv_a_layernorm_targets(text_cfg: Any, layer_idx: int) -> Optional[List[str]]:
    """QuaRot fuse only; IterSmooth uses ov(kv_b→o) instead."""
    if is_kda_layer(text_cfg, layer_idx):
        return None
    return [f"{attn_prefix(layer_idx)}.kv_b_proj"]


def post_attention_layernorm_targets(text_cfg: Any, layer_idx: int) -> List[str]:
    prefix = layer_prefix(layer_idx)
    if is_dense_mlp(text_cfg, layer_idx):
        return [
            f"{prefix}.mlp.gate_proj",
            f"{prefix}.mlp.up_proj",
        ]
    moe = moe_prefix(layer_idx)
    return [
        f"{moe}.gate",
        f"{moe}.shared_experts.gate_proj",
        f"{moe}.shared_experts.up_proj",
        f"{moe}.routed_expert_down_proj",
    ]


def routed_expert_norm_targets(
    text_cfg: Any,
    layer_idx: int,
    *,
    require_latent_hidden: bool = False,
) -> Optional[List[str]]:
    """Targets for ``routed_expert_norm``, or None when not applicable.

    ``require_latent_hidden=True`` (IterSmooth) uses ``latent_moe_use_norm`` (flag +
    latent hidden). QuaRot fuse historically only checks the config flag.
    """
    if is_dense_mlp(text_cfg, layer_idx):
        return None
    if require_latent_hidden:
        if not latent_moe_use_norm(text_cfg):
            return None
    elif not bool(getattr(text_cfg, "latent_moe_use_norm", False)):
        return None
    return [f"{moe_prefix(layer_idx)}.routed_expert_up_proj"]


def attn_res_norm_pairs(text_cfg: Any, layer_idx: int) -> List[Tuple[str, str]]:
    """``(norm_name, linear_name)`` pairs; QuaRot-only (not IterSmooth)."""
    if not use_attn_residuals(text_cfg):
        return []
    prefix = layer_prefix(layer_idx)
    return [
        (f"{prefix}.self_attention_res_norm", f"{prefix}.self_attention_res_proj"),
        (f"{prefix}.mlp_res_norm", f"{prefix}.mlp_res_proj"),
    ]


def hidden_rot_right(text_cfg: Any, layer_idx: int) -> List[str]:
    """Modules that take right-multiply by the hidden Hadamard."""
    names = list(input_layernorm_targets(text_cfg, layer_idx))
    prefix = layer_prefix(layer_idx)
    if is_dense_mlp(text_cfg, layer_idx):
        names.extend([f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"])
    else:
        moe = moe_prefix(layer_idx)
        names.extend(
            [
                f"{moe}.gate",
                f"{moe}.shared_experts.gate_proj",
                f"{moe}.shared_experts.up_proj",
                f"{moe}.routed_expert_down_proj",
            ]
        )
    if use_attn_residuals(text_cfg):
        names.extend([f"{prefix}.self_attention_res_proj", f"{prefix}.mlp_res_proj"])
    return names


def hidden_rot_left(text_cfg: Any, layer_idx: int) -> List[str]:
    """Modules that take left-multiply by the hidden Hadamard."""
    names = [f"{attn_prefix(layer_idx)}.o_proj"]
    prefix = layer_prefix(layer_idx)
    if is_dense_mlp(text_cfg, layer_idx):
        names.append(f"{prefix}.mlp.down_proj")
    else:
        moe = moe_prefix(layer_idx)
        names.extend([f"{moe}.shared_experts.down_proj", f"{moe}.routed_expert_up_proj"])
    return names


# ---------------------------------------------------------------------------
# QuaRot maps
# ---------------------------------------------------------------------------


def get_ln_fuse_map(config: Any, num_hidden_layers: Optional[int] = None):
    """Norm -> Linear fuse map for offline QuaRot (step 1)."""
    cfg = config.text_config
    if num_hidden_layers is None:
        num_hidden_layers = cfg.num_hidden_layers

    ln_linear_map: Dict[str, List[str]] = {}

    for layer_idx in range(num_hidden_layers):
        prefix = layer_prefix(layer_idx)
        attn = attn_prefix(layer_idx)

        ln_linear_map[f"{prefix}.input_layernorm"] = input_layernorm_targets(cfg, layer_idx)

        q_targets = q_a_layernorm_targets(cfg, layer_idx)
        if q_targets is not None:
            ln_linear_map[f"{attn}.q_a_layernorm"] = q_targets
        kv_targets = kv_a_layernorm_targets(cfg, layer_idx)
        if kv_targets is not None:
            ln_linear_map[f"{attn}.kv_a_layernorm"] = kv_targets

        ln_linear_map[f"{prefix}.post_attention_layernorm"] = post_attention_layernorm_targets(cfg, layer_idx)

        routed_norm = routed_expert_norm_targets(cfg, layer_idx, require_latent_hidden=False)
        if routed_norm is not None:
            ln_linear_map[f"{moe_prefix(layer_idx)}.routed_expert_norm"] = routed_norm

        for norm_name, linear_name in attn_res_norm_pairs(cfg, layer_idx):
            ln_linear_map[norm_name] = [linear_name]

    pre_run_fused_ln = {
        "language_model.model.norm": ["language_model.lm_head"],
    }
    if use_attn_residuals(cfg):
        pre_run_fused_ln["language_model.model.output_attn_res_norm"] = ["language_model.model.output_attn_res_proj"]

    return pre_run_fused_ln, ln_linear_map


def get_rotate_map(
    config: Any,
    block_size: int,
    num_hidden_layers: Optional[int] = None,
    *,
    enable_rot: bool = True,
    enable_rot_b_proj: bool = True,
    enable_rot_kv_b_proj: bool = True,
    enable_rot_latent: bool = True,
) -> Tuple[List[Any], List[Any]]:
    """Weight rotate map; ``rot_latent`` only maps local EP experts via ``_get_expert_range``."""
    cfg = config.text_config
    if num_hidden_layers is None:
        num_hidden_layers = cfg.num_hidden_layers

    hidden_size = cfg.hidden_size
    q_lora_rank = cfg.q_lora_rank
    kv_lora_rank = cfg.kv_lora_rank
    qk_rope_head_dim = cfg.qk_rope_head_dim
    h_lat = getattr(cfg, "routed_expert_hidden_size", None)

    rot_pairs: Dict[str, Any] = {}
    pre_run_pairs: List[Any] = []

    if enable_rot:
        rot = QuaRotInterface.get_rotate_command(
            size=hidden_size,
            mode=QuaRotInterface.QuaRotMode.HADAMARD,
            block_size=block_size,
        )
        pre_left, pre_right = {}, {}
        pre_right["language_model.model.embed_tokens"] = rot
        pre_left["mm_projector.rot_proj"] = rot
        pre_right["language_model.lm_head"] = rot
        if use_attn_residuals(cfg):
            pre_right["language_model.model.output_attn_res_proj"] = rot
        pre_run_pairs.append(QuaRotInterface.RotatePair(left_rot=pre_left, right_rot=pre_right))

        left_rot, right_rot = {}, {}
        for layer_idx in range(num_hidden_layers):
            for name in hidden_rot_right(cfg, layer_idx):
                right_rot[name] = rot
            for name in hidden_rot_left(cfg, layer_idx):
                left_rot[name] = rot

        rot_pairs["rot"] = QuaRotInterface.RotatePair(left_rot=left_rot, right_rot=right_rot)

    if enable_rot_b_proj:
        rot_b_proj = QuaRotInterface.get_rotate_command(
            size=q_lora_rank,
            mode=QuaRotInterface.QuaRotMode.BLOCK_HADAMARD_SHIFTED,
            block_size=block_size,
        )
        left_b, right_b = {}, {}
        for layer_idx in range(num_hidden_layers):
            if is_kda_layer(cfg, layer_idx):
                continue
            attn = attn_prefix(layer_idx)
            left_b[f"{attn}.q_a_proj"] = rot_b_proj
            right_b[f"{attn}.q_b_proj"] = rot_b_proj
        rot_pairs["rot_b_proj"] = QuaRotInterface.RotatePair(left_rot=left_b, right_rot=right_b)

    if enable_rot_kv_b_proj:
        rot_kv = QuaRotInterface.get_rotate_command(
            size=kv_lora_rank,
            mode=QuaRotInterface.QuaRotMode.HADAMARD,
            block_size=block_size,
        )
        left_kv, right_kv = {}, {}
        for layer_idx in range(num_hidden_layers):
            if is_kda_layer(cfg, layer_idx):
                continue
            attn = attn_prefix(layer_idx)
            left_kv[f"{attn}.kv_a_proj_with_mqa"] = [
                rot_kv,
                torch.eye(qk_rope_head_dim, dtype=rot_kv.dtype, device=rot_kv.device),
            ]
            right_kv[f"{attn}.kv_b_proj"] = rot_kv
        rot_pairs["rot_kv_b_proj"] = QuaRotInterface.RotatePair(left_rot=left_kv, right_rot=right_kv)

    # EP coupling: only rotate experts present on this rank (None slots elsewhere).
    if enable_rot_latent and h_lat is not None:
        rot_latent = QuaRotInterface.get_rotate_command(
            size=h_lat,
            mode=QuaRotInterface.QuaRotMode.HADAMARD,
            block_size=block_size,
        )
        left_lat, right_lat = {}, {}
        for layer_idx in range(num_hidden_layers):
            if is_dense_mlp(cfg, layer_idx):
                continue
            moe = moe_prefix(layer_idx)
            left_lat[f"{moe}.routed_expert_down_proj"] = rot_latent
            right_lat[f"{moe}.routed_expert_up_proj"] = rot_latent
            expert_start, expert_end = _get_expert_range(cfg)
            for expert_idx in range(expert_start, expert_end):
                expert = f"{moe}.experts.{expert_idx}"
                right_lat[f"{expert}.w1"] = rot_latent
                right_lat[f"{expert}.w3"] = rot_latent
                left_lat[f"{expert}.w2"] = rot_latent
        rot_pairs["rot_latent"] = QuaRotInterface.RotatePair(left_rot=left_lat, right_rot=right_lat)

    return pre_run_pairs, list(rot_pairs.values())
