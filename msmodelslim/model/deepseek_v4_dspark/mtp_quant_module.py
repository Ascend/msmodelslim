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

DSpark MTP 模块加载与包装。

与 Flash MTP 不同：
  - mtp.0 含 main_proj / main_norm（无 enorm/hnorm/e_proj）
  - 末层 mtp.{N-1} 含独立 norm 与 hc_head_fn
  - embed / head 与主模型共享原始权重；量化落盘时复制为 mtp.0.embed / mtp.{N-1}.head
"""

import os
from typing import Any, Optional

import torch
from torch import nn

from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import json_safe_load
from ..deepseek_v4.convert_fp8_to_bf16 import auto_dequant_state_dict
from ..deepseek_v4.model import ParallelEmbedding, RMSNorm
from ..deepseek_v4.mtp_quant_module import get_shared_weight
from ..common.weight_helper import get_state_dict, get_weight_map
from .model import DSparkMarkovHead, DSparkConfidenceHead


def ensure_config_n_mtp_layers(config: Any, model_path: str, config_data: dict | None = None) -> int:
    """写入正确的 n_mtp_layers（优先 weight_map，避免 config 误报为 1）。"""
    if config_data is None:
        config_path = os.path.join(model_path, "config.json")
        config_data = json_safe_load(config_path) if os.path.isfile(config_path) else {}
    n_mtp = detect_n_mtp_layers(model_path, config_data)
    config.n_mtp_layers = n_mtp
    return n_mtp


def detect_n_mtp_layers(model_path: str, config_data: dict) -> int:
    """从 weight_map 推断 MTP 层数（DSpark checkpoint 常为 mtp.0/1/2，config 可能仍为 1）。"""
    if config_data.get("dspark_block_size"):
        weight_map_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(weight_map_path):
            weight_map = json_safe_load(weight_map_path).get("weight_map", {})
            mtp_indices = sorted(
                {int(key.split(".")[1]) for key in weight_map if key.startswith("mtp.") and key.split(".")[1].isdigit()}
            )
            if mtp_indices:
                return mtp_indices[-1] + 1
        return 3
    return int(config_data.get("n_mtp_layers", config_data.get("num_nextn_predict_layers", 0)))


def prune_dspark_mtp_stage_modules(mtp_decoder: nn.Module, mtp_idx: int, n_mtp_layers: int) -> None:
    """移除当前 stage 不应存在的 DSpark 专有子模块。"""
    if mtp_idx != 0:
        for name in ("main_proj", "main_norm"):
            if hasattr(mtp_decoder, name):
                delattr(mtp_decoder, name)
                mtp_decoder._modules.pop(name, None)

    if n_mtp_layers <= 0 or mtp_idx != n_mtp_layers - 1:
        for name in ("norm", "markov_head", "confidence_head", "head"):
            if hasattr(mtp_decoder, name):
                delattr(mtp_decoder, name)
                mtp_decoder._modules.pop(name, None)
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            if hasattr(mtp_decoder, name):
                delattr(mtp_decoder, name)


def _exclude_missing_checkpoint_params(model_path: str, module: nn.Module, layer_prefix: str) -> list[str]:
    weight_map = get_weight_map(model_path)
    exclude: list[str] = []
    for name, _ in module.named_parameters():
        weight_key = f"{layer_prefix}.{name}" if layer_prefix else name
        if weight_key not in weight_map:
            exclude.append(name)
    return exclude


def load_dspark_mtp_state_dict(model_path: str, module: nn.Module, layer_prefix: str) -> dict[str, torch.Tensor]:
    exclude = _exclude_missing_checkpoint_params(model_path, module, layer_prefix)
    state_dict = get_state_dict(model_path, module, prefix=layer_prefix, exclude=exclude)
    auto_dequant_state_dict(layer_prefix, state_dict, model_path)
    return state_dict


def _install_detached_weight(
    module: nn.Module,
    source: torch.Tensor,
    device: torch.device | str | None = None,
) -> None:
    """写入独立 Parameter，避免与主模型 embed/head 共享 storage。"""
    dtype = module.weight.dtype if hasattr(module, "weight") else source.dtype
    if device is None:
        device = torch.device("cpu")
        if hasattr(module, "weight") and not module.weight.is_meta:
            device = module.weight.device
    data = source.detach().to(device=device, dtype=dtype).contiguous().clone()
    module.weight = nn.Parameter(data)


def _module_device(module: nn.Module | None) -> torch.device | None:
    if module is None:
        return None
    params = getattr(module, "parameters", None)
    if callable(params):
        for param in params(recurse=True):
            if not param.is_meta:
                return param.device
    weight = getattr(module, "weight", None)
    if torch.is_tensor(weight) and not weight.is_meta:
        return weight.device
    buffers = getattr(module, "buffers", None)
    if callable(buffers):
        for buf in buffers(recurse=True):
            if not buf.is_meta:
                return buf.device
    return None


def _place_module(module: nn.Module, device: torch.device | str | None) -> None:
    if device is None:
        return
    module.to(device)


def _load_original_embed_weight(model_path: str) -> torch.Tensor:
    """从 checkpoint 读取并 dequant 主模型 embed 原始权重（QuaRot 旋转前）。"""
    state = {"embed.weight": get_shared_weight(model_path, "embed.weight")}
    auto_dequant_state_dict("", state, model_path)
    return state["embed.weight"]


def _load_original_head_weight(model_path: str) -> torch.Tensor:
    """从 checkpoint 读取并 dequant 主模型 head 原始权重（QuaRot 旋转前）。"""
    state = {"head.weight": get_shared_weight(model_path, "head.weight")}
    auto_dequant_state_dict("", state, model_path)
    return state["head.weight"]


def resolve_original_embed_weight(
    model_path: str | None = None,
    original_weight: torch.Tensor | None = None,
    embed: ParallelEmbedding | None = None,
) -> torch.Tensor:
    """优先用旋转前缓存，否则从 checkpoint dequant，最后才退回内存权重。"""
    if original_weight is not None:
        return original_weight
    if model_path:
        return _load_original_embed_weight(model_path)
    if embed is not None and not embed.weight.is_meta:
        return embed.weight.data
    raise ValueError("resolve_original_embed_weight requires original_weight, model_path, or materialized embed")


def resolve_original_head_weight(
    model_path: str | None = None,
    original_weight: torch.Tensor | None = None,
    head: nn.Linear | None = None,
) -> torch.Tensor:
    """优先用旋转前缓存，否则从 checkpoint dequant，最后才退回内存权重。"""
    if original_weight is not None:
        return original_weight
    if model_path:
        return _load_original_head_weight(model_path)
    if head is not None and not head.weight.is_meta:
        return head.weight.data
    raise ValueError("resolve_original_head_weight requires original_weight, model_path, or materialized head")


def copy_embed_for_save(
    embed: ParallelEmbedding | None,
    model_path: str | None = None,
    original_weight: torch.Tensor | None = None,
) -> ParallelEmbedding:
    """复制主模型旋转前 embed，供落盘为 mtp.0.embed。"""
    source = resolve_original_embed_weight(model_path=model_path, original_weight=original_weight, embed=embed)
    vocab_size, dim = source.shape[0], source.shape[1]
    copied = ParallelEmbedding(vocab_size, dim)
    _install_detached_weight(copied, source)
    return copied


def refresh_mtp_embed_from_checkpoint(
    embed: ParallelEmbedding,
    model_path: str,
    original_weight: torch.Tensor | None = None,
) -> None:
    """将 mtp.0.embed 刷新为旋转前的原始 embed.weight（独立 Parameter）。"""
    source = resolve_original_embed_weight(model_path=model_path, original_weight=original_weight)
    _install_detached_weight(embed, source)


def refresh_mtp_head_from_checkpoint(
    head: nn.Linear,
    model_path: str,
    original_weight: torch.Tensor | None = None,
) -> None:
    """将 mtp.{N-1}.head 刷新为旋转前的原始 head.weight（独立 Parameter）。"""
    source = resolve_original_head_weight(model_path=model_path, original_weight=original_weight)
    _install_detached_weight(head, source)


def copy_head_for_quarot_fusion(
    head: nn.Linear | None,
    model_path: str | None = None,
    original_weight: torch.Tensor | None = None,
) -> nn.Linear:
    """复制主模型旋转前 head，供落盘为 mtp.{N-1}.head。"""
    source = resolve_original_head_weight(model_path=model_path, original_weight=original_weight, head=head)
    in_features, out_features = source.shape[1], source.shape[0]
    bias = head.bias is not None if head is not None else False
    dtype = source.dtype if original_weight is not None or model_path else head.weight.dtype
    copied = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    _install_detached_weight(copied, source)
    if bias and head is not None and head.bias is not None:
        copied.bias = nn.Parameter(head.bias.data.detach().clone())
    return copied


def ensure_dedicated_mtp_embed(
    mtp_decoder: nn.Module,
    embed: ParallelEmbedding | None,
    model_path: str | None = None,
    original_weight: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> ParallelEmbedding:
    target_device = device if device is not None else _module_device(mtp_decoder)
    dedicated = copy_embed_for_save(embed, model_path=model_path, original_weight=original_weight)
    if hasattr(mtp_decoder, "embed"):
        delattr(mtp_decoder, "embed")
        mtp_decoder._modules.pop("embed", None)
    mtp_decoder.add_module("embed", dedicated)
    _place_module(dedicated, target_device)
    get_logger().info(
        "Initialized dedicated mtp.0.embed from unrotated embed.weight, shape=%s, device=%s",
        tuple(dedicated.weight.shape),
        dedicated.weight.device,
    )
    return dedicated


def ensure_dedicated_mtp_head(
    mtp_decoder: nn.Module,
    model_path: str,
    layer_prefix: str,
    dim: int,
    vocab_size: int,
    original_weight: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> nn.Linear:
    """创建/刷新独立 head 模块，权重为旋转前的主模型 head.weight。"""
    target_device = device if device is not None else _module_device(mtp_decoder)
    if hasattr(mtp_decoder, "head"):
        delattr(mtp_decoder, "head")
        mtp_decoder._modules.pop("head", None)
    dedicated_head = nn.Linear(dim, vocab_size, bias=False, dtype=torch.float32)
    mtp_decoder.add_module("head", dedicated_head)

    shared = resolve_original_head_weight(model_path=model_path, original_weight=original_weight)
    _install_detached_weight(dedicated_head, shared, device=target_device or "cpu")
    get_logger().info(
        "Initialized dedicated %s from unrotated head.weight, shape=%s, device=%s",
        f"{layer_prefix}.head",
        tuple(dedicated_head.weight.shape),
        dedicated_head.weight.device,
    )
    return dedicated_head


def wrap_dspark_block(
    block: nn.Module,
    embed: ParallelEmbedding | None,
    head: Optional[nn.Linear] = None,
    copy_head: bool = False,
    copy_embed: bool = False,
    model_path: str | None = None,
    original_embed_weight: torch.Tensor | None = None,
    original_head_weight: torch.Tensor | None = None,
) -> None:
    """挂载 embed/head；copy_* 为 True 时复制旋转前的主模型权重为独立子模块。"""
    if copy_embed:
        ensure_dedicated_mtp_embed(block, embed, model_path=model_path, original_weight=original_embed_weight)
    else:
        if embed is None:
            raise ValueError("wrap_dspark_block requires embed when copy_embed is False")
        block.embed = embed

    if head is None:
        return
    if copy_head:
        dedicated = copy_head_for_quarot_fusion(head, model_path=model_path, original_weight=original_head_weight)
        if hasattr(block, "head"):
            delattr(block, "head")
        block.add_module("head", dedicated)
    else:
        block.head = head


def attach_dspark_mtp_stage0(mtp_decoder: nn.Module, config: Any, state_dict: dict[str, torch.Tensor]) -> None:
    target_ids = getattr(config, "dspark_target_layer_ids", ()) or ()
    in_dim = config.dim * len(target_ids)
    if not hasattr(mtp_decoder, "main_proj"):
        mtp_decoder.main_proj = nn.Linear(in_dim, config.dim, bias=False)
    if not hasattr(mtp_decoder, "main_norm"):
        mtp_decoder.main_norm = RMSNorm(config.dim, config.norm_eps)

    if "main_proj.weight" in state_dict:
        mtp_decoder.main_proj.weight.data.copy_(state_dict["main_proj.weight"])
    if "main_norm.weight" in state_dict:
        mtp_decoder.main_norm.weight.data.copy_(state_dict["main_norm.weight"])


def _attach_markov_and_confidence(mtp_decoder: nn.Module, config: Any, state_dict: dict[str, torch.Tensor]) -> None:
    markov_rank = getattr(config, "dspark_markov_rank", 0) or 256
    if not hasattr(mtp_decoder, "markov_head"):
        mtp_decoder.markov_head = DSparkMarkovHead(config.vocab_size, markov_rank)
    if not hasattr(mtp_decoder, "confidence_head"):
        mtp_decoder.confidence_head = DSparkConfidenceHead(config.dim + markov_rank)

    markov_sd = {
        key[len("markov_head.") :]: value for key, value in state_dict.items() if key.startswith("markov_head.")
    }
    if markov_sd:
        mtp_decoder.markov_head.load_state_dict(markov_sd, strict=False)

    conf_sd = {
        key[len("confidence_head.") :]: value for key, value in state_dict.items() if key.startswith("confidence_head.")
    }
    if conf_sd:
        mtp_decoder.confidence_head.load_state_dict(conf_sd, strict=False)


def attach_dspark_mtp_last_stage(
    mtp_decoder: nn.Module,
    config: Any,
    model_path: str,
    layer_prefix: str,
    state_dict: dict[str, torch.Tensor],
    original_head_weight: torch.Tensor | None = None,
) -> None:
    if not hasattr(mtp_decoder, "norm"):
        mtp_decoder.norm = RMSNorm(config.dim, config.norm_eps)
    if "norm.weight" in state_dict:
        mtp_decoder.norm.weight.data.copy_(state_dict["norm.weight"])

    hc_mult = getattr(config, "hc_mult", 4)
    hc_dim = hc_mult * config.dim
    if not hasattr(mtp_decoder, "hc_head_fn"):
        mtp_decoder.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
    if not hasattr(mtp_decoder, "hc_head_base"):
        mtp_decoder.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
    if not hasattr(mtp_decoder, "hc_head_scale"):
        mtp_decoder.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    for key in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        if key in state_dict:
            getattr(mtp_decoder, key).data.copy_(state_dict[key])

    _attach_markov_and_confidence(mtp_decoder, config, state_dict)
    ensure_dedicated_mtp_head(
        mtp_decoder=mtp_decoder,
        model_path=model_path,
        layer_prefix=layer_prefix,
        dim=config.dim,
        vocab_size=config.vocab_size,
        original_weight=original_head_weight,
    )


def wrap_dspark_mtp_decoder(
    mtp_decoder: nn.Module,
    config: Any,
    model_path: str,
    layer_prefix: str,
    mtp_idx: int,
    n_mtp_layers: int,
    backbone_embed: ParallelEmbedding,
    backbone_head: nn.Linear,
    original_embed_weight: torch.Tensor | None = None,
    original_head_weight: torch.Tensor | None = None,
) -> None:
    """按 DSpark stage 挂载额外结构、加载权重，并复制旋转前的共享 embed/head。"""
    del backbone_embed, backbone_head
    get_logger().debug("Wrap DSpark MTP decoder %s (stage %s/%s)", layer_prefix, mtp_idx, n_mtp_layers)
    state_dict = load_dspark_mtp_state_dict(model_path, mtp_decoder, layer_prefix)

    if mtp_idx == 0:
        attach_dspark_mtp_stage0(mtp_decoder, config, state_dict)
        wrap_dspark_block(
            mtp_decoder,
            None,
            copy_embed=True,
            model_path=model_path,
            original_embed_weight=original_embed_weight,
        )
    if mtp_idx == n_mtp_layers - 1:
        attach_dspark_mtp_last_stage(
            mtp_decoder,
            config,
            model_path,
            layer_prefix,
            state_dict,
            original_head_weight=original_head_weight,
        )

    get_logger().debug("Success to wrap DSpark MTP decoder %s", layer_prefix)
