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

# MoonViT3d vision tower + ``patchmergerv2`` projector + the VLM root of Kimi-K3.1.
#
# Named after the released ``modeling_kimi_k3.py``, which is the file holding the vision
# stack and ``KimiK3ForConditionalGeneration``; this module keeps that split: the vision
# tower, the projector, ``merge_input_ids_with_image_features`` and
# ``KimiK31ForConditionalGeneration``.  The ``language_model`` subtree lives in
# :mod:`msmodelslim.model.kimi_k3_1.modeling_kimi_linear`.
#
# Kimi-K3.1 keeps Kimi-K3's vision stack unchanged (``vision_config`` is byte-identical in
# both ``config.json``): 27 blocks over 1024 hidden / 1536 qkv / 4096 ffn, learnable
# ``divided_fixed`` 2D position embedding, 2D complex RoPE, ``sd2_tpool`` patch merge and a
# two-layer ``PatchMergerMLPV2`` that projects 4x1024 merged channels to 7168.
#
# The port is parameter-name compatible with the released index (``vision_tower.patch_embed
# .proj``, ``vision_tower.encoder.blocks.{i}.wqkv``, ``mm_projector.proj.{0,2}`` ...), but it
# runs on Ascend without ``flash_attn``: attention is the upstream ``eager_attention``
# semantics computed per image, which is exactly the block-diagonal mask the packed path
# builds.  ``torch.compile`` / transformer-only helpers are avoided on purpose.

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .modeling_kimi_linear import KimiLinearForCausalLM

#: ``PytorchGELUTanh`` of ``transformers.activations`` is exactly this.
_GELU_TANH = "tanh"


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_1d_sincos_pos_embed(embed_dim: int, t_size: int, cls_token: bool = False) -> np.ndarray:
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _resolve(vision_cfg, name: str, *fallbacks: str, default=None):
    """Read ``vt_*`` style keys with the plain ``transformers`` names as fallback."""
    for key in (name,) + fallbacks:
        value = getattr(vision_cfg, key, None)
        if value is not None:
            return value
    return default


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    """Learnable ``[H, W, C]`` grid interpolated to each image size (+ sincos time term)."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames)).float().unsqueeze(1),
            persistent=False,
        )
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for frames, height, width in grid_thws.tolist():
            if frames > self.num_frames:
                raise ValueError(f"t:{frames} exceeds num_frames:{self.num_frames}")
            if (height, width) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = (
                    F.interpolate(
                        self.weight.permute(2, 0, 1).unsqueeze(0),
                        size=(height, width),
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .flatten(end_dim=1)
                )
            if frames == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(frames, 1, 1) + self.time_weight[0:frames]
            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))
        return x + torch.cat(pos_embs)


class MoonVision3dPatchEmbed(nn.Module):
    """``patch_size`` conv over pre-cut patches plus the divided-fixed position embedding."""

    def __init__(self, vision_cfg) -> None:
        super().__init__()
        out_dim = int(_resolve(vision_cfg, "vt_hidden_size", "hidden_size"))
        patch_size = _resolve(vision_cfg, "patch_size", default=14)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(patch_size) != 2:
            raise ValueError(f"patch_size must have 2 entries, got {patch_size}")
        self.patch_size = tuple(int(item) for item in patch_size)
        self.proj = nn.Conv2d(
            3,
            out_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bool(getattr(vision_cfg, "patch_embed_proj_bias", True)),
        )
        pos_emb_type = getattr(vision_cfg, "pos_emb_type", "divided_fixed")
        if pos_emb_type != "divided_fixed":
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")
        self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
            height=int(getattr(vision_cfg, "init_pos_emb_height", 64)),
            width=int(getattr(vision_cfg, "init_pos_emb_width", 64)),
            num_frames=int(getattr(vision_cfg, "init_pos_emb_time", 4)),
            dim=out_dim,
            interpolation_mode=getattr(vision_cfg, "pos_emb_interpolation_mode", "bilinear"),
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        return self.pos_emb(x, grid_thws)


class Rope2DPosEmbRepeated(nn.Module):
    """2D complex RoPE: half the channels encode the height axis, half the width axis."""

    def __init__(self, dim: int, max_height: int = 512, max_width: int = 512, theta_base: float = 10000):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4, got {dim}")
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self._cache: Optional[Tuple[int, int, torch.Tensor]] = None

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}"

    def _precompute_freqs_cis(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """``[height, width, dim//2]`` complex64 cis table for the requested grid."""
        flat_pos = torch.arange(0, height * width, dtype=torch.float32, device=device)
        x_pos = flat_pos % width
        y_pos = torch.div(flat_pos, width, rounding_mode="floor")
        dim_range = torch.arange(0, self.dim, 4)[: self.dim // 4].float().to(device)
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()
        y_freqs = torch.outer(y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
        return freqs_cis.reshape(height, width, -1)

    def get_freqs_cis(self, grid_thws: torch.Tensor, device: torch.device) -> torch.Tensor:
        shapes = [tuple(int(item) for item in row) for row in grid_thws.tolist()]
        for _, height, width in shapes:
            if not (1 <= height <= self.max_height and 1 <= width <= self.max_width):
                raise ValueError(f"grid {(height, width)} exceeds the {self.max_height}x{self.max_width} RoPE table")
        max_height = max(item[1] for item in shapes)
        max_width = max(item[2] for item in shapes)
        if self._cache is None or self._cache[0] < max_height or self._cache[1] < max_width:
            self._cache = (max_height, max_width, self._precompute_freqs_cis(max_height, max_width, device))
        table = self._cache[2]
        return torch.cat(
            [table[:height, :width].reshape(-1, self.dim // 2).repeat(frames, 1) for frames, height, width in shapes],
            dim=0,
        )


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``(..., heads, head_dim)`` q/k with ``(..., head_dim/2)`` complex64 cis."""
    for tensor in (xq, xk):
        if tensor.ndim != freqs_cis.ndim + 1 or tensor.shape[:-2] != freqs_cis.shape[:-1]:
            raise ValueError(f"rope shape mismatch: {tuple(tensor.shape)} vs {tuple(freqs_cis.shape)}")
        if tensor.shape[-1] != 2 * freqs_cis.shape[-1]:
            raise ValueError(f"rope last dim mismatch: {tuple(tensor.shape)} vs {tuple(freqs_cis.shape)}")
    cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MLP2(nn.Module):
    """Two-layer vision FFN (``fc0`` -> tanh-GELU -> ``fc1``)."""

    def __init__(self, dims: Sequence[int], activation: nn.Module, bias: bool = True):
        super().__init__()
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.activation(self.fc0(x)))


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    """Vision norms mirror upstream: ``nn.RMSNorm(dim)`` keeps ``eps=None`` semantics."""
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    raise NotImplementedError(f"Not support norm_type: {norm_type}")


class MoonViTEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        qkv_hidden_size: Optional[int] = None,
        norm_type: str = "layernorm",
        mlp_type: str = "mlp2",
        *,
        attn_bias: bool = False,
        linear_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.qkv_hidden_size = hidden_dim if qkv_hidden_size is None else qkv_hidden_size
        self.hidden_size_per_attention_head = self.qkv_hidden_size // self.num_heads
        self.norm0 = _make_norm(norm_type, hidden_dim)
        self.norm1 = _make_norm(norm_type, hidden_dim)
        if mlp_type != "mlp2":
            raise NotImplementedError(f"Not support mlp_type: {mlp_type}")
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], nn.GELU(approximate=_GELU_TANH), bias=linear_bias)
        self.wqkv = nn.Linear(hidden_dim, self.qkv_hidden_size * 3, bias=attn_bias)
        self.wo = nn.Linear(self.qkv_hidden_size, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xqkv = self.wqkv(x).view(*x.shape[:-1], 3, self.num_heads, self.hidden_size_per_attention_head)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)
        if rope_freqs_cis is not None:
            xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        scale = self.hidden_size_per_attention_head**-0.5
        outputs = []
        offsets = [int(item) for item in cu_seqlens.tolist()]
        for start, end in zip(offsets[:-1], offsets[1:]):
            if end <= start:
                continue
            q_i = xq[start:end].transpose(0, 1).float()  # [H, L, D]
            k_i = xk[start:end].transpose(0, 1).float()
            v_i = xv[start:end].transpose(0, 1).float()
            weights = torch.softmax(torch.matmul(q_i, k_i.transpose(-1, -2)) * scale, dim=-1)
            outputs.append(torch.matmul(weights, v_i).transpose(0, 1).to(xq.dtype))
        attn_out = torch.cat(outputs, dim=0) if outputs else xq.new_zeros(xq.shape)
        return self.wo(attn_out.flatten(start_dim=-2))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention_qkvpacked(self.norm0(hidden_states), cu_seqlens, rope_freqs_cis)
        return hidden_states + self.mlp(self.norm1(hidden_states))


class MoonViT3dEncoder(nn.Module):
    def __init__(self, vision_cfg) -> None:
        super().__init__()
        hidden_dim = int(_resolve(vision_cfg, "vt_hidden_size", "hidden_size"))
        num_layers = int(_resolve(vision_cfg, "vt_num_hidden_layers", "num_hidden_layers"))
        num_heads = int(_resolve(vision_cfg, "vt_num_attention_heads", "num_attention_heads"))
        mlp_dim = int(_resolve(vision_cfg, "vt_intermediate_size", "intermediate_size"))
        qkv_hidden_size = getattr(vision_cfg, "qkv_hidden_size", None)
        norm_type = getattr(vision_cfg, "norm_type", "layernorm")
        qkv_dim = hidden_dim if qkv_hidden_size is None else int(qkv_hidden_size)

        self.rope_2d = Rope2DPosEmbRepeated(qkv_dim // num_heads, 512, 512)
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    qkv_hidden_size=qkv_hidden_size,
                    norm_type=norm_type,
                    mlp_type=getattr(vision_cfg, "mlp_type", "mlp2"),
                    attn_bias=bool(getattr(vision_cfg, "attn_bias", False)),
                    linear_bias=bool(getattr(vision_cfg, "linear_bias", True)),
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = _make_norm(norm_type, hidden_dim)

    @staticmethod
    def build_cu_seqlens(grid_thws: torch.Tensor, device: torch.device) -> torch.Tensor:
        """``[0, n0, n0 + n1, ...]`` int32 offsets over the packed patch sequence."""
        lengths = grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]
        lengths = torch.cat((torch.zeros(1, dtype=lengths.dtype, device=lengths.device), lengths)).to(device)
        return lengths.cumsum(dim=0, dtype=torch.int32)

    def forward(self, hidden_states: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws, hidden_states.device)
        cu_seqlens = self.build_cu_seqlens(grid_thws, hidden_states.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rope_freqs_cis)
        return self.final_layernorm(hidden_states)


def tpool_patch_merger(
    x: torch.Tensor, grid_thws: torch.Tensor, merge_kernel_size: Tuple[int, int] = (2, 2)
) -> List[torch.Tensor]:
    """``sd2_tpool``: average over time, then group ``kh x kw`` spatial patches."""
    d_model = x.size(-1)
    kernel_height, kernel_width = int(merge_kernel_size[0]), int(merge_kernel_size[1])
    outputs = []
    pre_sum = 0
    for frames, height, width in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + frames * height * width]
        new_height, new_width = height // kernel_height, width // kernel_width
        reshaped = seq.view(frames, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        outputs.append(reshaped.reshape(new_height * new_width, kernel_height * kernel_width, -1))
        pre_sum += frames * height * width
    return outputs


class MoonViT3dVisionTower(nn.Module):
    """``vision_tower`` subtree: patch embed -> 27 blocks -> final norm -> patch merge."""

    def __init__(self, vision_cfg) -> None:
        super().__init__()
        self.merge_kernel_size = tuple(int(item) for item in getattr(vision_cfg, "merge_kernel_size", (2, 2)))
        self.patch_size = getattr(vision_cfg, "patch_size", 14)
        self.merge_type = getattr(vision_cfg, "merge_type", "sd2_tpool")
        self.patch_embed = MoonVision3dPatchEmbed(vision_cfg)
        self.encoder = MoonViT3dEncoder(vision_cfg)

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> List[torch.Tensor]:
        if grid_thws.ndim != 2 or grid_thws.size(1) != 3:
            raise ValueError(f"grid_thws must be [N, 3], got {tuple(grid_thws.shape)}")
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type != "sd2_tpool":
            raise NotImplementedError(f"Not support {self.merge_type}")
        return tpool_patch_merger(hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size)


class PatchMergerMLPV2(nn.Module):
    """``mm_projector`` of ``patchmergerv2``: 2-layer MLP + RMSNorm post-norm."""

    def __init__(self, vision_cfg) -> None:
        super().__init__()
        mm_hidden_size = int(_resolve(vision_cfg, "mm_hidden_size", "text_hidden_size"))
        text_hidden_size = int(_resolve(vision_cfg, "text_hidden_size", "mm_hidden_size"))
        kernel_height, kernel_width = getattr(vision_cfg, "merge_kernel_size", (2, 2))
        self.hidden_size = mm_hidden_size * (int(kernel_height) * int(kernel_width))
        eps = float(getattr(vision_cfg, "projector_ln_eps", 1e-5))
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_size, text_hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(text_hidden_size, eps=eps)

    def forward(self, x) -> List[torch.Tensor]:
        if isinstance(x, (list, tuple)):
            lengths = [item.shape[0] for item in x]
            flat = torch.cat([item.reshape(item.shape[0], -1) for item in x], dim=0)
            return list(torch.split(self.post_norm(self.proj(flat)), lengths, dim=0))
        batch = x.shape[0]
        return self.post_norm(self.proj(x.reshape(batch, -1, self.hidden_size)))


def build_mm_projector(vision_cfg) -> nn.Module:
    projector_type = getattr(vision_cfg, "mm_projector_type", "patchmergerv2")
    if projector_type == "patchmergerv2":
        return PatchMergerMLPV2(vision_cfg)
    raise NotImplementedError(f"Unsupported mm_projector_type: {projector_type}")


def merge_input_ids_with_image_features(
    image_features: Sequence[torch.Tensor],
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    media_placeholder_token_id: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter projected media features into the text embedding stream.

    Mirrors ``KimiK3ForConditionalGeneration._merge_input_ids_with_image_features``: one
    ``<|media_placeholder|>`` token expands to ``image_features[i].shape[0]`` positions.
    Returns ``(embeddings, attention_mask, position_ids)``.
    """
    embed_dim = image_features[0].shape[-1]
    lengths = [item.shape[0] for item in image_features]
    features = torch.cat(image_features, dim=0)

    batch_size, sequence_length = input_ids.shape
    device = inputs_embeds.device
    occupation = torch.ones_like(input_ids.flatten())
    occupation[input_ids.flatten() == media_placeholder_token_id] = torch.tensor(
        lengths, dtype=torch.long, device=input_ids.device
    )
    occupation = occupation.reshape(input_ids.shape)
    max_embed_dim = int(occupation.sum(-1).max().item())

    left_padding = not bool(torch.sum(input_ids[:, -1] == pad_token_id))
    batch_indices, non_media_indices = torch.where(input_ids != media_placeholder_token_id)
    new_token_positions = torch.cumsum(occupation, -1) - 1
    nb_media_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions = new_token_positions + nb_media_pad[:, None]
    text_to_overwrite = new_token_positions[batch_indices, non_media_indices]

    final_embedding = torch.zeros(batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=device)
    final_attention_mask = torch.zeros(batch_size, max_embed_dim, dtype=attention_mask.dtype, device=device)
    batch_indices = batch_indices.to(device)
    non_media_indices = non_media_indices.to(device)
    text_to_overwrite = text_to_overwrite.to(device)
    attention_mask = attention_mask.to(device)

    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_media_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_media_indices]

    media_to_overwrite = torch.full((batch_size, max_embed_dim), True, dtype=torch.bool, device=device)
    media_to_overwrite[batch_indices, text_to_overwrite] = False
    media_to_overwrite &= media_to_overwrite.cumsum(-1) - 1 >= nb_media_pad[:, None].to(device)
    if media_to_overwrite.sum() != features.shape[:-1].numel():
        raise ValueError(
            f"Number of media placeholders ({int(media_to_overwrite.sum())}) does not match "
            f"the number of media features ({features.shape[:-1].numel()})."
        )
    final_embedding[media_to_overwrite] = features.reshape(-1, embed_dim).to(device)
    final_attention_mask |= media_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(final_attention_mask == 0, 1)

    pad_batch, pad_indices = torch.where(input_ids == pad_token_id)
    final_embedding[pad_batch, new_token_positions[pad_batch, pad_indices]] = 0
    return final_embedding, final_attention_mask, position_ids


class KimiK31ForConditionalGeneration(nn.Module):
    """Kimi-K3.1 graph: ``vision_tower`` -> ``mm_projector`` -> ``language_model``.

    The quantization pipeline walks the three roots in this order (vision as a whole, the
    projector as a whole, then the text decoder layer by layer), matching the released
    ``KimiK31ForConditionalGeneration``.  The vision stack is byte-compatible with
    Kimi-K3's, which is why the ``vision_config`` of both configs is identical.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.vision_tower = MoonViT3dVisionTower(config.vision_config)
        self.mm_projector = build_mm_projector(config.vision_config)
        self.language_model = KimiLinearForCausalLM(config.text_config)

    @property
    def text_config(self) -> Any:
        return self.config.text_config

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.lm_head

    def extract_image_features(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> List[torch.Tensor]:
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        return self.vision_tower(pixel_values.to(target_dtype), grid_thws)

    def merge_input_ids_with_image_features(
        self,
        image_features: Sequence[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return merge_input_ids_with_image_features(
            image_features=image_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            media_placeholder_token_id=int(self.config.media_placeholder_token_id),
            pad_token_id=int(self.config.pad_token_id),
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thws: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Full multimodal teacher-forced forward returning ``[B, S, vocab]`` logits."""
        del kwargs
        if inputs_embeds is None:
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        has_media = pixel_values is not None and pixel_values.numel() > 0 and input_ids.shape[1] > 1
        if has_media:
            image_features = self.mm_projector(self.extract_image_features(pixel_values, grid_thws))
            inputs_embeds = inputs_embeds.to(image_features[0].dtype)
            inputs_embeds, attention_mask, _ = self.merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=(attention_mask if attention_mask is not None else torch.ones_like(input_ids)),
            )
        hidden_states = self.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = self.language_model.lm_head(hidden_states)
        if labels is None:
            return logits
        shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1]).float()
        shift_labels = labels[..., 1:].reshape(-1).to(logits.device)
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
