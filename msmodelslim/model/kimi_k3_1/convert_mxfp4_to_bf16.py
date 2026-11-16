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

# MXFP4 (``mxfp4-pack-quantized``) -> BF16 dequantization for Kimi-K3.1.
#
# Kimi-K3.1 ships the *same* ``quantization_config`` as Kimi-K3: routed experts
# (``block_sparse_moe.experts.*.w{1,2,3}``) and the latent MoE projections
# (``routed_expert_{down,up}_proj``) are stored as
#
#   * ``weight_packed``: uint8 ``[N, K/2]``   -- two FP4 E2M1 nibbles per byte
#   * ``weight_scale``:  uint8 ``[N, K/32]``  -- E8M0 shared exponents (bias 127)
#
# while ``re:.*self_attn.*``, ``re:.*shared_experts.*``, ``re:.*mlp\.(gate|up|down)_proj.*``,
# ``lm_head``, ``vision_tower`` and ``mm_projector`` are excluded (plain BF16 on disk).
# The decoder is therefore format-level shared with Kimi-K3 and imported from there instead
# of being copied: the parameters it addresses are looked up by checkpoint key, which are
# identical for both models (``language_model.model.layers.{i}.…``).

from msmodelslim.model.kimi_k3.convert_mxfp4_to_bf16 import (  # noqa: F401
    dequant_mxfp4_ct,
    dequant_subtree_mxfp4_to_bf16,
    e8m0_uint8_to_scale,
    get_full_weight_map,
    get_mxfp4_weight_map,
    load_tensor_by_full_name,
    replace_compressed_linear_with_bf16,
    unpack_fp4_from_uint8,
)

__all__ = [
    "dequant_mxfp4_ct",
    "dequant_subtree_mxfp4_to_bf16",
    "e8m0_uint8_to_scale",
    "get_full_weight_map",
    "get_mxfp4_weight_map",
    "load_tensor_by_full_name",
    "replace_compressed_linear_with_bf16",
    "unpack_fp4_from_uint8",
]
