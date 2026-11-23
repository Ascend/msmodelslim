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

msModelSlim adapter for Qwen3.8-Flash-Next (qwen4_exp architecture).

The Qwen3.8-Flash-Next is a multimodal understanding model (VLM) whose top-level
architecture is `Qwen4ExpForConditionalGeneration`. For msModelSlim we treat only
the text/language_model path:

    model.model.language_model.layers[i] -> Qwen4ExpTextDecoderLayer

Notable structural traits (kept transparent to the quant pipeline via the default
generic visit/forward helpers):

- Hybrid attention: alternating linear_attention (Gated DeltaNet) and full_attention
  (Qwen Sparse Attention with indexer), per `text_config.layer_types`.
- MoE: 3D fused experts (`mlp.experts.{gate_up_proj,down_proj}`), plus shared expert.
- Hyper-connection (gated residual) with `hc_count=4` (`hc_count * hidden_size` flow).
- Per-Layer Embedding (PLE / n-gram) at the configured `ple_layer_ids` (optional).
- MTP: `mtp.layers.0` is exposed but `_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]`
  in `Qwen4ExpPreTrainedModel` so the base pretrained load ignores MTP by default.

Basic adapter scope (this file):
- Text-only calibration path (LLM family) via the text tokenizer.
- `_load_model` uses `AutoModelForImageTextToText` because the top-level
  architecture is `Qwen4ExpForConditionalGeneration`.
- `generate_model_visit` / `generate_model_forward` rely on the default generic
  helpers that match `decoderlayer` in the module class name (works for
  `Qwen4ExpTextDecoderLayer`).
"""

from .loader import Qwen4ExpFlashNextAdapterLoader
from .model_adapter import Qwen4ExpFlashNextModelAdapter

__all__ = ["Qwen4ExpFlashNextAdapterLoader", "Qwen4ExpFlashNextModelAdapter"]
