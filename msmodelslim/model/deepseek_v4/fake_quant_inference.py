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

DeepSeek-V4 shell for the existing layer-wise fake-quant evaluator.

Generation deliberately follows the engine's use_cache=False contract. Each
new token reruns the complete prefix. This adapter does not add an
incremental decoding implementation.
"""

from copy import deepcopy
from typing import List, Optional, Union

import torch

from msmodelslim.utils.exception import UnsupportedError
from .model import Block, Compressor, RMSNorm, Transformer


_DEFAULT_EOS = object()


class DeepSeekV4DecoderLayer(Block):
    """Keep checkpoint names while enabling the evaluator's layer staging."""


class _InferenceRMSNorm(RMSNorm):
    """Preserve the caller dtype around the evaluator's shared alignment hooks."""

    def __init__(self, norm):
        super().__init__(norm.dim, norm.eps)
        self.weight = norm.weight
        self._input_dtype: torch.dtype = norm.weight.dtype
        # Registered before WeightManager's hooks, so capture the original dtype.
        self.register_forward_pre_hook(self._capture_input_dtype, with_kwargs=True)

    def _capture_input_dtype(self, module, args, kwargs):
        self._input_dtype = (args[0] if args else kwargs["x"]).dtype

    def forward(self, x):
        return super().forward(x).to(self._input_dtype)


class DeepSeekV4FakeQuantModel(Transformer):
    def __init__(self, args):
        super().__init__(args)
        # The evaluator recognizes layers by the DecoderLayer suffix. Keep the
        # original Block instances, including their parameter and buffer names.
        for layer in self.layers:
            layer.__class__ = DeepSeekV4DecoderLayer
        self.norm = _InferenceRMSNorm(self.norm)
        self.eos_token_id: Optional[Union[int, List[int]]] = None

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=1,
        use_cache=False,
        do_sample=False,
        num_beams=1,
        eos_token_id=_DEFAULT_EOS,
        pad_token_id=None,
        streamer=None,
        **kwargs,
    ):
        if kwargs or use_cache or do_sample or num_beams != 1:
            raise UnsupportedError(
                "DeepSeek-V4 fake-quant dialogue supports greedy multi-prefill only "
                "(use_cache=False, do_sample=False, num_beams=1)",
                action="Use the documented msmodelslim eval command.",
            )
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("DeepSeek-V4 fake-quant dialogue expects one non-empty prompt")
        if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
            raise ValueError("Padded prompts are not supported; evaluate each prompt separately")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if input_ids.shape[1] + max_new_tokens > self.max_seq_len:
            raise ValueError(f"Prompt plus generation exceeds max_seq_len={self.max_seq_len}")
        if eos_token_id is _DEFAULT_EOS:
            eos_token_id = self.eos_token_id
        stop_ids = (
            set()
            if eos_token_id is None
            else set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        )
        sequences = input_ids
        if streamer is not None:
            streamer.put(sequences.detach().cpu())
        try:
            for _ in range(max_new_tokens):
                # Layer offload replaces the parent's runtime cache tensors. Clear
                # compressor views so the next prefill binds those new buffers.
                for module in self.modules():
                    if isinstance(module, Compressor):
                        module.kv_cache = None
                logits = self(sequences, start_pos=0)
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Non-finite logits during DeepSeek-V4 fake-quant dialogue")
                token = logits.argmax(dim=-1, keepdim=True).to(sequences.device)
                sequences = torch.cat((sequences, token), dim=1)
                if streamer is not None:
                    streamer.put(token.detach().cpu())
                if int(token.item()) in stop_ids:
                    break
        finally:
            if streamer is not None:
                streamer.end()
        return sequences


def build_meta_model(config):
    """Build full-depth meta parameters and CPU runtime buffers, without MTP.

    The standard autoregressive dialogue uses the main model; DSpark speculative
    decoding is not enabled by the generic evaluator. Exported MTP FA3 metadata
    may therefore be absent from this inference shell.
    """
    from accelerate import init_empty_weights

    args = deepcopy(config)
    args.max_batch_size = 1
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(include_buffers=False):
            model = DeepSeekV4FakeQuantModel(args)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.eval()
    return model
