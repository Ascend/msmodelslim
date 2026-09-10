# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn

from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import get_logger

from ..interface import FakeQuantInferenceInterface, InferenceResult


class _LoggingTextStreamer:
    """Stream generated text from ``model.generate`` into the msmodelslim logger.

    Wraps HF ``TextStreamer`` so incremental decoded text is routed to ``get_logger()``
    instead of stdout. Only generated tokens are streamed (``skip_prompt=True``).
    """

    def __init__(
        self,
        tokenizer,
        sample_idx: int,
        skip_special_tokens: bool = True,
    ) -> None:
        from transformers.generation.streamers import TextStreamer

        self._sample_idx = sample_idx
        self._acc = ""
        self._streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=skip_special_tokens,
        )
        self._streamer.on_finalized_text = self.on_finalized_text  # route output to logger

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        # Called by TextStreamer with each newly decoded chunk; log the full text-so-far.
        # The final text is not special-cased here: run() logs generated token ids separately.
        if text:
            self._acc += text
            get_logger().info(
                "Fake-quant stream sample[%d] text=%r",
                self._sample_idx + 1,
                self._acc,
            )

    # ---- BaseStreamer protocol: delegate to the wrapped TextStreamer.
    def put(self, value) -> None:
        self._streamer.put(value)

    def put_value(self, value) -> None:
        self._streamer.put_value(value)

    def end(self) -> None:
        self._streamer.end()


class PrefillLoop:
    """Drive generation via native ``model.generate`` and collect the new tokens.

    Layer-wise weight staging is already installed on the model by WeightManager, so a plain
    ``model.generate(**sample)`` is all the scheduling required. ``use_cache=False`` makes each
    decode step re-run the full sequence (multi-prefill, no KV cache on device). Samples run
    sequentially; batching with padding is a future optimization (keeps VLM samples correct).
    """

    def __init__(self, adapter: FakeQuantInferenceInterface, model: nn.Module) -> None:
        self.adapter = adapter
        self.model = model

    def run(
        self,
        inputs: List[Any],
        max_new_tokens: int,
        disable_eos: bool = False,
        global_sample_indices: Optional[List[int]] = None,
    ) -> InferenceResult:
        """Run one generate call per sample and return the collected new tokens.

        ``global_sample_indices`` maps this rank's shard-local samples to global order for
        logging (padded duplicates are marked ``[pad]``); single-card callers omit it.
        ``disable_eos`` makes ``model.generate`` not stop at EOS, required for multi-card DP
        so all ranks run the same number of forwards and MoE collectives stay synchronized;
        outputs are then truncated at the first EOS token.
        """
        if max_new_tokens < 1:
            raise InvalidModelError(
                f"max_new_tokens must be >= 1, got {max_new_tokens}",
                action="Please set InferenceConfig.max_new_tokens to a positive integer.",
            )

        num_total = len(global_sample_indices) if global_sample_indices is not None else len(inputs)
        t0 = time.perf_counter()
        generated_token_ids: List[List[int]] = [[] for _ in inputs]
        for local_i, sample in enumerate(inputs):
            if global_sample_indices is not None:
                is_pad = local_i >= len(global_sample_indices)
                global_i = global_sample_indices[local_i] if not is_pad else local_i
            else:
                is_pad = False
                global_i = local_i
            get_logger().info(
                "Fake-quant generate sample %d/%d (max_new_tokens=%d, use_cache=False, disable_eos=%s)",
                global_i + 1,
                num_total,
                max_new_tokens,
                disable_eos,
            )
            if global_sample_indices is not None:
                get_logger().debug(
                    "  local sample %d/%d on rank=%s%s",
                    local_i + 1,
                    len(inputs),
                    dist.get_rank() if dist.is_initialized() else "?",
                    " [pad]" if is_pad else "",
                )
            new_ids = self._generate_one(
                sample, max_new_tokens=max_new_tokens, sample_idx=global_i, disable_eos=disable_eos
            )
            generated_token_ids[local_i] = list(new_ids)
            get_logger().debug(
                "Fake-quant generate sample %d done generated=%s",
                global_i + 1,
                generated_token_ids[local_i],
            )

        generated_texts = self._decode_generated(generated_token_ids)
        get_logger().info("Fake-quant generated %d sample(s) in %.1fs", num_total, time.perf_counter() - t0)
        return InferenceResult(
            generated_token_ids=generated_token_ids,
            generated_texts=generated_texts,
        )

    # ------------------------------------------------------------------ private

    @torch.no_grad()
    def _generate_one(self, sample: Any, max_new_tokens: int, sample_idx: int, disable_eos: bool = False) -> List[int]:
        """Run ``model.generate`` for one sample and return the newly generated token ids.

        ``use_cache=False`` preserves multi-prefill semantics; greedy decoding
        (``do_sample=False``, ``num_beams=1``) matches the previous per-step ``argmax``.
        When the adapter exposes a ``tokenizer``, a ``_LoggingTextStreamer`` is attached to
        stream text into the log as it is produced. When ``disable_eos`` is set, generation
        never stops at EOS (all DP ranks stay aligned); the output is truncated afterwards.
        """
        gen_kwargs = self._to_generate_kwargs(sample)
        input_ids = gen_kwargs["input_ids"]
        input_len = int(input_ids.shape[1])
        gen_kwargs.update(
            max_new_tokens=max_new_tokens,
            use_cache=False,
            do_sample=False,
            num_beams=1,
        )
        if disable_eos:
            gen_kwargs["eos_token_id"] = None
        pad_id = self._pad_token_id()
        if pad_id is not None:
            gen_kwargs.setdefault("pad_token_id", pad_id)

        streamer = self._build_streamer(sample_idx)
        if streamer is not None:
            gen_kwargs["streamer"] = streamer

        outputs = self.model.generate(**gen_kwargs)
        sequences = getattr(outputs, "sequences", outputs)
        new_ids = sequences[0, input_len:]  # B=1: slice off the original prompt
        new_ids = new_ids.detach().cpu().tolist()
        if disable_eos:
            new_ids = self._truncate_at_eos(new_ids)
        return new_ids

    def _build_streamer(self, sample_idx: int):
        """Build a logger-backed streamer when the adapter has a tokenizer; else None."""
        tokenizer = getattr(self.adapter, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            return _LoggingTextStreamer(tokenizer, sample_idx=sample_idx)
        except ImportError as exc:
            get_logger().warning(
                "TextStreamer unavailable (%s); streaming disabled for sample[%d]",
                exc,
                sample_idx,
            )
            return None

    def _to_generate_kwargs(self, sample: Any) -> Dict[str, Any]:
        """Convert a ``handle_dataset`` sample into ``model.generate`` kwargs.

        Text samples are ``[input_ids, attention_mask, ...]`` lists; VLM samples are
        processor-ready dicts. Both map to generate kwargs; extra dict keys are forwarded
        to ``forward`` by generate.
        """
        if isinstance(sample, dict):
            if "input_ids" not in sample:
                raise InvalidModelError(
                    f"generate sample dict missing 'input_ids', keys={list(sample.keys())}",
                    action="Please ensure adapter.handle_dataset returns dicts with input_ids.",
                )
            return {k: v for k, v in sample.items() if v is not None}
        if isinstance(sample, (list, tuple)):
            if len(sample) < 2 or not isinstance(sample[0], torch.Tensor):
                raise InvalidModelError(
                    f"generate sample list expects [input_ids, attention_mask], got len={len(sample)}",
                    action="Override FakeQuantInferenceInterface for this input shape.",
                )
            return {"input_ids": sample[0], "attention_mask": sample[1]}
        if isinstance(sample, torch.Tensor):
            return {"input_ids": sample}
        raise InvalidModelError(
            f"Unsupported generate sample type {type(sample).__name__}",
            action="Please pass handle_dataset outputs unchanged into generate.",
        )

    def _pad_token_id(self) -> Optional[int]:
        """Resolve a pad_token_id for generate (tokenizer first, then model config)."""
        tokenizer = getattr(self.adapter, "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
        cfg = getattr(self.model, "config", None)
        return getattr(cfg, "pad_token_id", None)

    def _decode_generated(self, generated_token_ids: List[List[int]]) -> List[str]:
        tokenizer = getattr(self.adapter, "tokenizer", None)
        texts: List[str] = []
        for ids in generated_token_ids:
            if not ids:
                texts.append("")
            elif tokenizer is None:
                texts.append(str(ids))
            else:
                texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        return texts

    def _truncate_at_eos(self, token_ids: List[int]) -> List[int]:
        """Cut a token list at the first EOS (used when ``disable_eos=True``).

        Recover the true output length after ``eos_token_id=None`` ran generation to
        ``max_new_tokens``; returns the full list when no EOS was produced.
        """
        eos_token_ids = self._resolve_eos_token_ids()
        if not eos_token_ids:
            return token_ids
        for i, tid in enumerate(token_ids):
            if tid in eos_token_ids:
                return token_ids[:i]
        return token_ids

    def _resolve_eos_token_ids(self) -> set:
        """Resolve the set of EOS token ids from tokenizer or model config."""
        result: set = set()
        tokenizer = getattr(self.adapter, "tokenizer", None)
        if tokenizer is not None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                if isinstance(eos_id, (list, tuple)):
                    result.update(int(x) for x in eos_id)
                else:
                    result.add(int(eos_id))
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            eos_id = getattr(cfg, "eos_token_id", None)
            if eos_id is not None:
                if isinstance(eos_id, (list, tuple)):
                    result.update(int(x) for x in eos_id)
                else:
                    result.add(int(eos_id))
        return result
