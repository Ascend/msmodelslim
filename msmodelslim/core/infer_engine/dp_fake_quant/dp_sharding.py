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

import math
from typing import Dict, List, Sequence, TypeVar

from msmodelslim.core.context import get_current_context
from msmodelslim.utils.exception import SchemaValidateError

from ..interface import InferenceResult

T = TypeVar("T")

FAKE_QUANT_INFER_NAMESPACE = "fake_quant_infer"


def partial_key(rank: int) -> str:
    """Return the shared-context key under which DP worker ``rank`` stores its partial result."""
    return f"partial_{rank}"


def shard_samples_by_rank(samples: Sequence[T], rank: int, world_size: int) -> List[T]:
    """Return samples assigned to ``rank``.

    Uses the same round-robin rule as ``torch.utils.data.DistributedSampler`` with
    ``drop_last=False``: when ``len(samples)`` is not divisible by ``world_size``, the list
    is padded by duplicating from the beginning so every rank gets at least one sample.
    Padded entries are discarded during merge (see ``global_indices_for_rank``).
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    num_samples = len(samples)
    num_per_rank = math.ceil(num_samples / world_size)
    total_size = num_per_rank * world_size
    padded = list(samples) + list(samples[: total_size - num_samples])
    return [padded[i] for i in range(rank, total_size, world_size)]


def global_indices_for_rank(rank: int, num_samples: int, world_size: int) -> List[int]:
    """Global sample indices owned by ``rank`` (``range(rank, num_samples, world_size)``)."""
    return list(range(rank, num_samples, world_size))


def merge_inference_results(
    partial_by_rank: Dict[int, InferenceResult],
    num_samples: int,
    world_size: int,
) -> InferenceResult:
    """Merge per-rank ``InferenceResult`` lists back to global sample order."""
    merged_token_ids: List[List[int]] = [[] for _ in range(num_samples)]
    merged_texts: List[str] = [""] * num_samples

    for rank in range(world_size):
        partial = partial_by_rank.get(rank)
        if partial is None:
            continue
        global_indices = global_indices_for_rank(rank, num_samples, world_size)
        for local_i, global_i in enumerate(global_indices):
            if local_i < len(partial.generated_token_ids):
                merged_token_ids[global_i] = list(partial.generated_token_ids[local_i])
            if local_i < len(partial.generated_texts):
                merged_texts[global_i] = partial.generated_texts[local_i]

    return InferenceResult(
        generated_token_ids=merged_token_ids,
        generated_texts=merged_texts,
    )


def _infer_ns_state():
    ctx = get_current_context()
    if ctx is None:
        raise SchemaValidateError(
            "Fake-quant inference requires an active context",
            action="Ensure ContextManager wraps engine.run with the shared context (see DP runner).",
        )
    return ctx[FAKE_QUANT_INFER_NAMESPACE].state  # pylint: disable=unsubscriptable-object


def merge_partials_into_context(num_samples: int, world_size: int) -> InferenceResult:
    """Merge DP ``partial_{rank}`` dumps back to global sample order on the parent process."""
    state = _infer_ns_state()
    partial_by_rank: Dict[int, InferenceResult] = {}
    for rank in range(world_size):
        key = partial_key(rank)
        if key not in state:
            raise SchemaValidateError(
                f"Fake-quant context is missing partial result for rank {rank}",
                action="Ensure every DP worker stored its local inference output.",
            )
        partial_by_rank[rank] = InferenceResult.model_validate(state[key])
    return merge_inference_results(partial_by_rank, num_samples, world_size)
