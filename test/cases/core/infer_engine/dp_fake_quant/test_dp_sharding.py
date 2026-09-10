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

msmodelslim/core/infer_engine/dp_fake_quant/dp_sharding.py 的单元测试。
"""

import pytest

from msmodelslim.core.context import ContextManager, LocalDictContext
from msmodelslim.core.infer_engine.dp_fake_quant.dp_sharding import (
    FAKE_QUANT_INFER_NAMESPACE,
    global_indices_for_rank,
    merge_inference_results,
    merge_partials_into_context,
    partial_key,
    shard_samples_by_rank,
)
from msmodelslim.core.infer_engine.interface import InferenceResult
from msmodelslim.utils.exception import SchemaValidateError


class TestDpShardingFunctions:
    """对应 dp_sharding 中的纯函数。"""

    def test_partial_key_returns_rank_suffix_when_rank_given(self):
        assert partial_key(3) == "partial_3"

    def test_shard_samples_by_rank_round_robins_when_divisible(self):
        assert shard_samples_by_rank([0, 1, 2, 3], rank=1, world_size=2) == [1, 3]

    def test_shard_samples_by_rank_pads_from_start_when_not_divisible(self):
        assert shard_samples_by_rank([0, 1, 2], rank=0, world_size=2) == [0, 2]
        assert shard_samples_by_rank([0, 1, 2], rank=1, world_size=2) == [1, 0]

    def test_shard_samples_by_rank_raises_when_world_size_zero(self):
        with pytest.raises(ValueError):
            shard_samples_by_rank([0], rank=0, world_size=0)

    def test_shard_samples_by_rank_raises_when_rank_out_of_range(self):
        with pytest.raises(ValueError):
            shard_samples_by_rank([0], rank=2, world_size=2)

    def test_global_indices_for_rank_steps_by_world_size(self):
        assert global_indices_for_rank(1, num_samples=5, world_size=2) == [1, 3]

    def test_merge_inference_results_orders_by_global_when_all_ranks_present(self):
        partials = {
            0: InferenceResult(generated_token_ids=[[0], [2]], generated_texts=["a", "c"]),
            1: InferenceResult(generated_token_ids=[[1], [3]], generated_texts=["b", "d"]),
        }

        merged = merge_inference_results(partials, num_samples=4, world_size=2)

        assert merged.generated_token_ids == [[0], [1], [2], [3]]
        assert merged.generated_texts == ["a", "b", "c", "d"]

    def test_merge_inference_results_keeps_blank_when_rank_missing(self):
        partials = {0: InferenceResult(generated_token_ids=[[0]], generated_texts=["a"])}

        merged = merge_inference_results(partials, num_samples=2, world_size=2)

        assert merged.generated_texts == ["a", ""]

    def test_merge_inference_results_skips_short_partial_when_shorter(self):
        partials = {
            0: InferenceResult(generated_token_ids=[[0]], generated_texts=["a"]),
            1: InferenceResult(generated_token_ids=[], generated_texts=[]),
        }

        merged = merge_inference_results(partials, num_samples=3, world_size=2)

        assert merged.generated_token_ids == [[0], [], []]


class TestMergePartialsIntoContext:
    """对应 merge_partials_into_context。"""

    def test_merge_returns_global_order_when_all_partials_stored(self):
        ctx = LocalDictContext()
        with ContextManager(ctx):
            state = ctx[FAKE_QUANT_INFER_NAMESPACE].state
            state["partial_0"] = {"generated_token_ids": [[0], [2]], "generated_texts": ["a", "c"]}
            state["partial_1"] = {"generated_token_ids": [[1]], "generated_texts": ["b"]}

            merged = merge_partials_into_context(num_samples=3, world_size=2)

        assert merged.generated_token_ids == [[0], [1], [2]]
        assert merged.generated_texts == ["a", "b", "c"]

    def test_merge_raises_when_partial_missing(self):
        ctx = LocalDictContext()
        with ContextManager(ctx):
            with pytest.raises(SchemaValidateError):
                merge_partials_into_context(num_samples=2, world_size=2)

    def test_merge_raises_when_no_active_context(self):
        with pytest.raises(SchemaValidateError):
            merge_partials_into_context(num_samples=1, world_size=1)
