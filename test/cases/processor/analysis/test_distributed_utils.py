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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.processor.analysis.distributed_utils import (
    dedupe_layer_scores_keep_max,
    merge_layer_scores_across_ranks,
    publish_layer_analysis_result,
    read_layer_analysis_result,
    write_layer_analysis_result,
)


class TestDedupeLayerScoresKeepMax:
    def test_keeps_max_score_for_duplicate_names(self):
        scores = [
            {"name": "visual", "score": 0.09},
            {"name": "audio_tower", "score": 0.0},
            {"name": "visual", "score": 0.0},
        ]
        assert dedupe_layer_scores_keep_max(scores) == [
            {"name": "visual", "score": 0.09},
            {"name": "audio_tower", "score": 0.0},
        ]

    def test_preserves_order_when_no_duplicates(self):
        scores = [{"name": "a", "score": 1.0}, {"name": "b", "score": 2.0}]
        assert dedupe_layer_scores_keep_max(scores) == scores

    def test_skips_invalid_items_without_name_or_score(self):
        scores = [
            {"name": "a", "score": 1.0},
            {"name": None, "score": 2.0},
            {"name": "b", "score": None},
            {"score": 3.0},
        ]
        assert dedupe_layer_scores_keep_max(scores) == [{"name": "a", "score": 1.0}]


class TestMergeLayerScoresAcrossRanks:
    def test_merge_returns_local_when_dist_not_initialized(self):
        local = [{"name": "a", "score": 1.0}, {"name": "b", "score": 2.0}]
        with patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            assert merge_layer_scores_across_ranks(local) == local

    def test_merge_averages_scores_by_name_across_ranks(self):
        local = [{"name": "layer.0", "score": 2.0}]
        with patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 2

            def _all_gather_object(out_list, obj):
                out_list[0] = [{"name": "layer.0", "score": 2.0}, {"name": "layer.1", "score": 4.0}]
                out_list[1] = [{"name": "layer.0", "score": 4.0}]

            mock_dist.all_gather_object.side_effect = _all_gather_object
            merged = merge_layer_scores_across_ranks(local)

        assert merged == [
            {"name": "layer.0", "score": 3.0},
            {"name": "layer.1", "score": 4.0},
        ]

    def test_merge_skips_empty_rank_and_invalid_items(self):
        local = [{"name": "layer.0", "score": 2.0}]
        with patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 2

            def _all_gather_object(out_list, obj):
                out_list[0] = [{"name": None, "score": 1.0}, {"name": "layer.0", "score": 2.0}]
                out_list[1] = []

            mock_dist.all_gather_object.side_effect = _all_gather_object
            merged = merge_layer_scores_across_ranks(local)

        assert merged == [{"name": "layer.0", "score": 2.0}]


class TestPublishLayerAnalysisResult:
    def test_publish_dedupes_before_write_when_single_process(self):
        mock_ctx = MagicMock()
        mock_ns = MagicMock()
        mock_ns.state = {}
        mock_ns.debug = {}
        mock_ctx.__getitem__.return_value = mock_ns

        with (
            patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist,
            patch(
                "msmodelslim.processor.analysis.distributed_utils.get_current_context",
                return_value=mock_ctx,
            ),
        ):
            mock_dist.is_initialized.return_value = False
            scores = publish_layer_analysis_result(
                [
                    {"name": "visual", "score": 0.09},
                    {"name": "visual", "score": 0.0},
                ],
                "mse_layer_wise",
                quant_modules=["*"],
            )

        assert scores == [{"name": "visual", "score": 0.09}]
        assert mock_ns.state["layer_scores"] == scores
        assert mock_ns.debug["layer_scores"] == scores

    def test_publish_writes_on_rank0_under_dp(self):
        mock_ctx = MagicMock()
        mock_ns = MagicMock()
        mock_ns.state = {}
        mock_ns.debug = {}
        mock_ctx.__getitem__.return_value = mock_ns

        with (
            patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist,
            patch(
                "msmodelslim.processor.analysis.distributed_utils.get_current_context",
                return_value=mock_ctx,
            ),
            patch(
                "msmodelslim.processor.analysis.distributed_utils.merge_layer_scores_across_ranks",
                return_value=[{"name": "a", "score": 1.5}],
            ),
        ):
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 2
            mock_dist.get_rank.return_value = 0
            scores = publish_layer_analysis_result(
                [{"name": "a", "score": 1.0}],
                "kurtosis",
                patterns=["*"],
            )

        assert scores == [{"name": "a", "score": 1.5}]
        assert mock_ns.state["layer_scores"] == scores
        assert mock_ns.state["method"] == "kurtosis"
        assert mock_ns.debug["layer_scores"] == scores
        assert mock_ns.debug["method"] == "kurtosis"
        assert mock_ns.debug["patterns"] == ["*"]
        assert not mock_dist.barrier.called

    def test_publish_skips_write_on_non_rank0(self):
        mock_ctx = MagicMock()
        mock_getitem = MagicMock()
        mock_ctx.__getitem__ = mock_getitem
        with (
            patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist,
            patch(
                "msmodelslim.processor.analysis.distributed_utils.get_current_context",
                return_value=mock_ctx,
            ),
            patch(
                "msmodelslim.processor.analysis.distributed_utils.merge_layer_scores_across_ranks",
                return_value=[{"name": "a", "score": 1.5}],
            ),
        ):
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 2
            mock_dist.get_rank.return_value = 1
            scores = publish_layer_analysis_result([{"name": "a", "score": 1.0}], "kurtosis")

        assert scores == [{"name": "a", "score": 1.5}]
        mock_getitem.assert_not_called()


class TestWriteLayerAnalysisResult:
    def test_write_preserves_extra_fields(self):
        mock_ctx = MagicMock()
        mock_ns = MagicMock()
        mock_ns.state = {}
        mock_ns.debug = {}
        mock_ctx.__getitem__.return_value = mock_ns
        scores = [{"name": "a", "score": 1.0, "induction_heads": [0, 1]}]

        with (
            patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist,
            patch(
                "msmodelslim.processor.analysis.distributed_utils.get_current_context",
                return_value=mock_ctx,
            ),
        ):
            mock_dist.is_initialized.return_value = False
            out = write_layer_analysis_result(scores, "ra_compress", patterns=["*"])

        assert out == scores
        assert mock_ns.state["layer_scores"] == scores
        assert mock_ns.state["layer_scores"][0]["induction_heads"] == [0, 1]

    def test_write_includes_quant_modules(self):
        mock_ctx = MagicMock()
        mock_ns = MagicMock()
        mock_ns.state = {}
        mock_ns.debug = {}
        mock_ctx.__getitem__.return_value = mock_ns
        scores = [{"name": "a", "score": 1.0}]

        with (
            patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist,
            patch(
                "msmodelslim.processor.analysis.distributed_utils.get_current_context",
                return_value=mock_ctx,
            ),
        ):
            mock_dist.is_initialized.return_value = False
            write_layer_analysis_result(scores, "mse_model_wise", quant_modules=["*mlp*"])

        assert mock_ns.state["quant_modules"] == ["*mlp*"]
        assert mock_ns.debug["quant_modules"] == ["*mlp*"]

    def test_write_returns_input_when_context_missing(self):
        with (
            patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist,
            patch(
                "msmodelslim.processor.analysis.distributed_utils.get_current_context",
                return_value=None,
            ),
        ):
            mock_dist.is_initialized.return_value = False
            scores = [{"name": "a", "score": 1.0}]
            assert write_layer_analysis_result(scores, "kurtosis") == scores

    def test_write_skips_broken_store(self):
        mock_ctx = MagicMock()
        mock_ns = MagicMock()
        mock_ns.state = object()
        mock_ns.debug = {"layer_scores": []}
        mock_ctx.__getitem__.return_value = mock_ns
        scores = [{"name": "a", "score": 1.0}]

        with (
            patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist,
            patch(
                "msmodelslim.processor.analysis.distributed_utils.get_current_context",
                return_value=mock_ctx,
            ),
        ):
            mock_dist.is_initialized.return_value = False
            out = write_layer_analysis_result(scores, "kurtosis")

        assert out == scores
        assert mock_ns.debug["layer_scores"] == scores


class TestReadLayerAnalysisResult:
    def test_reads_from_state_first(self):
        ctx = {
            "layer_analysis": SimpleNamespace(
                state={
                    "layer_scores": [{"name": "a", "score": 1.0}],
                    "method": "kurtosis",
                    "patterns": ["*"],
                    "quant_modules": ["*"],
                },
                debug={"layer_scores": [{"name": "b", "score": 2.0}]},
            )
        }

        result = read_layer_analysis_result(ctx)
        assert result["layer_scores"] == [{"name": "a", "score": 1.0}]
        assert result["method"] == "kurtosis"
        assert result["patterns"] == ["*"]
        assert result["quant_modules"] == ["*"]

    def test_falls_back_to_debug_when_state_missing(self):
        ctx = {
            "layer_analysis": SimpleNamespace(
                state=None,
                debug={"layer_scores": [{"name": "b", "score": 2.0}], "method": "mse"},
            )
        }
        result = read_layer_analysis_result(ctx)
        assert result["layer_scores"] == [{"name": "b", "score": 2.0}]
        assert result["method"] == "mse"

    def test_raises_when_layer_scores_not_found(self):
        ctx = {"layer_analysis": SimpleNamespace(state={}, debug={})}
        with pytest.raises(KeyError, match="layer_scores not found"):
            read_layer_analysis_result(ctx)

    def test_skips_store_without_layer_scores_key(self):
        ctx = {
            "layer_analysis": SimpleNamespace(
                state={"method": "kurtosis"},
                debug={"layer_scores": "not-a-list"},
            )
        }
        with pytest.raises(KeyError, match="layer_scores not found"):
            read_layer_analysis_result(ctx)

    def test_store_get_handles_mapping_without_get(self):
        ctx = {
            "layer_analysis": SimpleNamespace(
                state={"layer_scores": [{"name": "a", "score": 1.0}]},
                debug=None,
            )
        }
        result = read_layer_analysis_result(ctx)
        assert result["method"] is None


class TestMaybeBarrierBeforeLinearQuant:
    def test_calls_barrier_when_distributed_multi_rank(self):
        with patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist:
            from msmodelslim.processor.analysis.distributed_utils import maybe_barrier_before_linear_quant

            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 4
            maybe_barrier_before_linear_quant()
            mock_dist.barrier.assert_called_once()

    def test_skips_barrier_when_single_rank(self):
        with patch("msmodelslim.processor.analysis.distributed_utils.dist") as mock_dist:
            from msmodelslim.processor.analysis.distributed_utils import maybe_barrier_before_linear_quant

            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 1
            maybe_barrier_before_linear_quant()
            mock_dist.barrier.assert_not_called()


class TestCheckDistributedAnalysisSupported:
    @patch("msmodelslim.processor.analysis.distributed_utils.dist")
    def test_raises_when_distributed_and_not_supported(self, mock_dist):
        from msmodelslim.processor.analysis.distributed_utils import check_distributed_analysis_supported
        from msmodelslim.utils.exception import UnsupportedError

        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4

        with pytest.raises(UnsupportedError, match="unsupported_dp"):
            check_distributed_analysis_supported(False, "unsupported_dp")
