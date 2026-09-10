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

msmodelslim/core/infer_engine/dp_fake_quant/dp_engine.py 的单元测试。
"""

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from msmodelslim.core.const import DeviceType
from msmodelslim.core.context import ContextManager, LocalDictContext
from msmodelslim.core.infer_engine.dp_fake_quant.dp_engine import DPFakeQuantInferenceEngine
from msmodelslim.core.infer_engine.dp_fake_quant.dp_sharding import FAKE_QUANT_INFER_NAMESPACE, partial_key
from msmodelslim.core.infer_engine.interface import InferenceConfig, InferenceResult
from msmodelslim.utils.exception import UnsupportedError


class TestDPFakeQuantInferenceEngine:
    """对应 DPFakeQuantInferenceEngine。"""

    @staticmethod
    def _result():
        return InferenceResult(generated_token_ids=[[0]], generated_texts=["a"])

    def test_run_delegates_to_single_engine_when_one_device(self):
        engine = DPFakeQuantInferenceEngine()
        with patch.object(engine, "_run_single", return_value=self._result()) as m:
            got = engine.run(Mock(), Mock(), [], InferenceConfig(), device_indices=[0])

        assert got.generated_texts == ["a"]
        m.assert_called_once()

    def test_run_delegates_to_single_engine_when_no_device_indices(self):
        engine = DPFakeQuantInferenceEngine()
        adapter, loader, inputs = Mock(), Mock(), [Mock()]
        config = InferenceConfig()
        with patch.object(engine, "_run_single", return_value=self._result()) as m:
            engine.run(adapter, loader, inputs, config, device_indices=None)

        m.assert_called_once_with(adapter, loader, inputs, config, DeviceType.NPU, None)

    def test_run_delegates_to_dp_when_multiple_devices(self):
        engine = DPFakeQuantInferenceEngine()
        adapter, loader, inputs = Mock(), Mock(), [Mock()]
        config = InferenceConfig()
        with patch.object(engine, "_run_dp", return_value=self._result()) as m:
            engine.run(adapter, loader, inputs, config, device_indices=[0, 1])

        m.assert_called_once_with(adapter, loader, inputs, config, DeviceType.NPU, [0, 1])

    def test_run_dp_raises_unsupported_when_loader_has_no_model_path(self):
        engine = DPFakeQuantInferenceEngine()
        loader = SimpleNamespace()  # 无 model_path 属性

        with pytest.raises(UnsupportedError):
            engine._run_dp(Mock(), loader, [Mock(), Mock()], InferenceConfig(), DeviceType.NPU, [0, 1])

    @patch("msmodelslim.core.infer_engine.dp_fake_quant.dp_engine.mp")
    def test_run_dp_spawns_workers_and_merges_when_master_port_set(self, mock_mp):
        engine = DPFakeQuantInferenceEngine()
        loader = SimpleNamespace(model_path="/export")
        merged = InferenceResult(generated_token_ids=[[0], [1]], generated_texts=["a", "b"])
        os.environ["MASTER_PORT"] = "12345"
        ctx = LocalDictContext()
        try:
            with ContextManager(ctx):
                with (
                    patch(
                        "msmodelslim.core.infer_engine.dp_fake_quant.dp_engine.shard_samples_by_rank",
                        return_value=[[0], [1]],
                    ) as mock_shard,
                    patch(
                        "msmodelslim.core.infer_engine.dp_fake_quant.dp_engine.merge_partials_into_context",
                        return_value=merged,
                    ),
                ):
                    got = engine._run_dp(Mock(), loader, [0, 1], InferenceConfig(), DeviceType.NPU, [0, 1])

            assert got is merged
            mock_shard.assert_called()
            mock_mp.spawn.assert_called_once()
            mock_mp.set_start_method.assert_called_once_with("spawn", force=True)
        finally:
            os.environ.pop("MASTER_PORT", None)

    @patch("msmodelslim.core.infer_engine.dp_fake_quant.dp_engine.mp")
    def test_run_dp_sets_master_port_when_missing_in_env(self, mock_mp):
        os.environ.pop("MASTER_PORT", None)
        engine = DPFakeQuantInferenceEngine()
        loader = SimpleNamespace(model_path="/export")
        ctx = LocalDictContext()
        try:
            with ContextManager(ctx):
                with (
                    patch(
                        "msmodelslim.core.infer_engine.dp_fake_quant.dp_engine.find_free_port",
                        return_value=4321,
                    ),
                    patch(
                        "msmodelslim.core.infer_engine.dp_fake_quant.dp_engine.merge_partials_into_context",
                        return_value=self._result(),
                    ),
                    patch(
                        "msmodelslim.core.infer_engine.dp_fake_quant.dp_engine.shard_samples_by_rank",
                        side_effect=lambda s, r, w: s,
                    ),
                ):
                    engine._run_dp(Mock(), loader, [0], InferenceConfig(), DeviceType.NPU, [0, 1])

            assert os.environ["MASTER_PORT"] == "4321"
        finally:
            os.environ.pop("MASTER_PORT", None)

    def test_dp_worker_stores_partial_result_when_run(self):
        ctx = LocalDictContext()
        result = InferenceResult(generated_token_ids=[[1], [3]], generated_texts=["x", "z"])
        worker_engine = Mock()
        worker_engine._run_single.return_value = result
        loader = Mock()

        def _fake_engine():
            return worker_engine

        with (
            patch(
                "msmodelslim.core.infer_engine.dp_fake_quant.dp_engine.FakeQuantInferenceEngine",
                side_effect=_fake_engine,
            ),
            patch("msmodelslim.core.infer_engine.dp_fake_quant.dp_engine.setup_distributed"),
            patch("msmodelslim.core.infer_engine.dp_fake_quant.dp_engine.set_logger_level"),
            patch("msmodelslim.format.format_handler.build_default_format_chain") as mock_chain,
        ):
            mock_chain.return_value.handle.return_value = loader
            DPFakeQuantInferenceEngine._dp_worker(
                rank=0,
                world_size=2,
                device_indices=[0, 1],
                adapter=Mock(),
                model_path="/export",
                shards=[[10, 30], [20]],
                inference_config=InferenceConfig(),
                device=DeviceType.NPU,
                shared_ctx=ctx,
                backend="nccl",
                master_port=9999,
                num_samples=3,
            )

        state = ctx[FAKE_QUANT_INFER_NAMESPACE].state
        assert state[partial_key(0)] == result.model_dump()
        worker_engine._run_single.assert_called_once()
        args = worker_engine._run_single.call_args
        assert args.kwargs["inputs"] == [10, 30]
        assert args.kwargs["global_sample_indices"] == [0, 2]
