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

msmodelslim/core/infer_engine/fake_quant/engine.py 的单元测试。
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from msmodelslim.core.const import DeviceType
from msmodelslim.core.infer_engine.fake_quant.engine import FakeQuantInferenceEngine
from msmodelslim.core.infer_engine.interface import InferenceConfig, InferenceResult
from msmodelslim.utils.exception import UnsupportedError


class TestFakeQuantInferenceEngine:
    """对应 FakeQuantInferenceEngine。"""

    @staticmethod
    def _make_result():
        return InferenceResult(generated_token_ids=[[1]], generated_texts=["a"])

    def test_run_delegates_to_single_when_single_device(self):
        engine = FakeQuantInferenceEngine()
        result = self._make_result()
        with patch.object(engine, "_run_single", return_value=result) as m:
            got = engine.run(Mock(), Mock(), [Mock()], InferenceConfig(), device_indices=[0])

        assert got is result
        m.assert_called_once()

    def test_run_delegates_to_single_when_no_device_indices(self):
        engine = FakeQuantInferenceEngine()
        adapter, loader, inputs = Mock(), Mock(), [Mock()]
        config = InferenceConfig()
        with patch.object(engine, "_run_single", return_value=self._make_result()) as m:
            engine.run(adapter, loader, inputs, config, device=DeviceType.CPU)

        m.assert_called_once_with(adapter, loader, inputs, config, DeviceType.CPU, None)

    def test_run_raises_unsupported_when_multiple_devices(self):
        engine = FakeQuantInferenceEngine()

        with pytest.raises(UnsupportedError):
            engine.run(Mock(), Mock(), [], InferenceConfig(), device_indices=[0, 1])

    def test_run_single_returns_result_when_full_pipeline_happy(self):
        engine = FakeQuantInferenceEngine()
        adapter = Mock()
        adapter.handle_dataset.side_effect = lambda inputs, device: list(inputs)
        session = Mock()
        model = Mock()
        session.build.return_value = model
        wm = Mock()
        loop = Mock()
        result = self._make_result()
        loop.run.return_value = result
        with (
            patch("msmodelslim.core.infer_engine.fake_quant.engine.Session", return_value=session),
            patch("msmodelslim.core.infer_engine.fake_quant.engine.WeightManager", return_value=wm),
            patch("msmodelslim.core.infer_engine.fake_quant.engine.PrefillLoop", return_value=loop),
        ):
            got = engine._run_single(
                adapter,
                Mock(format_loader=None),
                ["raw"],
                InferenceConfig(max_new_tokens=3),
                DeviceType.CPU,
                None,
            )

        assert got is result
        session.build.assert_called_once()
        adapter.handle_dataset.assert_called_once_with(["raw"], DeviceType.CPU)
        wm.install_hooks.assert_called_once_with(model)
        wm.remove_all_hooks.assert_called_once()
        loop.run.assert_called_once_with(["raw"], 3, disable_eos=False, global_sample_indices=None)

    def test_run_single_cleans_store_when_pipeline_raises(self):
        engine = FakeQuantInferenceEngine()
        adapter = Mock()
        adapter.handle_dataset.side_effect = lambda inputs, device: list(inputs)
        with (
            patch("msmodelslim.core.infer_engine.fake_quant.engine.Session", side_effect=RuntimeError("boom")),
            patch("msmodelslim.core.infer_engine.fake_quant.engine.WeightManager"),
            patch("msmodelslim.core.infer_engine.fake_quant.engine.PrefillLoop"),
        ):
            with pytest.raises(RuntimeError):
                engine._run_single(adapter, Mock(), [], InferenceConfig(), DeviceType.CPU, None)

    def test_run_single_disables_eos_when_dist_initialized(self):
        engine = FakeQuantInferenceEngine()
        adapter = Mock()
        adapter.handle_dataset.side_effect = lambda inputs, device: list(inputs)
        session = Mock()
        model = Mock()
        session.build.return_value = model
        loop = Mock()
        result = self._make_result()
        loop.run.return_value = result
        mock_dist = Mock()
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 2
        with (
            patch("msmodelslim.core.infer_engine.fake_quant.engine.dist", mock_dist),
            patch("msmodelslim.core.infer_engine.fake_quant.engine.Session", return_value=session),
            patch("msmodelslim.core.infer_engine.fake_quant.engine.WeightManager"),
            patch("msmodelslim.core.infer_engine.fake_quant.engine.PrefillLoop", return_value=loop),
        ):
            engine._run_single(adapter, Mock(), ["raw"], InferenceConfig(), DeviceType.CPU, None)

        loop.run.assert_called_once_with(["raw"], 1, disable_eos=True, global_sample_indices=None)

    def test_resolve_device_str_returns_npu_with_index_when_npu(self):
        npu_mock = MagicMock()
        with patch.object(torch, "npu", npu_mock, create=True):
            assert FakeQuantInferenceEngine._resolve_device_str(DeviceType.NPU, [3]) == "npu:3"

        npu_mock.set_device.assert_called_once_with("npu:3")

    def test_resolve_device_str_uses_zero_when_no_indices(self):
        npu_mock = MagicMock()
        with patch.object(torch, "npu", npu_mock, create=True):
            assert FakeQuantInferenceEngine._resolve_device_str(DeviceType.NPU, None) == "npu:0"

        npu_mock.set_device.assert_called_once_with("npu:0")

    def test_resolve_device_str_returns_enum_value_when_cpu(self):
        assert FakeQuantInferenceEngine._resolve_device_str(DeviceType.CPU, None) == "cpu"
