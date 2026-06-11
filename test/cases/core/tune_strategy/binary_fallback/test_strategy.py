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
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.core.const import DeviceType
from msmodelslim.core.practice import Metadata, PracticeConfig
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1ServiceConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.tune_strategy.binary_fallback.strategy import (
    BinaryFallbackStrategy,
    BinaryFallbackStrategyConfig,
    get_plugin,
)
from msmodelslim.core.tune_strategy.interface import EvaluateResult
from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.quant.linear import LinearProcessorConfig
from msmodelslim.utils.exception import SchemaValidateError, SpecError, UnsupportedError
from msmodelslim.utils.pydantic_model_path import resolve_pydantic_model_path


def _default_quant_save_list():
    return [AscendV1QuantFormatConfig(part_file_size=4)]


def _default_v1_spec() -> ModelslimV1ServiceConfig:
    return ModelslimV1ServiceConfig(
        process=[
            LinearProcessorConfig(
                type="linear_quant",
                qconfig=LinearQConfig(
                    act=QConfig(
                        scope=QScope.PER_TENSOR,
                        dtype=QDType.INT8,
                        symmetric=False,
                        method="minmax",
                    ),
                    weight=QConfig(
                        scope=QScope.PER_CHANNEL,
                        dtype=QDType.INT8,
                        symmetric=True,
                        method="minmax",
                    ),
                ),
                include=["*"],
                exclude=[],
            ),
        ],
        save=_default_quant_save_list(),
        dataset="mix_calib.jsonl",
    )


def _default_practice_template() -> PracticeConfig:
    return PracticeConfig(
        apiversion="modelslim_v1",
        metadata=Metadata(
            config_id="binary_fallback",
            label={"w_bit": 8, "a_bit": 8, "is_sparse": False, "kv_cache": False},
        ),
        spec=_default_v1_spec(),
    )


class _MockModel(PipelineInterface):
    """Mock model implementing PipelineInterface."""

    @property
    def model_type(self):
        return "test"

    @property
    def model_path(self):
        return Path("/tmp/test")

    @property
    def trust_remote_code(self):
        return False

    def handle_dataset(self, dataset, device=DeviceType.NPU):
        return list(dataset) if dataset else []

    def init_model(self, device: DeviceType = DeviceType.NPU):
        return MagicMock()

    def generate_model_visit(self, model):
        yield from ()

    def generate_model_forward(self, model, inputs):
        yield from ()

    def enable_kv_cache(self, model, need_kv_cache: bool) -> None:
        pass


class TestBinaryFallbackStrategyConfig:
    def test_BinaryFallbackStrategyConfig_passes_when_valid_template_and_list_path(self):
        cfg = BinaryFallbackStrategyConfig(
            template=_default_practice_template(),
            rollback_path="spec.process.0.exclude",
        )
        assert cfg.type == "binary_fallback"
        assert resolve_pydantic_model_path(cfg.template, "spec.process.0.exclude") == []

    def test_BinaryFallbackStrategyConfig_raises_SchemaValidateError_when_apiversion_not_v1(self):
        template = PracticeConfig(
            apiversion="modelslim_v0",
            metadata=Metadata(config_id="v0"),
            spec={"calib_dataset": "mix_calib.jsonl"},
        )
        with pytest.raises(SchemaValidateError) as exc_info:
            BinaryFallbackStrategyConfig(template=template, rollback_path="spec.process.0.exclude")
        assert "modelslim_v1" in str(exc_info.value)

    def test_BinaryFallbackStrategyConfig_raises_SchemaValidateError_when_rollback_path_not_list(self):
        template = _default_practice_template()
        with pytest.raises(SchemaValidateError) as exc_info:
            BinaryFallbackStrategyConfig(template=template, rollback_path="metadata.config_id")
        assert "list" in str(exc_info.value).lower()


class TestBinaryFallbackStrategy:
    def _make_config(self, rollback_candidates=None):
        return BinaryFallbackStrategyConfig(
            template=_default_practice_template(),
            rollback_path="spec.process.0.exclude",
            rollback_candidates=rollback_candidates,
        )

    def _make_dataset_loader(self):
        loader = MagicMock()
        loader.get_dataset_by_name = MagicMock(return_value=[])
        return loader

    def test_generate_practice_stops_after_one_yield_when_zero_rollback_satisfied(self):
        strategy = BinaryFallbackStrategy(
            config=self._make_config(rollback_candidates=["layer.a"]),
            dataset_loader=self._make_dataset_loader(),
        )
        gen = strategy.generate_practice(_MockModel(), device=DeviceType.NPU)
        practice = next(gen)
        assert practice.metadata.config_id.startswith("binary_fallback_")
        assert resolve_pydantic_model_path(practice, "spec.process.0.exclude") == []
        with pytest.raises(StopIteration):
            gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=True))

    def test_generate_practice_yields_final_prefix_when_user_candidates_binary_search_hits(self):
        strategy = BinaryFallbackStrategy(
            config=self._make_config(rollback_candidates=["layer.a", "layer.b", "layer.c"]),
            dataset_loader=self._make_dataset_loader(),
        )
        gen = strategy.generate_practice(_MockModel(), device=DeviceType.NPU)

        _ = next(gen)
        _ = gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=False))

        _ = gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=True))
        _ = gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=False))
        final = gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=True))
        assert resolve_pydantic_model_path(final, "spec.process.0.exclude") == ["*layer.a.*", "*layer.b.*"]

        with pytest.raises(StopIteration):
            gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=True))

    def test_generate_practice_raises_SpecError_when_max_rollback_still_not_satisfied(self):
        strategy = BinaryFallbackStrategy(
            config=self._make_config(rollback_candidates=["*layer.a.*", "*layer.b.*"]),
            dataset_loader=self._make_dataset_loader(),
        )
        gen = strategy.generate_practice(_MockModel(), device=DeviceType.NPU)

        _ = next(gen)
        with pytest.raises(SpecError):
            gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=False))
            gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=False))
            gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=False))

    def test_generate_practice_skips_analysis_when_user_candidates_provided(self):
        strategy = BinaryFallbackStrategy(
            config=self._make_config(rollback_candidates=["layer.a"]),
            dataset_loader=self._make_dataset_loader(),
        )
        with patch.object(strategy, "_run_sensitive_layer_analysis") as mock_analysis:
            gen = strategy.generate_practice(_MockModel(), device=DeviceType.NPU)
            next(gen)
            mock_analysis.assert_not_called()
            with pytest.raises(StopIteration):
                gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=True))

    def test_generate_practice_raises_UnsupportedError_when_no_candidates_and_model_not_interface(self):
        strategy = BinaryFallbackStrategy(
            config=self._make_config(rollback_candidates=None),
            dataset_loader=self._make_dataset_loader(),
        )
        model = MagicMock()
        gen = strategy.generate_practice(model, device=DeviceType.NPU)
        with pytest.raises(UnsupportedError) as exc_info:
            next(gen)
            gen.send(EvaluateResult(accuracies=[], expectations=[], is_satisfied=False))
        assert "PipelineInterface" in str(exc_info.value)

    def test_get_plugin_returns_config_and_strategy_classes_when_called(self):
        config_cls, strategy_cls = get_plugin()
        assert config_cls is BinaryFallbackStrategyConfig
        assert strategy_cls is BinaryFallbackStrategy
