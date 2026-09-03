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
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1ServiceConfig
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.tune_strategy.interface import EvaluateResult
from msmodelslim.core.tune_strategy.standing_high.strategy import (
    StandingHighStrategy,
    StandingHighStrategyConfig,
    get_plugin,
)
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.quant.linear import LinearProcessorConfig
from msmodelslim.utils.exception import SchemaValidateError, SpecError, UnsupportedError


def _default_quant_save_list():
    return [AscendV1QuantFormatConfig(part_file_size=4)]


def _standing_high_default_template() -> ModelslimV1ServiceConfig:
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


def _make_anti_outlier_strategies():
    """Minimal anti_outlier_strategies for config (at least one element)."""
    return [[{"type": "flex_smooth_quant"}]]


class TestStandingHighStrategyConfig:
    """StandingHighStrategyConfig 单元测试。"""

    def test_StandingHighStrategyConfig_field_match_when_valid_anti_outlier_strategies_and_default_template(self):
        """
        场景：构造配置时传入合法 anti_outlier_strategies 与结构等同默认 template 的配置
        （须显式传入 template：Pydantic default_factory 在类定义时已绑定，monkeypatch 无效；
        生产 _create_default_template 仍含 AscendV1Config，与 ModelslimV1ServiceConfig.save 新 schema 不兼容）。
        预期：type=standing_high，anti_outlier_strategies、template.process、metadata.config_id 符合预期。
        """
        cfg = StandingHighStrategyConfig(
            anti_outlier_strategies=_make_anti_outlier_strategies(),
            template=_standing_high_default_template(),
        )
        assert cfg.type == "standing_high"
        assert len(cfg.anti_outlier_strategies) >= 1
        assert len(cfg.template.process) >= 1
        assert cfg.metadata.config_id == "standing_high"

    def test_StandingHighStrategyConfig_raises_SchemaValidateError_when_anti_outlier_strategies_empty(self):
        """
        场景：anti_outlier_strategies 为空列表。
        预期：抛出 SchemaValidateError 且消息含 least one。
        """
        with pytest.raises(SchemaValidateError) as exc_info:
            StandingHighStrategyConfig(anti_outlier_strategies=[], template=_standing_high_default_template())
        assert "least one" in str(exc_info.value).lower()

    def test_StandingHighStrategyConfig_raises_SchemaValidateError_when_template_has_no_linear_quant(self):
        """
        场景：template 中不含 linear_quant 类型的 process。
        预期：抛出 SchemaValidateError 且消息含 linear_quant。
        """
        template = ModelslimV1ServiceConfig(
            process=[{"type": "flex_smooth_quant"}],
            save=_default_quant_save_list(),
            dataset="mix_calib.jsonl",
        )
        with pytest.raises(SchemaValidateError) as exc_info:
            StandingHighStrategyConfig(
                anti_outlier_strategies=_make_anti_outlier_strategies(),
                template=template,
            )
        assert "linear_quant" in str(exc_info.value).lower()


class TestStandingHighStrategy:
    """StandingHighStrategy 单元测试。"""

    def _make_config(self):
        return StandingHighStrategyConfig(
            anti_outlier_strategies=_make_anti_outlier_strategies(),
            template=_standing_high_default_template(),
        )

    def _make_dataset_loader(self):
        loader = MagicMock()
        loader.get_dataset_by_name = MagicMock(return_value=[])
        return loader

    def test_generate_practice_raises_UnsupportedError_when_model_not_implement_PipelineInterface(self):
        """
        场景：generate_practice 传入未实现 PipelineInterface 的 model。
        预期：抛出 UnsupportedError 且消息含 PipelineInterface。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)
        non_interface_model = MagicMock()
        gen = strategy.generate_practice(non_interface_model, device=DeviceType.NPU)
        with pytest.raises(UnsupportedError) as exc_info:
            next(gen)
        assert "PipelineInterface" in str(exc_info.value)

    def test_generate_practice_yields_float_full_fallback_practice_as_first_yield(self):
        """
        场景：generate_practice 首次 yield。
        预期：产出 config_id 以 standing_high_float_ 开头的 practice，且每个 linear_quant 的 exclude 包含 "*"。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)

        model = _MockModel()
        with patch.object(strategy, "_run_sensitive_layer_analysis", return_value=None):
            gen = strategy.generate_practice(model, device=DeviceType.NPU)
            practice = next(gen)

        assert practice is not None
        assert practice.metadata.config_id.startswith("standing_high_float_")
        linear_procs = [p for p in practice.spec.process if getattr(p, "type", None) == "linear_quant"]
        assert linear_procs, "全回退 practice 应至少保留一个 linear_quant"
        assert all("*" in (proc.exclude or []) for proc in linear_procs)

    def test_generate_practice_raises_SpecError_when_float_full_fallback_unsatisfied(self):
        """
        场景：首轮 yield 浮点全回退 practice 后，send 一个 is_satisfied=False 的 EvaluateResult。
        预期：抛出 SpecError，Message 与 action 与 binary_fallback 风格一致（"with full float fallback" / "Relax accuracy expectations"）。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)

        model = _MockModel()
        with patch.object(strategy, "_run_sensitive_layer_analysis", return_value=None):
            gen = strategy.generate_practice(model, device=DeviceType.NPU)
            _ = next(gen)

        unsatisfied = EvaluateResult(accuracies=[], expectations=[], is_satisfied=False)
        with pytest.raises(SpecError) as exc_info:
            gen.send(unsatisfied)

        message = str(exc_info.value)
        assert "full float fallback" in message
        assert "relax" in message.lower()

    def test_generate_practice_yields_zero_practice_then_stops_when_send_is_satisfied_true(self):
        """
        场景：浮点全回退预检通过（send is_satisfied=True）后，再 send 一个 is_satisfied=True，触发结束。
        预期：首项为 standing_high_float_ 前缀，第二次 yield 为 standing_high_ 前缀，最终 StopIteration。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)

        model = _MockModel()
        with patch.object(strategy, "_run_sensitive_layer_analysis", return_value=None):
            gen = strategy.generate_practice(model, device=DeviceType.NPU)
            float_practice = next(gen)

        assert float_practice.metadata.config_id.startswith("standing_high_float_")

        satisfied = EvaluateResult(accuracies=[], expectations=[], is_satisfied=True)
        zero_practice = gen.send(satisfied)
        assert zero_practice.metadata.config_id.startswith("standing_high_")
        assert not zero_practice.metadata.config_id.startswith("standing_high_float_")

        try:
            gen.send(satisfied)
        except StopIteration:
            pass
        else:
            pytest.fail("zero evaluation 满足时应该结束")

    def test_build_practice_config_returns_PracticeConfig_with_metadata_and_spec_when_valid_anti_outlier(self):
        """
        场景：_build_practice_config 传入合法 anti_outlier、linear_quant_exclude=[]。
        预期：返回 apiversion、metadata.config_id、spec.process、spec.dataset 与 template 一致。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)
        anti_outlier = config.anti_outlier_strategies[0]
        practice = strategy._build_practice_config(anti_outlier, linear_quant_exclude=[])
        assert practice.apiversion == "modelslim_v1"
        assert practice.metadata.config_id == "standing_high_0"
        assert len(practice.spec.process) >= 1
        assert practice.spec.dataset == config.template.dataset

    def test_build_practice_config_appends_exclude_to_linear_quant_when_linear_quant_exclude_provided(self):
        """
        场景：_build_practice_config 传入非空 linear_quant_exclude。
        预期：对应 linear_quant 的 exclude 中包含传入的项。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)
        anti_outlier = config.anti_outlier_strategies[0]
        practice = strategy._build_practice_config(
            anti_outlier,
            linear_quant_exclude=["layer.0.linear"],
        )
        assert practice.spec is not None
        linear_procs = [p for p in practice.spec.process if getattr(p, "type", None) == "linear_quant"]
        assert len(linear_procs) >= 1
        excludes = linear_procs[0].exclude or []
        assert any("layer.0.linear" in pat for pat in excludes)

    def test_generate_practice_yields_multiple_practices_when_send_is_satisfied_false_then_true(self):
        """
        场景：generate_practice 后多次 send EvaluateResult。
        流程：首轮浮点全回退通过 → zero 不满足进入二分 → 二分找到最小级别 → 摸高迭代 → 结束。
        预期：可依次取到多个 practice，最后 send(None) 触发 StopIteration。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighStrategy(config=config, dataset_loader=loader)

        model = _MockModel()
        # Avoid running real analysis / binary search; focus on generator control flow.
        with (
            patch.object(strategy, "_run_sensitive_layer_analysis", return_value=None),
            patch.object(strategy, "_find_satisfied_disable_level") as mock_find_level,
            patch.object(strategy, "_stand_high") as mock_stand_high,
        ):

            def _fake_find_level():
                _ = yield strategy._build_practice_config(
                    config.anti_outlier_strategies[0],
                    linear_quant_exclude=["g2"],
                )
                return 1

            def _fake_stand_high(_init_level: int):
                _ = yield strategy._build_practice_config(
                    config.anti_outlier_strategies[0],
                    linear_quant_exclude=["g2"],
                )
                _ = yield strategy._build_practice_config(
                    config.anti_outlier_strategies[0],
                    linear_quant_exclude=[],
                )

            mock_find_level.side_effect = _fake_find_level
            mock_stand_high.side_effect = _fake_stand_high

            satisfied = EvaluateResult(accuracies=[], expectations=[], is_satisfied=True)
            unsatisfied = EvaluateResult(accuracies=[], expectations=[], is_satisfied=False)

            gen = strategy.generate_practice(model, device=DeviceType.NPU)
            p1 = next(gen)
            # 首轮必为浮点全回退预检
            assert p1.metadata.config_id.startswith("standing_high_float_")

            # 浮点全回退满足 → 进入 zero 评估
            p2 = gen.send(satisfied)
            assert p2.metadata.config_id.startswith("standing_high_")
            assert not p2.metadata.config_id.startswith("standing_high_float_")

            # zero 不满足 → 进入二分搜索
            p3 = gen.send(unsatisfied)
            assert p3 is not None

            # 二分搜索满足 → 进入摸高循环
            p4 = gen.send(satisfied)
            assert p4 is not None

            p5 = gen.send(satisfied)
            assert p5 is not None

            with pytest.raises(StopIteration):
                gen.send(None)

    def test_get_plugin_returns_config_and_strategy_classes_when_called(self):
        """
        场景：调用 get_plugin()。
        预期：返回 (StandingHighStrategyConfig, StandingHighStrategy) 元组。
        """
        config_cls, strategy_cls = get_plugin()
        assert config_cls is StandingHighStrategyConfig
        assert strategy_cls is StandingHighStrategy

    def test_select_layers_returns_empty_when_disable_level_zero(self):
        """场景：disable_level=0。预期：空列表。"""
        strategy = StandingHighStrategy.__new__(StandingHighStrategy)
        strategy.config = self._make_config()
        strategy._analysis_layer_scores = [
            {"name": "model.layers.2", "score": 0.9},
            {"name": "model.layers.1", "score": 0.5},
        ]
        assert strategy.select_layers_by_disable_level(0) == []

    def test_select_layers_returns_top_k_names_when_disable_level_positive(self):
        """场景：disable_level=1。预期：最高分一层。"""
        strategy = StandingHighStrategy.__new__(StandingHighStrategy)
        strategy.config = self._make_config()
        strategy._analysis_layer_scores = [
            {"name": "model.layers.2", "score": 0.9},
            {"name": "model.layers.1", "score": 0.5},
        ]
        names = strategy.select_layers_by_disable_level(1)
        assert names == ["model.layers.2"]

    def test_build_practice_config_appends_wildcard_exclude_when_layers_given(self):
        """场景：指定回退层。预期：linear_quant exclude 含通配符。"""
        strategy = StandingHighStrategy.__new__(StandingHighStrategy)
        strategy.config = self._make_config()
        strategy._StandingHighStrategy__counter = 0
        practice = strategy._build_practice_config(
            strategy.config.anti_outlier_strategies[0],
            ["model.layers.1"],
        )
        linear_procs = [p for p in practice.spec.process if p.type == "linear_quant"]
        assert any("*model.layers.1.*" in (p.exclude or []) for p in linear_procs)

    def test_permute_yields_anti_outlier_strategy_when_called(self):
        """场景：_permute 首次 yield。预期：返回配置中的策略列表。"""
        strategy = StandingHighStrategy.__new__(StandingHighStrategy)
        strategy.config = self._make_config()
        strategy._StandingHighStrategy__counter = 0
        strategy._current_index = 0
        chosen = next(strategy._permute())
        assert chosen is strategy.config.anti_outlier_strategies[0]
