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

# pylint: disable=duplicate-code

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.core.const import DeviceType, QuantType
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.tune_strategy.common.config_builder.expert_experience.expert_experience import (
    ExpertExperienceConfigBuilder,
)
from msmodelslim.core.tune_strategy.common.config_builder.expert_experience import StructureConfig
from msmodelslim.core.tune_strategy.standing_high_with_experience.strategy import (
    StandingHighWithExperienceStrategy,
    StandingHighWithExperienceStrategyConfig,
    get_plugin,
)
from msmodelslim.core.tune_strategy.standing_high_with_experience.standing_high_with_experience_interface import (
    StandingHighWithExperienceInterface,
)
from msmodelslim.format.ascendV1_format.ascendV1 import AscendV1QuantFormatConfig
from msmodelslim.core.tune_strategy.standing_high.standing_high_interface import StandingHighInterface
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


def _default_quant_save_list():
    return [AscendV1QuantFormatConfig(part_file_size=4)]


@pytest.fixture(autouse=True)
def _patch_expert_builder_spec_save(monkeypatch):
    """ExpertExperienceConfigBuilder.build_spec_save 使用 QuantFormatConfig。"""
    monkeypatch.setattr(
        ExpertExperienceConfigBuilder,
        "build_spec_save",
        lambda self, **kwargs: _default_quant_save_list(),
    )


class _MockModel(StandingHighWithExperienceInterface, StandingHighInterface, PipelineInterface):
    """Mock model implementing StandingHighWithExperienceInterface."""

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

    def load_model(self, device=DeviceType.NPU):
        m = MagicMock()
        m.to = MagicMock(return_value=None)
        return m

    def init_model(self, device: DeviceType = DeviceType.NPU):
        return MagicMock()

    def generate_model_visit(self, model):
        yield from ()

    def generate_model_forward(self, model, inputs):
        yield from ()

    def enable_kv_cache(self, model, need_kv_cache: bool) -> None:
        pass


class TestStandingHighWithExperienceStrategyConfig:
    """StandingHighWithExperienceStrategyConfig 单元测试。"""

    def test_StandingHighWithExperienceStrategyConfig_field_match_when_valid_w8a8_and_structure_configs(self):
        """
        场景：构造配置时传入合法 quant_type=w8a8 与 structure_configs。
        预期：type、quant_type、structure_configs 与入参一致。
        """
        cfg = StandingHighWithExperienceStrategyConfig(
            structure_configs=[StructureConfig(type="MHA", include=["*"], exclude=[])],
            quant_type=QuantType.W8A8,
        )
        assert cfg.type == "standing_high_with_experience"
        assert cfg.quant_type == QuantType.W8A8
        assert len(cfg.structure_configs) == 1
        assert cfg.structure_configs[0].type == "MHA"

    def test_StandingHighWithExperienceStrategyConfig_raises_SchemaValidateError_when_quant_type_not_in_expert(self):
        """
        场景：quant_type 不在专家经验 supported_quant_types 内（如 w8a16）。
        预期：抛出 SchemaValidateError 且消息含 quant_type 或 supported。
        """
        with pytest.raises(SchemaValidateError) as exc_info:
            StandingHighWithExperienceStrategyConfig(
                structure_configs=[StructureConfig(type="MHA", include=["*"], exclude=[])],
                quant_type=QuantType.W8A16,  # w8a16 不在 expert_experience 的 w8a8/w4a8 中
            )
        assert "quant_type" in str(exc_info.value) or "supported" in str(exc_info.value).lower()


class TestStandingHighWithExperienceStrategy:
    """StandingHighWithExperienceStrategy 单元测试。"""

    def _make_config(self, structure_configs=None, quant_type=QuantType.W8A8):
        if structure_configs is None:
            structure_configs = [StructureConfig(type="MHA", include=["*"], exclude=[])]
        return StandingHighWithExperienceStrategyConfig(
            structure_configs=structure_configs,
            quant_type=quant_type,
        )

    def _make_dataset_loader(self):
        loader = MagicMock()
        loader.get_dataset_by_name = MagicMock(return_value=[])
        return loader

    def test_init_raises_SchemaValidateError_when_config_not_StandingHighWithExperienceStrategyConfig(self):
        """
        场景：策略初始化时传入非 StandingHighWithExperienceStrategyConfig 的 config。
        预期：抛出 SchemaValidateError 且消息含 StandingHighWithExperienceStrategyConfig。
        """
        from msmodelslim.core.tune_strategy.interface import StrategyConfig

        wrong_cfg = MagicMock(spec=StrategyConfig)
        wrong_cfg.type = "other"
        loader = self._make_dataset_loader()
        with pytest.raises(SchemaValidateError) as exc_info:
            StandingHighWithExperienceStrategy(config=wrong_cfg, dataset_loader=loader)
        assert "StandingHighWithExperienceStrategyConfig" in str(exc_info.value)

    def test_generate_practice_raises_UnsupportedError_when_model_not_implement_interface(self):
        """
        场景：generate_practice 传入未实现 StandingHighWithExperienceInterface 的 model。
        预期：抛出 UnsupportedError 且消息含 StandingHighWithExperienceInterface。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighWithExperienceStrategy(config=config, dataset_loader=loader)
        non_interface_model = MagicMock()
        gen = strategy.generate_practice(non_interface_model, device=DeviceType.NPU)
        with pytest.raises(UnsupportedError) as exc_info:
            next(gen)
        assert "StandingHighWithExperienceInterface" in str(exc_info.value)

    @patch("msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.StandingHighStrategy")
    @patch(
        "msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.StandingHighWithExperienceStrategy._filter_supported_anti_outlier_strategies"
    )
    def test_generate_practice_yields_inner_practice_when_model_implements_interface_and_delegates_to_standing_high(
        self, mock_filter, mock_standing_high_cls
    ):
        """
        场景：model 实现接口，策略内部委托 StandingHighStrategy 生成 practice。
        预期：yield 的 practice 与内层策略产出一致，且只调用一次。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighWithExperienceStrategy(config=config, dataset_loader=loader)
        mock_sh = MagicMock()
        from msmodelslim.core.practice import PracticeConfig

        mock_practice = MagicMock(spec=PracticeConfig)
        mock_sh.generate_practice.return_value = iter([mock_practice])
        mock_standing_high_cls.return_value = mock_sh
        # 避免 mock 模型不实现 FlexSmoothQuant 等接口导致策略被过滤光
        mock_filter.side_effect = lambda strategies, _: strategies

        model = _MockModel()
        gen = strategy.generate_practice(model, device=DeviceType.NPU)
        out = next(gen)
        assert out is mock_practice
        mock_standing_high_cls.assert_called_once()
        mock_sh.generate_practice.assert_called_once_with(model, DeviceType.NPU)

    @patch("msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.StandingHighStrategy")
    @patch(
        "msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.StandingHighWithExperienceStrategy._filter_supported_anti_outlier_strategies"
    )
    def test_generate_practice_uses_default_structure_configs_when_structure_configs_none_and_auto_detect_fails(
        self, mock_filter, mock_sh_cls
    ):
        """
        场景：structure_configs 为 None，auto_detect 失败后回退默认。
        预期：传入内层 StandingHigh 的 base_config 含 type=standing_high、template、anti_outlier_strategies。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighWithExperienceStrategy(config=config, dataset_loader=loader)
        strategy.config.structure_configs = None  # 模拟未提供

        mock_sh = MagicMock()
        mock_sh.generate_practice.return_value = iter([])
        mock_sh_cls.return_value = mock_sh
        mock_filter.side_effect = lambda strategies, _: strategies

        model = _MockModel()
        list(strategy.generate_practice(model, device=DeviceType.NPU))
        call_kw = mock_sh_cls.call_args[1]
        base_config = call_kw["config"]
        assert base_config.type == "standing_high"
        assert base_config.template is not None
        assert base_config.anti_outlier_strategies is not None

    @patch("msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.StandingHighStrategy")
    @patch(
        "msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.StandingHighWithExperienceStrategy._auto_detect_structure_configs"
    )
    @patch(
        "msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.StandingHighWithExperienceStrategy._filter_supported_anti_outlier_strategies"
    )
    def test_generate_practice_uses_detected_structure_configs_when_structure_configs_none_and_auto_detect_returns_non_empty(
        self, mock_filter, mock_auto_detect, mock_sh_cls
    ):
        """
        场景：structure_configs 为 None，_auto_detect_structure_configs 返回非空列表。
        预期：策略将 config.structure_configs 更新为检测结果，且 auto_detect 被调用一次。
        """
        detected = [StructureConfig(type="GQA", include=["*"], exclude=[])]
        mock_auto_detect.return_value = detected
        mock_filter.side_effect = lambda strategies, _: strategies
        mock_sh = MagicMock()
        mock_sh.generate_practice.return_value = iter([])
        mock_sh_cls.return_value = mock_sh

        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighWithExperienceStrategy(config=config, dataset_loader=loader)
        strategy.config.structure_configs = None
        model = _MockModel()
        list(strategy.generate_practice(model, device=DeviceType.NPU))
        assert strategy.config.structure_configs == detected
        mock_auto_detect.assert_called_once_with(model)

    def test_generate_base_config_returns_standing_high_config_with_template_and_anti_outlier_when_valid_input(self):
        """
        场景：_generate_base_config 传入合法 config、structure_configs，model=None。
        预期：返回 StandingHighStrategyConfig，type=standing_high，含 template、metadata、anti_outlier_strategies。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighWithExperienceStrategy(config=config, dataset_loader=loader)
        structure_configs = [StructureConfig(type="MHA", include=["*"], exclude=[])]
        # model=None 时不走 _filter_supported_anti_outlier_strategies，直接使用 builder 产出
        base = strategy._generate_base_config(config, structure_configs, model=None)
        assert base.type == "standing_high"
        assert base.template is not None
        assert base.metadata is not None
        assert base.anti_outlier_strategies is not None
        assert len(base.anti_outlier_strategies) >= 1

    def test_generate_base_config_raises_SchemaValidateError_when_builder_returns_no_anti_outlier_strategies(self):
        """
        场景：builder.get_tuning_search_space 返回的 anti_outlier_strategies 为 None。
        预期：抛出 SchemaValidateError 且消息含 anti_outlier。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighWithExperienceStrategy(config=config, dataset_loader=loader)
        structure_configs = [StructureConfig(type="MHA", include=["*"], exclude=[])]
        with patch(
            "msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.ExpertExperienceConfigBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.build.return_value = MagicMock(spec=MagicMock, metadata=MagicMock())
            mock_builder.get_tuning_search_space.return_value = MagicMock(anti_outlier_strategies=None)
            mock_builder_cls.return_value = mock_builder
            with pytest.raises(SchemaValidateError) as exc_info:
                strategy._generate_base_config(config, structure_configs, model=None)
            assert "anti_outlier" in str(exc_info.value).lower()

    @patch(
        "msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.StandingHighWithExperienceStrategy._filter_supported_anti_outlier_strategies"
    )
    def test_generate_base_config_raises_SchemaValidateError_when_filter_returns_empty(self, mock_filter):
        """
        场景：model 非 None，_filter_supported_anti_outlier_strategies 返回空列表。
        预期：抛出 SchemaValidateError 且消息含 No supported 或 filtering。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighWithExperienceStrategy(config=config, dataset_loader=loader)
        structure_configs = [StructureConfig(type="MHA", include=["*"], exclude=[])]
        mock_filter.return_value = []  # 全部被过滤掉
        with pytest.raises(SchemaValidateError) as exc_info:
            strategy._generate_base_config(config, structure_configs, model=_MockModel())
        assert "No supported" in str(exc_info.value) or "filtering" in str(exc_info.value).lower()

    def test_filter_supported_anti_outlier_strategies_returns_all_unchanged_when_load_model_fails(self):
        """
        场景：model.load_model 抛出异常。
        预期：返回原始 strategies 列表，不进行过滤。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighWithExperienceStrategy(config=config, dataset_loader=loader)
        model = _MockModel()
        model.load_model = MagicMock(side_effect=RuntimeError("load failed"))
        strategies = [[{"type": "flex_smooth_quant"}]]
        filtered = strategy._filter_supported_anti_outlier_strategies(strategies, model)
        assert filtered == strategies

    @patch("msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.AutoSessionProcessor.from_config")
    def test_filter_supported_anti_outlier_strategies_skips_group_when_from_config_raises_UnsupportedError(
        self, mock_from_config
    ):
        """
        场景：某组策略的 AutoSessionProcessor.from_config 抛出 UnsupportedError。
        预期：该组被过滤掉，其余组保留。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighWithExperienceStrategy(config=config, dataset_loader=loader)
        model = _MockModel()
        # 第一组第一个 processor 不支持，第二组支持
        mock_from_config.side_effect = [UnsupportedError("not supported"), None]
        strategies = [
            [{"type": "flex_smooth_quant"}],
            [{"type": "iter_smooth"}],
        ]
        filtered = strategy._filter_supported_anti_outlier_strategies(strategies, model)
        assert len(filtered) == 1
        assert filtered[0] == strategies[1]

    def test_filter_supported_anti_outlier_strategies_returns_filtered_when_to_meta_raises(self):
        """
        场景：loaded_model.to('meta') 抛出异常。
        预期：仍返回已过滤的策略列表（不因 to_meta 失败而丢失结果）。
        """
        config = self._make_config()
        loader = self._make_dataset_loader()
        strategy = StandingHighWithExperienceStrategy(config=config, dataset_loader=loader)
        model = _MockModel()
        loaded = MagicMock()
        loaded.to = MagicMock(side_effect=RuntimeError("to meta failed"))
        model.load_model = MagicMock(return_value=loaded)
        with patch(
            "msmodelslim.core.tune_strategy.standing_high_with_experience.strategy.AutoSessionProcessor.from_config"
        ) as mock_from_config:
            mock_from_config.return_value = None
            strategies = [[{"type": "flex_smooth_quant"}]]
            filtered = strategy._filter_supported_anti_outlier_strategies(strategies, model)
        assert len(filtered) == 1
        assert filtered[0] == strategies[0]

    def test_get_plugin_returns_config_and_strategy_classes_when_called(self):
        """
        场景：调用 get_plugin()。
        预期：返回 (StandingHighWithExperienceStrategyConfig, StandingHighWithExperienceStrategy) 元组。
        """
        config_cls, strategy_cls = get_plugin()
        assert config_cls is StandingHighWithExperienceStrategyConfig
        assert strategy_cls is StandingHighWithExperienceStrategy
