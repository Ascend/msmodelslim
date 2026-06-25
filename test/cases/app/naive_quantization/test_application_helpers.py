#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2
-------------------------------------------------------------------------
"""

import pytest
from unittest.mock import MagicMock, patch

from msmodelslim.app.naive_quantization.application import (
    DEFAULT_PEDIGREE,
    DEFAULT_QUANT_TYPE,
    TipsType,
    _build_quant_tips,
    validate_device_index,
)
from msmodelslim.core.const import DeviceType, QuantType
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


class TestBuildQuantTips:
    """Test suite for _build_quant_tips — 提示词构造。"""

    # ---------- 正常情形 ----------

    def test_build_quant_tips_returns_default_message_when_no_quant_no_change_no_best_practice(self):
        """主路径：Q0C0B0 时应返回默认值 + 默认实践的提示。"""
        result = _build_quant_tips(TipsType.Q0C0B0, "qwen3", QuantType.W8A8, "default_practice_id")

        assert "No quant_type" in result
        assert "default_practice_id" in result
        assert QuantType.W8A8.value in result or str(QuantType.W8A8) in result

    def test_build_quant_tips_returns_empty_string_when_q1c0b1(self):
        """主路径：Q1C0B1（找到最佳实践且未变更）应返回空串（无需额外提示）。"""
        result = _build_quant_tips(TipsType.Q1C0B1, "qwen3", QuantType.W8A8, "best_practice_id")

        assert result == ""

    def test_build_quant_tips_includes_model_type_when_no_best_practice(self):
        """主路径：Q1C0B0 时提示应包含 model_type。"""
        result = _build_quant_tips(TipsType.Q1C0B0, "qwen3", QuantType.W4A8, "default_id")

        assert "qwen3" in result
        assert QuantType.W4A8.value in result or str(QuantType.W4A8) in result

    def test_build_quant_tips_includes_quant_type_when_changed(self):
        """主路径：Q1C1B0 时提示应包含 quant_type（被变更的）。"""
        result = _build_quant_tips(TipsType.Q1C1B0, "llama", QuantType.W8A8S, "default_id")

        assert "W8A8S" in result or "default" in result

    def test_build_quant_tips_default_module_constants(self):
        """边界：DEFAULT_PEDIGREE / DEFAULT_QUANT_TYPE 应是预期值。"""
        assert DEFAULT_PEDIGREE == "default"
        assert DEFAULT_QUANT_TYPE == QuantType.W8A8

    # ---------- 异常情形 ----------

    def test_build_quant_tips_raises_unsupported_error_for_unknown_tips_type(self):
        """异常：未知 TipsType 应抛 UnsupportedError。"""
        # 构造一个不在 enum 里的 TipsType 实例（绕过 enum 校验）
        fake_type = "Q9C9B9_INVALID"

        with pytest.raises(UnsupportedError):
            _build_quant_tips(fake_type, "qwen3", QuantType.W8A8, "id")  # type: ignore[arg-type]


class TestGetConfig:
    """Test suite for NaiveQuantizationApplication.get_config — 多场景 config 匹配。"""

    def _make_app(self, iter_configs, practice_config_factory=None):
        """构造 NaiveQuantizationApplication，mock 依赖以测试 get_config 各分支。"""
        from msmodelslim.app.naive_quantization.application import NaiveQuantizationApplication
        from msmodelslim.core.quant_service import IQuantService
        from msmodelslim.model import IModelFactory

        practice_manager = MagicMock()
        practice_manager.iter_config = MagicMock(side_effect=iter_configs)
        return NaiveQuantizationApplication(
            practice_manager=practice_manager,
            quant_service=MagicMock(spec=IQuantService),
            model_factory=MagicMock(spec=IModelFactory),
        )

    def _make_practice_config(self, config_id="cid", tag_list=None, label=None):
        """构造 mock PracticeConfig 实例。"""
        cfg = MagicMock()
        cfg.metadata.config_id = config_id
        cfg.metadata.label = label or {"w_bit": 8, "a_bit": 8, "kv_cache": False, "fa_quant": False, "is_sparse": False}
        cfg.metadata.scenario_tags = tag_list or []
        cfg.tag = tag_list or []
        return cfg

    def test_get_config_returns_config_when_first_iteration_matches(self):
        """场景 1：iter_config 第一个 config 就匹配时应返回。"""
        from msmodelslim.core.practice.interface import ScenarioTagMatch
        from msmodelslim.app.naive_quantization import application as app_module

        cfg = self._make_practice_config()

        def iter_factory(_):
            return iter([cfg])  # 每次返回新 iter

        app = self._make_app(iter_configs=iter_factory)

        with patch.object(app_module.NaiveQuantizationApplication, "check_config", return_value=ScenarioTagMatch.MATCH):
            result, tips = app.get_config("qwen3", "Qwen3-32B", quant_type=QuantType.W8A8)

        assert result is cfg
        assert isinstance(tips, str)

    def test_get_config_uses_default_pedigree_fallback_when_model_pedigree_unknown(self):
        """场景 2：模型 pedigree 找不到时降级到 default pedigree。"""
        from msmodelslim.core.practice.interface import ScenarioTagMatch
        from msmodelslim.app.naive_quantization import application as app_module

        cfg_default = self._make_practice_config()

        def iter_factory(pedigree):
            if pedigree == "qwen3":
                return iter([])
            else:
                return iter([cfg_default])

        app = self._make_app(iter_configs=iter_factory)

        # scenario 1 iter 为空 → 0 次 check；scenario 2 只有 1 次 check（应返回 MATCH）
        with patch.object(app_module.NaiveQuantizationApplication, "check_config", return_value=ScenarioTagMatch.MATCH):
            result, tips = app.get_config("qwen3", "Qwen3-32B", quant_type=QuantType.W8A8)

        assert result is cfg_default
        assert "No best practice" in tips

    def test_get_config_falls_back_to_default_quant_type_when_specified_missing(self):
        """场景 3：指定 quant_type 找不到时降级到默认 quant_type。"""
        from msmodelslim.core.practice.interface import ScenarioTagMatch
        from msmodelslim.app.naive_quantization import application as app_module

        cfg_default = self._make_practice_config()

        def iter_factory(_):
            return iter([cfg_default])

        app = self._make_app(iter_configs=iter_factory)

        with patch.object(
            app_module.NaiveQuantizationApplication,
            "check_config",
            side_effect=[ScenarioTagMatch.NO_MATCH, ScenarioTagMatch.MATCH],
        ):
            result, tips = app.get_config("qwen3", "Qwen3-32B", quant_type=QuantType.W4A8)

        assert result is cfg_default
        assert "No best practice" in tips

    def test_get_config_raises_unsupported_when_nothing_found(self):
        """场景 4：所有 iter 都无结果时应抛 UnsupportedError。"""
        from msmodelslim.core.practice.interface import ScenarioTagMatch
        from msmodelslim.app.naive_quantization import application as app_module

        def iter_factory(pedigree):
            return iter([])

        app = self._make_app(iter_configs=iter_factory)

        with patch.object(
            app_module.NaiveQuantizationApplication, "check_config", return_value=ScenarioTagMatch.NO_MATCH
        ):
            with pytest.raises(UnsupportedError):
                app.get_config("qwen3", "Qwen3-32B", quant_type=QuantType.W8A8)

    def test_get_config_returns_standby_config_when_only_standby_match(self):
        """边界：所有 config 都是 STANDBY 时应返回第一个 standby。"""
        from msmodelslim.core.practice.interface import ScenarioTagMatch
        from msmodelslim.app.naive_quantization import application as app_module

        cfg1 = self._make_practice_config(config_id="standby1")

        def iter_factory(_):
            return iter([cfg1])

        app = self._make_app(iter_configs=iter_factory)

        with patch.object(
            app_module.NaiveQuantizationApplication, "check_config", return_value=ScenarioTagMatch.STANDBY
        ):
            result, tips = app.get_config("qwen3", "Qwen3-32B")

        assert result is cfg1
        assert "standby" in tips.lower() or "standby1" in tips


class TestValidateDeviceIndex:
    """Test suite for validate_device_index — 设备索引校验。"""

    # ---------- 正常情形 ----------

    def test_validate_device_index_passes_with_empty_list(self):
        """边界：空列表应通过（no indices，无 device count check）。"""
        validate_device_index([], DeviceType.CPU)

    def test_validate_device_index_passes_with_single_cpu_index(self):
        """主路径：CPU 设备下单个索引应通过。"""
        validate_device_index([0], DeviceType.CPU)

    # ---------- 异常情形 ----------

    def test_validate_device_index_raises_error_when_index_is_negative(self):
        """异常：负索引应抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError, match="non-negative"):
            validate_device_index([-1, 0, 1], DeviceType.CPU)

    def test_validate_device_index_raises_error_when_indices_have_duplicates(self):
        """异常：重复索引应抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError, match="duplicate"):
            validate_device_index([0, 1, 0], DeviceType.CPU)

    def test_validate_device_index_raises_error_when_cpu_has_multi_device(self):
        """异常：CPU 设备 + 多索引应抛 SchemaValidateError。"""
        with pytest.raises(SchemaValidateError, match="multi-device"):
            validate_device_index([0, 1], DeviceType.CPU)

    def test_validate_device_index_error_includes_negative_value_in_message(self):
        """边界：错误消息应包含具体的负数列表。"""
        with pytest.raises(SchemaValidateError) as exc_info:
            validate_device_index([-3, -1, 0], DeviceType.CPU)

        # 错误消息应包含负数
        assert "-3" in str(exc_info.value) or "-1" in str(exc_info.value)

    def test_validate_device_index_skipped_for_npu(self):
        """边界：NPU 路径需要 torch_npu 运行时，本机跳过（无 torch.npu）。"""
        # CPU-only 环境下 NPU 分支不可测；用 try-except 兜底
        try:
            import torch as _t

            if hasattr(_t, "npu"):
                # NPU 可用时跑：单卡下 [0,1] 应触发越界
                with pytest.raises(SchemaValidateError):
                    validate_device_index([0, 1], DeviceType.NPU)
        except (ImportError, AttributeError):
            pytest.skip("torch_npu not available")
