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

msmodelslim.infra.yaml_practice_accuracy_manager 模块的单元测试
"""

from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from pydantic import BaseModel
from msmodelslim.infra.yaml_practice_accuracy_manager import (
    YamlTuningAccuracy,
    YamlTuningAccuracyManager,
)
from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.tune_strategy import EvaluateResult, EvaluateAccuracy
from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateServiceConfig
from msmodelslim.utils.yaml_database import YamlDatabase


class MockEvaluateServiceConfig(EvaluateServiceConfig, BaseModel):
    type: str = "test_type"
    test_field: str = "test_value"


def create_test_practice_config(config_id: str = "test_config") -> PracticeConfig:
    return PracticeConfig.model_validate(
        {
            "apiversion": "modelslim_v1",
            "metadata": {"config_id": config_id},
            "spec": {},
        }
    )


def create_test_evaluate_result(accuracy: Decimal = Decimal('0.95')) -> EvaluateResult:
    return EvaluateResult(accuracies=[EvaluateAccuracy(dataset="test_dataset", accuracy=accuracy)], is_satisfied=True)


def create_test_evaluation_config(config_id: str = "test_eval") -> MockEvaluateServiceConfig:
    return MockEvaluateServiceConfig(type="test_type", test_field=config_id)


class TestYamlTuningAccuracy:
    """测试YamlTuningAccuracy类"""

    def test_init_set_empty_cache_when_initialize(self, tmp_path: Path):
        """测试初始化空精度管理器"""
        accuracy_dir = tmp_path / "accuracy"
        accuracy = YamlTuningAccuracy(str(accuracy_dir))

        assert accuracy.database == str(accuracy_dir)  # 校验database正确设置
        assert accuracy.history_dir == accuracy_dir  # 校验history_dir正确设置
        assert accuracy.get_accuracy_count() == 0  # 校验get_accuracy_count返回0

    def test_load_accuracy_database_return_cache_when_data_valid_none_invalid(self, tmp_path: Path):
        """A valid flat entry loads one record; None / non-dict content loads an empty cache."""
        accuracy_dir = tmp_path / "accuracy"
        accuracy_dir.mkdir()
        accuracy_file = accuracy_dir / "accuracy.yaml"

        valid_entry = (
            "eval-md5-quant-md5:\n  accuracies:\n  - dataset: test1\n    accuracy: 0.9\n  is_satisfied: true\n"
        )
        accuracy_file.write_text(valid_entry, encoding="utf-8")
        accuracy = YamlTuningAccuracy(str(accuracy_dir))
        assert accuracy.get_accuracy_count() == 1  # one valid entry loaded

        accuracy_file.write_text("key: null\n", encoding="utf-8")
        accuracy2 = YamlTuningAccuracy(str(accuracy_dir))
        assert accuracy2.get_accuracy_count() == 0  # None content loads empty cache

        accuracy_file.write_text("key: invalid_string\n", encoding="utf-8")
        accuracy3 = YamlTuningAccuracy(str(accuracy_dir))
        assert accuracy3.get_accuracy_count() == 0  # non-dict content loads empty cache

    def test_get_accuracy_return_none_when_not_exist_and_return_result_when_exist(self, tmp_path: Path):
        """测试精度缓存为空和已添加精度记录时获取精度"""
        accuracy_dir = tmp_path / "accuracy"
        accuracy = YamlTuningAccuracy(str(accuracy_dir))

        practice = create_test_practice_config("test_config")
        evaluation_config = create_test_evaluation_config()
        evaluate_result = create_test_evaluate_result(Decimal('0.95'))

        assert accuracy.get_accuracy(practice, evaluation_config) is None  # 校验未添加精度时返回None

        accuracy.append_accuracy(practice, evaluation_config, evaluate_result)
        result = accuracy.get_accuracy(practice, evaluation_config)
        assert isinstance(result, EvaluateResult)  # 校验添加精度后返回结果
        assert result.accuracies[0].accuracy == Decimal('0.95')  # 校验精度值正确

    def test_append_accuracy_adds_new_record_when_first_time(self, tmp_path: Path):
        """Add one record on first append."""
        accuracy_dir = tmp_path / "accuracy"
        accuracy = YamlTuningAccuracy(str(accuracy_dir))

        practice = create_test_practice_config("test_config")
        evaluation_config = create_test_evaluation_config()
        evaluate_result = create_test_evaluate_result(Decimal('0.9'))

        accuracy.append_accuracy(practice, evaluation_config, evaluate_result)

        assert accuracy.get_accuracy_count() == 1  # one record stored
        result = accuracy.get_accuracy(practice, evaluation_config)
        assert isinstance(result, EvaluateResult)  # retrievable via public API
        assert result.accuracies[0].accuracy == Decimal('0.9')  # stored value is returned

    def test_append_accuracy_overwrites_existing_record(self, tmp_path: Path):
        """Same composite key is overwritten in place without growing the cache."""
        accuracy_dir = tmp_path / "accuracy"
        accuracy = YamlTuningAccuracy(str(accuracy_dir))

        practice = create_test_practice_config("test_config")
        evaluation_config = create_test_evaluation_config()
        first_result = create_test_evaluate_result(Decimal('0.9'))
        second_result = create_test_evaluate_result(Decimal('0.95'))

        accuracy.append_accuracy(practice, evaluation_config, first_result)
        accuracy.append_accuracy(practice, evaluation_config, second_result)

        assert accuracy.get_accuracy_count() == 1  # still one record after overwrite
        result = accuracy.get_accuracy(practice, evaluation_config)
        assert isinstance(result, EvaluateResult)  # retrievable via public API
        assert result.accuracies[0].accuracy == Decimal('0.95')  # overwritten value is returned

    def test_get_accuracy_count_return_zero_when_empty_and_return_one_when_added(self, tmp_path: Path):
        """测试精度缓存为空和添加一条精度记录时获取精度缓存数量"""
        accuracy_dir = tmp_path / "accuracy"
        accuracy = YamlTuningAccuracy(str(accuracy_dir))

        assert accuracy.get_accuracy_count() == 0  # 校验初始状态返回0

        practice = create_test_practice_config("config1")
        evaluation_config = create_test_evaluation_config("eval1")
        evaluate_result = create_test_evaluate_result(Decimal('0.95'))

        accuracy.append_accuracy(practice, evaluation_config, evaluate_result)
        assert accuracy.get_accuracy_count() == 1  # 校验添加记录后返回1

    def test_composite_key_format_evaluation_md5_quant_config_md5_when_append_accuracy(self, tmp_path: Path):
        """测试添加精度记录后验证复合键格式和持久化功能"""
        accuracy_dir = tmp_path / "accuracy"
        accuracy_dir.mkdir()
        accuracy = YamlTuningAccuracy(str(accuracy_dir))

        practice = create_test_practice_config("test_config")
        evaluation_config = create_test_evaluation_config()
        evaluate_result = create_test_evaluate_result(Decimal('0.95'))

        accuracy.append_accuracy(practice, evaluation_config, evaluate_result)

        result = accuracy.get_accuracy(practice, evaluation_config)
        assert isinstance(result, EvaluateResult)  # 校验能通过复合键获取精度
        assert result.accuracies[0].accuracy == Decimal('0.95')  # 校验精度值正确

        reloaded = YamlTuningAccuracy(str(accuracy_dir))
        persisted_result = reloaded.get_accuracy(practice, evaluation_config)
        assert isinstance(persisted_result, EvaluateResult)  # persisted record reloads via public API
        assert persisted_result.accuracies[0].accuracy == Decimal('0.95')  # persisted value matches


class TestYamlTuningAccuracyManager:
    """测试YamlTuningAccuracyManager类"""

    def test_load_accuracy_return_yaml_tuning_accuracy_when_empty_and_with_data(self, tmp_path: Path):
        """测试空目录和包含已存在精度数据的目录下加载精度缓存"""
        manager = YamlTuningAccuracyManager()
        accuracy_dir = tmp_path / "accuracy"

        accuracy = manager.load_accuracy(str(accuracy_dir))
        assert isinstance(accuracy, YamlTuningAccuracy)  # 校验返回YamlTuningAccuracy实例
        assert accuracy.database == str(accuracy_dir)  # 校验database属性正确

        accuracy_dir.mkdir(exist_ok=True)
        accuracy_data = {"eval1-prac1": {"accuracies": [{"dataset": "test1", "accuracy": "0.9"}], "is_satisfied": True}}
        db = YamlDatabase(config_dir=accuracy_dir, read_only=False)
        db["accuracy"] = accuracy_data
        accuracy2 = manager.load_accuracy(str(accuracy_dir))
        assert isinstance(accuracy2, YamlTuningAccuracy)  # 校验能加载包含已存在数据的精度记录

    @patch('msmodelslim.infra.yaml_practice_accuracy_manager.YamlTuningAccuracy')
    @patch('msmodelslim.infra.yaml_practice_accuracy_manager.get_logger')
    def test_load_accuracy_wraps_constructor_failure_in_runtime_error(self, mock_logger, mock_accuracy, tmp_path: Path):
        """Constructor failures surface as a RuntimeError with a clear message and error log."""
        mock_accuracy.side_effect = ValueError("boom")
        manager = YamlTuningAccuracyManager()

        with pytest.raises(RuntimeError, match="Failed to create or load accuracy database"):
            manager.load_accuracy(str(tmp_path / "accuracy"))

        mock_logger.return_value.error.assert_called_once()
