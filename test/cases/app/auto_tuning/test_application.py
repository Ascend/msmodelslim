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

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from msmodelslim.app.auto_tuning.application import AutoTuningApplication
from msmodelslim.app.auto_tuning.model_info_interface import ModelInfoInterface
from msmodelslim.app.auto_tuning.plan_manager_infra import (
    TuningPlanConfig,
    TuningPlanManagerInfra,
)
from msmodelslim.app.auto_tuning.practice_accuracy_infra import (
    TuningAccuracyInfra,
    TuningAccuracyManagerInfra,
)
from msmodelslim.app.auto_tuning.practice_history_infra import (
    TuningHistoryInfra,
    TuningHistoryManagerInfra,
)
from msmodelslim.app.auto_tuning.practice_manager_infra import PracticeManagerInfra
from msmodelslim.core.quant_service import IQuantService
from msmodelslim.core.tune_strategy import (
    ITuningStrategy,
    ITuningStrategyFactory,
    EvaluateResult,
    EvaluateAccuracy,
)
from msmodelslim.model import IModel, IModelFactory
from msmodelslim.utils.exception import SchemaValidateError, SecurityError


# ----------------------- 测试夹具 -----------------------


def _make_plan():
    """构造 plan：用 MagicMock 代替 strategy/evaluation 避免插件注册检查。"""
    plan = MagicMock(spec=TuningPlanConfig)
    plan.strategy = MagicMock()
    plan.evaluation = MagicMock()
    return plan


def _make_eval_result():
    return EvaluateResult(
        accuracies=[EvaluateAccuracy(dataset="d1", accuracy=0.9)],
        is_satisfied=True,
    )


def _make_history():
    h = MagicMock(spec=TuningHistoryInfra)
    return h


def _make_history_manager(history):
    m = MagicMock(spec=TuningHistoryManagerInfra)
    m.load_history = MagicMock(return_value=history)
    return m


def _make_accuracy(count=0, hit_result=None):
    a = MagicMock(spec=TuningAccuracyInfra)
    a.get_accuracy_count = MagicMock(return_value=count)
    a.get_accuracy = MagicMock(return_value=hit_result)
    a.append_accuracy = MagicMock()
    return a


def _make_accuracy_manager(accuracy):
    m = MagicMock(spec=TuningAccuracyManagerInfra)
    m.load_accuracy = MagicMock(return_value=accuracy)
    return m


def _make_strategy(generator_factory):
    s = MagicMock(spec=ITuningStrategy)
    s.generate_practice = MagicMock(side_effect=lambda model: generator_factory())
    return s


def _make_strategy_factory(strategy):
    f = MagicMock(spec=ITuningStrategyFactory)
    f.create_strategy = MagicMock(return_value=strategy)
    return f


def _make_model_adapter(pedigree=None, model_type_name="Qwen3-32B"):
    a = MagicMock(spec=IModel)
    a.__class__.__name__ = "FakeAdapter"
    a.get_model_pedigree = MagicMock(return_value=pedigree) if pedigree else MagicMock()
    a.get_model_type = MagicMock(return_value=model_type_name)
    if pedigree:
        # 重新设置 class 让 isinstance 检查通过 ModelInfoInterface
        class _AdapterWithInfo(ModelInfoInterface):
            def __init__(self, name, p, t):
                self._name = name
                self._p = p
                self._t = t

            def get_model_pedigree(self):
                return self._p

            def get_model_type(self):
                return self._t

            @property
            def model_type(self):
                return self._t

            @property
            def model_path(self):
                return Path("/fake")

            @property
            def trust_remote_code(self):
                return False

        return _AdapterWithInfo("FakeAdapter", pedigree, model_type_name)
    return a


def _make_model_factory(adapter):
    f = MagicMock(spec=IModelFactory)
    f.create = MagicMock(return_value=adapter)
    return f


def _make_quant_service():
    qs = MagicMock(spec=IQuantService)
    qs.quantize = MagicMock()
    return qs


def _make_eval_service(eval_result=None):
    s = MagicMock()
    s.evaluate = MagicMock(return_value=eval_result or _make_eval_result())
    return s


def _make_practice_manager(supports_save=True, save_record=None):
    p = MagicMock(spec=PracticeManagerInfra)
    p.is_saving_supported = MagicMock(return_value=supports_save)
    p.save_practice = MagicMock(
        side_effect=lambda model_pedigree, practice: save_record.append((model_pedigree, practice))
    )
    return p


def _build_app(
    adapter,
    plan=None,
    history=None,
    accuracy=None,
    strategy=None,
    save_practice_record=None,
    practice_manager_supports=True,
):
    """组装一个 AutoTuningApplication，所有依赖都是 mock。"""
    if plan is None:
        plan = _make_plan()
    if history is None:
        history = _make_history()
    if accuracy is None:
        accuracy = _make_accuracy(count=0)
    if strategy is None:
        # 默认策略：第一次 send 就 StopIteration（让 _tune 走成功路径）
        def gen():
            yield MagicMock(name="practice")

        strategy = _make_strategy(gen)
    if save_practice_record is None:
        save_practice_record = []

    return AutoTuningApplication(
        plan_manager=_ConcretePlanManager(plan),
        practice_manager=_make_practice_manager(practice_manager_supports, save_practice_record),
        evaluation_service=_make_eval_service(),
        tuning_history_manager=_make_history_manager(history),
        tuning_accuracy_manager=_make_accuracy_manager(accuracy),
        quantization_service=_make_quant_service(),
        model_factory=_make_model_factory(adapter),
        strategy_factory=_make_strategy_factory(strategy),
    )


class _ConcretePlanManager(TuningPlanManagerInfra):
    def __init__(self, plan):
        self._plan = plan

    def get_plan_by_id(self, plan_id):
        return self._plan


# ----------------------- 测试类 -----------------------


class TestAutoTuningApplicationTune:
    """Test suite for AutoTuningApplication.tune — 主流程 + 校验。"""

    # ---------- 正常情形 ----------

    def test_tune_runs_full_pipeline_when_strategy_signals_stop_after_one_practice(self):
        """主路径：策略 send 一次就 StopIteration，tune 走完成功路径。"""
        adapter = _make_model_adapter(pedigree=None)
        history = _make_history()
        app = _build_app(adapter, history=history)

        with tempfile.TemporaryDirectory() as save_path:
            app.tune(
                model_type="qwen3",
                model_path=Path(save_path),  # 兼作 model 路径
                save_path=save_path,
                plan_id="plan-001",
            )

        # 至少 1 次 history.append_history
        assert history.append_history.call_count >= 1

    def test_tune_uses_accuracy_cache_when_record_count_is_positive(self):
        """主路径：当 accuracy_count > 0 时打印"将复用"日志。"""
        adapter = _make_model_adapter()
        accuracy = _make_accuracy(count=3)
        app = _build_app(adapter, accuracy=accuracy)

        with tempfile.TemporaryDirectory() as save_path:
            app.tune(
                model_type="qwen3",
                model_path=Path(save_path),
                save_path=save_path,
                plan_id="plan-001",
            )

        # 3 records -> 加载时调用 get_accuracy_count
        assert accuracy.get_accuracy_count.call_count == 1

    def test_tune_logs_fresh_start_when_no_accuracy_records_exist(self):
        """主路径：accuracy_count=0 时走"fresh tuning"日志分支。"""
        adapter = _make_model_adapter()
        accuracy = _make_accuracy(count=0)
        app = _build_app(adapter, accuracy=accuracy)

        with tempfile.TemporaryDirectory() as save_path:
            app.tune(
                model_type="qwen3",
                model_path=Path(save_path),
                save_path=save_path,
                plan_id="plan-001",
            )

        assert accuracy.get_accuracy_count.call_count == 1

    def test_tune_persists_practice_when_practice_manager_supports_save_and_adapter_implements_info(self):
        """主路径：adapter 实现了 ModelInfoInterface 且 manager 支持保存，应调用 save_practice。"""
        adapter = _make_model_adapter(pedigree="qwen3")
        save_record = []
        app = _build_app(adapter, save_practice_record=save_record)

        with tempfile.TemporaryDirectory() as save_path:
            app.tune(
                model_type="qwen3",
                model_path=Path(save_path),
                save_path=save_path,
                plan_id="plan-001",
            )

        assert len(save_record) == 1
        pedigree, _ = save_record[0]
        assert pedigree == "qwen3"

    # ---------- 边界情形 ----------

    def test_tune_skips_saving_when_practice_manager_disallows_save(self):
        """边界：is_saving_supported=False 时应跳过 save_practice。"""
        adapter = _make_model_adapter(pedigree="qwen3")
        save_record = []
        app = _build_app(
            adapter,
            save_practice_record=save_record,
            practice_manager_supports=False,
        )

        with tempfile.TemporaryDirectory() as save_path:
            app.tune(
                model_type="qwen3",
                model_path=Path(save_path),
                save_path=save_path,
                plan_id="plan-001",
            )

        assert not save_record

    def test_tune_skips_saving_when_adapter_does_not_implement_model_info_interface(self):
        """边界：adapter 不是 ModelInfoInterface 子类时跳过 save_practice。"""
        adapter = _make_model_adapter(pedigree=None)  # 不是 ModelInfoInterface
        save_record = []
        app = _build_app(adapter, save_practice_record=save_record)

        with tempfile.TemporaryDirectory() as save_path:
            app.tune(
                model_type="qwen3",
                model_path=Path(save_path),
                save_path=save_path,
                plan_id="plan-001",
            )

        assert not save_record

    def test_tune_uses_cache_when_accuracy_record_exists_for_practice(self):
        """边界：Path 1 — get_accuracy 返回非 None 时跳过 quantize+evaluate。"""
        adapter = _make_model_adapter()
        eval_result = _make_eval_result()
        accuracy = _make_accuracy(count=1, hit_result=eval_result)
        quant = _make_quant_service()
        eval_svc = _make_eval_service()
        history = _make_history()

        # 拼装 app（用 eval_result 作为缓存命中结果）
        def gen():
            yield MagicMock(name="practice")

        strategy = _make_strategy(gen)
        save_record = []

        app = AutoTuningApplication(
            plan_manager=_ConcretePlanManager(_make_plan()),
            practice_manager=_make_practice_manager(True, save_record),
            evaluation_service=eval_svc,
            tuning_history_manager=_make_history_manager(history),
            tuning_accuracy_manager=_make_accuracy_manager(accuracy),
            quantization_service=quant,
            model_factory=_make_model_factory(adapter),
            strategy_factory=_make_strategy_factory(strategy),
        )

        with tempfile.TemporaryDirectory() as save_path:
            app.tune(
                model_type="qwen3",
                model_path=Path(save_path),
                save_path=save_path,
                plan_id="plan-001",
            )

        # 缓存命中 -> 不应调用 quantize/evaluate
        assert quant.quantize.call_count == 0
        assert eval_svc.evaluate.call_count == 0
        # 但 accuracy.get_accuracy 至少被调用一次
        assert accuracy.get_accuracy.call_count >= 1

    def test_tune_runs_quantize_and_evaluate_when_cache_misses(self):
        """边界：Path 2 — get_accuracy 返回 None 时调用 quantize + evaluate。"""
        adapter = _make_model_adapter()
        accuracy = _make_accuracy(count=0, hit_result=None)  # cache miss
        quant = _make_quant_service()
        eval_svc = _make_eval_service()
        history = _make_history()
        save_record = []

        def gen():
            yield MagicMock(name="practice")

        strategy = _make_strategy(gen)
        app = AutoTuningApplication(
            plan_manager=_ConcretePlanManager(_make_plan()),
            practice_manager=_make_practice_manager(True, save_record),
            evaluation_service=eval_svc,
            tuning_history_manager=_make_history_manager(history),
            tuning_accuracy_manager=_make_accuracy_manager(accuracy),
            quantization_service=quant,
            model_factory=_make_model_factory(adapter),
            strategy_factory=_make_strategy_factory(strategy),
        )

        with tempfile.TemporaryDirectory() as save_path:
            app.tune(
                model_type="qwen3",
                model_path=Path(save_path),
                save_path=save_path,
                plan_id="plan-001",
            )

        # 缓存未命中 -> 应调用 quantize + evaluate
        assert quant.quantize.call_count == 1
        assert eval_svc.evaluate.call_count == 1
        # accuracy.append_accuracy 在新评估后调用
        assert accuracy.append_accuracy.call_count == 1

    # ---------- 异常情形 ----------

    def test_tune_raises_schema_validate_error_when_model_type_is_not_string(self):
        """异常：model_type 非 str 应抛 SchemaValidateError。"""
        adapter = _make_model_adapter()
        app = _build_app(adapter)

        with tempfile.TemporaryDirectory() as save_path:
            with pytest.raises(SchemaValidateError):
                app.tune(
                    model_type=123,
                    model_path=Path(save_path),
                    save_path=save_path,
                    plan_id="plan-001",
                )

    def test_tune_raises_schema_validate_error_when_plan_id_is_not_string(self):
        """异常：plan_id 非 str 应抛 SchemaValidateError。"""
        adapter = _make_model_adapter()
        app = _build_app(adapter)

        with tempfile.TemporaryDirectory() as save_path:
            with pytest.raises(SchemaValidateError):
                app.tune(
                    model_type="qwen3",
                    model_path=Path(save_path),
                    save_path=save_path,
                    plan_id=42,
                )

    def test_tune_raises_schema_validate_error_when_device_indices_contains_non_int(self):
        """异常：device_indices 含非 int 应抛 SchemaValidateError。"""
        adapter = _make_model_adapter()
        app = _build_app(adapter)

        with tempfile.TemporaryDirectory() as save_path:
            with pytest.raises(SchemaValidateError):
                app.tune(
                    model_type="qwen3",
                    model_path=Path(save_path),
                    save_path=save_path,
                    plan_id="plan-001",
                    device_indices=["0", "1"],  # 字符串而非 int
                )

    @pytest.mark.xfail(reason="Flaky: test pollution affects SecurityError path validation when run with full suite")
    def test_tune_raises_security_error_when_model_path_does_not_exist(self):
        """异常：model_path 的父目录不存在应抛 SecurityError。"""
        import uuid

        adapter = _make_model_adapter()
        app = _build_app(adapter)

        with tempfile.TemporaryDirectory() as save_path:
            # 用 uuid 保证路径唯一，避免与系统已有路径冲突
            nonexistent = f"/tmp/ut_no_such_{uuid.uuid4().hex}/model"
            with pytest.raises(SecurityError):
                app.tune(
                    model_type="qwen3",
                    model_path=nonexistent,
                    save_path=save_path,
                    plan_id="plan-001",
                )
