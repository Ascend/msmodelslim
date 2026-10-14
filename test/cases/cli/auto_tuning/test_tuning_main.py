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

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch


from msmodelslim.cli.auto_tuning.__main__ import get_dataset_dir, get_practice_dir, main


class TestGetDirs:
    """Test suite for get_practice_dir / get_dataset_dir — auto_tuning 路径解析。"""

    def test_get_practice_dir_returns_existing_path(self):
        """主路径：返回路径应存在。"""
        result = get_practice_dir()

        assert isinstance(result, Path)
        assert result.exists()
        assert "lab_practice" in str(result)

    def test_get_dataset_dir_returns_existing_path(self):
        """主路径：返回路径应存在。"""
        result = get_dataset_dir()

        assert isinstance(result, Path)
        assert result.exists()
        assert "lab_calib" in str(result)


class TestAutoTuningMain:
    """Test suite for main(args) — auto_tuning 入口编排。"""

    def _make_args(self, **overrides):
        defaults = dict(
            model_type="qwen3",
            model_path="/fake/model",
            save_path="/fake/save",
            config="plan-001",
            device="npu",
            timeout=None,
            trust_remote_code=False,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    @patch("msmodelslim.cli.auto_tuning.__main__.AutoTuningApplication")
    @patch("msmodelslim.cli.auto_tuning.__main__.PluginTuningStrategyFactory")
    @patch("msmodelslim.cli.auto_tuning.__main__.YamlTuningAccuracyManager")
    @patch("msmodelslim.cli.auto_tuning.__main__.YamlTuningHistoryManager")
    @patch("msmodelslim.cli.auto_tuning.__main__.YamlPracticeManager")
    @patch("msmodelslim.cli.auto_tuning.__main__.ServiceOrientedEvaluateService")
    @patch("msmodelslim.cli.auto_tuning.__main__.YamlTuningPlanManager")
    def test_main_invokes_application_tune_with_parsed_args(
        self,
        mock_plan_cls,
        mock_eval_cls,
        mock_pm_cls,
        mock_hist_cls,
        mock_acc_cls,
        mock_strat_cls,
        mock_app_cls,
    ):
        """主路径：main 应构造 AutoTuningApplication 并调用 tune。"""
        mock_app_instance = MagicMock()
        mock_app_cls.return_value = mock_app_instance

        args = self._make_args(device="npu:0,1,2")
        main(args)

        # tune 应被调用
        mock_app_instance.tune.assert_called_once()
        call_kwargs = mock_app_instance.tune.call_args.kwargs
        assert call_kwargs["model_type"] == "qwen3"
        assert call_kwargs["plan_id"] == "plan-001"
        assert call_kwargs["device_indices"] == [0, 1, 2]
        assert call_kwargs["timeout"] is None

    @patch("msmodelslim.cli.auto_tuning.__main__.AutoTuningApplication")
    @patch("msmodelslim.cli.auto_tuning.__main__.PluginTuningStrategyFactory")
    @patch("msmodelslim.cli.auto_tuning.__main__.YamlTuningAccuracyManager")
    @patch("msmodelslim.cli.auto_tuning.__main__.YamlTuningHistoryManager")
    @patch("msmodelslim.cli.auto_tuning.__main__.YamlPracticeManager")
    @patch("msmodelslim.cli.auto_tuning.__main__.ServiceOrientedEvaluateService")
    @patch("msmodelslim.cli.auto_tuning.__main__.YamlTuningPlanManager")
    def test_main_parses_timeout_string_when_provided(
        self,
        mock_plan_cls,
        mock_eval_cls,
        mock_pm_cls,
        mock_hist_cls,
        mock_acc_cls,
        mock_strat_cls,
        mock_app_cls,
    ):
        """边界：timeout 字符串应被 AutoTuningApplication.tune 接收（内部转换）。"""
        mock_app_instance = MagicMock()
        mock_app_cls.return_value = mock_app_instance

        args = self._make_args(timeout="1D")
        main(args)

        call_kwargs = mock_app_instance.tune.call_args.kwargs
        assert call_kwargs["timeout"] == "1D"
