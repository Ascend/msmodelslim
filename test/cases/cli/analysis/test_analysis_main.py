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

from argparse import Namespace
from unittest.mock import patch

import pytest

from msmodelslim.cli.analysis.__main__ import get_dataset_dir, main


class TestGetDatasetDir:  # pylint: disable=duplicate-code
    """Test suite for get_dataset_dir — analysis 模块数据集目录解析。"""

    def test_get_dataset_dir_returns_existing_path(self):
        """主路径：返回路径应存在。"""
        from pathlib import Path

        result = get_dataset_dir()

        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()
        assert "lab_calib" in str(result)


class TestAnalysisMain:
    """Test suite for main(args) — analysis 入口编排。"""

    def _make_args(self, **overrides):
        defaults = dict(
            model_type="qwen3",
            model_path="/fake/model",
            device="npu",
            scope="linear",
            metrics="kurtosis",
            pattern=["*"],
            quant_modules=["*"],
            calib_dataset="mix_calib.jsonl",
            topk=15,
            trust_remote_code=False,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    @patch("msmodelslim.cli.analysis.__main__.LayerAnalysisApplication")
    @patch("msmodelslim.cli.analysis.__main__.LoggingAnalysisResultDisplayer")
    @patch("msmodelslim.cli.analysis.__main__.YamlAnalysisPipelineLoader")
    def test_main_dispatches_to_linear_scope_when_scope_is_linear(self, mock_pl_cls, mock_disp_cls, mock_app_cls):
        """主路径：scope=linear 时应构造 LinearArgs 并调用 analyze。"""
        from unittest.mock import MagicMock

        mock_pl_instance = MagicMock()
        mock_pl_cls.return_value = mock_pl_instance
        mock_disp_instance = MagicMock()
        mock_disp_cls.return_value = mock_disp_instance
        mock_app_instance = MagicMock()
        mock_app_cls.return_value = mock_app_instance

        args = self._make_args(scope="linear", metrics="kurtosis")
        main(args)

        # analyze 应被调用
        mock_app_instance.analyze.assert_called_once()
        call_kwargs = mock_app_instance.analyze.call_args.kwargs
        assert call_kwargs["model_type"] == "qwen3"
        assert call_kwargs["topk"] == 15

    @patch("msmodelslim.cli.analysis.__main__.LayerAnalysisApplication")
    @patch("msmodelslim.cli.analysis.__main__.LoggingAnalysisResultDisplayer")
    @patch("msmodelslim.cli.analysis.__main__.YamlAnalysisPipelineLoader")
    def test_main_dispatches_to_layer_scope_when_scope_is_layer(self, mock_pl_cls, mock_disp_cls, mock_app_cls):
        """主路径：scope=layer 时应构造 LayerArgs。"""
        from unittest.mock import MagicMock

        mock_pl_cls.return_value = MagicMock()
        mock_disp_cls.return_value = MagicMock()
        mock_app_instance = MagicMock()
        mock_app_cls.return_value = mock_app_instance

        args = self._make_args(scope="layer", metrics="mse_layer_wise")
        main(args)

        mock_app_instance.analyze.assert_called_once()

    @patch("msmodelslim.cli.analysis.__main__.LayerAnalysisApplication")
    @patch("msmodelslim.cli.analysis.__main__.LoggingAnalysisResultDisplayer")
    @patch("msmodelslim.cli.analysis.__main__.YamlAnalysisPipelineLoader")
    def test_main_dispatches_to_attn_scope_when_scope_is_attn(self, mock_pl_cls, mock_disp_cls, mock_app_cls):
        """主路径：scope=attn 时应构造 AttnArgs。"""
        from unittest.mock import MagicMock

        mock_pl_cls.return_value = MagicMock()
        mock_disp_cls.return_value = MagicMock()
        mock_app_instance = MagicMock()
        mock_app_cls.return_value = mock_app_instance

        args = self._make_args(scope="attn", metrics="mse")
        main(args)

        mock_app_instance.analyze.assert_called_once()

    def test_main_raises_value_error_when_scope_unsupported(self):
        """异常：scope 非 linear/layer/attn 时应抛 ValueError。"""
        args = self._make_args(scope="unknown_scope")

        with pytest.raises(ValueError, match="Unsupported analyze scope"):
            main(args)
