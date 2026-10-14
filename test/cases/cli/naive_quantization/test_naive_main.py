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
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch


from msmodelslim.cli.naive_quantization.__main__ import get_dataset_dir, get_practice_dir, main


class TestGetPracticeDir:
    """Test suite for get_practice_dir — 解析官方 practice 目录。"""

    def test_get_practice_dir_returns_existing_path(self):
        """主路径：返回的路径应存在（lab_practice/ 已在仓库内）。"""
        result = get_practice_dir()

        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()

    def test_get_practice_dir_points_to_lab_practice_subdir(self):
        """主路径：路径名应含 'lab_practice'。"""
        result = get_practice_dir()

        assert "lab_practice" in str(result)


class TestGetDatasetDir:  # pylint: disable=duplicate-code
    """Test suite for get_dataset_dir — 解析校准数据集目录。"""

    def test_get_dataset_dir_returns_existing_path(self):
        """主路径：返回的路径应存在（lab_calib/ 已在仓库内）。"""
        result = get_dataset_dir()

        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()

    def test_get_dataset_dir_points_to_lab_calib_subdir(self):
        """主路径：路径名应含 'lab_calib'。"""
        result = get_dataset_dir()

        assert "lab_calib" in str(result)


class TestMainDispatch:
    """Test suite for main(args) — naive_quantization 入口编排。"""

    def _make_args(self, **overrides):
        defaults = dict(
            model_type="qwen3",
            model_path="/fake/model",
            save_path="/fake/save",
            device="npu",
            config_path=None,
            quant_type="w8a8",
            trust_remote_code=False,
            debug=False,
            tag=None,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    @patch("msmodelslim.cli.naive_quantization.__main__.NaiveQuantizationApplication")
    @patch("msmodelslim.cli.naive_quantization.__main__.QuantServiceProxy")
    @patch("msmodelslim.cli.naive_quantization.__main__.YamlPracticeManager")
    @patch("msmodelslim.cli.naive_quantization.__main__.discover_plugin_practice_dirs")
    def test_main_invokes_application_quant_with_parsed_args_when_called(
        self, mock_discover, mock_pm_cls, mock_qs_cls, mock_app_cls
    ):
        """主路径：main 应构造 app 并调用 app.quant(args)。"""
        mock_discover.return_value = []
        mock_pm_instance = MagicMock()
        mock_pm_cls.return_value = mock_pm_instance
        mock_qs_instance = MagicMock()
        mock_qs_cls.return_value = mock_qs_instance
        mock_app_instance = MagicMock()
        mock_app_cls.return_value = mock_app_instance

        with tempfile.TemporaryDirectory() as tmp:
            args = self._make_args(
                model_path=tmp,  # 用真实存在路径
                save_path=tmp,
            )
            main(args)

        # 应调用 app.quant 一次
        mock_app_instance.quant.assert_called_once()
        # 透传的关键参数
        call_kwargs = mock_app_instance.quant.call_args.kwargs
        assert call_kwargs["model_type"] == "qwen3"
        assert call_kwargs["quant_type"] == "w8a8"

    @patch("msmodelslim.cli.naive_quantization.__main__.NaiveQuantizationApplication")
    @patch("msmodelslim.cli.naive_quantization.__main__.QuantServiceProxy")
    @patch("msmodelslim.cli.naive_quantization.__main__.YamlPracticeManager")
    @patch("msmodelslim.cli.naive_quantization.__main__.discover_plugin_practice_dirs")
    def test_main_creates_debug_persistence_when_debug_flag_set(
        self, mock_discover, mock_pm_cls, mock_qs_cls, mock_app_cls
    ):
        """边界：debug=True 时应构造 DebugInfoPersistence 并传给 QuantServiceProxy。"""
        mock_discover.return_value = []
        mock_pm_instance = MagicMock()
        mock_pm_cls.return_value = mock_pm_instance
        mock_qs_instance = MagicMock()
        mock_qs_cls.return_value = mock_qs_instance
        mock_app_instance = MagicMock()
        mock_app_cls.return_value = mock_app_instance

        with tempfile.TemporaryDirectory() as tmp:
            args = self._make_args(
                model_path=tmp,
                save_path=tmp,
                debug=True,
            )
            main(args)

        # QuantServiceProxy 应被调用，配置中应含 debug_info_persistence
        mock_qs_cls.assert_called_once()
        call_kwargs = mock_qs_cls.call_args.kwargs
        assert call_kwargs["debug_info_persistence"] is not None

    @patch("msmodelslim.cli.naive_quantization.__main__.NaiveQuantizationApplication")
    @patch("msmodelslim.cli.naive_quantization.__main__.QuantServiceProxy")
    @patch("msmodelslim.cli.naive_quantization.__main__.YamlPracticeManager")
    @patch("msmodelslim.cli.naive_quantization.__main__.discover_plugin_practice_dirs")
    def test_main_skips_debug_persistence_when_debug_flag_false(
        self, mock_discover, mock_pm_cls, mock_qs_cls, mock_app_cls
    ):
        """边界：debug=False 时 debug_info_persistence 应为 None。"""
        mock_discover.return_value = []
        mock_pm_instance = MagicMock()
        mock_pm_cls.return_value = mock_pm_instance
        mock_qs_instance = MagicMock()
        mock_qs_cls.return_value = mock_qs_instance
        mock_app_instance = MagicMock()
        mock_app_cls.return_value = mock_app_instance

        with tempfile.TemporaryDirectory() as tmp:
            args = self._make_args(
                model_path=tmp,
                save_path=tmp,
                debug=False,
            )
            main(args)

        call_kwargs = mock_qs_cls.call_args.kwargs
        assert call_kwargs["debug_info_persistence"] is None

    @patch("msmodelslim.cli.naive_quantization.__main__.NaiveQuantizationApplication")
    @patch("msmodelslim.cli.naive_quantization.__main__.QuantServiceProxy")
    @patch("msmodelslim.cli.naive_quantization.__main__.YamlPracticeManager")
    @patch("msmodelslim.cli.naive_quantization.__main__.discover_plugin_practice_dirs")
    def test_main_parses_device_string_to_extract_indices(self, mock_discover, mock_pm_cls, mock_qs_cls, mock_app_cls):
        """边界：device='npu:0,1' 应被 parse_device_string 解析后传给 app.quant。"""
        mock_discover.return_value = []
        mock_pm_instance = MagicMock()
        mock_pm_cls.return_value = mock_pm_instance
        mock_qs_instance = MagicMock()
        mock_qs_cls.return_value = mock_qs_instance
        mock_app_instance = MagicMock()
        mock_app_cls.return_value = mock_app_instance

        with tempfile.TemporaryDirectory() as tmp:
            args = self._make_args(
                model_path=tmp,
                save_path=tmp,
                device="npu:0,1",
            )
            main(args)

        call_kwargs = mock_app_instance.quant.call_args.kwargs
        # device_index 应该是 [0, 1]
        assert call_kwargs["device_index"] == [0, 1]
        assert call_kwargs["device_type"].value == "npu"
