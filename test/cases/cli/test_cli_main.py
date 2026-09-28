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

from unittest.mock import patch

import pytest

from msmodelslim.cli.__main__ import _normalize_analyze_argv, main


class TestNormalizeAnalyzeArgv:
    """Test suite for _normalize_analyze_argv — analyze 子命令的 argv 规范化。"""

    # ---------- 正常情形 ----------

    def test_normalize_returns_unchanged_when_no_analyze_subcommand(self):
        """主路径：非 analyze 命令应原样返回。"""
        argv = ["quant", "--model_type", "x"]

        result = _normalize_analyze_argv(argv)

        assert result == argv

    def test_normalize_returns_unchanged_when_empty_argv(self):
        """边界：空 argv 应原样返回。"""
        assert _normalize_analyze_argv([]) == []

    def test_normalize_injects_linear_scope_when_scope_omitted(self):
        """主路径：analyze 不带 scope 时应自动注入 'linear'。"""
        argv = ["analyze", "--model_type", "qwen3"]

        result = _normalize_analyze_argv(argv)

        # 注入后 'linear' 应紧跟 'analyze'
        assert result[0] == "analyze"
        assert result[1] == "linear"
        assert "--model_type" in result

    def test_normalize_returns_unchanged_when_explicit_scope_layer(self):
        """主路径：scope=layer 时应原样保留。"""
        argv = ["analyze", "layer", "--model_type", "x"]

        result = _normalize_analyze_argv(argv)

        assert result == argv

    def test_normalize_returns_unchanged_when_explicit_scope_attn(self):
        """主路径：scope=attn 时应原样保留。"""
        argv = ["analyze", "attn", "--model_type", "x"]

        result = _normalize_analyze_argv(argv)

        assert result == argv

    def test_normalize_returns_unchanged_when_explicit_scope_linear(self):
        """主路径：scope=linear 时应原样保留。"""
        argv = ["analyze", "linear", "--model_type", "x"]

        result = _normalize_analyze_argv(argv)

        assert result == argv

    # ---------- 边界情形 ----------

    def test_normalize_returns_unchanged_when_help_requested(self):
        """边界：analyze -h 时不应自动注入 scope（保留 help 干净）。"""
        argv = ["analyze", "-h"]

        result = _normalize_analyze_argv(argv)

        assert result == argv
        assert "linear" not in result

    def test_normalize_returns_unchanged_when_help_requested_long_form(self):
        """边界：analyze --help 时也不应注入 scope。"""
        argv = ["analyze", "--help"]

        result = _normalize_analyze_argv(argv)

        assert result == argv

    def test_normalize_converts_legacy_attention_mse_to_attn_scope(self):
        """主路径：`--metrics attention_mse` 应转为 `attn --metrics mse`。"""
        argv = ["analyze", "--model_type", "qwen3", "--metrics", "attention_mse"]

        result = _normalize_analyze_argv(argv)

        # 转换后：analyze attn --model_type qwen3 --metrics mse
        assert "attn" in result
        # attention_mse 应被改为 mse
        assert "attention_mse" not in result
        assert result[result.index("--metrics") + 1] == "mse"

    def test_normalize_drops_pattern_arg_when_converting_legacy_attention_mse(self):
        """边界：legacy attention_mse → attn 转换时，attn scope 不接受 --pattern，应被丢弃。"""
        argv = ["analyze", "--metrics", "attention_mse", "--pattern", "layer.*", "--model_type", "x"]

        result = _normalize_analyze_argv(argv)

        # --pattern 及其值 'layer.*' 应被移除
        assert "--pattern" not in result
        assert "layer.*" not in result

    def test_normalize_does_not_modify_args_before_analyze(self):
        """边界：analyze 之前的参数应原样保留。"""
        argv = ["--config", "/etc/conf", "analyze", "--model_type", "x"]

        result = _normalize_analyze_argv(argv)

        # --config /etc/conf 应原样保留
        assert "--config" in result
        assert "/etc/conf" in result
        # 'linear' 应被注入
        assert "linear" in result

    def test_normalize_returns_unchanged_when_metrics_value_malformed(self):
        """边界：--metrics 后无值（或 ValueError）时不转换。"""
        # 模拟 metrics 后面是 - 开头的标志（不是值）
        argv = ["analyze", "--metrics", "--model_type", "x"]

        result = _normalize_analyze_argv(argv)

        # 不应触发 legacy 转换
        assert "attn" not in result
        # 应注入 'linear'
        assert "linear" in result

    def test_normalize_returns_unchanged_when_metrics_not_attention_mse(self):
        """边界：--metrics 取非 legacy 值时不应转换。"""
        argv = ["analyze", "--metrics", "kurtosis", "--model_type", "x"]

        result = _normalize_analyze_argv(argv)

        # 不应转换（attn 仅针对 attention_mse）
        assert "attn" not in result
        # 注入 linear
        assert "linear" in result
        # kurtosis 仍保留
        assert "kurtosis" in result


class TestMainDispatcher:
    """Test suite for main() — 顶层 CLI dispatcher（按子命令路由）。"""

    @patch("msmodelslim.cli.__main__.sys")
    @patch("msmodelslim.cli.naive_quantization.__main__.main")
    def test_main_dispatches_to_naive_quantization_when_command_is_quant(self, mock_nq_main, mock_sys):
        """主路径：command=quant 时应调用 cli.naive_quantization.__main__.main。"""
        mock_sys.argv = ["msmodelslim", "quant", "--model_type", "qwen3", "--model_path", "/x", "--save_path", "/y"]
        mock_nq_main.return_value = None

        main()

        mock_nq_main.assert_called_once()

    @patch("msmodelslim.cli.__main__.sys")
    @patch("msmodelslim.cli.analysis.__main__.main")
    def test_main_dispatches_to_analysis_when_command_is_analyze(self, mock_a_main, mock_sys):
        """主路径：command=analyze 时应调用 cli.analysis.__main__.main。"""
        mock_sys.argv = [
            "msmodelslim",
            "analyze",
            "linear",
            "--model_type",
            "qwen3",
            "--model_path",
            "/x",
        ]
        mock_a_main.return_value = None

        main()

        mock_a_main.assert_called_once()

    @patch("msmodelslim.cli.__main__.sys")
    @patch("msmodelslim.cli.auto_tuning.__main__.main")
    def test_main_dispatches_to_auto_tuning_when_command_is_tune(self, mock_t_main, mock_sys):
        """主路径：command=tune 时应调用 cli.auto_tuning.__main__.main。"""
        mock_sys.argv = [
            "msmodelslim",
            "tune",
            "--model_type",
            "qwen3",
            "--model_path",
            "/x",
            "--save_path",
            "/y",
            "--config",
            "/p",
        ]
        mock_t_main.return_value = None

        main()

        mock_t_main.assert_called_once()

    @patch("msmodelslim.cli.__main__.sys")
    def test_main_prints_help_when_no_command_given(self, mock_sys):
        """边界：未指定子命令时应正常退出（调用 parser.print_help()，不抛错）。"""
        mock_sys.argv = ["msmodelslim"]

        # 不应抛错（parser.print_help 内部 stdout 打印）
        main()


class TestAnalyzeAttnHeadExcludedParams:
    """attn_head 不提供 --top_k / --calibration_dataset：帮助信息不列出，显式传入即报错。"""

    EXCLUDED_ARGS = (
        ["--top_k", "10"],
        ["--calibration_dataset", "mix_calib.jsonl"],
    )

    @staticmethod
    def _help_output(mock_sys, argv, capsys):
        mock_sys.argv = argv
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        return capsys.readouterr().out

    @patch("msmodelslim.cli.__main__.sys")
    def test_attn_head_help_does_not_list_excluded_params(self, mock_sys, capsys):
        """主路径：attn_head 不支持 --top_k / --calibration_dataset，帮助信息中不应出现。"""
        out = self._help_output(mock_sys, ["msmodelslim", "analyze", "attn_head", "--help"], capsys)

        assert "--top_k" not in out
        assert "--topk" not in out
        assert "--calibration_dataset" not in out
        assert "--calib_dataset" not in out

    @pytest.mark.parametrize("scope", ["linear", "layer", "attn"])
    @patch("msmodelslim.cli.__main__.sys")
    def test_other_scopes_help_lists_excluded_params(self, mock_sys, capsys, scope):
        """主路径：linear/layer/attn 输出 disable_names 且消费校准集，帮助信息中保留这两个参数。"""
        out = self._help_output(mock_sys, ["msmodelslim", "analyze", scope, "--help"], capsys)

        assert "--top_k" in out
        assert "--calibration_dataset" in out

    @pytest.mark.parametrize("extra_argv", EXCLUDED_ARGS)
    @patch("msmodelslim.cli.__main__.sys")
    def test_attn_head_rejects_excluded_params(self, mock_sys, capsys, extra_argv):
        """异常：attn_head 显式传入这两个参数时应被 argparse 拒绝。"""
        mock_sys.argv = [
            "msmodelslim",
            "analyze",
            "attn_head",
            "--model_type",
            "Qwen2.5-7B-Instruct",
            "--model_path",
            "/tmp/model",
        ] + extra_argv

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code != 0
        assert "unrecognized arguments" in capsys.readouterr().err
