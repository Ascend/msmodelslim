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

import argparse
from types import SimpleNamespace
from unittest import mock

import pytest

import msmodelslim.cli.__main__ as cli


class TestHelpFormatter:
    @staticmethod
    def _invocation(*opts, **kw):
        parser = argparse.ArgumentParser(formatter_class=cli._UnifiedHelpFormatter)
        parser.add_argument(*opts, **kw)
        return cli._UnifiedHelpFormatter(prog='prog')._format_action_invocation(parser._actions[-1])

    def test_show_only_short_and_canonical(self):
        assert self._invocation('-v', '--verbose', '--old_verbose', action='store_true') == '-v, --verbose'
        assert self._invocation('--model_path', dest='model_path', metavar='<PATH>', type=str) == (
            '--model_path <PATH>'
        )


class TestCliConvertToBool:
    def test_valid_values(self):
        assert all(cli._cli_convert_to_bool(v) is True for v in ('true', 'yes', 'on', 'True'))
        assert all(cli._cli_convert_to_bool(v) is False for v in ('false', 'no', 'off', 'FALSE'))

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            cli._cli_convert_to_bool('maybe')


class TestParseTimeout:
    def test_seconds(self):
        assert cli._parse_timeout('3600') == 3600

    def test_legacy_duration(self):
        assert cli._parse_timeout('1D2H') == '1D2H'


class TestWarnDeprecated:
    @staticmethod
    def _warn(argv):
        with mock.patch.object(cli, 'get_logger') as g:
            cli._warn_deprecated(argv)
        return g.return_value.warning

    def test_legacy_spellings(self):
        for old, new in (
            ('--topk', '--top_k'),
            ('--calib_dataset', '--calibration_dataset'),
            ('--tag', '--tags'),
            ('--pattern', '--patterns'),
            ('--config_path', '--config'),
        ):
            w = self._warn(['quant', old, 'x'])
            assert w.call_count == 1
            assert w.call_args[0][1] == old
            assert w.call_args[0][2] == new

    def test_dedup_and_canonical_silent(self):
        assert self._warn(['quant', '--topk', 'a', '--topk', 'b']).call_count == 1
        assert self._warn(['quant', '--top_k', '5']).call_count == 0


class TestApplyLogLevel:
    @staticmethod
    def _level(**kw):
        with mock.patch.object(cli, 'set_logger_level') as s:
            cli._apply_log_level(SimpleNamespace(**kw))
        return s.call_args[0][0]

    def test_precedence(self):
        assert self._level(log_level='error', verbose=False, quiet=False, debug=False) == 'error'
        assert self._level(log_level=None, verbose=True, quiet=False, debug=False) == 'debug'
        assert self._level(log_level=None, verbose=False, quiet=True, debug=False) == 'error'
        assert self._level(log_level='warning', verbose=True, quiet=False, debug=False) == 'warning'


class TestArgvDetectors:
    def test_help_and_version_detection(self):
        assert cli._is_help_request(['quant', '--help'])
        assert cli._is_version_request(['quant', '--version'])
        assert not cli._is_help_request(['quant', '--version'])
        assert not cli._is_version_request(['quant', '--help'])


class TestNormalizeDeviceArgv:
    def test_space_form(self):
        assert cli._normalize_device_argv(['quant', '--device', 'npu:0,1,2,3']) == [
            'quant',
            '--device',
            'npu',
            '--device_id',
            '0',
            '1',
            '2',
            '3',
        ]

    def test_equals_form(self):
        assert cli._normalize_device_argv(['tune', '--device=cpu:0']) == [
            'tune',
            '--device',
            'cpu',
            '--device_id',
            '0',
        ]

    def test_canonical_and_missing_value_untouched(self):
        argv = ['quant', '--device', 'npu', '--device_id', '0', '1']
        assert cli._normalize_device_argv(argv) == argv
        assert cli._normalize_device_argv(['quant', '--device', '--model_path', 'm']) == [
            'quant',
            '--device',
            '--model_path',
            'm',
        ]

    def test_explicit_new_args_win(self):
        argv = ['quant', '--device', 'npu:0,1', '--device_id', '2']
        assert cli._normalize_device_argv(argv) == argv

    def test_analyze_untouched(self):
        argv = ['analyze', '--device', 'npu:0,1']
        assert cli._normalize_device_argv(argv) == argv

    def test_warns_once(self):
        with mock.patch.object(cli, 'get_logger') as g:
            cli._normalize_device_argv(['quant', '--device', 'npu:0,1', '--device', 'cpu:2'])
        assert g.return_value.warning.call_count == 1
