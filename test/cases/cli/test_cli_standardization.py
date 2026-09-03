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


class TestPlaceholderFormatting:
    """Help placeholders: single values use <NAME>, multi-value uses [<NAME> ...]."""

    @staticmethod
    def _placeholder(*opts, **kw):
        parser = argparse.ArgumentParser(formatter_class=cli._UnifiedHelpFormatter)
        parser.add_argument(*opts, **kw)
        action = parser._actions[-1]
        fmt = cli._UnifiedHelpFormatter(prog='prog')
        default = fmt._get_default_metavar_for_optional(action)
        return fmt._format_args(action, default)

    def test_multi_value_space_separated(self):
        assert self._placeholder('--device_id', nargs='*', metavar='<ID>') == '[<ID> ...]'
        assert self._placeholder('--tags', nargs='*', metavar='<TAG>') == '[<TAG> ...]'

    def test_optional_single_value_no_brackets(self):
        assert self._placeholder('--trust_remote_code', nargs='?', const=True) == 'TRUST_REMOTE_CODE'
        assert self._placeholder('--model_path', metavar='<PATH>') == '<PATH>'

    def test_flag_no_placeholder(self):
        assert self._placeholder('-v', '--verbose', action='store_true') == ''
