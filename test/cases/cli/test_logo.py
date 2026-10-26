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

from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.cli.logo import CliLogo, print_logo


SIMPLE_LOGO = (
    "=================================================================\n"
    "                   >>>>>   MindStudio   <<<<<\n"
    "    THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE\n"
    "=================================================================\n\n"
)

COLORED_LOGO = (
    "\033[38;5;240m=================================================================\033[0m\n"
    "\033[1;97m                   >>>>>  \033[48;5;21;38;5;46m MindStudio \033[0m"
    "\033[1;97m  <<<<<\033[0m\n"
    "\033[1;97m    THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE\033[0m\n"
    "\033[38;5;240m=================================================================\033[0m\n\n"
)


@pytest.mark.parametrize(
    ("isatty", "term", "expected_logo"),
    [
        (False, "xterm-256color", SIMPLE_LOGO),
        (True, None, SIMPLE_LOGO),
        (True, "dumb", SIMPLE_LOGO),
        (True, "unknown", SIMPLE_LOGO),
        (True, "xterm-256color", COLORED_LOGO),
    ],
)
def test_print_logo_writes_logo_selected_from_terminal_capability_when_called(isatty, term, expected_logo):
    stderr = MagicMock()
    stderr.isatty.return_value = isatty
    environment = {} if term is None else {"TERM": term}

    with (
        patch("msmodelslim.cli.logo.sys.stderr", stderr),
        patch.dict("msmodelslim.cli.logo.os.environ", environment, clear=True),
    ):
        CliLogo().print_logo()

    stderr.write.assert_called_once_with(expected_logo)
    stderr.flush.assert_called_once_with()


def test_print_logo_delegates_to_cli_logo_instance_when_called():
    logo_instance = MagicMock()

    with patch("msmodelslim.cli.logo.CliLogo", return_value=logo_instance) as logo_class:
        print_logo()

    logo_class.assert_called_once_with()
    logo_instance.print_logo.assert_called_once_with()
