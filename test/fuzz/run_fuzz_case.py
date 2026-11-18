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

import runpy
import sys
from pathlib import Path


def install_source_path():
    """Make the repository importable before running a fuzz case."""
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)

    # Import the package to apply its source-layout compatibility.
    import msmodelslim  # noqa: F401  # pylint: disable=import-outside-toplevel


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_fuzz_case.py FUZZ_TEST [ARGS...]")

    install_source_path()
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
