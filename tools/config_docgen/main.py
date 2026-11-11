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

配置文档生成器的稳定命令行入口。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from tools.config_docgen.generate import main as generate_main  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the generator with the canonical command-line interface."""
    return generate_main(argv)


if __name__ == "__main__":
    sys.exit(main())
