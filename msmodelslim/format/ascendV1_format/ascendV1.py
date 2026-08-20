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

AscendV1 落盘格式配置（占位）；运行时由 core ``AscendV1Saver`` / ``DistributedAscendV1Saver`` 处理。
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import Field

from msmodelslim.format.base import QuantFormatConfig


class AscendV1QuantFormatConfig(QuantFormatConfig):
    """AscendV1 保存格式配置，导出昇腾落盘格式的权重文件。"""

    def set_save_directory(self, save_directory: str):
        self.save_directory = str(save_directory)

    type: Literal['ascendv1_saver'] = Field(
        default="ascendv1_saver", description="保存格式类型，固定为 `ascendv1_saver`。"
    )
    save_directory: str = Field(default=".", exclude=True)
    part_file_size: int = Field(default=4, description="分片文件大小，单位 GB；0 表示不分片。")
    ext: Dict[str, Any] = Field(
        default_factory=dict,
        exclude_if=lambda v: not v,
        description="保存格式扩展参数，随实现而定；空对象表示无扩展参数。",
    )


__all__ = [
    "AscendV1QuantFormatConfig",
]
