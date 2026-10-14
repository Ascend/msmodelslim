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

from decimal import Decimal

from pydantic import BaseModel

from msmodelslim.utils.hash import calculate_md5


class _DecimalModel(BaseModel):
    score: Decimal
    name: str


def test_calculate_md5_return_32_length_hex_when_base_model_has_decimal_field():
    """
    calculate_md5 内部会 model_dump 后 json.dumps；Decimal 在 Python 模式下不可被 JSON 序列化。
    使用 mode='json' 后应能正常得到 32 位十六进制 MD5，且不因 Decimal 字段抛错。
    """
    value = _DecimalModel(score=Decimal("0.95"), name="acc")
    md5_value = calculate_md5(value)

    assert isinstance(md5_value, str)  # 断言：返回字符串形式的摘要
    assert len(md5_value) == 32  # 断言：MD5 为 32 个十六进制字符
