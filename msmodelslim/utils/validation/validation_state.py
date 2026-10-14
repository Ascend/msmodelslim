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

import threading

# 存储校验上下文（如 YAML 路径前缀），由 BeforeValidator 设置
_validation_context = threading.local()

# 存储 BeforeValidator 等阶段收集的额外错误，由 model_validate 统一合并报出
_additional_validation_errors = threading.local()


def set_validation_context(path_prefix: str = ""):
    """设置校验上下文，model_validate 会将前缀加到所有错误路径上。"""
    _validation_context.path_prefix = path_prefix


def get_validation_context() -> dict:
    """获取当前校验上下文。"""
    return {"path_prefix": getattr(_validation_context, 'path_prefix', "")}


def add_validation_error(error_msg: str):
    """添加额外的校验错误，由 model_validate 统一合并报出。"""
    if not hasattr(_additional_validation_errors, 'value'):
        _additional_validation_errors.value = []
    _additional_validation_errors.value.append(error_msg)


def get_additional_validation_errors() -> list:
    """获取额外的校验错误列表。"""
    return getattr(_additional_validation_errors, 'value', [])


def clear_additional_validation_errors():
    """清空额外的校验错误。"""
    _additional_validation_errors.value = []
