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

import inspect
import sys

from pydantic import BaseModel as PydanticBaseModel, ConfigDict, ValidationError

from msmodelslim.utils.validation.validation_state import (
    get_validation_context,
    get_additional_validation_errors,
    clear_additional_validation_errors,
)
from msmodelslim.utils.exception import SchemaValidateError


def _format_validation_error(e: ValidationError, extra_lines: list = None) -> str:
    prefix = get_validation_context().get('path_prefix', '')
    errors = e.errors()
    lines = [f"{len(errors) + len(extra_lines or [])} validation error(s) found"]
    for err in errors:
        loc = '.'.join(str(x) for x in err['loc'])
        # discriminated union 的 tag 不匹配错误，路径补上 .type
        if err['type'] == 'union_tag_invalid' and not loc.endswith('.type'):
            loc = f"{loc}.type"
        lines.append(f"  {prefix}{loc}")
        lines.append(f"    {err['msg']}")
    if extra_lines:
        for line in extra_lines:
            if prefix and not line.startswith(' '):
                lines.append(f"  {prefix}{line}")
            else:
                lines.append(f"  {line}")
    return '\n'.join(lines)


def _in_model_validate():
    for frame in inspect.stack():
        if frame.function == 'model_validate':
            return True
    return False


def patch_pydantic():
    """
    通过patch的方式替换pydantic的BaseModel，将ValidationError转换为SchemaValidateError。

    Pydantic 原生 model_validate 会收集所有嵌套校验错误并带上完整路径，
    __init__ 通过检测调用栈判断是否为嵌套调用：嵌套时不转换，保留 Pydantic 错误收集能力；
    直接调用时转换为 SchemaValidateError 并带上完整路径。
    """

    # 保存原始的BaseModel
    original_base_model = PydanticBaseModel

    class PatchedBaseModel(original_base_model):
        """
        自定义BaseModel，将pydantic的ValidationError转换为项目的SchemaValidateError。
        默认 extra='forbid'，所有子类拒绝未知字段；需要允许时显式设置 extra='allow'。
        """

        model_config = ConfigDict(extra='forbid')

        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except ValidationError as e:
                if _in_model_validate():
                    raise
                raise SchemaValidateError(
                    _format_validation_error(e),
                    action="Please fix the data schema error and retry.",
                ) from e

        @classmethod
        def model_validate(cls, *args, **kwargs):
            clear_additional_validation_errors()
            try:
                result = super().model_validate(*args, **kwargs)
                extra = get_additional_validation_errors()
                if extra:
                    raise SchemaValidateError(
                        _format_validation_error(None, extra_lines=extra),
                        action="Please fix the data schema error and retry.",
                    )
                return result
            except ValidationError as e:
                extra = get_additional_validation_errors()
                raise SchemaValidateError(
                    _format_validation_error(e, extra_lines=extra),
                    action="Please fix the data schema error and retry.",
                ) from e

        @classmethod
        def model_validate_json(cls, *args, **kwargs):
            try:
                return super().model_validate_json(*args, **kwargs)
            except ValidationError as e:
                raise SchemaValidateError(
                    _format_validation_error(e),
                    action="Please fix the data schema error and retry.",
                ) from e

    # 全局替换pydantic的BaseModel
    import pydantic

    pydantic.BaseModel = PatchedBaseModel

    # 替换sys.modules中的BaseModel，确保所有import都能获取到patched版本
    if 'pydantic' in sys.modules:
        sys.modules['pydantic'].BaseModel = PatchedBaseModel

    # 为了确保所有可能的导入路径都被覆盖
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('pydantic'):
            module = sys.modules[module_name]
            if hasattr(module, 'BaseModel'):
                module.BaseModel = PatchedBaseModel
