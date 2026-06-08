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

from typing import Iterable, Union

from typing_extensions import Self, Type


class ModelslimError(Exception):
    code = 0
    default_message = 'modelslim error'

    def __init__(self, *args, action=''):
        super().__init__(*args)
        self.action = action

    def __str__(self):
        message = super().__str__()
        if not message or message == "None":
            message = self.default_message
        desc = f"Code: {self.code}, Message: {message}"
        if self.action:
            desc += f"\nTIP: {self.action}"
        return desc

    def __repr__(self):
        """重写repr方法，用于日志打印时显示错误类型、错误信息、错误码和解决推荐"""
        error_type = self.__class__.__name__
        message = super().__str__()
        if not message or message == "None":
            message = self.default_message

        desc = f"[{error_type}] Code: {self.code}, Message: {message}"
        if self.action:
            desc += f", TIP: {self.action}"

        return desc

    @classmethod
    def create_exception(cls, name: str, code: int, default_message: str = '') -> Type[Self]:
        return type(name, (cls,), {"code": code, "default_message": default_message})


# EnvironmentError
EnvError: Type[ModelslimError] = ModelslimError.create_exception(
    "EnvError", 100, "Environment failed to meet the requirements."
)
VersionError: Type[ModelslimError] = EnvError.create_exception(
    "VersionError", 101, "Version of dependencies mismatched."
)
EnvVarError: Type[ModelslimError] = EnvError.create_exception("EnvVarError", 102, "Environment variable not set right.")
ConfigError: Type[ModelslimError] = EnvError.create_exception("ConfigError", 103, "Config file is invalid.")

# MisbehaviorError
MisbehaviorError: Type[ModelslimError] = ModelslimError.create_exception("MisbehaviorError", 200, "User misbehavior.")
InvalidModelError: Type[ModelslimError] = MisbehaviorError.create_exception(
    "InvalidModelError", 201, "Invalid model to load or inference."
)
InvalidDatasetError: Type[ModelslimError] = MisbehaviorError.create_exception(
    "InvalidDatasetError", 202, "Invalid dataset to load."
)
SchemaValidateError: Type[ModelslimError] = MisbehaviorError.create_exception(
    "SchemaValidateError", 203, "Argument schema validation failed."
)
SecurityError: Type[ModelslimError] = MisbehaviorError.create_exception(
    "SecurityError", 204, "Potential security risk."
)

# TrivialError
TrivialError: Type[ModelslimError] = ModelslimError.create_exception(
    "TrivialError", 300, "Trivial error, no need to resolve."
)
UnsupportedError: Type[ModelslimError] = TrivialError.create_exception(
    "UnsupportedError", 301, "Unsupported operation."
)
SpecError: Type[ModelslimError] = TrivialError.create_exception("SpecError", 302, "Specific scenario error.")
ModelslimTimeoutError: Type[ModelslimError] = TrivialError.create_exception(
    "ModelslimTimeoutError", 303, "Timeout error."
)

# ToDoError
ToDoError: Type[ModelslimError] = ModelslimError.create_exception("ToDoError", 400, "Bug to be fixed soon.")


# UnexpectedError
class UnexpectedError(ModelslimError):
    code = 500
    default_message = "Unexpected error."
    tips = []

    @classmethod
    def inject_tips(cls, tips: Union[str, Iterable[str]]) -> None:
        if isinstance(tips, str):
            cls.tips.append(tips)
        elif isinstance(tips, Iterable):
            cls.tips.extend(tips)

    @classmethod
    def clear_tips(cls) -> None:
        cls.tips.clear()

    def _all_tips(self):
        all_tips = []
        if self.action:
            all_tips.append(self.action)
        all_tips.extend(self.tips)
        return all_tips

    def __str__(self):
        message_only = Exception.__str__(self)
        if not message_only or message_only == "None":
            message_only = self.default_message

        desc = f"Code: {self.code}, Message: {message_only}"
        for tip in self._all_tips():
            desc += f"\nTIP: {tip}"
        return desc

    def __repr__(self):
        error_type = self.__class__.__name__
        message_only = Exception.__str__(self)
        if not message_only or message_only == "None":
            message_only = self.default_message

        desc = f"[{error_type}] Code: {self.code}, Message: {message_only}"
        for tip in self._all_tips():
            desc += f", TIP: {tip}"
        return desc
