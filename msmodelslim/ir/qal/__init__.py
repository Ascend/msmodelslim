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
# pylint: disable=pointless-string-statement

"""
Quantization Abstract Layer，量化抽象层，简称QAL，是针对量化业务进行抽象的模块，将量化业务划分为业务逻辑与计算逻辑两部分。

QAL主要包含两部分：
1、基础类型声明：提供了量化领域所需要的基础类型的抽象。
2、动态函数分发：引入了一套动态的函数分发机制。

基础类型声明主要包含以下内容：
1、定义了基础的量化数据类型，如QDType/QScope/QScheme/QStorage/QParam等。
4、定义了基础的量化函数接口，如quantize/dequantize/fake_quantize等。

动态函数分发：
1、接口与实现分离，使用register_api声明接口，依赖接口抽象而不是具体实现。
2、动态加载实现，使用register注册实现，实现可以动态加载接口实现。
3、动态分发机制，基于dispatch_key的动态函数分发，便于管理相近语义的多个函数版本的实现。
4、一定程度解决组合爆炸问题，。对于包含多个维度拓展的组合函数，实际上并非所有组合都是合法的，基于动态函数分发，可以方便的管理有效组合。

"""

__all__ = [
    "QDType",
    "QParam",
    "QStorage",
    "QScope",
    "QScheme",
    "QFuncRegistry",
    "QABCRegistry",
]

from .qbase import QDType, QParam, QStorage, QScope, QScheme  # noqa: E402
from .qregistry import QFuncRegistry, QABCRegistry  # noqa: E402
