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
# pylint: disable=useless-import-alias

from typing import Tuple, Optional, Iterator, Set

import torch
from torch import nn as nn

from msmodelslim.ir.qal import QParam, QStorage
from msmodelslim.ir.qal.qbase import QScheme
from msmodelslim.ir.qal.qregistry import QABCRegistry


@QABCRegistry.register_abc(dispatch_key=Tuple[QScheme, QScheme])
class AutoFakeQuantLinear(nn.Module):
    """

    AutoFakeQuantLinear用于快速创建相应的伪量化IR。

    伪量化IR提供了某种量化方式的所有参数描述，例如对于W8A8量化，其所对应的伪量化IR为W8A8FakeQuantLinear。

    """

    @staticmethod
    def is_atomic() -> bool:
        """
        如果该伪量化IR是原子性的，则返回True，否则返回False。
        原子性伪量化IR是指该IR应当被视为一个整体，不能被拆分，哪怕其内部包含其他伪量化IR。
        """

        return True

    @classmethod
    def create(cls, x_q_param: QParam, w_q_param: QParam, w_q: QStorage, bias: Optional[torch.Tensor] = None):
        return QABCRegistry.create(
            AutoFakeQuantLinear, (x_q_param.scheme, w_q_param.scheme), *(x_q_param, w_q_param, w_q, bias)
        )

    def named_modules(
        self,
        memo: Optional[Set[nn.Module]] = None,
        prefix: str = '',
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, nn.Module]]:
        if self.is_atomic():
            yield prefix, self
            return

        yield from super().named_modules(memo, prefix, remove_duplicate)


@QABCRegistry.register_abc(dispatch_key=Tuple[QScheme])
class AutoFakeQuantActivation(nn.Module):
    """
    单激活伪量化IR的抽象基类（仅有一个激活量化方案）。
    """

    @staticmethod
    def is_atomic() -> bool:
        return True

    @classmethod
    def create(cls, x_q_param: QParam):
        return QABCRegistry.create(AutoFakeQuantActivation, x_q_param.scheme, x_q_param)

    def named_modules(
        self, memo: Optional[Set[nn.Module]] = None, prefix: str = '', remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Module]]:
        if self.is_atomic():
            yield prefix, self
            return

        yield from super().named_modules(memo, prefix, remove_duplicate)


@QABCRegistry.register_abc(dispatch_key=Tuple[QScheme])
class AutoFakeQuantDynamicCache(nn.Module):
    @staticmethod
    def is_atomic() -> bool:
        return True

    @classmethod
    def create(cls, x_q_param: QParam):
        return QABCRegistry.create(AutoFakeQuantDynamicCache, x_q_param.scheme, x_q_param)

    def named_modules(
        self, memo: Optional[Set[nn.Module]] = None, prefix: str = '', remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Module]]:
        if self.is_atomic():
            yield prefix, self
            return

        yield from super().named_modules(memo, prefix, remove_duplicate)
