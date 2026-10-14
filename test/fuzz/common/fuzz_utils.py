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
# pylint: disable=consider-using-from-import,pointless-string-statement

"""
模糊测试公共工具函数
提供了一组在模糊测试中常用的辅助函数
"""
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402


def consume_bool(fdp):
    """生成随机布尔值
    Args:
        fdp: FuzzedDataProvider对象
    Returns:
        bool: 随机生成的布尔值
    """
    return fdp.ConsumeIntInRange(0, 1) == 1


def consume_int(fdp, min_v=0, max_v=100):
    """生成指定范围内的随机整数
    Args:
        fdp: FuzzedDataProvider对象
        min_v: 最小值，默认0
        max_v: 最大值，默认100
    Returns:
        int: 随机生成的整数
    """
    return fdp.ConsumeIntInRange(min_v, max_v)


def consume_float(fdp, min_v=0.0, max_v=1.0):
    """生成指定范围内的随机浮点数
    Args:
        fdp: FuzzedDataProvider对象
        min_v: 最小值，默认0.0
        max_v: 最大值，默认1.0
    Returns:
        float: 随机生成的浮点数
    """
    return fdp.ConsumeFloatInRange(min_v, max_v)


def consume_pick_list(fdp, pick_list):
    """从列表中随机选择一个元素
    Args:
        fdp: FuzzedDataProvider对象
        pick_list: 列表
    Returns:
        any: 随机选择的元素
    """
    return fdp.PickValueInList(pick_list)


@dataclass
class DummyModelConfig:
    model_type: str = 'dummy'
    torch_dtype: torch.dtype = torch.float32
    architectures: list[str] = field(default_factory=lambda: ['DummyModel'])


class DummyModel(nn.Module):
    """用于测试的简单线性模型
    包含一个线性层和必要的配置信息
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.config = DummyModelConfig()
        self.device = torch.device('cpu')
        self.dtype = torch.float32

    def forward(self, x):
        return self.linear(x)


def consume_str(fdp):
    """生成随机Unicode字符串
    Args:
        fdp: FuzzedDataProvider对象
    Returns:
        str: 随机生成的8字符长度的字符串
    """
    return fdp.ConsumeUnicodeNoSurrogates(8)


def random_all_tensors(fdp):
    """随机生成不同类型的张量数据
    可能返回：None、字典类型张量、列表类型张量或单个张量
    Args:
        fdp: FuzzedDataProvider对象
    Returns:
        Union[None, dict, list, torch.Tensor]: 随机生成的张量数据
    """
    choice = consume_int(fdp, 0, 3)
    if choice == 0:
        return None
    elif choice == 1:
        return {consume_str(fdp): torch.tensor(consume_int(fdp, 0, 100))}
    elif choice == 2:
        return [torch.tensor(consume_int(fdp, 0, 100))]
    else:
        return torch.tensor(consume_int(fdp, 0, 100))
