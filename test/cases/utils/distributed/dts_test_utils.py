#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""DTS 单测用：构造与用例中 ``dependencies`` 路径一致的最小 ``nn.Module`` 树。"""

import torch.nn as nn


def build_dts_dependency_mock_model(num_unique_modules: int = 100) -> nn.Module:
    """返回一棵子模块路径覆盖 DTS 测试用例所需名字的模型树。"""
    root = nn.Module()
    for name in (
        "module1",
        "module2",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
        "A",
        "B",
        "C",
        "D",
        "shared",
        "shared_module",
    ):
        setattr(root, name, nn.Linear(2, 2))
    for i in range(num_unique_modules):
        setattr(root, f"module{i}", nn.Linear(2, 2))

    model = nn.Module()
    layer0 = nn.Module()
    self_attn = nn.Module()
    self_attn.q_proj = nn.Linear(2, 2)
    layer0.self_attn = self_attn
    model.layers = nn.ModuleList([layer0])
    root.model = model

    layer_list = nn.ModuleList()
    for _i in range(4):
        li = nn.Module()
        li.q_proj = nn.Linear(2, 2)
        li.k_proj = nn.Linear(2, 2)
        li.v_proj = nn.Linear(2, 2)
        li.o_proj = nn.Linear(2, 2)
        layer_list.append(li)
    root.layer = layer_list
    return root
