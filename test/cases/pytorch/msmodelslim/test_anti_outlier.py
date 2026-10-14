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

import torch
import unittest

from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig

setattr(AntiOutlier, 'init_dag', lambda *args, **kargs: None)
setattr(AntiOutlier, '_process', lambda *args, **kargs: None)


class OneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.l1 = torch.nn.Linear(8, 8, bias=False)
        self.l2 = torch.nn.Linear(8, 8, bias=False)
        self.l3 = torch.nn.Linear(8, 8, bias=False)
        self.l4 = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        x = self.l1(x)

        return x


def test_anti_outlier_m1():
    model = OneModel()

    dataset_calib = [[torch.FloatTensor(8, 8)]]
    anti_config = AntiOutlierConfig(anti_method="m1")
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()


def test_anti_outlier_m2():
    model = OneModel()

    dataset_calib = [[torch.FloatTensor(8, 8)]]
    anti_config = AntiOutlierConfig(anti_method="m2")
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()


def test_anti_outlier_m3():
    model = OneModel()

    dataset_calib = [[torch.FloatTensor(8, 8)]]
    anti_config = AntiOutlierConfig(w_bit=8, a_bit=16, anti_method="m3")
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()


class TestAdapterMethods(unittest.TestCase):
    def test_check_all_names_not_disable_anti(self):
        """测试 check_all_names_not_disable_anti 方法"""

        model = OneModel()
        disable_anti_names = ["l1", "l3"]
        dataset_calib = [[torch.FloatTensor(8, 8)]]
        anti_config = AntiOutlierConfig(disable_anti_names=disable_anti_names)
        anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)

        # 验证所有名称都不在禁用列表中
        names = ["l2", "l4"]
        result = anti_outlier.check_all_names_not_disable_anti(names)
        self.assertTrue(result)

        # 验证至少有一个名称在禁用列表中
        names = ["l1", "l2"]
        result = anti_outlier.check_all_names_not_disable_anti(names)
        self.assertFalse(result)

        # 测试禁用列表为空的情况
        anti_outlier.cfg.disable_anti_names = []
        names = ["l1", "l2"]
        result = anti_outlier.check_all_names_not_disable_anti(names)
        self.assertTrue(result)
