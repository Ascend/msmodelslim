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

from ascend_utils.common.security import check_int, check_number


class RACompressConfig:
    """ 
    The configuration for compression.
    config = RACompressConfig(theta=0.00001,alpha=100)
    """

    def __init__(self, theta=0.00001, alpha=100):
        """
        Args:
            theta:attention score贡献度,保证校准后模型推理精度
            alpha:校准偏置,用于保证泛化性,控制窗口大小
        """
        self.theta = theta
        self.alpha = alpha
        self._check_params()

    def _check_params(self):
        check_int(self.alpha, 0, 10000, param_name="alpha")
        check_number(self.theta, float, 0.00001, 0.001, param_name="theta")
