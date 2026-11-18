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

from ascend_utils.common.security import check_number


class RARopeCompressConfig:
    """ 
    The configuration for compression.
    config = RACompressConfig(induction_head_ratio=0.14,
                              echo_head_ratio=0.01)
    """

    def __init__(self, induction_head_ratio=0.14, echo_head_ratio=0.01):
        """
        Args:
            induction_head_ratio:控制induction head的判定比例
            echo_head_ratio:控制echoing head的判定比例
        """
        self.induction_head_ratio = induction_head_ratio
        self.echo_head_ratio = echo_head_ratio
        self._check_params()

    def _check_params(self):
        check_number(self.induction_head_ratio, float, 0, 1, param_name="induction_head_ratio")
        check_number(self.echo_head_ratio, float, 0, 1, param_name="echo_head_ratio")
