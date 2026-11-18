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

from enum import Enum

from ascend_utils.common.security.pytorch import validate_device
from ascend_utils.common.security import check_type
from msmodelslim import logger as msmodelslim_logger

_ANTI_METHODS = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
_SUPPORTED_DEVICES = ["cpu", "npu", 'gpu']


class AntiMethods(str, Enum):
    M1 = "m1"
    M2 = "m2"
    M3 = "m3"
    M4 = "m4"
    M5 = "m5"
    M6 = "m6"


class AntiOutlierConfig:
    def __init__(
            self,
            w_bit=8,
            a_bit=8,
            anti_method="m2",
            dev_type='cpu',
            dev_id=None,
            w_sym=True,
            disable_anti_names=None,
            flex_config: dict = None,
    ):
        if disable_anti_names is None:
            disable_anti_names = []
        # Basic setting
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.anti_method = anti_method
        self.dev_type = dev_type
        self.dev_id = dev_id
        self.w_sym = w_sym
        self.disable_anti_names = disable_anti_names
        self.w_signed = True
        self.a_signed = True
        self.a_sym = False
        self.alpha = 0.5
        self.os_k = 100
        self.ch_align = False
        self.w_adjust = True
        self.flex_config = self.setup_flex_config(flex_config)

        self.device, self.dev_id = validate_device(dev_type, dev_id, _SUPPORTED_DEVICES)
        check_type(self.w_bit, int, param_name='w_bit')
        check_type(self.a_bit, int, param_name='a_bit')
        check_type(self.anti_method, str, param_name='anti_method')
        check_type(self.w_sym, bool, param_name='w_sym')
        check_type(self.disable_anti_names, list, param_name='disable_anti_names')
        check_type(self.flex_config, dict, param_name='flex_config')

        if self.anti_method not in _ANTI_METHODS:
            raise ValueError("Configuration param `anti_method` must be in choices {}"
                             .format(_ANTI_METHODS))
        if self.anti_method == "m5":
            self.ch_align = False

        if self.anti_method != AntiMethods.M3 and self.w_bit not in [4, 8]:
            # 如果anti_method是非m3的情况下，anti outlier只能使用再w8a8的场景下
            raise ValueError(f"w_bit must be 8 or 4, but got {self.w_bit}, please check it")
        elif self.anti_method == AntiMethods.M3 and self.w_bit not in [4, 8]:
            # 如果anti_method是m3的情况下，anti outlier可以用在w4a16和w8a16基于Minmax的data-free量化
            raise ValueError(f"w_bit must be 4 or 8 (anti_method='m3'), but got {self.w_bit}, please check it")

        if self.a_bit == 8:
            pass
        elif self.a_bit == 16 and self.anti_method == 'm3':
            pass
        else:
            raise ValueError(f"a_bit must be 8 or 16(anti_method='m3'), but got {self.a_bit}, please check it.")

        if self.w_sym:
            pass
        elif self.anti_method == 'm3':
            msmodelslim_logger.warning("running AWQ, set w_sym=False, please keep it same in quantization!")
            pass
        else:
            raise ValueError("w_sym can only be True when running anti_method='m3', please check it.")


    @staticmethod
    def setup_flex_config(flex_config):

        flex_config_map = {'alpha': {'type': float, 'default': None},
                           'beta': {'type': float, 'default': None}}

        flex_config = {} if flex_config is None else flex_config

        for key, val in flex_config_map.items():
            if key not in flex_config:
                flex_config[key] = val['default']

        for key, _ in flex_config.items():
            if key not in flex_config_map:
                raise ValueError(f"{key} in flex config is not supported")

        return flex_config
