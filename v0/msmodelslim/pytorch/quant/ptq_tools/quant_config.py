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

from ascend_utils.common.security import check_type, check_int, check_element_type, \
    check_dict_character

# quant_mode
# 0 : data-free
# 1 : min-max
QUANT_MODE_LIST = [0, 1]

# act_method
# 0 : data-free
# 1 : min-max
# 2 : histogram
ACT_METHOD_LIST = [0, 1, 2]
DEVICE_LIST = ['cpu', 'npu']
VALID_INPUT_SHAPE_LIST = [0, 3, 4, 5]


class QuantConfig:
    """ The configuration for knowledge distillation."""

    def __init__(
        self,
        w_bit=8,
        a_bit=8,
        w_signed=True,
        a_signed=False,
        w_sym=True,
        a_sym=False,
        input_shape=None,
        act_quant=True,
        act_method=0,
        quant_mode=0,
        disable_names=None,
        amp_num=0,
        keep_acc=None,
        sigma=25,
        device='cpu',
    ):
        # Basic setting
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.w_signed = w_signed
        self.a_signed = a_signed
        self.w_sym = w_sym
        self.a_sym = a_sym
        if input_shape is None:
            input_shape = []
        self.input_shape = input_shape
        self.act_quant = act_quant
        self.sigma = sigma
        # name list for disabled modules
        if disable_names is None:
            disable_names = []
        self.disable_names = disable_names
        # number of layers for AMP fallback
        self.amp_num = amp_num
        self.device = device
        # Keep accuracy control, [bool, int] or bool
        # admm is for data-free/label-free, easy_quant is for data-free
        # round_opt is for label-free
        if keep_acc is None:
            self.keep_acc = {'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}
        else:
            self.keep_acc = keep_acc

        # act_method
        # 0 : data-free
        # 1 : min-max
        # 2 : histogram
        self.act_method = act_method
        # quant_mode
        # 0 : data-free
        # 1 : min-max
        self.quant_mode = quant_mode
        self._check_params()

    def _check_params(self):

        if not isinstance(self.w_bit, int) or self.w_bit != 8:
            raise TypeError("w_bit must be 8, please check it.")
        if not isinstance(self.a_bit, int) or self.a_bit != 8:
            raise TypeError("a_bit must be 8, please check it.")

        check_dict_character(dict_value=self.keep_acc, param_name="keep_acc")
        if 'admm' not in self.keep_acc:
            raise ValueError("admm should be in keep_accuracy")
        else:
            check_type(self.keep_acc['admm'], list, param_name="the value of key 'admm'")
            check_type(self.keep_acc['admm'][0], bool, param_name="first element in the value of keyword 'admm'")
            check_type(self.keep_acc['admm'][1], int, param_name="second element in the value of keyword 'admm'")
        if 'easy_quant' not in self.keep_acc:
            raise ValueError("easy_quant should be in keep_accuracy")
        else:
            check_type(self.keep_acc['easy_quant'], list, param_name="the value of key 'easy_quant'")
            check_type(self.keep_acc['easy_quant'][0],
                       bool,
                       param_name="first element in the value of keyword 'easy_quant'")
            check_type(self.keep_acc['easy_quant'][1],
                       int,
                       param_name="second element in the value of keyword 'easy_quant'")
        if 'round_opt' not in self.keep_acc:
            raise ValueError("round_opt should be in keep_accuracy")
        else:
            check_type(self.keep_acc['round_opt'], bool, param_name="the value of key 'round_opt'")

        check_int(self.amp_num, min_value=0, param_name='amp_num')
        check_int(self.sigma, min_value=0, max_value=100, param_name='sigma')

        check_element_type(self.disable_names, element_type=str, value_type=list, param_name="disable_names")

        if not isinstance(self.disable_names, list):
            raise TypeError("disable_names must be list, please check it.")

        if not isinstance(self.input_shape, list):
            raise TypeError("input_shape must be list, please check it.")
        if len(self.input_shape) not in VALID_INPUT_SHAPE_LIST:
            raise ValueError("input_shape must be empty(has calib data), 3D (unbatched), 4D (batched) or 5D "
                             "(for video Diffusion Models inputs), please check it.")
        check_element_type(self.input_shape, element_type=int, value_type=list, param_name="input_shape")

        if not isinstance(self.act_quant, bool):
            raise ValueError("act_quant is invalid, please check it.")
        if not isinstance(self.w_signed, bool):
            raise ValueError("w_signed is invalid, please check it.")
        if not isinstance(self.a_signed, bool):
            raise ValueError("a_signed is invalid, please check it.")
        if not isinstance(self.w_sym, bool):
            raise ValueError("w_sym is invalid, please check it.")
        if not isinstance(self.a_sym, bool):
            raise ValueError("a_sym is invalid, please check it.")

        if self.act_method not in ACT_METHOD_LIST:
            raise ValueError("act_method is invalid, please check it.")

        if self.quant_mode not in QUANT_MODE_LIST:
            raise ValueError("quant_mode is invalid, please check it.")

        if self.device not in DEVICE_LIST:
            raise ValueError("device is invalid, please check it.")