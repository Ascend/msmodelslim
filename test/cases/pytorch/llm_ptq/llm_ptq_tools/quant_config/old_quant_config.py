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
# pylint: disable=consider-using-in

from ascend_utils.common.security import check_number, check_element_type, check_type
from ascend_utils.common.security.pytorch import validate_device
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType, WeightQuantMethod

A_BIT_LIST = [8, 16]
W_BIT_LIST = [8]
W_BIT_LIST_LOW_BIT = [4, 8]
GROUP_SIZE_LIST = [64, 128]

_SUPPORTED_DEVICES = ["cpu", "npu"]


class OldQuantConfig:
    """The configuration for quantization"""

    def __init__(
        self,
        w_bit: int = 8,
        a_bit: int = 8,
        act_method: int = 1,
        w_method: str = 'MinMax',
        disable_names: object = None,
        pr: float = 1.0,
        mm_tensor: bool = True,
        dev_type: object = 'cpu',
        dev_id: int = None,
        fraction: float = 0.01,
        nonuniform: bool = False,
        co_sparse: bool = False,
        w_sym: bool = True,
        is_lowbit: bool = False,
        do_smooth: bool = False,
        use_sigma: bool = False,
        sigma_factor: float = 3.0,
        disable_last_linear: bool = True,
        use_kvcache_quant: bool = False,
        is_dynamic: bool = False,
        open_outlier: bool = True,
        group_size: int = 64,
        percdamp: float = 0.01,
    ):
        """
        Args:
            w_bit: quantization bits for weight
            a_bit: quantization bits for activation
            act_method: activation quantization Method
                1: method1 for Lable-Free
                2: method2 for Lable-Free
                3: method3 for Lable-Free (suggested)
            w_method: weight quantization method
                'MinMax': support Data-Free
                'GPTQ'
                'HQQ':support Data-Free
            disable_names: list of layer names to disable
            pr: probability to choose quantization
            mm_tensor: per-tensor or per-channel
                True: per-tensor
                False: per-channel
            dev_type: device type: 'cpu'
            fraction: accuracy adjustment parameter
            nonuniform: whether to uniform quantization
            w_sym: symmetry of weight quantization
                True: symmetrical
                False: nonsymmetrical
        """
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.act_method = act_method
        self.w_method = w_method
        if disable_names is None:
            disable_names = []
        self.disable_names = disable_names
        self.dev_type = dev_type
        self.dev_id = dev_id
        self.pr = pr
        self.mm_tensor = mm_tensor
        self.w_sym = w_sym
        self.fraction = fraction
        self.nonuniform = nonuniform
        self.co_sparse = co_sparse
        self.is_lowbit = is_lowbit
        self.do_smooth = do_smooth
        self.use_sigma = use_sigma
        self.sigma_factor = sigma_factor
        self.disable_last_linear = disable_last_linear
        self.use_kvcache_quant = use_kvcache_quant
        self.open_outlier = open_outlier
        self.group_size = group_size
        self.percdamp = percdamp

        self.set_quant_param()
        self.set_lowbit_param()
        self.set_per_group_param()

        self._check_sparse_config()
        self.device, self.dev_id = validate_device(dev_type, dev_id, _SUPPORTED_DEVICES)
        self.is_dynamic = is_dynamic

        check_number(self.act_method, int, 1, 3, param_name="act_method")
        check_element_type(self.disable_names, str, list, param_name="disable_names")
        check_type(self.mm_tensor, bool, param_name="mm_tensor")
        check_type(self.w_sym, bool, param_name="w_sym")
        check_type(self.disable_last_linear, bool, param_name='disable_last_linear')
        if self.a_bit == 8 and not self.w_sym:
            raise TypeError("w_sym should be True when a_bit = 8, please check it")
        check_type(self.co_sparse, bool, param_name="co_sparse")
        check_type(self.is_lowbit, bool, param_name='is_lowbit')
        check_type(self.open_outlier, bool, param_name="open_outlier")
        self._check_sparse_config()
        self._check_lowbit_config()
        self._check_dynamic_config()

        params = {
            'w_bit': self.w_bit,
            'a_bit': self.a_bit,
            'w_method': "None",  # 根据实际状况填写
            'is_sparse': self.co_sparse,
            'is_dynamic': self.is_dynamic,
            'is_lowbit': self.is_lowbit,
        }
        self.model_quant_type = QuantType.get_quant_type(params)
        self.w_hessian, self.hqq = WeightQuantMethod.get_wmethod_config(w_method)
        WeightQuantMethod.check_quant_type(self.model_quant_type, w_method=w_method)
        check_type(self.disable_last_linear, bool, param_name='disable_last_linear')
        check_type(self.use_kvcache_quant, bool, param_name="use_kvcache_quant")

    def set_quant_param(self):
        self.w_signed = True
        self.a_signed = True
        self.a_sym = False
        self.input_shape = []
        self.act_quant = True
        self.amp_num = 0
        self.keep_acc = {'admm': [False, 1000], 'round_opt': False}
        self.hessian_optim = {'flag': False, 'std_threshold': -1, 'sigma_weight': 5}
        self.seq_opt = {'en': False, 'n': 10, 'bin': 0.01}
        self.int_bias = False
        self.calib_mode = 0
        self.model_quant_type = QuantType.UNKNOWN

    def set_lowbit_param(self):
        self.use_amp = False
        self.optimize_input_amax = False
        self.smooth_down = False
        self.ch_align = True
        self.search_smooth = True
        self.default_smooth_alpha = 0.5
        self.search_intervals = 10
        self.quant_down = True
        self.offload = False
        self.down_proj_type = ['c_proj', 'down_proj', 'dense_4h_to_h']
        self.norm_class_name = 'RMSNorm'
        self.lwc = False
        self.lac = False
        self.search_omniquant = False
        self.use_low_rank = False
        self.optimizate_msd = False
        self.fold = 3
        self.do_msd = False

    def set_per_group_param(self):
        is_per_group = self.a_bit == 16 and not self.open_outlier and self.is_lowbit
        if is_per_group:
            if self.group_size not in GROUP_SIZE_LIST:
                raise ValueError(f"group_size must be among choice {GROUP_SIZE_LIST}, please check it.")
            self.do_msd = True
            self.fold = 3
            self.optimizate_msd = True
            self.do_smooth = False
            self.amp_num = 0
            self.mm_tensor = False
        else:
            self.group_size = -1
            if self.is_lowbit and (self.w_method == 'HQQ' or self.w_method == 'GPTQ'):
                raise ValueError(
                    "When is_lowbit is set to True, the w_method configuration is only supported in a "
                    "per-group scenario. Please check the config."
                )

    def _check_sparse_config(self):
        check_type(self.nonuniform, bool, param_name="nonuniform")
        check_number(self.fraction, float, 0.0, 0.1, param_name="fraction")

        if self.co_sparse and self.is_lowbit:
            self.co_sparse = False

        if self.co_sparse:
            self.pr = 1.0
            if self.w_bit != 4:
                self.w_bit = 4
                msmodelslim_logger.warning(
                    "Running in sparse requires `w_bit` of 4, your config `w_bit` will replaced to 4 instead."
                )
            if self.a_bit != 8:
                self.a_bit = 8
                msmodelslim_logger.warning(
                    "Running in sparse requires `a_bit` of 8, your config `a_bit` will replaced to 8 instead."
                )
        elif self.is_lowbit:
            if self.w_bit not in W_BIT_LIST_LOW_BIT:
                raise ValueError(f"w_bit must be among choice {W_BIT_LIST_LOW_BIT}, please check it.")
            if self.a_bit not in A_BIT_LIST:
                raise ValueError(f"a_bit must be among choice {A_BIT_LIST}, please check it.")
        else:
            check_number(self.pr, float, 0, 1, param_name="pr")
            if self.w_bit not in W_BIT_LIST:
                raise ValueError(f"w_bit must be among choice {W_BIT_LIST}, please check it.")
            if self.a_bit not in A_BIT_LIST:
                raise ValueError(f"a_bit must be among choice {A_BIT_LIST}, please check it.")

    def _check_lowbit_config(self):
        check_type(self.do_smooth, bool, param_name="do_smooth")
        check_type(self.use_sigma, bool, param_name="use_sigma")
        check_number(self.sigma_factor, float, 3.0, 4.0, param_name="sigma factor")
        if self.act_method == 3 and self.is_lowbit:
            raise ValueError("act_method can not be 3 when running lowbit.")

    def _check_dynamic_config(self):
        check_type(self.is_dynamic, bool, param_name='is_dynamic')
        if self.is_dynamic:
            if self.w_bit != 8:
                raise ValueError("w_bit must be 8 when running dynamic quantization.")
            if self.a_bit != 8:
                raise ValueError("a_bit must be 8 when running dynamic quantization.")
