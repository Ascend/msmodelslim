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


import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from ascend_utils.common.security import check_type
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import TensorQuantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config import QuantConfig
from msmodelslim.pytorch.llm_sparsequant.atomic_power_outlier import quant_one_weight_by_outliers


class LinearSparseQuantizer(nn.Module):
    """
    Class to quantize given linear layer weights
    """

    def __init__(self, cfg=None, logger=None):
        """
        cfg: quantization configuration
        """
        super(LinearSparseQuantizer, self).__init__()
        check_type(cfg, QuantConfig, param_name="cfg")
        self.in_features = None
        self.out_features = None
        self.weight = None
        self.new_weight = None
        self.weight_quant_flag = False
        self.bias = None
        self.quant_input = TensorQuantizer(
            bit=cfg.a_bit, is_signed=cfg.a_signed, is_enable=True,
            is_input=True, cfg=cfg, logger=logger
        )
        self.quant_weight = TensorQuantizer(
            bit=cfg.w_bit, is_signed=cfg.w_signed, is_enable=True,
            is_input=False, cfg=cfg, logger=logger
        )

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data)
        try:
            self.bias = nn.Parameter(linear.bias.data)
        except AttributeError:
            self.bias = None

    def forward(self, x):
        if self.quant_weight.int_infer and (not self.quant_weight.is_calib):
            dtype, fp_out = self.int_infer_no_calib_quant(x)
            return fp_out.to(dtype)
        if self.quant_input.bit <= 8:
            x = self.quant_input(x)

        if self.quant_weight.cfg.co_sparse:
            if not self.weight_quant_flag:
                per_channel = False if self.quant_weight.cfg.mm_tensor else True
                recovered_weight, s_w, i8_w, offset_w = \
                    quant_one_weight_by_outliers(self.weight, powerquant=self.quant_weight.cfg.nonuniform,
                                                 fraction=self.quant_weight.cfg.fraction,
                                                 num_bits=self.quant_weight.cfg.w_bit,
                                                 per_channel=per_channel)

                self.new_weight = recovered_weight.to(self.weight.device).type_as(self.weight)
                self.quant_weight.weight_scale = s_w.to(self.weight.device).type_as(self.weight)
                self.quant_weight.weight_offset = offset_w.to(self.weight.device).type_as(self.weight)
                self.weight_quant_flag = True
                self.weight.data = self.weight.data.cpu()  # Do not delete, for cpu offload

                del recovered_weight, s_w, i8_w
                gc.collect()
            return F.linear(x, self.new_weight, self.bias)
        else:
            if self.quant_input.w_hessian:  # gptq
                weight = self.quant_weight(self.weight, y=x.clone())
            else:
                weight = self.quant_weight(self.weight)
        return F.linear(x, weight, self.bias)

    def int_infer_no_calib_quant(self, x):
        # half
        dtype = x.dtype
        x = x.to(torch.float16)
        x = self.quant_input(x)  # 量化范围-127~127
        weight = self.quant_weight(self.weight)
        x = x.to(torch.float32)
        weight = weight.to(torch.float32)
        int_out = F.linear(x, weight)
        input_scale, input_offset = self.quant_input.get_scale_offset()
        weight_scale, _ = self.quant_weight.get_scale_offset()
        if (input_scale is not None) and (weight_scale is not None):
            if len(weight_scale.shape) > 1:
                weight_scale = weight_scale.reshape((len(x.shape) - 2) * (1,) + (1, weight.shape[0]))
            # offset correction, offline calibration
            correction = weight.sum(dim=1) * input_offset.to(torch.float32)
            if self.quant_weight.int_bias:
                # int32 bias, offline calibration
                fp_scale = input_scale * weight_scale
                fp_scale = fp_scale.to(torch.float32)
                if self.bias is not None:
                    bias_int = self.cal_bias_int(fp_scale)
                else:
                    bias_int = torch.zeros(correction.size(0)).to(x.divice)
                bias_int -= correction

                # int32 biasadd -> dequant
                fp_out = (int_out + bias_int) * fp_scale
                fp_out = fp_out.to(torch.float16)
            else:
                # dequant
                fp_scale = input_scale * weight_scale
                fp_scale = fp_scale.to(torch.float32)

                fp_out = int_out * fp_scale
                fp_out = fp_out.to(torch.float16)

                # fp32 bias, offline calibration
                correction_fp = correction * fp_scale
                correction_fp = correction_fp.to(torch.float16)
                if self.bias is not None:
                    bias_fp = self.bias.data.to(torch.float16) - correction_fp
                else:
                    bias_fp = - correction_fp
                # fp32 biasadd
                fp_out += bias_fp
        else:
            fp_out = int_out
            if self.bias is not None:
                fp_out += self.bias.data
        return dtype, fp_out

    def cal_bias_int(self, fp_scale):
        if fp_scale != 0:
            bias_int = (self.bias.data / fp_scale).round()
        else:
            raise ValueError("dequant scale is 0, please check your quant config.")
        return bias_int
