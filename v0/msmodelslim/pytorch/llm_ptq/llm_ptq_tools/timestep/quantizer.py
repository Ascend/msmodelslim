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


import torch
import torch.nn as nn
import torch.nn.functional as F

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import TimestepQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import LinearQuantizer, TensorQuantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
from msmodelslim import logger as ms_logger


class TimestepQuantMixin:
    def __init__(self, *args, **kwargs):
        """
        Mixin for handling timestep-aware quantization.
        必须确保子类继承自 nn.Module 并具备以下属性：
            - self.cfg: 配置对象，包含 use_timestep_quant 和 max_dynamic_step
            - self.input_scale / self.input_offset: 存储动态 scale/offset
            - self.is_dynamic: 控制是否使用动态范围量化
        """
        super().__init__(*args, **kwargs)

        # 校验必须的属性是否存在
        required_attrs = ['cfg', 'input_scale', 'input_offset', 'is_dynamic']
        missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing_attrs:
            raise TypeError(
                f"Missing required attributes {missing_attrs} in class {self.__class__.__name__}. "
                "TimestepQuantMixin must be used with a class that defines these attributes."
            )

        # 添加类型提示和注释
        self.cfg: TimestepQuantConfig = self.cfg  # 配置对象，包含时间步量化相关参数
        if not getattr(self.cfg, 'use_timestep_quant', False):
            raise TypeError("Please check the Quant config. It must be TimestepQuantConfig "
                            "and self.cfg.use_timestep_quant must be True")

        # 初始化 timestep 对应的 scale/offset 容器
        self._timestep_scales = dict()  # t_idx -> scale (Tensor)
        self._timestep_offsets = dict()  # t_idx -> offset (Tensor)

        self.fake_quant_print_flag = True

    @staticmethod
    def get_current_timestep():
        """
        获取当前时间步索引
        """
        return TimestepManager.get_timestep_idx()

    def update_timestep_scale_offset(self, device='cpu'):
        t_idx = self.get_current_timestep()
        scale, offset = self.input_scale, self.input_offset
        self._timestep_scales[t_idx] = scale.to(device)
        self._timestep_offsets[t_idx] = offset.to(device)

    def apply_timestep_quant_settings(self, device='cpu'):
        """
        根据当前 timestep 决定是否使用固定 scale/offset
        """
        if not getattr(self.cfg, 'use_timestep_quant', False):
            return

        try:
            t_idx = self.get_current_timestep()
        except AttributeError as e:
            raise RuntimeError("get_current_timestep() failed. "
                               "Make sure TimestepManager is properly initialized.") from e

        max_t = self.cfg.max_dynamic_step

        if self.fake_quant_print_flag:
            ms_logger.info('Do fake quantization using max_dynamic_step=%s', self.cfg.max_dynamic_step)
            self.fake_quant_print_flag = False

        if t_idx >= max_t:
            # 使用最后一步的 scale/offset
            if max_t not in self._timestep_scales or max_t not in self._timestep_offsets:
                raise KeyError(f"Timestep {max_t} not found in stored scales/offsets.")
            self.input_scale = self._timestep_scales[max_t].to(device)
            self.input_offset = self._timestep_offsets[max_t].to(device)
            self.is_dynamic = False
        else:
            self.is_dynamic = True

    def update_config(self, max_dynamic_step: int):
        self.cfg.max_dynamic_step = max_dynamic_step

    def get_timestep_scale_offset_dict(self):
        """
        导出所有已记录的时间步 scale/offset
        """
        t_list = sorted(self._timestep_scales.keys())
        scale_tensor = torch.stack([self._timestep_scales[t].cpu() for t in t_list])
        offset_tensor = torch.stack([self._timestep_offsets[t].cpu() for t in t_list])

        return scale_tensor, offset_tensor

    def update_timestep_scale_offset_from_dict(self, state_dict):
        """
        Update timestep scale and offset parameters from a state dictionary.
        
        Args:
            state_dict: Dictionary containing  'input_scale', and 'input_offset' keys
        
        Raises:
            KeyError: If required keys are missing
            ValueError: If input arrays have mismatched lengths
            TypeError: If timestep index is not an integer
        """
        # Check if state_dict contains required keys
        required_keys = ['input_scale', 'input_offset']
        missing_keys = [key for key in required_keys if key not in state_dict]
        if missing_keys:
            raise KeyError(f"Required keys {missing_keys} not found in state_dict")

        scale = state_dict['input_scale']
        offset = state_dict['input_offset']

        if not isinstance(scale, torch.Tensor) or not isinstance(offset, torch.Tensor):
            raise TypeError(
                f"scale and offset must be torch.Tensor, got {type(scale).__name__} and {type(offset).__name__}")

        # Check lengths match
        if not (len(scale) == len(offset)):
            raise ValueError(f"Length mismatch: scale({len(scale)}), offset({len(offset)})")

        # get t_idx
        t_idx = list(range(scale.shape[0]))

        # Update parameters
        for t, scale, offset in zip(t_idx, scale, offset):
            if not isinstance(t, int):
                raise TypeError(f"Timestep index must be an integer, got {type(t).__name__}")

            try:
                self._timestep_scales[t] = nn.Parameter(scale)
                self._timestep_offsets[t] = nn.Parameter(offset)
            except Exception as e:
                raise RuntimeError(f"Failed to update parameters for timestep {t}: {str(e)}") from e


class TimestepAwareTensorQuantizer(TimestepQuantMixin, TensorQuantizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 先初始化 TensorQuantizer，然后初始化 Mixin


class LinearQuantizerTimestep(LinearQuantizer):
    def __init__(self, cfg=None, logger=None):
        super().__init__(cfg=cfg, logger=logger)

        if not getattr(cfg, 'use_timestep_quant', False):
            raise ValueError('TimestepQuantConfig error, use_timestep_quant must be True.')
        self.quant_input = TimestepAwareTensorQuantizer(
            bit=cfg.a_bit, is_signed=cfg.a_signed, is_enable=True,
            is_input=True, cfg=cfg, logger=logger, is_dynamic=cfg.is_dynamic
        )

    @staticmethod
    def reshape_x_to_blc(f):
        def forward_with_reshape(x: torch.Tensor):
            if x.dim() in {3}:
                return f(x)
            elif x.dim() == 1:
                raise ValueError("x.dim() == 1 is not supported")
            else:
                shape = x.shape
                # xx, ..., C_in -> B, L, C_in
                x_reshape = x.reshape(shape[0], -1, shape[-1])
                # B, L, C_out -> xx, ..., C_out
                return f(x_reshape).reshape(*shape[:-1], -1)

        return forward_with_reshape

    def forward(self, x):
        if self.quant_weight.int_infer and (not self.quant_weight.is_calib):
            return self.reshape_x_to_blc(self._int_infer_forward)(x)
        else:
            if not self.quant_weight.is_calib:
                # when is_calib is False, do fake quantization
                self.quant_input.apply_timestep_quant_settings(device=x.device)

            if self.quant_input.w_hessian:  # gptq
                weight = self.quant_weight(self.weight, y=x.clone())
            else:
                weight = self.quant_weight(self.weight)
            if self.quant_input.bit <= 8:
                x = self.quant_input(x)
            return F.linear(x, weight, self.bias)

    def get_quant_weight(self, device):
        # fix fake quantize, avoid to have duplicated float weight
        # offload float weight to cpu

        if self.quant_weight.int_weight_tensor is None:
            self.quant_weight(self.weight)

        if self.weight.device != 'cpu':
            self.weight = nn.Parameter(self.weight.cpu(), requires_grad=False)

        if not self.quant_weight.int_weight_tensor.device != device:
            self.quant_weight.int_weight_tensor = self.quant_weight.int_weight_tensor.to(device)

        return self.quant_weight.int_weight_tensor

    def set_quant_weight(self, quant_weight):
        self.quant_weight.int_weight_tensor = quant_weight.cpu()

    def load_layer_params(self, params_to_load: dict, device):
        # Check if state_dict contains required keys
        required_keys = ['weight_offset', 'weight_scale', 'weight',
                         'input_scale', 'input_offset',
                         'deq_scale', 'quant_bias']
        missing_keys = [key for key in required_keys if key not in params_to_load]
        if missing_keys:
            raise KeyError(f"Required keys {missing_keys} not found in state_dict")

        self.quant_input.update_timestep_scale_offset_from_dict(params_to_load)
        self.quant_input.is_calib = False
        self.quant_input.int_infer = True

        self.quant_weight.weight_offset = params_to_load['weight_offset'].to(device)
        self.quant_weight.weight_scale = params_to_load['weight_scale'].to(device)
        self.quant_weight.int_weight_tensor = params_to_load['weight'].to(device).to(torch.bfloat16)
        self.quant_weight.int_infer = True
        self.quant_weight.int_weight_flag = False
        self.quant_weight.has_init_quant_para = True
        self.quant_weight.is_calib = False

        self.weight = nn.Parameter(self.weight.cpu(), requires_grad=False)

    def _int_infer_forward(self, x):
        ori_dtype = torch.bfloat16
        self.quant_input.apply_timestep_quant_settings(device=x.device)

        quant_param_dtype = torch.float32
        x_dtype = torch.float16
        if ori_dtype == torch.bfloat16:
            quant_param_dtype = torch.bfloat16
            x_dtype = torch.bfloat16

        x = x.to(x_dtype)
        x = self.quant_input(x)
        x = x.to(quant_param_dtype)

        weight = self.get_quant_weight(x.device)
        weight = weight.to(quant_param_dtype)

        int_out = F.linear(x, weight)

        input_scale, input_offset = self.quant_input.get_scale_offset()
        weight_scale, _ = self.quant_weight.get_scale_offset()
        if (input_scale is not None) and (weight_scale is not None):
            input_scale = input_scale.to(quant_param_dtype)
            input_offset = input_offset.to(quant_param_dtype)
            weight_scale = weight_scale.to(quant_param_dtype)
            if len(weight_scale.shape) > 1:
                weight_scale = weight_scale.reshape((len(x.shape) - 2) * (1,) + (1, weight.shape[0]))
            # offset correction, offline calibration
            correction = weight.sum(dim=1) * input_offset.to(quant_param_dtype)
            fp_scale = input_scale * weight_scale
            fp_scale = fp_scale.to(quant_param_dtype)
            fp_out = self._bias_and_dequant_process(correction, int_out, fp_scale, x.device, x_dtype)
        else:
            fp_out = int_out
            if self.bias is not None:
                fp_out += self.bias.data

        return fp_out.to(ori_dtype)

