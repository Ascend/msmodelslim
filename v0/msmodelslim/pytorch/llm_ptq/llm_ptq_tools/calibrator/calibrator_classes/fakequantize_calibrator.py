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


import sys
import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module, remove_hook_from_module

from ascend_utils.common.security.pytorch import validate_device
from ascend_utils.common.security import check_type
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.calibrator.calibrator_classes. \
    fake_quantize import FakeLinearQuantizerOfW8A16OrW4A16, FakeLinearQuantizerOfW8A8
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantModelJsonDescription
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim import logger as msmodelslim_logger

_SUPPORTED_DEVICES = ["cpu", "npu", 'gpu']


class FakeQuantizeCalibrator(object):
    def __init__(
            self,
            model,
            dev_id: int,
            dev_type: object,
            description: dict,
            safetensor: dict,
    ):
        check_type(description, dict, param_name='description')
        check_type(safetensor, dict, param_name='safetensor')
        self.logger = msmodelslim_logger
        self.dev_type, self.dev_id = validate_device(dev_type, dev_id, _SUPPORTED_DEVICES)
        self.model = self.init_model_device(model)
        self.rollback_names = []
        self.safetensor = safetensor
        self.description = description
        self.init_tensor_list()
        self.init_quantize_type_model()
        QuantModelJsonDescription.check_description_match(quant_model_json_description=self.description,
                                                          quant_model_safetensor=self.safetensor)

    def init_quantize_type_model(self):
        quant_type = self.description.get("model_quant_type", None)
        if quant_type in [QuantType.W8A16, QuantType.W4A16]:
            self.init_quantize_model(FakeLinearQuantizerOfW8A16OrW4A16)
        elif quant_type in [QuantType.W8A8, QuantType.W8A8_DYNAMIC]:
            self.init_quantize_model(FakeLinearQuantizerOfW8A8)
        else:
            raise ValueError(f'fake quantization only support {QuantType.W8A16},{QuantType.W4A16},{QuantType.W8A8}'
                             f',{QuantType.W8A8_DYNAMIC}, but got {quant_type}.')

    def init_tensor_list(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
                self.processing_rollback_names(name)

    def processing_rollback_names(self, name):
        if name + '.weight' in self.description:
            if self.description[name + '.weight'] == QuantType.FLOAT:
                self.rollback_names.append(name)
        else:
            raise ValueError(f"{name + '.weight'} not found in description")

    def init_model_device(self, model):
        if not isinstance(model, nn.Module):
            raise TypeError("Model should be a PyTorch model, please check it.")
        if self.dev_type == "cpu":
            same_device = self.dev_type == model.device.type
        else:
            same_device = self.dev_type == model.device
        if not same_device:
            self.logger.warning("Model is not on the device indicated in `dev_type`, "
                                "Model is on the device `{}` while `dev_type` "
                                "indicates `{}`".format(model.device, self.dev_type))
            self.logger.info("Transferring model from `{}` to `{}`...".format(model.device, self.dev_type))
            model = model.to(self.dev_type)
            self.logger.info("Transfer done. Suggest to check model.")
        return model

    def init_quantize_model(self, new_module):

        def _set_module(ori_mod, submodule_key, mod):
            tokens = submodule_key.split('.')
            sub_tokens = tokens[:-1]
            cur_mod = ori_mod
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            setattr(cur_mod, tokens[-1], mod)

        for name, module in self.model.named_modules():
            if name in self.rollback_names:
                continue
            if isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
                quant_type = self.description['model_quant_type']
                quant_mod = new_module()
                quant_params = dict
                quant_params_init = {'quant_weight': self.safetensor[name + '.weight']}
                if quant_type in [QuantType.W8A16, QuantType.W4A16]:
                    quant_params = {
                        'weight_offset': self.safetensor[name + '.weight_offset'],
                        'weight_scale': self.safetensor[name + '.weight_scale']
                    }
                elif quant_type == QuantType.W8A8:
                    quant_params = {
                        'deq_scale': self.safetensor[name + '.deq_scale'],
                        'input_offset': self.safetensor[name + '.input_offset'],
                        'input_scale': self.safetensor[name + '.input_scale'],
                    }
                elif quant_type == QuantType.W8A8_DYNAMIC:
                    quant_params = {
                        'weight_scale': self.safetensor[name + '.weight_scale'],
                        'is_dynamic': True
                    }
                quant_params.update(quant_params_init)
                quant_mod.set_param(module, **quant_params)
                # 拷贝accelerate定义的hook
                if hasattr(module, "_hf_hook"):
                    add_hook_to_module(quant_mod, module._hf_hook)
                    remove_hook_from_module(module)

                _set_module(self.model, name, quant_mod)
                del module.weight
