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

import copy
import fnmatch

from typing import Dict, Any, Optional, List

import torch

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param


class LayerConfigManager:
    DEFAULT_CFG_STORE = {
        'rollback': QuantConfig(w_bit=16, a_bit=16),
        'default': QuantConfig(w_bit=8, a_bit=8, mm_tensor=False),
        'w8a8': QuantConfig(w_bit=8, a_bit=8, mm_tensor=False),
        'w8a8_dynamic': QuantConfig(w_bit=8, a_bit=8, is_dynamic=True, mm_tensor=False),
        'w8a16': QuantConfig(w_bit=8, a_bit=16, mm_tensor=False)
    }

    def __init__(self,
                 mix_cfg: Optional[Dict[str, Any]] = None,
                 cfg_store: Optional[Dict[str, Any]] = None,
                 rollback_names: Optional[List[str]] = None):
        self.logger = msmodelslim_logger
        self.mix_cfg: Dict[str, Any] = mix_cfg if mix_cfg is not None else {}
        self.cfg_store: Dict[str, Any] = cfg_store if cfg_store is not None else self.DEFAULT_CFG_STORE
        self.rollback_names: List[str] = rollback_names if rollback_names is not None else []
        self.layer_cfg: Dict[str, QuantConfig] = {}

    @staticmethod
    def convert_to_w8a8(cfg: QuantConfig) -> QuantConfig:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.w_bit = 8
        cfg_copy.a_bit = 8
        cfg_copy.is_dynamic = False
        check_and_generate_config_param(cfg_copy)
        return cfg_copy

    @staticmethod
    def convert_to_w816(cfg: QuantConfig) -> QuantConfig:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.w_bit = 8
        cfg_copy.a_bit = 16
        cfg_copy.is_dynamic = False
        check_and_generate_config_param(cfg_copy)
        return cfg_copy

    @staticmethod
    def convert_to_w8a8_dynamic(cfg: QuantConfig) -> QuantConfig:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.w_bit = 8
        cfg_copy.a_bit = 8
        cfg_copy.is_dynamic = True
        check_and_generate_config_param(cfg_copy)
        return cfg_copy


    """
        temperal method for deepseek v3/r1, change to w8a8 per token
    """
    @staticmethod
    def w4a8_dynamic_convert_to_w8a8_dynamic(cfg: QuantConfig) -> QuantConfig:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.w_bit = 8
        cfg_copy.a_bit = 8
        cfg_copy.is_dynamic = True
        cfg_copy.group_size = -1
        check_and_generate_config_param(cfg_copy)
        cfg_copy.a_sym = True
        cfg_copy.is_stage_quant = True
        return cfg_copy


    """
        temperal method for deepseek v3/r1, change to w8a8 per tensor
    """
    @staticmethod
    def w4a8_dynamic_convert_to_w8a8(cfg: QuantConfig) -> QuantConfig:
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.w_bit = 8
        cfg_copy.a_bit = 8
        cfg_copy.is_dynamic = False
        cfg_copy.group_size = -1
        check_and_generate_config_param(cfg_copy)
        cfg_copy.a_sym = False
        cfg_copy.is_stage_quant = True
        return cfg_copy


    @staticmethod
    def resolve_cfg_reference(name, cfg_store):
        if name not in cfg_store:
            raise ValueError(
                f"LayerConfigManager: config key '{name}' not found in cfg_store. "
                "Please check your mix_cfg or rollback settings."
            )
        return cfg_store[name]

    def build_layer_config(self, name) -> QuantConfig:
        if name in self.rollback_names:
            cfg = self.resolve_cfg_reference('rollback', self.cfg_store)
            self.logger.info(f'layer {name} was rolled back with {cfg.model_quant_type}')
            return cfg

        if name in self.mix_cfg:
            cfg = self.resolve_cfg_reference(self.mix_cfg[name], self.cfg_store)
            self.logger.info(f'layer {name} was specified with {cfg.model_quant_type}')
            return cfg

        for rule, quant_type in self.mix_cfg.items():
            if fnmatch.fnmatchcase(name, rule):
                cfg = self.resolve_cfg_reference(quant_type, self.cfg_store)
                self.logger.info(f'layer {name} match {rule}, will use {cfg.model_quant_type}')
                return cfg

        cfg = self.resolve_cfg_reference('default', self.cfg_store)
        self.logger.info(f'layer {name} not match any rule, will use {cfg.model_quant_type}')
        return cfg

    def get_layer_config(self, name) -> QuantConfig:

        if name in self.layer_cfg:
            return self.layer_cfg[name]

        return self.resolve_cfg_reference('default', self.cfg_store)

    def build_config_map(self, model: torch.nn.Module):
        self.layer_cfg = {
            name: self.build_layer_config(name) 
            for name, module in model.named_modules() 
            if isinstance(module, torch.nn.Linear)
        }
        