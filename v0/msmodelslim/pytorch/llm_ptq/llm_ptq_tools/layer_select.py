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


import functools
from abc import ABC, abstractmethod
from itertools import groupby

import torch
import torch.nn as nn
from tqdm import tqdm
from ascend_utils.common.security import check_type, check_number, check_element_type
from msmodelslim import logger as msmodelslim_logger
_SUPPORT_QUANTILE = False
try:
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
        get_quantile_score
    )
except ImportError:
    msmodelslim_logger.warning(
        "The current CANN version does not support LayerSelector quantile method."
    )
else:
    _SUPPORT_QUANTILE = True

DISABLE_LEVEL = "disable_level"
THRESHOLD = "threshold"
PERCENT = "percent"
QUANTILE = "quantile"
STD = "std"
SUPPORT_METHOD = (QUANTILE, STD)


def check_tensor_list(calib_data_item):
    for item in calib_data_item:
        if not isinstance(item, torch.Tensor):
            return True
    return False


def check_tensor_dict(calib_data_item):
    for _, value in calib_data_item.items():
        if not isinstance(value, torch.Tensor):
            return True
    return False


def check_calib_data(calib_data):
    check_type(calib_data, list, param_name='calib_data')
    for i, calib_data_item in enumerate(calib_data):
        element_not_tensor = False
        check_type(calib_data_item, (list, dict), param_name=f"calib_data[{i}]")
        if isinstance(calib_data_item, list):
            element_not_tensor = check_tensor_list(calib_data_item)
        elif isinstance(calib_data_item, dict):
            element_not_tensor = check_tensor_dict(calib_data_item)
        if element_not_tensor:
            raise ValueError("Not all elements in calib_data are list or dict of torch.Tensor, "
                                "please make sure that the model can run with model(*(calib_data[0]))")


def check_layer_names(model, layer_names):
    quant_name_list = get_all_linear_conv_module(model)
    for name in layer_names:
        if name not in quant_name_list:
            raise ValueError(f"`disable_names` has invalid key `{name}`, please check your model configurations.")


def get_all_linear_conv_module(model):
    quant_name_list = []
    conv_name_list = []
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
            quant_name_list.append(name)
        elif isinstance(module, nn.Conv2d):
            conv_name_list.append(name)
    return quant_name_list + conv_name_list


class RangeMethod(ABC):
    @abstractmethod
    def get_hook(self):
        # hook activation to compute score of layer
        pass

    @abstractmethod
    def get_layer_score(self, layer):
        # compute score of a layer given by info of hook
        pass

    def get_scores(self, act_scales):
        named_act = []
        for name in tqdm(act_scales, desc="LayerSelector computing scores"):
            res = self.get_layer_score(act_scales[name])
            named_act.append({'name': name, 'score': res})
        return named_act
    

class QuantileMethod(RangeMethod):
    name = QUANTILE

    def __init__(self, sample_step=100):
        self.sample_step = sample_step

    def get_hook(self):
        def activation_hook(m, x, y, name, act_stats):
            if isinstance(x, tuple):
                x = x[0]

            x = x.reshape(-1)
            x = torch.sort(x)[0]
            sample_step = self.sample_step
            if x.numel() < sample_step:
                x_sample = x
            else:
                x_sample = x[sample_step // 2::sample_step].view(-1, 1)
            if name not in act_stats:
                act_stats[name] = {}
                act_stats[name]['tensor'] = [x_sample.to('cpu')]
            else:
                act_stats[name]['tensor'].append(x_sample.to("cpu"))
            act_stats[name]['device'] = x.device

        return activation_hook
    
    def get_layer_score(self, layer):
        act = torch.cat(layer['tensor']).view(-1).float()
        device = layer['device']
        res = get_quantile_score(act, device)
        return res
    

class StdMethod(RangeMethod):
    name = STD

    def get_hook(self):
        def activation_hook(m, x, y, name, act_stats):
            if isinstance(x, tuple):
                x = x[0]
            
            tensor_in = x.float()
            hidden_dim = tensor_in.shape[-1]
            tensor = tensor_in.reshape(-1, hidden_dim).detach()
            comming_max = torch.max(tensor, dim=0)[0]
            comming_min = torch.min(tensor, dim=0)[0]

            if name not in act_stats:
                act_stats[name] = {}

            act_stats[name]['shift'] = (comming_max + comming_min)

            tensor_max = torch.max(tensor)
            if 't_max' in act_stats[name].keys():
                act_stats[name]['t_max'] = torch.max(act_stats[name]['t_max'], tensor_max)
            else:
                act_stats[name]['t_max'] = tensor_max
            
            tensor_min = torch.min(tensor)
            if 't_min' in act_stats[name].keys():
                act_stats[name]['t_min'] = torch.min(act_stats[name]['t_min'], tensor_min)
            else:
                act_stats[name]['t_min'] = tensor_min

            tensor_std = torch.std(tensor - act_stats[name]['shift'])
            if 'std' in act_stats[name].keys():
                act_stats[name]['std'] = torch.max(act_stats[name]['std'], tensor_std)
            else:
                act_stats[name]['std'] = tensor_std
        return activation_hook
    
    def get_layer_score(self, layer):
        abs_max_tensor = max(abs(layer['t_max']), abs(layer['t_min']))
        range_param = abs_max_tensor / layer['std']
        return range_param.item()
    

class RangeMethodFactory:
    @staticmethod
    def create_method(range_method: str):
        if range_method not in SUPPORT_METHOD:
            raise ValueError("range method is invalid!")
        if range_method == QUANTILE:
            if not _SUPPORT_QUANTILE:
                raise ImportError("The current CANN version does not support LayerSelector quantile method!")
            range_method = QuantileMethod()
        elif range_method == STD:
            range_method = StdMethod()
        else:
            raise NotImplementedError()
        return range_method
    

class LayerSelector:
    def __init__(self, model: nn.Module, layer_names=None, range_method="quantile"):
        '''
        Initialize the LayerSelector object.

        Parameters:
        - model (nn.Module): The PyTorch model whose layers will be analyzed.
        - layer_names (list, optional): List of layer names to analyze. 
                                        If None, all linear and convolutional layers will be used.
        - range_method (str): Method to compute the range for quantization, default is "quantile".

        Raises:
        - TypeError: If the model is not of type nn.Module.
        - ValueError: If the range_method or layer_names do not match expected types or the model structure.
        '''

        self.logger = msmodelslim_logger
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module, please check it.")
        self.model = model
        if layer_names is None:
            self.layer_names = get_all_linear_conv_module(model)
        else:
            check_element_type(layer_names, element_type=str, value_type=list, param_name='layer_names')
            self.layer_names = list(set(layer_names))
        check_type(range_method, str, param_name="range_method")
        check_layer_names(model, self.layer_names)
        self.range_method = RangeMethodFactory.create_method(range_method)
        self.layer_scores = None
        self.layer_groups = None

    def run(self, calib_data):
        '''
        Run the LayerSelector to collect activation statistics and compute layer parameters.

        Parameters:
        - calib_data (iterable): Calibration data used to compute activation statistics.
        
        Process:
        - Registers hooks to the specified layers in the model to collect activation data.
        - Runs the model using the calibration data to record activation statistics.
        - Removes hooks after data collection and computes quantization difficulty scores for the layers.

        Raises:
        - ValueError: If the calibration data format is unsupported.
        '''
        model = self.model
        check_calib_data(calib_data)

        act_stats = {}
        hooks = []
        act_hook = self.range_method.get_hook()

        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                if name in self.layer_names:
                    hooks.append(
                        m.register_forward_hook(functools.partial(act_hook, name=name, act_stats=act_stats))
                    )
        for data in tqdm(calib_data, desc="LayerSelector running model"):
            if isinstance(data, tuple) or isinstance(data, list):
                with torch.no_grad():
                    model(*data)
            elif isinstance(data, dict):
                with torch.no_grad():
                    model(**data)
            else:
                raise NotImplementedError()
            
        for h in hooks:
            h.remove()

        self.compute_layer_param(act_stats)

    def compute_layer_param(self, act_scales):
        '''
        Compute quantization difficulty scores for each layer based on activation statistics.

        Parameters:
        - act_scales (dict): Collected activation statistics from the model run.

        Process:
        - Uses the range method to compute scores for each layer.
        - Logs the quantization difficulty scores for each layer.
        - Groups layers based on their scores for easier selection later.
        '''
        named_act = self.range_method.get_scores(act_scales)
        for it in named_act:
            self.logger.info(f"{it['name']}'s quantization difficulty score: {it['score']:.4f}")
        self.layer_scores = sorted(named_act, key=lambda x: x['score'], reverse=True)
        self.layer_groups = [list(group) for key, group in groupby(self.layer_scores, key=lambda x: x['score'])]

    def select_layers_by_threshold(self, threshold):
        '''
        Select layers based on a threshold value for their quantization difficulty score.

        Parameters:
        - threshold (float): Minimum score for a layer to be selected.

        Returns:
        - auto_disable_names (list): List of layer names that exceed the threshold.

        Raises:
        - ValueError: If the threshold is not a valid number or below zero.
        '''
        check_number(threshold, min_value=0, param_name=THRESHOLD)
        ret = list(filter(lambda x: x['score'] > threshold, self.layer_scores))
        auto_disable_names = [layer['name'] for layer in ret]
        return auto_disable_names
    
    def select_layers_by_disable_level(self, disable_level):
        '''
        Select layers based on their disable levels.

        Parameters:
        - disable_level (int): Number of groups to include in the selection.

        Returns:
        - auto_disable_names (list): List of layer names in the selected disable groups.

        Process:
        - Iterates through the grouped layers based on their scores.
        - Logs a message if multiple layers in a group share the same input.

        Raises:
        - ValueError: If the disable_level is not a valid integer or is less than zero.
        '''
        check_number(disable_level, int, 0, param_name=DISABLE_LEVEL)
        groups = self.layer_groups[:disable_level]
        auto_disable_names = []
        for group in groups:
            temp = [layer['name'] for layer in group]
            auto_disable_names += temp
            if len(temp) > 1:
                info = ", ".join(temp)
                self.logger.info(f"{info} have the same input, will be selected together!")
        return auto_disable_names
    