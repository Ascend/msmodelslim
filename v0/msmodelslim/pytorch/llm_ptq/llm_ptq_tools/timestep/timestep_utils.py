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


from collections import defaultdict
import itertools
from typing import List

from torch import nn
from tqdm import tqdm

from msmodelslim import logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import TimestepQuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.quantizer import LinearQuantizerTimestep, TimestepQuantMixin, \
    TimestepAwareTensorQuantizer


def _add_timestep_statistic(model: nn.Module):
    update_cnt = 0
    for _, mod in model.named_modules():
        if isinstance(mod, TimestepQuantMixin):
            mod.update_timestep_scale_offset()
            update_cnt += 1
    if update_cnt == 0:
        logger.warning("No timestep-aware module found in the model")


def _preprocess_data(data):
    # check intergrity
    if not data or data[0].get('timestep_idx', None) is None:
        raise ValueError("Please check the calibration data. calib_data must include timestep_idx")
    data.sort(key=lambda x: -x['timestep_idx'])

    data_grouped = [(k, list(v)) for k, v in itertools.groupby(data, key=lambda x: x['timestep_idx'])]
    return data_grouped


def _run_calib_timestep(model: nn.Module, data: List[dict]):
    data_group = _preprocess_data(data)
    for timestep_idx, group in tqdm(data_group, desc="Calibrating timestep-aware modules"):
        TimestepManager.set_timestep_idx(timestep_idx)

        for item in group:
            model(*item['args'])

            _add_timestep_statistic(model)


def change_timestep_aware_quantizer_config(model: nn.Module, max_dynamic_step: int):
    for _, mod, in model.named_modules():
        if isinstance(mod, TimestepAwareTensorQuantizer):
            mod.update_config(max_dynamic_step=max_dynamic_step)


def run_calib_timestep(model: nn.Module, data: List[dict], cfg: TimestepQuantConfig):
    """
    对支持 timestep-aware 量化的模块进行时间步粒度的校准。

    Parameters
    ----------
    model : nn.Module
        待校准的量化模型，其中部分线性层模块应继承自 `TimestepQuantMixin`
    data : List[dict]
        校准数据列表，每个元素是一个 dict，必须包含以下字段：
        - 'timestep_idx' (int): 时间步索引，表示当前样本所属的时间步；
        - ’kwargs dict[str, Any]‘, 用作 model(**kwargs)
    cfg : TimestepQuantConfig
        包含 timestep-aware 量化配置的实例

    Raises
    ------
    ValueError
        如果 data 中缺少 'timestep_idx' 或其类型不符合要求
    """
    if not isinstance(cfg, QuantConfig):
        raise ValueError("Please check the Quant config. It must be QuantConfig")

    _run_calib_timestep(model, data)

    return


def load_quant_weight(params_to_load: dict, model: nn.Module, device):
    """
    params_to_load: dict
        The parameters to load into the model.
        {
            "..layer_name.weight": tensor,
            "..layer_name.bias": tensor,
            "..layer_name.weight_scale": tensor,
            "..layer_name.weight_offset": tensor,
            "..layer_name.input_scale": tensor,
            "..layer_name.input_offset": tensor,
            ...
        }


    model: nn.Module
        The model to load the parameters into.
    device: str
        The device to load the parameters into.
    """

    def get_para_layer_name(name: str):
        if '.' not in name:
            return '', name

        module_name, para_name = name.rsplit('.', 1)
        return module_name, para_name

    quant_layer = {name: mod for name, mod in model.named_modules() if isinstance(mod, LinearQuantizerTimestep)}

    # get params grouped by layer
    params_grouped = defaultdict(dict)
    for name, tensor in params_to_load.items():
        module_name, para_name = get_para_layer_name(name)
        params_grouped[module_name][para_name] = tensor

    # load params into model
    for mod_name in params_grouped:
        if mod_name not in quant_layer:
            continue

        cur_layer = quant_layer[mod_name]
        cur_layer.load_layer_params(params_grouped[mod_name], device)
