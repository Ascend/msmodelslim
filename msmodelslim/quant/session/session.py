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
# pylint: disable=consider-using-from-import

from typing import Dict, Optional, List, Any

import torch.nn as nn
from pydantic import BaseModel

from msmodelslim.utils.logging import get_logger
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig, Calibrator
from msmodelslim.utils.exception import SchemaValidateError

M3 = 'm3'
M4 = 'm4'
M6 = 'm6'
W8A8 = 'w8a8'
FA3 = 'fa3'
W8A8_DYNAMIC = 'w8a8_dynamic'
W8A8_TIMESTEP = 'w8a8_timestep'

ANTI_OUTLIER_METHOD_LIST = [M3, M4, M6]
QUANT_METHOD_LIST = [W8A8, FA3, W8A8_DYNAMIC, W8A8_TIMESTEP]
SAVE = 'save'

SD3_TRANSFORMER = 'SD3Transformer2DModel'
FLUX_TRANSFORMER = 'FluxTransformer2DModel'
HUNYUANVIDEO_TRANSFORMER = 'HYVideoDiffusionTransformer'

ACT_METHOD = {'minmax': 1, 'histogram': 2, 'mix': 3}


class M3ProcessorConfig(BaseModel):
    pass


class M4ProcessorConfig(BaseModel):
    pass


class M6Config(BaseModel):
    alpha: float = None
    beta: float = None


class M6ProcessorConfig(BaseModel):
    cfg: M6Config


class W8A8QuantConfig(BaseModel):
    act_method: str = 'minmax'


class W8A8ProcessorConfig(BaseModel):
    cfg: W8A8QuantConfig
    disable_names: list


class FA3ProcessorConfig(BaseModel):
    pass


class W8A8DynamicQuantConfig(BaseModel):
    act_method: str = 'minmax'


class W8A8DynamicProcessorConfig(BaseModel):
    cfg: W8A8DynamicQuantConfig
    disable_names: list


class W8A8TimeStepQuantConfig(BaseModel):
    act_method: str = 'minmax'


class W8A8TimeStepProcessorConfig(BaseModel):
    cfg: W8A8TimeStepQuantConfig
    disable_names: list
    timestep_sep: int


class SaveProcessorConfig(BaseModel):
    output_path: str
    safetensors_name: Optional[str] = None
    json_name: Optional[str] = None
    save_type: list = ['safe_tensor']
    part_file_size: Optional[int] = None


class SessionConfig(BaseModel):
    processor_cfg_map: Dict[str, BaseModel] = {}
    calib_data: Optional[List[Any]] = None
    device: str = 'cpu'


def process_session_cfg(session_cfg: SessionConfig, device_id):
    check_flag = True
    anti_cfg = None
    quant_cfg = None
    save_cfg = None
    processor_config_name_list = session_cfg.processor_cfg_map.keys()

    for processor_name in processor_config_name_list:
        if processor_name in ANTI_OUTLIER_METHOD_LIST:
            anti_cfg = AntiOutlierConfig(dev_type=session_cfg.device, dev_id=device_id)
            anti_cfg.anti_method = processor_name

            if anti_cfg.anti_method == M6:
                flex_config = {
                    'alpha': session_cfg.processor_cfg_map[anti_cfg.anti_method].cfg.alpha,
                    'beta': session_cfg.processor_cfg_map[anti_cfg.anti_method].cfg.beta,
                }
                anti_cfg.flex_config = anti_cfg.setup_flex_config(flex_config)

            # 将量化方法中的回退层添加入异常值抑制回退
            disable_quant_add_anti = next(
                (
                    session_cfg.processor_cfg_map[key].disable_names
                    for key in session_cfg.processor_cfg_map
                    if key in QUANT_METHOD_LIST
                ),
                None,
            )
            if disable_quant_add_anti:
                if not hasattr(anti_cfg, 'disable_anti_names') or anti_cfg.disable_anti_names is None:
                    anti_cfg.disable_anti_names = []
                anti_cfg.disable_anti_names.extend(disable_quant_add_anti)

        elif processor_name in QUANT_METHOD_LIST:
            if processor_name == FA3:
                if not session_cfg.processor_cfg_map.get(W8A8_DYNAMIC, None):
                    check_flag = False
                    break

                quant_cfg = QuantConfig(
                    dev_type=session_cfg.device,
                    dev_id=device_id,
                    is_dynamic=True,
                    mm_tensor=False,
                    act_method=ACT_METHOD[session_cfg.processor_cfg_map[W8A8_DYNAMIC].cfg.act_method],
                    disable_names=session_cfg.processor_cfg_map[W8A8_DYNAMIC].disable_names,
                ).fa_quant()

            elif processor_name == W8A8_TIMESTEP:
                quant_cfg = QuantConfig(
                    dev_type=session_cfg.device,
                    dev_id=device_id,
                    is_dynamic=False,
                    mm_tensor=False,
                    act_method=ACT_METHOD[session_cfg.processor_cfg_map[processor_name].cfg.act_method],
                    disable_names=session_cfg.processor_cfg_map[processor_name].disable_names,
                ).timestep_quant(session_cfg.processor_cfg_map[processor_name].timestep_sep)

            else:
                if not session_cfg.processor_cfg_map.get(FA3, None):
                    is_dynamic = processor_name == W8A8_DYNAMIC

                    quant_cfg = QuantConfig(
                        dev_type=session_cfg.device,
                        dev_id=device_id,
                        is_dynamic=is_dynamic,
                        mm_tensor=False,
                        act_method=ACT_METHOD[session_cfg.processor_cfg_map[processor_name].cfg.act_method],
                        disable_names=session_cfg.processor_cfg_map[processor_name].disable_names,
                    )

        elif processor_name == SAVE:
            save_cfg = session_cfg.processor_cfg_map[processor_name]
        else:
            check_flag = False
    if not check_flag:
        raise SchemaValidateError(
            "The processor_cfg_map in session_config is not supported", action="Please check session_config."
        )
    return anti_cfg, quant_cfg, save_cfg


def quant_model(model: nn.Module, session_cfg: SessionConfig):
    anti_cfg, quant_cfg, save_cfg = process_session_cfg(session_cfg, device_id=model.device.index)
    calib_data = session_cfg.calib_data
    if anti_cfg:
        norm_class_name = None
        if model.__class__.__name__ in (SD3_TRANSFORMER, FLUX_TRANSFORMER, HUNYUANVIDEO_TRANSFORMER):
            norm_class_name = 'layernorm'
        anti_outlier = AntiOutlier(model, calib_data, anti_cfg, norm_class_name=norm_class_name)
        anti_outlier.process()

    calibrator = Calibrator(model, quant_cfg, calib_data)
    calibrator.run()
    if save_cfg:
        calibrator.save(
            output_path=save_cfg.output_path,
            safetensors_name=save_cfg.safetensors_name,
            json_name=save_cfg.json_name,
            save_type=save_cfg.save_type,
            part_file_size=save_cfg.part_file_size,
        )
    else:
        get_logger().warning("The save config is None, the quantized model will not be saved.")
