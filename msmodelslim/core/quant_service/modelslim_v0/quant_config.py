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

from pydantic import BaseModel
from typing_extensions import Self

from msmodelslim.core.quant_service.interface import BaseQuantConfig
from msmodelslim.utils.exception import SchemaValidateError


class QuantSpec(BaseModel):
    # anti
    anti_cfg: dict = None  # anti-outlier config
    anti_params: dict = None  # anti-outlier params

    # calib
    calib_cfg: dict = None  # calib config
    calib_params: dict = None  # calib params
    calib_save_params: dict = None  # calib save params

    # quantization parameters
    batch_size: int = 4  # batch size
    anti_dataset: str = None  # anti-outlier dataset
    calib_dataset: str = None  # calib dataset


class ModelslimV0QuantConfig(BaseQuantConfig):
    spec: QuantSpec  # quantization config specification

    @classmethod
    def from_base(cls, quant_config: BaseQuantConfig) -> Self:
        return cls(
            apiversion=quant_config.apiversion,
            spec=load_specific_config(quant_config.spec),
        )


def load_specific_config(yaml_spec: object) -> QuantSpec:
    """Load specific configuration from YAML spec"""
    if isinstance(yaml_spec, QuantSpec):
        return yaml_spec
    if not isinstance(yaml_spec, dict):
        raise SchemaValidateError("task spec must be dict", action='Please make sure the task spec is a dictionary')

    config = QuantSpec()
    config.anti_cfg = yaml_spec.get('anti_cfg', None)
    config.anti_params = yaml_spec.get('anti_params', {})
    config.calib_cfg = yaml_spec.get('calib_cfg', {})
    config.calib_params = yaml_spec.get('calib_params', {})
    config.calib_save_params = yaml_spec.get('calib_save_params', {})
    config.batch_size = yaml_spec.get('batch_size', 4)
    config.anti_dataset = yaml_spec.get('anti_dataset', None)
    config.calib_dataset = yaml_spec.get('calib_dataset', 'teacher_qualification.jsonl')
    return config
