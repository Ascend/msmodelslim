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

import re
from enum import Enum
from pathlib import Path
from typing import Optional, List

import torch

from msmodelslim.core.const import DeviceType
from msmodelslim.core.const import QuantType
from msmodelslim.core.practice.interface import PracticeConfig, ScenarioTagMatch
from msmodelslim.core.quant_service import IQuantService
from msmodelslim.model import IModelFactory, IModel
from msmodelslim.utils.exception import SchemaValidateError, ToDoError, UnsupportedError
from msmodelslim.utils.exception_decorator import exception_catcher
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import yaml_safe_load
from msmodelslim.utils.validation.conversion import (
    convert_to_readable_file,
    convert_to_writable_dir,
    convert_to_readable_dir,
)
from msmodelslim.utils.validation.value import validate_str_length
from .model_info_interface import ModelInfoInterface
from .practice_manager_infra import PracticeManagerInfra, QuantConfigExportInfra

DEFAULT_PEDIGREE = 'default'
DEFAULT_QUANT_TYPE = QuantType.W8A8


class TipsType(str, Enum):
    """
    Q1_C0_B0:
    含义解读：
    Q：量化方式是否指定 Q1-指定；Q0-未指定
    C：量化方式是否更改 C1-更改；C0-未更改
    B：是否是最佳实践  B1-是最佳实践；B0-非最佳实践
    """

    Q0C0B0 = "Q0_C0_B0"  # 未指定量化方式，未更改量化方式，非最佳实践，未指定量化方式场景不存在量化方式变更场景
    Q0C0B1 = "Q0_C0_B1"  # 未指定量化方式，未更改量化方式，是最佳实践，未指定量化方式场景不存在量化方式变更场景
    Q1C0B0 = "Q1_C0_B0"  # 已指定量化方式，未更改量化方式，非最佳实践
    Q1C0B1 = "Q1_C0_B1"  # 已指定量化方式，未更改量化方式，是最佳实践，原正常匹配最佳实践场景
    Q1C1B0 = "Q1_C1_B0"  # 已指定量化方式，已更改量化方式，非最佳实践
    Q1C1B1 = "Q1_C1_B1"  # 已指定量化方式，已更改量化方式，是最佳实践


def _build_quant_tips(tips_type: TipsType, model_type: str, quant_type: QuantType, config_id: str) -> str:
    """
    Args:
        tips_type: 提示类型
        model_type:模型类型
        quant_type:量化方式
        config_id:最佳实践文件config_id

    Returns: 提示词
    """

    if tips_type == TipsType.Q0C0B0:
        return (
            f"No quant_type or config_path provided. Default quant_type:{DEFAULT_QUANT_TYPE} will be used."
            f"The default practice:{config_id} for {DEFAULT_QUANT_TYPE} will be used."
        )
    elif tips_type == TipsType.Q0C0B1:
        return (
            f"No quant_type or config_path provided. Default quant_type:{DEFAULT_QUANT_TYPE} will be used."
            f"The best practice:{config_id} for {DEFAULT_QUANT_TYPE} will be used."
        )
    elif tips_type == TipsType.Q1C0B0:
        return (
            f"No best practice found for model_type={model_type} and quant_type={quant_type}. "
            f"The default practice:{config_id} for {quant_type} will be used."
        )
    elif tips_type == TipsType.Q1C0B1:
        return ""
    elif tips_type == TipsType.Q1C1B0:
        return (
            f"No best practice found for model_type={model_type} and quant_type={quant_type}. "
            f"The default practice:{config_id} for {DEFAULT_QUANT_TYPE} will be used."
        )
    elif tips_type == TipsType.Q1C1B1:
        return (
            f"No best practice found for model_type={model_type} and quant_type={quant_type}. "
            f"The best practice:{config_id} for {DEFAULT_QUANT_TYPE} will be used."
        )
    else:
        raise UnsupportedError("Get best practice error", action="Please use the correct msmodelslim version.")


def validate_device_index(device_index: Optional[List[int]], device_type: DeviceType):
    """
    Validate device_index parameter.

    Args:
        device_index: Device indices to validate
        device_type: Device type for context validation

    Raises:
        SchemaValidateError: If device_index is invalid
    """

    # Value validation: check if indices are non-negative
    if any(idx < 0 for idx in device_index):
        negative_indices = [idx for idx in device_index if idx < 0]
        raise SchemaValidateError(
            f"Device indices must be non-negative integers, but got negative values: {negative_indices}"
        )

    # Value validation: check for duplicates
    if len(device_index) != len(set(device_index)):
        duplicates = [idx for idx in set(device_index) if device_index.count(idx) > 1]
        raise SchemaValidateError(f"Device indices must be unique, but found duplicates: {duplicates}")

    # CPU does not support multi-device
    if device_type == DeviceType.CPU and len(device_index) > 1:
        raise SchemaValidateError(
            f"CPU does not support multi-device configuration. "
            f"Got device indices: {device_index}. "
            f"Please use NPU for multi-device parallel, or use single CPU device."
        )

    # Value validation: check device availability
    if device_type == DeviceType.NPU:
        max_device_count = torch.npu.device_count()
    else:
        # CPU doesn't need device count validation
        max_device_count = None

    # Check if indices exceed available devices
    if max_device_count is not None:
        invalid_indices = [idx for idx in device_index if idx >= max_device_count]
        if invalid_indices:
            raise SchemaValidateError(
                f"Device indices {invalid_indices} exceed maximum available device index "
                f"({max_device_count - 1}). Available device indices: 0 to {max_device_count - 1}"
            )


@logger_setter('msmodelslim.app.naive_quantization')
class NaiveQuantizationApplication:
    def __init__(
        self,
        practice_manager: PracticeManagerInfra,
        quant_service: IQuantService,
        model_factory: IModelFactory,
        quant_config_export_infra: Optional[QuantConfigExportInfra] = None,
    ):
        self.practice_manager = practice_manager
        self.quant_service = quant_service
        self.model_factory = model_factory
        self.quant_config_export_infra = quant_config_export_infra

    @staticmethod
    def check_config(
        config: PracticeConfig,
        model_type: str,
        quant_type: QuantType,
        scenario_tags: Optional[List[str]] = None,
        is_default=False,
    ) -> ScenarioTagMatch:
        label = config.metadata.label
        # Parse quant_type parameters
        match_result = re.match(r'^w(\d+)a(\d+)((?:c8|f[48])?)(s?)$', quant_type.value)
        if not match_result:
            raise ValueError(f"Invalid quant_type format: {quant_type.value}")
        w_bit = int(match_result.group(1))
        a_bit = int(match_result.group(2))
        suffix = match_result.group(3) or ''
        fa_bit = {'f4': 4, 'f8': 8}.get(suffix)
        is_sparse = bool(match_result.group(4))

        # Check if the label matches the quantization parameters
        if label.get('w_bit') != w_bit:
            return ScenarioTagMatch.NO_MATCH
        if label.get('a_bit') != a_bit:
            return ScenarioTagMatch.NO_MATCH
        if is_sparse ^ label.get('is_sparse', False):
            return ScenarioTagMatch.NO_MATCH
        if (suffix == 'c8') ^ label.get('kv_cache', False):
            return ScenarioTagMatch.NO_MATCH

        # FA3 精度匹配：fa_bit 精确匹配，若无 fa_bit 则 fa_quant=True 默认 8bit
        label_fa_bit = label.get('fa_bit')
        if label_fa_bit is None:
            label_fa_bit = 8 if label.get('fa_quant', False) else None
        if fa_bit != label_fa_bit:
            return ScenarioTagMatch.NO_MATCH
        if is_default:
            return ScenarioTagMatch.MATCH

        verified_model_types = getattr(config.metadata, 'verified_model_types', None)
        if verified_model_types:
            if model_type in verified_model_types:
                return ScenarioTagMatch.MATCH
            return ScenarioTagMatch.NO_MATCH

        return config.matches_scenario_tags(model_type, scenario_tags)

    def get_best_practice(
        self,
        model_adapter: IModel,
        quant_type: Optional[QuantType] = None,
        config_path: Optional[Path] = None,
        tag: Optional[List[str]] = None,
    ) -> PracticeConfig:
        """
        获取最佳实践匹配规则如下：
        场景1：指定config_path配置文件，直接采用，忽略quant_type配置
        场景2：未指定config_path和quant_type，将quant_type置为默认quant_type，然后按照场景3处理
        场景3：指定quant_type，查找最佳实践规则如下（优先级从高到低）：
        当前pedigree + 指定quant_type > 默认pedigree + 指定quant_type
        > 当前pedigree + 默认quant_type > 默认pedigree + 默认quant_type
        """
        # Handle explicit config path
        if config_path is not None:
            config_dict = yaml_safe_load(str(config_path))
            config = PracticeConfig.model_validate(config_dict)
            get_logger().info("Naive Quant apply config_path: %s", config_path)
            return config

        if not isinstance(model_adapter, ModelInfoInterface):
            raise ToDoError(
                f"Model adapter {model_adapter.__class__.__name__} does NOT implement ModelInfoInterface",
                action="Please implement ModelInfoInterface to support get best practice.",
            )

        model_type = model_adapter.get_model_type()
        model_pedigree = model_adapter.get_model_pedigree()

        # Handle unknown model
        if model_pedigree not in self.practice_manager:
            raise ToDoError(
                f"model_pedigree {model_pedigree} does NOT exist",
                action=f"Maybe you need change model_pedigree of model_adapter "
                f"or add {model_pedigree} in lab_practice.",
            )

        config, tips = self.get_config(model_pedigree, model_type, quant_type, tag)

        if tips != "":
            user_input = input(tips + "(Enter y to continue, otherwise it will exit): ").strip().lower()[:3]
            if user_input != 'y':
                raise UnsupportedError(
                    f"No best practice found for model_type={model_type} and quant_type={quant_type}",
                    action="You can specify the quantization configuration through config_path or change quant_type.",
                )
        return config

    def get_config(
        self,
        model_pedigree: str,
        model_type: str,
        quant_type: Optional[QuantType] = None,
        tag: Optional[List[str]] = None,
    ):
        has_quant_type = quant_type is not None
        use_quant_type = quant_type if quant_type is not None else DEFAULT_QUANT_TYPE
        standby_configs: List[PracticeConfig] = []
        is_default = model_pedigree == DEFAULT_PEDIGREE

        def _check(
            config: PracticeConfig, model_type: str, qt: QuantType, tag: Optional[List[str]] = None, is_default=False
        ):
            return self.check_config(config, model_type, qt, tag, is_default)

        def _build_return(config: PracticeConfig, tips_type: TipsType, qt: QuantType):
            tips = _build_quant_tips(tips_type, model_type, quant_type, config.metadata.config_id)
            return config, tips

        # 场景1：【指定量化方式】在模型适配器的最佳实践目录搜索指定量化类型的最佳实践
        for config in self.practice_manager.iter_config(model_pedigree):
            result = _check(config, model_type, use_quant_type, tag, is_default)
            if result == ScenarioTagMatch.NO_MATCH:
                continue
            if result == ScenarioTagMatch.STANDBY:
                standby_configs.append(config)
                continue
            # 默认模型适配器（未知模型）
            if has_quant_type:
                tips_type = TipsType.Q1C0B0 if model_pedigree == DEFAULT_PEDIGREE else TipsType.Q1C0B1
            else:
                tips_type = TipsType.Q0C0B0 if model_pedigree == DEFAULT_PEDIGREE else TipsType.Q0C0B1
            return _build_return(config, tips_type, use_quant_type)

        if standby_configs:
            tips = (
                f"No config verified for tags {tag}, including device_type and inference_engine. "
                f"Using standby config: {standby_configs[0].metadata.config_id}. "
            )
            return standby_configs[0], tips

        # 场景2：【指定量化方式】在最佳实践的default目录搜索指定量化类型的最佳实践
        if model_pedigree != DEFAULT_PEDIGREE:
            for config in self.practice_manager.iter_config(DEFAULT_PEDIGREE):
                result = _check(config, model_type, use_quant_type, tag, is_default=True)
                if result == ScenarioTagMatch.NO_MATCH:
                    continue
                tips_type = TipsType.Q1C0B0 if has_quant_type else TipsType.Q0C0B0
                return _build_return(config, tips_type, use_quant_type)

        if use_quant_type == DEFAULT_QUANT_TYPE or not has_quant_type:
            raise UnsupportedError("Get best practice error", action="Please use the correct msmodelslim version.")

        # 场景3：【默认量化方式】在模型适配器的最佳实践目录搜索默认量化类型的最佳实践
        for config in self.practice_manager.iter_config(model_pedigree):
            result = _check(config, model_type, DEFAULT_QUANT_TYPE, tag, is_default)
            if result == ScenarioTagMatch.NO_MATCH:
                continue
            if result == ScenarioTagMatch.STANDBY:
                standby_configs.append(config)
                continue
            tips_type = TipsType.Q1C1B0 if model_pedigree == DEFAULT_PEDIGREE else TipsType.Q1C1B1
            return _build_return(config, tips_type, DEFAULT_QUANT_TYPE)

        if standby_configs:
            tips = (
                f"No config verified for tags {tag}, including device_type and inference_engine. "
                f"Using standby config: {standby_configs[0].metadata.config_id}. "
            )
            return standby_configs[0], tips

        # 场景4：【默认量化方式】在最佳实践的default目录搜索默认量化类型的最佳实践
        if model_pedigree != DEFAULT_PEDIGREE:
            for config in self.practice_manager.iter_config(DEFAULT_PEDIGREE):
                result = _check(config, model_type, DEFAULT_QUANT_TYPE, tag, is_default=True)
                if result == ScenarioTagMatch.NO_MATCH:
                    continue
                return _build_return(config, TipsType.Q1C1B0, DEFAULT_QUANT_TYPE)

        raise UnsupportedError("Get best practice error", action="Please use the correct msmodelslim version.")

    @exception_catcher
    def quant(
        self,
        model_type: Optional[str],
        model_path: str,
        save_path: str,
        device_type: DeviceType = DeviceType.NPU,
        device_index: Optional[List[int]] = None,
        quant_type: Optional[QuantType] = None,
        config_path: Optional[str] = None,
        trust_remote_code: bool = False,
        tag: Optional[List[str]] = None,
    ):
        """
        Run the naive quantization application.
        Args:
            model_type: Optional[str], the type of the model; omit when config_path uses apiversion modelslim_convert
            model_path: str, the path of the model
            save_path: str, the path to save the quantized model
            device_type: DeviceType, the type of device (e.g., DeviceType.NPU, DeviceType.CPU)
                        Default: DeviceType.NPU
            device_index: Optional[List[int]], list of device indices to use (e.g., [0, 1, 2, 3])
                         If None, uses single default device
                         Default: None
            quant_type: Optional[QuantType], the quantization type, config_path and quant_type only one can be provided
            config_path: Optional[str], the path to config file, config_path and quant_type only one can be provided
            trust_remote_code: bool, whether to trust the remote code
            tag: Optional[List[str]], e.g. ['vLLM-Ascend','Atlas_A2_Inference'], tags to match configs with verified_tags
        """
        # 字符串类型与长度校验
        for param_name, value in [("model_path", model_path), ("save_path", save_path)]:
            if not isinstance(value, str):
                raise SchemaValidateError(f"{param_name} must be a string, but got {type(value)}")
            validate_str_length(input_str=value, str_name=param_name)
        if model_type is not None:
            if not isinstance(model_type, str):
                raise SchemaValidateError(f"model_type must be a string, but got {type(model_type)}")
            validate_str_length(input_str=model_type, str_name="model_type")

        model_path = convert_to_readable_dir(model_path)
        if not isinstance(model_path, Path):
            raise SchemaValidateError(f"model_path must be a Path, but got {type(model_path)}")
        save_path = convert_to_writable_dir(save_path)
        if not isinstance(save_path, Path):
            raise SchemaValidateError(f"save_path must be a Path, but got {type(save_path)}")
        if not isinstance(device_type, DeviceType):
            raise SchemaValidateError(f"device_type must be a DeviceType, but got {type(device_type)}")
        if device_index is not None:
            validate_device_index(device_index, device_type)
        if config_path is not None:
            validate_str_length(input_str=config_path, str_name='config_path')
            config_path = convert_to_readable_file(config_path)
        # 允许quant_type和config_path均为空的场景
        if quant_type is not None and config_path is not None:
            raise SchemaValidateError("quant_type and config_path only one can be provided")
        if quant_type is not None and not isinstance(quant_type, QuantType):
            raise SchemaValidateError("quant_type must be a QuantType")
        if config_path is not None and not isinstance(config_path, Path):
            raise SchemaValidateError(f"config_path must be a Path, but got {type(config_path)}")
        if model_type is None:
            from msmodelslim.core.quant_service.modelslim_convert.config_detect import is_modelslim_convert_config

            if config_path is None or not is_modelslim_convert_config(config_path):
                raise SchemaValidateError("model_type is required unless config_path uses apiversion modelslim_convert")
        if not isinstance(trust_remote_code, bool):
            raise SchemaValidateError("trust_remote_code must be a bool")
        if tag is not None:
            if not isinstance(tag, list):
                raise SchemaValidateError(f"tag must be a list or None, but got {type(tag)}")
            if len(tag) == 0:
                tag = None

        # Log parameters
        get_logger().info("quantization with following parameters:")
        get_logger().info("model_type: %s", model_type)
        get_logger().info("model_path: %s", model_path)
        get_logger().info("save_path: %s", save_path)
        get_logger().info("device_type: %s", device_type)
        if device_index is not None and len(device_index) > 1:
            device_list = ','.join(map(str, device_index))
            get_logger().info(
                "using %d devices: %s:%s",
                len(device_index),
                device_type.value,
                device_list,
            )
        elif device_index is not None and len(device_index) == 1:
            get_logger().info("using single device: %s:%s", device_type.value, device_index[0])
        else:
            get_logger().info("using single device (default): %s", device_type.value)
        if quant_type is not None:
            get_logger().info("quant_type: %s", quant_type)
        if config_path is not None:
            get_logger().info("config_path: %s", config_path)
        get_logger().info("trust_remote_code: %s", trust_remote_code)
        if tag:
            get_logger().info("tag: %s", tag)

        self._quant(
            model_type,
            model_path,
            save_path,
            device_type,
            device_index,
            quant_type,
            config_path,
            trust_remote_code,
            tag,
        )

    def _quant(
        self,
        model_type: Optional[str],
        model_path: Path,
        save_path: Path,
        device_type: DeviceType = DeviceType.NPU,
        device_index: Optional[List[int]] = None,
        quant_type: Optional[QuantType] = None,
        config_path: Optional[Path] = None,
        trust_remote_code: bool = False,
        tag: Optional[List[str]] = None,
    ):
        get_logger().info("===========ANALYSE MODEL===========")
        from msmodelslim.core.quant_service.modelslim_convert.config_detect import is_modelslim_convert_config
        from msmodelslim.model.base import BaseModelAdapter

        use_convert_adapter = config_path is not None and is_modelslim_convert_config(config_path)
        if use_convert_adapter:
            model_adapter = BaseModelAdapter(
                model_type=model_type or "convert",
                model_path=model_path,
                trust_remote_code=trust_remote_code,
            )
            get_logger().info("Using BaseModelAdapter for modelslim_convert (no model code load).")
        else:
            model_adapter = self.model_factory.create(model_type, model_path, trust_remote_code)
            get_logger().info("Using model adapter %s.", model_adapter.__class__.__name__)

        get_logger().info("===========GET BEST PRACTICE===========")
        practice_config = self.get_best_practice(
            model_adapter=model_adapter, quant_type=quant_type, config_path=config_path, tag=tag
        )
        # 使用量化配置导出基础设施导出配置
        export_model_type = model_type or "convert"
        self.quant_config_export_infra.export_quant_config(practice_config, export_model_type, save_path)

        get_logger().info("Get best practice %s success.", practice_config.metadata.config_id)

        get_logger().info("===========QUANTIZE MODEL===========")
        self.quant_service.quantize(
            quant_config=practice_config.extract_quant_config(),
            model_adapter=model_adapter,
            save_path=save_path,
            device=device_type,
            device_indices=device_index,
        )
        get_logger().info("===========SUCCESS===========")
