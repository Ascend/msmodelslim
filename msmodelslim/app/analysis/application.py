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

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from msmodelslim.core.analysis_service import IAnalysisService, AnalysisConfig, AnalysisScope
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.multimodal_sd_v1.legacy_pipeline_interface import (
    LegacyMultimodalPipelineInterface,
)
from msmodelslim.core.quant_service.multimodal_sd_v1.pipeline_interface import MultimodalPipelineInterface
from msmodelslim.model import IModelFactory
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.exception_decorator import exception_catcher
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.validation.conversion import convert_to_readable_dir
from msmodelslim.utils.validation.value import validate_str_length
from .result_displayer_infra import AnalysisResultDisplayerInfra


def _validate_device_indices(device_indices: Optional[List[int]], device: DeviceType) -> Optional[List[int]]:
    """Normalize/validate multi-device indices for analysis (same rules as quant)."""
    if device_indices is None:
        return None
    if not isinstance(device_indices, list):
        raise SchemaValidateError(f"device_indices must be a list, but got {type(device_indices)}")
    if any(not isinstance(idx, int) for idx in device_indices):
        raise SchemaValidateError("device_indices must contain integers only")
    if any(idx < 0 for idx in device_indices):
        raise SchemaValidateError(f"Device indices must be non-negative integers, but got: {device_indices}")
    if len(device_indices) != len(set(device_indices)):
        raise SchemaValidateError(f"Device indices must be unique, but got: {device_indices}")
    if device == DeviceType.CPU and len(device_indices) > 1:
        raise SchemaValidateError("CPU does not support multi-device analysis. Use NPU with --device npu:0,1,...")
    return device_indices


class AnalysisMetrics(str, Enum):
    """Enumeration of valid analysis metrics"""

    STD = 'std'
    QUANTILE = 'quantile'
    KURTOSIS = 'kurtosis'
    ATTENTION_MSE = 'mse'
    MSE_LAYER_WISE = 'mse_layer_wise'
    MSE_MODEL_WISE = 'mse_model_wise'
    RA_COMPRESS = 'ra_compress'


@dataclass
class LinearArgs:
    pattern: List[str] = field(default_factory=lambda: ['*'])
    metrics: AnalysisMetrics = AnalysisMetrics.KURTOSIS


@dataclass
class AttnArgs:
    metrics: AnalysisMetrics = AnalysisMetrics.ATTENTION_MSE


@dataclass
class AttnHeadArgs:
    metrics: AnalysisMetrics = AnalysisMetrics.RA_COMPRESS


@dataclass
class LayerArgs:
    quant_modules: List[str] = field(default_factory=lambda: ['*'])
    metrics: AnalysisMetrics = AnalysisMetrics.MSE_LAYER_WISE


ScopeAnalysisArgs = Union[LinearArgs, AttnArgs, AttnHeadArgs, LayerArgs]


def _reject_multimodal_generation_adapter(model_adapter: PipelineInterface) -> None:
    """敏感层分析不支持多模态生成模型（SD/DiT 等）；VLM 理解模型不在此接口体系内。"""
    if isinstance(model_adapter, (MultimodalPipelineInterface, LegacyMultimodalPipelineInterface)):
        raise UnsupportedError(
            f'Model adapter {model_adapter.__class__.__name__} is a multimodal generation model; '
            'sensitive layer analysis is not supported.',
        )


def _validate_calib_dataset_name(calib_dataset: str) -> None:
    """Accept LLM json/jsonl files and VLM directory-style dataset names (e.g. calibImages)."""
    path = Path(calib_dataset)
    suffix = path.suffix.lower()
    if suffix in {'.json', '.jsonl'}:
        return
    if path.is_dir() or (path.suffix == '' and not path.is_file()):
        return
    raise SchemaValidateError(
        f'Unsupported calib_dataset format: {calib_dataset}. '
        'Use .json/.jsonl for LLM text calib, or a directory name/path '
        '(e.g. calibImages under lab_calib) for VLM multimodal calib.',
        action='Please provide a .json/.jsonl file or a VLM calib directory name',
    )


def _analysis_config_from_scope_args(scope_args: ScopeAnalysisArgs, calib_dataset: str) -> AnalysisConfig:
    """由 CLI/App 的 scope_args 构造带区分字段的 ``AnalysisConfig``。"""
    if isinstance(scope_args, LinearArgs):
        pat = scope_args.pattern
        if not isinstance(pat, list):
            raise SchemaValidateError(f'pattern must be a list, got {type(pat)}')
        return AnalysisConfig(
            scope=AnalysisScope.LINEAR,
            metrics=scope_args.metrics.value,
            calib_dataset=calib_dataset,
            linear_pattern=list(pat),
        )
    if isinstance(scope_args, AttnHeadArgs):
        return AnalysisConfig(
            scope=AnalysisScope.ATTN_HEAD,
            metrics=scope_args.metrics.value,
            calib_dataset=calib_dataset,
        )
    if isinstance(scope_args, AttnArgs):
        return AnalysisConfig(
            scope=AnalysisScope.ATTN,
            metrics=scope_args.metrics.value,
            calib_dataset=calib_dataset,
        )
    if isinstance(scope_args, LayerArgs):
        qm = scope_args.quant_modules
        if not isinstance(qm, list):
            raise SchemaValidateError(f'quant_modules must be a list, got {type(qm)}')
        return AnalysisConfig(
            scope=AnalysisScope.LAYER,
            metrics=scope_args.metrics.value,
            calib_dataset=calib_dataset,
            quant_modules=list(qm),
        )
    raise SchemaValidateError(
        f'scope_args must be LinearArgs, AttnArgs, AttnHeadArgs, or LayerArgs, got {type(scope_args)!r}',
    )


@logger_setter('msmodelslim.app.analysis.application')
class LayerAnalysisApplication:
    """Application for analyzing model layer sensitivity"""

    def __init__(
        self,
        analysis_service: IAnalysisService,
        model_factory: IModelFactory,
        result_manager: AnalysisResultDisplayerInfra,
    ):
        self.analysis_service = analysis_service
        self.model_factory = model_factory
        self.result_manager = result_manager

    @exception_catcher
    def analyze(
        self,
        model_type: str,
        model_path: str,
        scope_args: ScopeAnalysisArgs,
        device: DeviceType = DeviceType.NPU,
        calib_dataset: str = 'mix_calib.jsonl',
        topk: int = 15,
        trust_remote_code: bool = False,
        save_path: Optional[str] = None,
        device_indices: Optional[List[int]] = None,
    ):
        """
        Run layer analysis on a model

        Args:
            model_type: Type of the model (e.g., 'Qwen2.5-7B-Instruct')
            model_path: Path to the model
            scope_args: Scope-specific options: ``LinearArgs``, ``AttnArgs``, or ``LayerArgs``.
                Each scope uses its own default ``AnalysisMetrics`` when omitted on the dataclass.
            device: Device to run analysis on
            calib_dataset: Dataset path for calibration
            topk: Number of top layers to output for disable_names
            trust_remote_code: Whether to trust remote code
            device_indices: Optional NPU indices; length > 1 enables DP multi-device analysis
        """
        # Validate string inputs with length checks
        str_params = [("model_type", model_type), ("model_path", model_path), ("calib_dataset", calib_dataset)]
        for param_name, value in str_params:
            if not isinstance(value, str):
                raise SchemaValidateError(f"{param_name} must be a string, but got {type(value)}")
            validate_str_length(input_str=value, str_name=param_name)

        analysis_config = _analysis_config_from_scope_args(scope_args, calib_dataset)
        model_path = convert_to_readable_dir(model_path)
        if not isinstance(model_path, Path):
            raise SchemaValidateError(f"model_path must be a Path, but got {type(model_path)}")
        if not isinstance(device, DeviceType):
            raise SchemaValidateError("device must be a DeviceType")
        if not isinstance(calib_dataset, str):
            raise SchemaValidateError(f"calib_dataset must be a string, but got {type(calib_dataset)}")
        # LLM: .json/.jsonl text files; VLM: directory short names (e.g. calibImages) or paths without suffix
        _validate_calib_dataset_name(calib_dataset)
        device_indices = _validate_device_indices(device_indices, device)
        if not isinstance(topk, int) or topk <= 0:
            raise SchemaValidateError(f"topk must be a integer greater than 0, but got {topk}")
        if not isinstance(trust_remote_code, bool):
            raise SchemaValidateError("trust_remote_code must be a bool")

        log = get_logger()
        log.info('Layer analysis with following parameters:')
        log.info('model_type: %s', model_type)
        log.info('model_path: %s', model_path)
        log.info('analyze_scope: %s', analysis_config.scope.value)
        if analysis_config.scope == AnalysisScope.LINEAR:
            log.info('linear_pattern: %s', analysis_config.linear_pattern)
        elif analysis_config.scope == AnalysisScope.LAYER:
            log.info('quant_modules: %s', analysis_config.quant_modules)
        elif analysis_config.scope == AnalysisScope.ATTN_HEAD:
            log.info('attn_head: attention head analysis (ra_compress)')
        else:
            log.info('attn: all attention modules')
        log.info('metrics: %s', analysis_config.metrics)
        log.info('device: %s', device)
        log.info('device_indices: %s', device_indices)
        log.info('calib_dataset: %s', calib_dataset)
        log.info('topk: %s', topk)
        log.info('trust_remote_code: %s', trust_remote_code)

        return self._analyze(
            model_type,
            model_path,
            analysis_config,
            device,
            topk,
            trust_remote_code,
            save_path=save_path,
            device_indices=device_indices,
        )

    def _analyze(
        self,
        model_type: str,
        model_path: Path,
        analysis_config: AnalysisConfig,
        device: DeviceType,
        topk: int,
        trust_remote_code: bool,
        save_path: Optional[str] = None,
        device_indices: Optional[List[int]] = None,
    ):
        """Internal analysis implementation"""
        # Run analysis
        get_logger().info("===========RUN ANALYSIS===========")

        get_logger().info("===========ANALYSE MODEL===========")
        model_adapter = self.model_factory.create(model_type, model_path, trust_remote_code)
        if not isinstance(model_adapter, PipelineInterface):
            raise UnsupportedError(
                f'Model adapter {model_adapter.__class__.__name__} does NOT support analyze',
                action='Please implement PipelineInterface for model analyzing',
            )
        _reject_multimodal_generation_adapter(model_adapter)
        get_logger().info("Using model adapter %s.", model_adapter.__class__.__name__)

        result = self.analysis_service.analyze(
            device=device,
            model_adapter=model_adapter,
            analysis_config=analysis_config,
            device_indices=device_indices,
        )

        # display results using service-specific formatter (only when result is not None)
        if result is not None:
            self.result_manager.display_result(
                result,
                topk,
                analysis_config.scope,
                save_path=save_path,
                model_type=model_type,
            )

        get_logger().info("===========ANALYSIS COMPLETE===========")
        return result
