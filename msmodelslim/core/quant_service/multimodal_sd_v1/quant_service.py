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

# pylint: disable=logging-fstring-interpolation

import copy
import functools
import os
from pathlib import Path
from typing import Optional, List, Literal


from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.dataset_loader_infra import DatasetLoaderInfra
from msmodelslim.core.quant_service import KeyInfoPersistenceInfra
from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner
from msmodelslim.model import IModel
from msmodelslim.utils.cache import load_cached_data_for_models, to_device
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.core.context import IContextFactory, ContextManager
from .legacy_pipeline_interface import LegacyMultimodalPipelineInterface
from .pipeline_interface import MultimodalPipelineInterface
from .quant_config import (
    MultimodalSDModelslimV1QuantConfig,
    MultiExpertQuantConfig,
    MultimodalSDPipelineAdapter,
    validate_inference_config,
)
from ..interface import BaseQuantConfig, QuantServiceConfig, IQuantService


def _require_expert_calib_in_dict(
    expert_name: str,
    calib_data: dict,
    models: dict,
) -> None:
    if expert_name in calib_data:
        return
    expected = ", ".join(sorted(models.keys()))
    available = ", ".join(sorted(calib_data.keys())) if calib_data else "(empty)"
    raise SchemaValidateError(
        f"calib data missing for expert '{expert_name}'. "
        f"init_model experts: [{expected}]; calib_data keys: [{available}]. "
        f"Check multimodal_sd_config.dump_config (enable_dump, dump_data_dir), "
        f"calibration pth paths (calib_data_<task>_<expert>.pth), "
        f"and that prepare_calib_data or run_calib_inference completed for every expert.",
    )


class MultimodalSDModelslimV1QuantServiceConfig(QuantServiceConfig):
    """multimodal_sd_modelslim_v1 量化服务配置，用于插件选择与 QuantService 初始化。"""

    apiversion: Literal["multimodal_sd_modelslim_v1"] = "multimodal_sd_modelslim_v1"


@logger_setter(
    prefix='msmodelslim.core.quant_service.multimodal_sd_v1'
)  # 4-level: msmodelslim.core.quant_service.multimodal_sd_v1
class MultimodalSDModelslimV1QuantService(IQuantService):
    backend_name: str = "multimodal_sd_modelslim_v1"

    def __init__(
        self,
        quant_service_config: MultimodalSDModelslimV1QuantServiceConfig,
        dataset_loader: DatasetLoaderInfra,
        context_factory: IContextFactory,
        debug_info_persistence: Optional[KeyInfoPersistenceInfra] = None,
        **kwargs,
    ):
        self.quant_service_config = quant_service_config
        self.dataset_loader = dataset_loader
        self.context_factory = context_factory
        self.debug_info_persistence = debug_info_persistence

    def quantize(
        self,
        quant_config: BaseQuantConfig,
        model_adapter: IModel,
        save_path: Optional[Path] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ) -> None:
        if not isinstance(quant_config, BaseQuantConfig):
            raise SchemaValidateError("task must be a BaseTask")
        if not isinstance(model_adapter, (LegacyMultimodalPipelineInterface, MultimodalPipelineInterface)):
            raise SchemaValidateError(
                "model must implement LegacyMultimodalPipelineInterface or MultimodalPipelineInterface",
            )
        if save_path is not None and not isinstance(save_path, Path):
            raise SchemaValidateError("save_path must be a Path or None")
        if not isinstance(device, DeviceType):
            raise SchemaValidateError("device must be a DeviceType")

        if device_indices is not None:
            get_logger().warning(
                "Specifying device indices is not supported in %s quant_service. Device indices will be ignored.",
                self.backend_name,
            )

        return self.quant_process(
            MultimodalSDModelslimV1QuantConfig.from_base(quant_config),
            model_adapter,
            save_path,
            device,
        )

    def quant_process(
        self,
        quant_config: MultimodalSDModelslimV1QuantConfig,
        model_adapter: MultimodalSDPipelineAdapter,
        save_path: Optional[Path],
        device: DeviceType = DeviceType.NPU,
    ):
        if isinstance(model_adapter, LegacyMultimodalPipelineInterface):
            return self._quant_process_legacy(quant_config, model_adapter, save_path, device)
        return self._quant_process(quant_config, model_adapter, save_path, device)

    def quantize_multi_expert_models(self, config: MultiExpertQuantConfig):
        model_adapter = config.model_adapter
        if isinstance(model_adapter, LegacyMultimodalPipelineInterface):
            return self._quantize_multi_expert_models_legacy(config)
        return self._quantize_multi_expert_models(config)

    def _quantize_multi_expert_models_legacy(self, config: MultiExpertQuantConfig):
        """主仓兼容：通过 model_adapter.transformer 切换专家。"""
        model_adapter = config.model_adapter
        models = config.models
        calib_data = config.calib_data
        quant_config = config.quant_config
        save_path = config.save_path
        device = config.device

        original_transformer = model_adapter.transformer

        for expert_name, expert_model in models.items():
            get_logger().info(f"========== Quantizing {expert_name} ==========")

            _require_expert_calib_in_dict(expert_name, calib_data, models)

            model_adapter.transformer = expert_model

            if expert_name != '':
                expert_save_path = save_path.joinpath(f"{expert_name}")
                expert_save_path.mkdir(parents=True, exist_ok=True)
            else:
                expert_save_path = save_path

            final_process_cfg = copy.copy(quant_config.spec.process)

            if expert_save_path is not None:
                get_logger().warning("========== QUANTIZATION: Prepare Save Path ==========")
                for save_cfg in quant_config.spec.save:
                    save_cfg.set_save_directory(expert_save_path)
                final_process_cfg += quant_config.spec.save
                get_logger().warning(f"prepare Persistence to {expert_save_path} success")

            if quant_config.spec.runner != "layer_wise":
                get_logger().warning("runner for multimodal_sd_v1 is not layer_wise, will be converted to layer_wise.")

            runner = LayerWiseRunner(adapter=model_adapter)
            ctx = self.context_factory.create()
            with ContextManager(ctx=ctx):
                for process_cfg in final_process_cfg:
                    runner.add_processor(processor_cfg=process_cfg)

                try:
                    model_adapter.apply_quantization(
                        functools.partial(
                            runner.run,
                            calib_data=calib_data[expert_name],
                            device=device,
                            model=expert_model,
                        )
                    )
                    get_logger().info(f"========== {expert_name} quantized, save to {expert_save_path} ==========")
                except Exception as e:
                    get_logger().error(f"========== {expert_name} quantization failed: {str(e)} ==========")
                    raise RuntimeError(f"========== {expert_name} quantization failed: {str(e)} ==========") from e

            if self.debug_info_persistence is not None:
                get_logger().info(f"==========SAVE DEBUG INFO for {expert_name}==========")
                try:
                    self.debug_info_persistence.save_from_context(ctx=ctx)
                except Exception as e:
                    get_logger().warning(f"Failed to save debug info: {e}")

        model_adapter.transformer = original_transformer

    def _quantize_multi_expert_models(self, config: MultiExpertQuantConfig):
        """按 expert 使用子适配器与 quantization_context。"""
        model_adapter = config.model_adapter
        models = config.models
        calib_data = config.calib_data
        quant_config = config.quant_config
        save_path = config.save_path
        device = config.device

        for expert_name, expert_model in models.items():
            get_logger().info(f"========== Quantizing {expert_name} ==========")

            _require_expert_calib_in_dict(expert_name, calib_data, models)

            expert_adapter = model_adapter.get_expert_adapter(expert_name)

            if expert_name != '':
                expert_save_path = save_path.joinpath(f"{expert_name}")
                expert_save_path.mkdir(parents=True, exist_ok=True)
            else:
                expert_save_path = save_path

            final_process_cfg = copy.copy(quant_config.spec.process)

            if expert_save_path is not None:
                get_logger().warning("========== QUANTIZATION: Prepare Save Path ==========")
                for save_cfg in quant_config.spec.save:
                    save_cfg.set_save_directory(expert_save_path)
                final_process_cfg += quant_config.spec.save
                get_logger().warning(f"prepare Persistence to {expert_save_path} success")

            if quant_config.spec.runner != "layer_wise":
                get_logger().warning("runner for multimodal_sd_v1 is not layer_wise, will be converted to layer_wise.")

            runner = LayerWiseRunner(adapter=expert_adapter)
            ctx = self.context_factory.create()
            with ContextManager(ctx=ctx):
                for process_cfg in final_process_cfg:
                    runner.add_processor(processor_cfg=process_cfg)

                try:
                    with expert_adapter.quantization_context():
                        runner.run(
                            calib_data=calib_data[expert_name],
                            device=device,
                            model=expert_model,
                        )
                    get_logger().info(f"========== {expert_name} quantized, save to {expert_save_path} ==========")
                except Exception as e:
                    get_logger().error(f"========== {expert_name} quantization failed: {str(e)} ==========")
                    raise RuntimeError(f"========== {expert_name} quantization failed: {str(e)} ==========") from e

            if self.debug_info_persistence is not None:
                get_logger().info(f"==========SAVE DEBUG INFO for {expert_name}==========")
                try:
                    self.debug_info_persistence.save_from_context(ctx=ctx)
                except Exception as e:
                    get_logger().warning(f"Failed to save debug info: {e}")

    def _quant_process_legacy(
        self,
        quant_config: MultimodalSDModelslimV1QuantConfig,
        model_adapter: LegacyMultimodalPipelineInterface,
        save_path: Optional[Path],
        device: DeviceType = DeviceType.NPU,
    ):
        """主仓兼容编排：set_model_args → load_pipeline → run_calib_inference。"""
        sd_config = quant_config.spec.multimodal_sd_config
        inference_raw = sd_config.resolve_inference_raw()
        if not inference_raw:
            raise SchemaValidateError(
                "Legacy pipeline requires inference_config or model_config in multimodal_sd_config.",
            )
        model_adapter.set_model_args(inference_raw)
        model_adapter.load_pipeline()

        get_logger().info("==========QUANTIZATION: Prepare Dataset==========")

        models = model_adapter.init_model(device)

        dump_config = quant_config.spec.multimodal_sd_config.dump_config
        if dump_config.enable_dump:
            config_dump_data_dir = dump_config.dump_data_dir
            if config_dump_data_dir:
                base_dir = config_dump_data_dir
            else:
                base_dir = save_path

            pth_file_path_list = {}
            for expert_name, _ in models.items():
                pth_file_path_list[expert_name] = os.path.join(
                    base_dir,
                    f"calib_data_{model_adapter.model_args.task_config}_{expert_name}.pth",
                )

            calib_data = load_cached_data_for_models(
                pth_file_path_list=pth_file_path_list,
                generate_func=model_adapter.run_calib_inference,
                models=models,
                dump_config=dump_config,
            )

            get_logger().info(f"prepare calib_data from {base_dir} success")
        else:
            tips = (
                "With enable_dump=False in the current config, calibration data will not be loaded/dumped. "
                "Please confirm whether your use case requires dump data: pure dynamic quantization does not need "
                "calibration data; static quantization or outlier suppression requires it. "
                "If you don't need it, enter y to continue."
            )
            user_input = input(tips + " (Enter y to continue, otherwise it will exit): ").strip().lower()[:3]
            if user_input != 'y':
                raise UnsupportedError(
                    tips,
                    action="To dump calibration data, set multimodal_sd_config.dump_config.enable_dump: True in your config.",
                )
            get_logger().info("enable_dump=False, skipping calibration data load/dump")
            calib_data = {expert_name: None for expert_name in models}

        calib_data = to_device(calib_data, device.value)
        get_logger().info("==========QUANTIZATION: Run Quantization==========")

        config = MultiExpertQuantConfig(
            model_adapter=model_adapter,
            models=models,
            calib_data=calib_data,
            quant_config=quant_config,
            save_path=save_path,
            device=device,
        )

        self.quantize_multi_expert_models(config)

        get_logger().info("==========QUANTIZATION: END==========")

    def _quant_process(
        self,
        quant_config: MultimodalSDModelslimV1QuantConfig,
        model_adapter: MultimodalPipelineInterface,
        save_path: Optional[Path],
        device: DeviceType = DeviceType.NPU,
    ):
        """inference_config + dataset + inference_dump_calib_data。"""
        sd_config = quant_config.spec.multimodal_sd_config
        inference_config = validate_inference_config(model_adapter, sd_config)
        model_adapter.configure_runtime(inference_config)

        models = model_adapter.init_model(device)

        get_logger().info("==========QUANTIZATION: Prepare Dataset==========")
        dataset_name = quant_config.spec.dataset
        raw_dataset = self.dataset_loader.get_dataset_by_name(dataset_name)
        calib_dataset = model_adapter.handle_dataset(raw_dataset)
        dump_config = quant_config.spec.multimodal_sd_config.dump_config
        calib_data = model_adapter.prepare_calib_data(
            models=models,
            dump_config=dump_config,
            save_path=save_path,
            dataset=calib_dataset,
            inference_config=inference_config,
        )

        calib_data = to_device(calib_data, device.value)
        get_logger().info("==========QUANTIZATION: Run Quantization==========")

        config = MultiExpertQuantConfig(
            model_adapter=model_adapter,
            models=models,
            calib_data=calib_data,
            quant_config=quant_config,
            save_path=save_path,
            device=device,
        )

        self.quantize_multi_expert_models(config)

        get_logger().info("==========QUANTIZATION: END==========")


def get_plugin():
    """
    获取 multimodal_sd_modelslim_v1 量化服务插件（返回配置类与组件类，由框架完成注册）。
    Returns:
        (MultimodalSDModelslimV1QuantServiceConfig, MultimodalSDModelslimV1QuantService) 元组
    """
    return MultimodalSDModelslimV1QuantServiceConfig, MultimodalSDModelslimV1QuantService
