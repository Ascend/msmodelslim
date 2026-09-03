#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
ModelslimConvertQuantService：将 convert 流水线注册为 quant_service 插件。

通过 ``msmodelslim quant --config_path <yaml>`` 使用，YAML 中 ``apiversion: modelslim_convert``。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.interface import BaseQuantConfig, IQuantService, QuantServiceConfig
from msmodelslim.model import IModel
from msmodelslim.utils.logging import get_logger, logger_setter
from .config_mapper import spec_to_convert_config
from .factory import create_convert_application
from .quant_config import ModelslimConvertQuantConfig

logger = get_logger()


class ModelslimConvertQuantServiceConfig(QuantServiceConfig):
    """modelslim_convert 量化服务配置，用于插件选择与 QuantService 初始化。"""

    apiversion: Literal["modelslim_convert"] = "modelslim_convert"


@logger_setter(prefix="msmodelslim.core.quant_service.modelslim_convert")
class ModelslimConvertQuantService(IQuantService):
    """离线权重转换服务：data-free，不经 runner 校准。"""

    backend_name: str = "modelslim_convert"

    def __init__(
        self,
        quant_service_config: ModelslimConvertQuantServiceConfig,
        **kwargs,
    ) -> None:
        self.quant_service_config = quant_service_config

    def quantize(
        self,
        quant_config: BaseQuantConfig,
        model_adapter: IModel,
        save_path: Optional[Path] = None,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ) -> None:
        if save_path is None:
            raise ValueError("modelslim_convert requires save_path")

        convert_quant_config = ModelslimConvertQuantConfig.from_base(quant_config)
        convert_config = spec_to_convert_config(
            spec=convert_quant_config.spec,
            model_path=str(model_adapter.model_path),
            save_path=str(save_path),
            model_family=getattr(model_adapter, "model_type", None),
            device_indices=device_indices,
        )

        logger.info(
            "==========CONVERT: model_path=%s save_path=%s device_indices=%s==========",
            convert_config.model_path,
            convert_config.save_path,
            device_indices,
        )
        # 数据安全：禁止 save_path 指向源模型目录或其子目录，避免清理逻辑误删源权重。
        model_path_res = Path(model_adapter.model_path).resolve()
        save_path_res = Path(save_path).resolve()
        if save_path_res == model_path_res or model_path_res in save_path_res.parents:
            raise ValueError(
                f"save_path ({save_path}) must not be the model_path or inside it "
                f"({model_adapter.model_path}); refusing to overwrite source weights"
            )
        # save_path 已禁止落在源模型目录内，此处清掉旧分片以免总数变化后残留同名/旧序号文件。
        # 先删后转，转换失败会留下空目录/旧输出，需由用户在全新 save_path 上重跑。
        if save_path.exists():
            for item in save_path.iterdir():
                if item.is_file() and item.suffix == ".safetensors":
                    item.unlink()
            logger.info("Cleared previous convert safetensors under save_path: %s", save_path)
        app = create_convert_application()
        app.run(convert_config)
        logger.info("==========CONVERT: END==========")


def get_plugin():
    """获取 modelslim_convert 量化服务插件。"""
    return ModelslimConvertQuantServiceConfig, ModelslimConvertQuantService
