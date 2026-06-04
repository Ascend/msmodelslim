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

from abc import abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Type

from pydantic import BaseModel

from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.utils.exception import ToDoError


class MultimodalPipelineInterface(PipelineInterface):
    """
    Interface for the multimodal pipeline inference.
    Multimodal has non transformer part, so we need to handle the non transformer part.

    调用顺序（quant_process）::
        inference_config = validate_inference_config(adapter, sd_config)  # quant_service 内
        model_adapter.configure_runtime(inference_config)
        models = model_adapter.init_model(device)
        calib_data = model_adapter.prepare_calib_data(...)
        with model_adapter.quantization_context():
            ...
    """

    @abstractmethod
    def get_inference_config_class(self) -> Type[BaseModel]:
        """返回本适配器对应的 InferenceConfig Pydantic 配置类。"""
        raise ToDoError(
            "This model does not support get_inference_config_class.",
            action="Please declare an InferenceConfig class and implement get_inference_config_class.",
        )

    @abstractmethod
    def configure_runtime(self, inference_config: Any) -> None:
        """
        将已通过校验的 inference_config 落到适配器运行态（model_args 等）。
        不执行 generate / dump；pipeline 构造在 init_model 中完成。
        """
        raise ToDoError(
            "This model does not support configure_runtime.",
            action="Please implement configure_runtime for your model.",
        )

    @abstractmethod
    def inference_dump_calib_data(self, dataset: Any = None, inference_config: Any = None):
        raise ToDoError(
            "This model does not support inference_dump_calib_data.",
            action="Please implement inference_dump_calib_data for your model.",
        )

    @abstractmethod
    def prepare_calib_data(
        self,
        models: Dict[str, Any],
        dump_config: Any,
        save_path: Path,
        dataset: Any,
        inference_config: Any,
    ) -> Dict[str, Any]:
        """
        由模型适配器自定义校准数据 dump/cache 机制（例如 pth/safetensors 等）。
        返回按 expert_name 对齐的 calib_data 字典，供量化阶段直接使用。
        """
        raise ToDoError(
            "This model does not support prepare_calib_data.",
            action="Please implement prepare_calib_data for your model.",
        )

    @abstractmethod
    def quantization_context(self) -> AbstractContextManager:
        """
        量化运行上下文（autocast、no_grad、no_sync、设备布置等）。

        当前专家模型与推理参数由量化服务在调用前写入 adapter（如 transformer、model_args），
        子类从 self 读取即可，无需额外入参。
        """
        raise ToDoError(
            "This model does not support quantization_context.",
            action="Please implement quantization_context for your model.",
        )

    def get_expert_adapter(self, expert_name: str) -> PipelineInterface:
        """
        获取某个 expert 对应的 runner adapter。
        默认返回 self（单 expert 或未拆分子适配器的模型）。
        """
        _ = expert_name
        return self
