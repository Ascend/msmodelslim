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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import ConfigDict, Field

from msmodelslim.app.train.proxy.task import TrainTaskConfig
from msmodelslim.app.train.qat.algorithm_train_hook import AlgorithmTrainHook
from msmodelslim.core.const import DeviceType
from msmodelslim.core.convert.config import (
    ConvertConfig,
    ConvertRule,
    ModuleRule,
    ParallelConfig,
)
from msmodelslim.core.convert.protocol import IIRTransformProcessor
from msmodelslim.core.convert.types import IRKind
from msmodelslim.core.train.protocol import ITrainHook
from msmodelslim.core.train.task import (
    TRAIN_ALGORITHM_PLUGIN_GROUP,
    CoreTrainTaskConfig,
    DatasetConfig,
    NestedTrainConfig,
    TrainAlgorithmConfig,
    TrainHyperparametersConfig,
    CheckpointingConfig,
)
from msmodelslim.format.checkpoint_format.constants import STRUCTURE_FILE
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.utils.plugin import TypedFactory
from msmodelslim.utils.plugin.plugin_utils import load_plugin_component_class
from msmodelslim.utils.security import json_safe_load


__all__ = [
    "CheckpointingConfig",
    "DatasetConfig",
    "NestedTrainConfig",
    "QATTask",
    "QATTaskConfig",
    "TrainHyperparametersConfig",
]

_PARETOQ_TENSOR_MAP = {
    "weight": "{module}.weight",
    "bias": "{module}.bias",
    "weight_clip_val": "{module}.weight_clip_val",
}

_DEPLOY_ACT_TO_IR = {
    "a8": IRKind.W4A8_DYNAMIC,
    "a4": IRKind.W4A4_DYNAMIC,
}

_SAVE_TYPE_TO_DST = {
    "ascend_v1": "ascendv1",
    "ascendv1": "ascendv1",
    "ascendv1_saver": "ascendv1",
    "huggingface": "huggingface",
    "hf": "huggingface",
    "compressed_tensors": "compressed_tensors",
}


class QATTaskConfig(TrainTaskConfig, CoreTrainTaskConfig):
    """
    ``type: qat`` 任务配置。

    - 继承 ``TrainTaskConfig``：proxy 按 ``type`` 插件分发
    - 继承 ``CoreTrainTaskConfig``：通用训练字段（setup 侧校验后原样下发 worker）
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["qat"] = Field(
        default="qat",
        description="训练范式，固定为 qat。",
    )


class QATTask:
    """
    解析 YAML 中的算法配置，组装 ``ITrainHook``，并注册 convert ``IIRTransformProcessor``。

    协议适配类来自 ``qat.algorithm_train_hook``；convert 边由算法侧 ``get_convert_processors`` 提供，
    本 Task 仅查询并注册到路由图（开闭原则 / IoC）。
    """

    def __init__(self, config: QATTaskConfig):
        self.config = config
        self.hooks: List[ITrainHook] = []
        self.convert_processors: List[IIRTransformProcessor] = []
        self._algo_factory = TypedFactory[Any](config_base_class=TrainAlgorithmConfig)

    @classmethod
    def build_from(cls, config: QATTaskConfig) -> "QATTask":
        task = cls(config)
        task.assemble_hooks()
        task.register_convert()
        return task

    def assemble_hooks(self) -> None:
        """解析 algorithm[]，组装为 ``ITrainHook`` 列表。"""
        if len(self.config.algorithm) != 1:
            raise SchemaValidateError(
                f"MVP supports exactly one algorithm, got {len(self.config.algorithm)}",
                action="Please set algorithm to a single paretoq block.",
            )
        self.hooks = [
            AlgorithmTrainHook(self.config.algorithm[0], self._algo_factory),
        ]

    def register_convert(self) -> List[IIRTransformProcessor]:
        """从算法插件查询 convert 边，并注册到默认 ``IRRouter``。"""
        from msmodelslim.app.train.qat.qat_linear_algorithm import QATLinearAlgorithm
        from msmodelslim.core.convert.router import IRRouter

        processors: List[IIRTransformProcessor] = []
        for algo_cfg in self.config.algorithm:
            impl_cls = load_plugin_component_class(
                TRAIN_ALGORITHM_PLUGIN_GROUP,
                algo_cfg.type,
            )
            if not issubclass(impl_cls, QATLinearAlgorithm):
                raise SchemaValidateError(
                    f"Algorithm plugin '{algo_cfg.type}' must implement QATLinearAlgorithm (got {impl_cls!r})",
                    action="Please register a QATLinearAlgorithm plugin for this algorithm type.",
                )
            processors.extend(impl_cls.get_convert_processors())

        router = IRRouter.default()
        for proc in processors:
            router.register_processor(proc)
        self.convert_processors = processors
        return processors

    def build_convert_config(
        self,
        checkpoint_path: Union[str, Path],
        deploy_save_path: Union[str, Path],
        *,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ) -> ConvertConfig:
        """从 ``qat_checkpoint.json`` 组装 modelslim_convert 用的 ``ConvertConfig``。"""
        ckpt = Path(checkpoint_path)
        structure = ckpt / STRUCTURE_FILE
        payload = json_safe_load(str(structure))
        if not isinstance(payload, dict):
            raise SchemaValidateError(
                f"{STRUCTURE_FILE} root must be a mapping",
                action="Please re-export the QAT checkpoint with CheckpointSaverFormat.",
            )
        modules: Dict[str, Any] = payload.get("modules") or {}
        if not modules:
            raise SchemaValidateError(
                f"No ParetoQ modules found in {structure}",
                action="Please ensure training wrote qat_checkpoint.json with a non-empty modules map.",
            )

        module_rules: List[ModuleRule] = []
        convert_rules: List[ConvertRule] = []
        for module_name, meta in modules.items():
            if not isinstance(meta, dict):
                raise SchemaValidateError(
                    f"Invalid module meta for {module_name!r}: expected mapping",
                    action="Please check qat_checkpoint.json modules entries.",
                )
            deploy_act = str(meta.get("deploy_act") or "a8")
            target_ir = _DEPLOY_ACT_TO_IR.get(deploy_act)
            if target_ir is None:
                raise SchemaValidateError(
                    f"Unsupported deploy_act={deploy_act!r} for module {module_name!r}",
                    action="Please use deploy_act a8 or a4.",
                )
            defaults = {
                "w_bits": int(meta.get("w_bits", 4)),
                "deploy_act": deploy_act,
                "weight_layerwise": bool(meta.get("weight_layerwise", False)),
            }
            module_rules.append(
                ModuleRule(
                    match=module_name,
                    module_kind="linear",
                    source_format="qat_paretoq",
                    source_ir=IRKind.PARETOQ_FLOAT,
                    tensor_map=dict(_PARETOQ_TENSOR_MAP),
                    defaults=defaults,
                )
            )
            convert_rules.append(
                ConvertRule(
                    match=module_name,
                    target_ir=target_ir,
                    route=[IRKind.PARETOQ_FLOAT, target_ir],
                )
            )

        if device == DeviceType.CPU:
            effective_indices: List[int] = []
        else:
            effective_indices = list(device_indices) if device_indices else [0]

        return ConvertConfig(
            model_path=str(ckpt),
            save_path=str(deploy_save_path),
            dst_format=self._resolve_dst_format(),
            part_file_size=self._resolve_part_file_size(),
            module_rules=module_rules,
            convert_rules=convert_rules,
            parallel=ParallelConfig(device_indices=effective_indices),
        )

    def _resolve_dst_format(self) -> str:
        if not self.config.save:
            return "ascendv1"
        first = self.config.save[0]
        type_name = str(getattr(first, "type", None) or "ascendv1").lower()
        return _SAVE_TYPE_TO_DST.get(type_name, type_name.replace("_saver", ""))

    def _resolve_part_file_size(self) -> int:
        if not self.config.save:
            return 4
        first = self.config.save[0]
        size = getattr(first, "part_file_size", None)
        return int(size) if size is not None else 4
