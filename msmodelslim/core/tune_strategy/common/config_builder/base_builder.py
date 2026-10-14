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

from abc import ABC, abstractmethod
from typing import Any, List

from msmodelslim.core.practice import Metadata
from msmodelslim.core.quant_service.modelslim_v1.quant_config import ModelslimV1ServiceConfig
from msmodelslim.core.quant_service.modelslim_v1.save import AscendV1Config

from msmodelslim.core.tune_strategy.common.config_builder.quantization_config import (
    QuantizationConfig,
    TuningSearchSpace,
)


class QuantizationConfigBuilder(ABC):
    """
    量化配置建造者基类。

    负责根据输入（如结构配置、量化类型等）构建 QuantizationConfig 的各组件，
    并定义调优搜索空间 get_tuning_search_space()。子类按需实现各 build_* 方法，
    未实现的组件由基类提供默认或由调用方补齐。当前仅支持 modelslim_v1（spec 为 ModelslimV1ServiceConfig）。
    """

    DEFAULT_APIVERSION = "modelslim_v1"
    DEFAULT_DATASET = "mix_calib.jsonl"

    def build_metadata(self, **kwargs: Any) -> Metadata:
        """构建 metadata 组件。"""
        return Metadata(config_id="unknown", label={})

    @abstractmethod
    def build_spec_process(self, **kwargs: Any) -> List[Any]:
        """构建 spec.process（处理器配置列表）"""
        ...

    def build_spec_dataset(self, **kwargs: Any) -> str:
        """构建 spec.dataset。默认返回 DEFAULT_DATASET。"""
        return self.DEFAULT_DATASET

    def build_spec_save(self, **kwargs: Any) -> List[Any]:
        """构建 spec.save。默认返回 ascendv1_saver。"""
        return [AscendV1Config(type="ascendv1_saver", part_file_size=4)]

    def build(
        self,
        *,
        apiversion: str = DEFAULT_APIVERSION,
        metadata: Metadata | None = None,
        spec_process: List[Any] | None = None,
        spec_dataset: str | None = None,
        spec_save: List[Any] | None = None,
        **kwargs: Any,
    ) -> QuantizationConfig:
        """
        组装完整 QuantizationConfig（spec 为 ModelslimV1ServiceConfig）。
        若某组件未传入，则调用对应 build_* 方法生成（传入 **kwargs）。
        """
        if metadata is None:
            metadata = self.build_metadata(**kwargs)
        if spec_process is None:
            spec_process = self.build_spec_process(**kwargs)
        if spec_dataset is None:
            spec_dataset = self.build_spec_dataset(**kwargs)
        if spec_save is None:
            spec_save = self.build_spec_save(**kwargs)

        spec = ModelslimV1ServiceConfig(
            runner="auto",
            process=spec_process,
            dataset=spec_dataset,
            save=spec_save,
        )
        return QuantizationConfig(
            apiversion=apiversion,
            metadata=metadata,
            spec=spec,
        )

    def get_tuning_search_space(self, **kwargs: Any) -> TuningSearchSpace:
        """
        限定调优的搜索范围

        返回 TuningSearchSpace，策略仅在此范围内搜索。例如摸高算法使用
        anti_outlier_strategies、后续可扩展 allowed_datasets、max_rollback_layers、
        anti_outlier_param_bounds 等。子类按需覆盖并设置相应字段。kwargs 由调用方传入，子类按需使用。
        """
        return TuningSearchSpace()
