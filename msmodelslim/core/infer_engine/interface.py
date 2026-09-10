# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.format.interface import IFormatLoader


class InferenceConfig(BaseModel):
    """Inference task config shared by the fake-quant inference engines."""

    max_new_tokens: int = Field(default=1, ge=1, description="Number of new tokens to generate")


class InferenceResult(BaseModel):
    """Inference outputs consumed by Application / CLI."""

    generated_token_ids: List[List[int]] = Field(default_factory=list)
    generated_texts: List[str] = Field(default_factory=list)


class FakeQuantInferenceInterface(ABC):
    """Adapter contract for AscendV1 fake-quant inference via ``FakeQuantInferenceEngine``.

    The engine drives native ``model.forward`` with per-layer weight hooks, so the adapter
    only has to build the empty-weights shell; layer-wise hydrate / move / offload and the
    multi-prefill loop are owned by the engine.

    Required:
      - ``build_meta_model``
      - ``handle_dataset``

    Optional:
      - ``tokenizer`` : used to decode generated ids; falls back to ``str(ids)`` when absent.
    """

    @abstractmethod
    def build_meta_model(self) -> nn.Module:
        """Build the empty-weights (meta) CausalLM skeleton for fake-quant.

        Parameters live on meta; non-persistent buffers (e.g. RoPE ``inv_freq``) must stay on
        CPU so the engine can snapshot them into ``RuntimeBufferStore`` and restore them after
        each decoder ``.to(meta)``.
        """

    @abstractmethod
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """Tokenize raw samples into model-ready inputs.

        Signature and return shape match ``PipelineInterface.handle_dataset``: convert the
        dataset into items passable to ``model(*data)`` / ``model(**data)`` (typically
        ``[input_ids, attention_mask]`` lists or processor dicts).
        """


class IInferenceEngine(ABC):
    """Protocol for fake-quant inference engines."""

    @abstractmethod
    def run(
        self,
        adapter: FakeQuantInferenceInterface,
        format_loader: IFormatLoader,
        inputs: List[Any],
        inference_config: InferenceConfig,
        device: DeviceType = DeviceType.NPU,
        device_indices: Optional[List[int]] = None,
    ) -> InferenceResult:
        """Run fake-quant inference for raw ``inputs`` and return the result.

        Tokenization happens inside ``_run_single`` via ``adapter.handle_dataset``, so
        multi-card workers tokenize after ``mp.spawn``.

        Args:
            adapter: Model adapter exposing ``build_meta_model``, ``handle_dataset``,
                and optional ``tokenizer``.
            format_loader: Resolved weight loader (AscendV1 today). Must be rebuilt per
                DP worker because safetensors handles are not picklable.
            inputs: Raw samples from ``--prompt_file`` (text / dataset items), not
                tokenized tensors.
            inference_config: Task config such as ``max_new_tokens``.
            device: Execution device for layer-wise load.
            device_indices: Physical device indices for multi-card sample DP.
        """
        raise NotImplementedError
