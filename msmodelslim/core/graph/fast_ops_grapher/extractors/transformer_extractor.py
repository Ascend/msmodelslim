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

from typing import Any, Tuple, Dict
import torch
from msmodelslim.core.graph.fast_ops_grapher.extractors.base_extractor import BaseExtractor


class TransformerExtractor(BaseExtractor):
    """TransformerExtractor 用于从 transformers 库的模型中提取计算图。"""

    def __init__(self, model, tokenizer):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer

    @staticmethod
    # pylint: disable=arguments-differ
    def create(model, tokenizer) -> 'TransformerExtractor':
        return TransformerExtractor(model, tokenizer)

    @property
    def target_module(self):
        return self._model

    @property
    def dummy_inputs(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        inputs = self._tokenizer(".", return_tensors="pt")
        target_device = next(self._model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        kwargs = {'use_cache': False}
        return ((), {**inputs, **kwargs})


class TransformerAutoExtractor(TransformerExtractor):
    """TransformerAutoExtractor 用于从模型路径加载AutoModel并提取计算图。"""

    @staticmethod
    # pylint: disable=arguments-renamed
    def create(model_path, trust_remote_code=False, device="cpu", revision='main') -> 'TransformerExtractor':
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code, revision=revision)
        with torch.device(device):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, revision=revision
            )
        return TransformerExtractor(model, tokenizer)
