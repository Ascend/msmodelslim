#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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

# pylint: disable=arguments-differ,use-implicit-booleaness-not-comparison

import sys
from types import SimpleNamespace

import torch
from torch import nn

from msmodelslim.core.graph.fast_ops_grapher.extractors.transformer_extractor import (
    TransformerAutoExtractor,
    TransformerExtractor,
)


class _FakeTokenizer:
    def __call__(self, text, return_tensors):
        assert text == "."
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model_path, trust_remote_code=False, revision="main"):
        assert model_path == "/tmp/fake-model"
        assert trust_remote_code is True
        assert revision == "test-revision"
        return _FakeTokenizer()


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(model_path, trust_remote_code=False, revision="main"):
        assert model_path == "/tmp/fake-model"
        assert trust_remote_code is True
        assert revision == "test-revision"
        return nn.Linear(3, 3)


class TestTransformerExtractor:
    def test_create_exposes_target_and_dummy_inputs_on_model_device(self):
        model = nn.Linear(3, 3)
        extractor = TransformerExtractor.create(model=model, tokenizer=_FakeTokenizer())

        assert extractor.target_module is model
        args, kwargs = extractor.dummy_inputs
        assert args == ()
        assert kwargs["use_cache"] is False
        assert kwargs["input_ids"].device == model.weight.device
        assert kwargs["attention_mask"].tolist() == [[1, 1, 1]]


class TestTransformerAutoExtractor:
    def test_create_forwards_options_and_returns_transformer_extractor(self, monkeypatch):
        fake_transformers = SimpleNamespace(
            AutoTokenizer=_FakeAutoTokenizer,
            AutoModelForCausalLM=_FakeAutoModel,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        extractor = TransformerAutoExtractor.create(
            model_path="/tmp/fake-model",
            trust_remote_code=True,
            revision="test-revision",
        )

        assert isinstance(extractor, TransformerExtractor)
        assert isinstance(extractor.target_module, nn.Linear)


class _FakeCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        _ = (attention_mask, use_cache)
        return self.projection(input_ids.float())


def test_extract_dag_returns_non_empty_graph_for_supported_model():
    extractor = TransformerExtractor.create(model=_FakeCausalModel(), tokenizer=_FakeTokenizer())

    graph = extractor.extract_dag()

    assert graph.node_count == 1
    assert {node.operator.op_name for node in graph.iter_nodes()} == {"addmm.default"}
