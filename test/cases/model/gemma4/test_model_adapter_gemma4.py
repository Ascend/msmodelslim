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

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from msmodelslim.model.gemma4.model_adapter import (
    Gemma4ModelAdapter,
    _cast_floating_state_dict,
    _promote_float_buffers_to_parameters,
)
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.security import json_safe_dump, json_safe_load


class TestPromoteFloatBuffersToParameters:
    def test_promote_float_buffers_to_parameters_creates_fp32_parameter_when_supported_buffer_exists(self):
        module = torch.nn.Module()
        module.register_buffer("layer_scalar", torch.tensor([1.5], dtype=torch.bfloat16))

        _promote_float_buffers_to_parameters(module)

        assert "layer_scalar" not in module._buffers
        assert isinstance(module.layer_scalar, torch.nn.Parameter)
        assert module.layer_scalar.dtype == torch.float32
        assert not module.layer_scalar.requires_grad


class TestCastFloatingStateDict:
    def test_cast_floating_state_dict_preserves_non_floating_tensors_when_dtype_is_changed(self):
        state_dict = {
            "weight": torch.ones(2, dtype=torch.float32),
            "indices": torch.ones(2, dtype=torch.int64),
            "mask": torch.ones(2, dtype=torch.bool),
        }

        converted = _cast_floating_state_dict(state_dict, torch.bfloat16)

        assert converted["weight"].dtype == torch.bfloat16
        assert converted["indices"].dtype == torch.int64
        assert converted["mask"].dtype == torch.bool


class TestGemma4ModelAdapter:
    embed_key = "model.language_model.embed_tokens.weight"
    embed_file = "quant_model_weights-00001-of-00001.safetensors"

    @classmethod
    def create_quant_export(cls, tmp_path):
        save_directory = tmp_path / "quant"
        save_directory.mkdir()
        embed_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        json_safe_dump(
            {
                "metadata": {"total_size": embed_weight.numel() * embed_weight.element_size()},
                "weight_map": {cls.embed_key: cls.embed_file},
            },
            str(save_directory / "quant_model_weights.safetensors.index.json"),
            indent=2,
        )
        json_safe_dump(
            {cls.embed_key: "FLOAT"},
            str(save_directory / "quant_model_description.json"),
            indent=2,
        )
        save_file({cls.embed_key: embed_weight}, save_directory / cls.embed_file)
        return save_directory, embed_weight

    @staticmethod
    def create_adapter(tie_word_embeddings):
        adapter = Gemma4ModelAdapter.__new__(Gemma4ModelAdapter)
        adapter.config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        return adapter

    def test_ascendv1_save_postprocess_exports_float_lm_head_when_word_embeddings_are_tied(self, tmp_path):
        save_directory, embed_weight = self.create_quant_export(tmp_path)
        adapter = self.create_adapter(tie_word_embeddings=True)

        adapter.ascendv1_save_postprocess(model=None, save_directory=str(save_directory))

        desc = json_safe_load(str(save_directory / "quant_model_description.json"))
        index = json_safe_load(str(save_directory / "quant_model_weights.safetensors.index.json"))
        weight_file = save_directory / index["weight_map"]["lm_head.weight"]
        with safe_open(str(weight_file), framework="pt", device="cpu") as f:
            lm_head_weight = f.get_tensor("lm_head.weight")

        assert desc["lm_head.weight"] == "FLOAT"
        torch.testing.assert_close(lm_head_weight, embed_weight)
        assert index["metadata"]["total_size"] == 2 * embed_weight.numel() * embed_weight.element_size()

    def test_generate_model_visit_excludes_final_norm_and_lm_head_when_dynamic_w8a8_is_used(self):
        adapter = Gemma4ModelAdapter.__new__(Gemma4ModelAdapter)
        adapter.generate_decoder_layer = lambda model: iter(())
        model = SimpleNamespace(
            model=SimpleNamespace(
                vision_tower=torch.nn.Identity(),
                embed_vision=torch.nn.Identity(),
            )
        )

        request_names = [request.name for request in adapter.generate_model_visit(model)]

        assert request_names == ["model.vision_tower", "model.embed_vision"]

    def test_ascendv1_save_postprocess_keeps_index_size_when_tied_lm_head_already_exists(self, tmp_path):
        save_directory, embed_weight = self.create_quant_export(tmp_path)
        adapter = self.create_adapter(tie_word_embeddings=True)
        adapter.ascendv1_save_postprocess(model=None, save_directory=str(save_directory))

        adapter.ascendv1_save_postprocess(model=None, save_directory=str(save_directory))

        index = json_safe_load(str(save_directory / "quant_model_weights.safetensors.index.json"))
        assert index["metadata"]["total_size"] == 2 * embed_weight.numel() * embed_weight.element_size()

    def test_ascendv1_save_postprocess_skips_export_when_word_embeddings_are_not_tied(self, tmp_path):
        save_directory = tmp_path / "quant"
        adapter = self.create_adapter(tie_word_embeddings=False)

        adapter.ascendv1_save_postprocess(model=None, save_directory=str(save_directory))

        assert not save_directory.exists()

    def test_ascendv1_save_postprocess_raises_invalid_model_error_when_tied_embedding_is_missing(self, tmp_path):
        save_directory = tmp_path / "quant"
        save_directory.mkdir()
        json_safe_dump(
            {"metadata": {"total_size": 0}, "weight_map": {}},
            str(save_directory / "quant_model_weights.safetensors.index.json"),
            indent=2,
        )
        adapter = self.create_adapter(tie_word_embeddings=True)

        with pytest.raises(InvalidModelError):
            adapter.ascendv1_save_postprocess(model=None, save_directory=str(save_directory))
