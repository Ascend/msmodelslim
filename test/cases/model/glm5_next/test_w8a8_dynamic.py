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

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from msmodelslim.model.glm5_next.w8a8_dynamic import (
    DESCRIPTION_FILE,
    WEIGHTS_INDEX_FILE,
    convert_to_w8a8_dynamic,
    get_quant_layer_count,
    quantize_per_channel,
    should_quantize,
)
from msmodelslim.utils.exception import InvalidModelError

NUM_QUANT_LAYERS = 2
QUANT_LAYERS = (0, 1)
MTP_LAYER = 2

CONFIG = {
    "architectures": ["Glm5NextForConditionalGeneration"],
    "model_type": "glm5_next",
    "text_config": {
        "num_hidden_layers": 3,
        "num_nextn_predict_layers": 1,
        "first_k_dense_replace": 1,
        "hidden_size": 4,
    },
}

# (tensor name, shape) of the tiny checkpoint used by the tests.
TENSORS = {
    # dense FFN of the first_k_dense_replace layer -> quantized
    "model.language_model.layers.0.mlp.gate_proj.weight": (4, 4),
    "model.language_model.layers.0.mlp.up_proj.weight": (4, 4),
    "model.language_model.layers.0.mlp.down_proj.weight": (4, 4),
    # KDA linear attention -> kept in BF16
    "model.language_model.layers.0.self_attn.q_proj.weight": (4, 4),
    "model.language_model.layers.0.self_attn.q_conv1d.weight": (4, 1),
    "model.language_model.layers.0.hc_attn_base.weight": (4,),
    # MoE experts and shared expert -> quantized
    "model.language_model.layers.1.mlp.experts.0.gate_proj.weight": (4, 4),
    "model.language_model.layers.1.mlp.experts.287.up_proj.weight": (4, 4),
    "model.language_model.layers.1.mlp.experts.0.down_proj.weight": (4, 4),
    "model.language_model.layers.1.mlp.shared_experts.down_proj.weight": (4, 4),
    # sparse-MLA and indexer -> kept in BF16
    "model.language_model.layers.1.self_attn.q_a_proj.weight": (4, 4),
    "model.language_model.layers.1.self_attn.kv_b_proj.weight": (4, 4),
    "model.language_model.layers.1.self_attn.indexer.weights_proj.weight": (4, 4),
    # MTP draft layer -> kept in BF16
    f"model.language_model.layers.{MTP_LAYER}.mlp.experts.0.gate_proj.weight": (4, 4),
    f"model.language_model.layers.{MTP_LAYER}.mlp.experts.0.up_proj.weight": (4, 4),
    f"model.language_model.layers.{MTP_LAYER}.enorm.weight": (4,),
    # other modules
    "model.language_model.embed_tokens.weight": (4, 4),
    "lm_head.weight": (4, 4),
}


def _build_model(model_path: Path) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    model_path.joinpath("config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    model_path.joinpath("tokenizer_config.json").write_text("{}", encoding="utf-8")

    names = sorted(TENSORS)
    first_shard, second_shard = names[: len(names) // 2], names[len(names) // 2 :]
    weight_map = {}
    for idx, shard_names in enumerate((first_shard, second_shard), start=1):
        file_name = f"model-{idx:05d}-of-00002.safetensors"
        save_file(
            {name: torch.randn(TENSORS[name]) for name in shard_names},
            str(model_path.joinpath(file_name)),
            metadata={"format": "pt"},
        )
        weight_map.update({name: file_name for name in shard_names})
    model_path.joinpath("model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}), encoding="utf-8"
    )


class TestShouldQuantize(unittest.TestCase):
    def test_should_quantize_shouldBeTrue_when_ffn_projection_of_a_quantized_layer(self):
        for name in (
            "model.layers.1.mlp.experts.287.down_proj.weight",
            "model.layers.1.mlp.shared_experts.up_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ):
            with self.subTest(name=name):
                self.assertTrue(should_quantize(name, NUM_QUANT_LAYERS))

    def test_should_quantize_shouldBeFalse_when_module_is_out_of_the_scope(self):
        for name in (
            "model.layers.1.self_attn.q_b_proj.weight",
            "model.layers.1.self_attn.indexer.wk.weight",
            "model.layers.1.mlp.gate.weight",
            "model.layers.1.mlp.experts.0.gate_proj.weight_scale",
            "model.language_model.layers.1.mlp.experts.0.gate_proj.bias",
        ):
            with self.subTest(name=name):
                self.assertFalse(should_quantize(name, NUM_QUANT_LAYERS))

    def test_should_quantize_shouldBeFalse_when_layer_is_the_mtp_draft_layer(self):
        self.assertFalse(should_quantize("model.language_model.layers.2.mlp.experts.0.gate_proj.weight", 2))
        self.assertTrue(should_quantize("model.language_model.layers.1.mlp.experts.0.gate_proj.weight", 2))


class TestQuantizePerChannel(unittest.TestCase):
    def test_quantize_per_channel_shouldReconstructWeights_when_dequantized(self):
        weight = torch.randn(8, 16, dtype=torch.bfloat16)

        quantized, scale, offset = quantize_per_channel(weight)

        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(torch.equal(offset, torch.zeros_like(scale)))
        self.assertEqual(quantized.shape, weight.shape)
        self.assertEqual(scale.shape, (weight.shape[0], 1))
        dequantized = quantized.float() * scale
        self.assertTrue(torch.allclose(dequantized, weight.float(), atol=scale.max().item() / 2))

    def test_quantize_per_channel_shouldRaise_when_weight_is_not_2d(self):
        with self.assertRaises(InvalidModelError):
            quantize_per_channel(torch.randn(2, 2, 2))

    def test_quantize_per_channel_shouldRaise_when_weight_is_not_finite(self):
        weight = torch.randn(4, 4)
        weight[0, 0] = float("nan")

        with self.assertRaises(InvalidModelError):
            quantize_per_channel(weight)


class TestGetQuantLayerCount(unittest.TestCase):
    def test_get_quant_layer_count_shouldExcludeMtpLayers_when_config_is_valid(self):
        self.assertEqual(get_quant_layer_count(CONFIG["text_config"]), NUM_QUANT_LAYERS)
        self.assertEqual(get_quant_layer_count({"num_hidden_layers": 5}), 5)

    def test_get_quant_layer_count_shouldRaise_when_num_hidden_layers_is_invalid(self):
        for text_config in ({}, {"num_hidden_layers": 0}, {"num_hidden_layers": "45"}):
            with self.subTest(text_config=text_config), self.assertRaises(InvalidModelError):
                get_quant_layer_count(text_config)


class TestConvertToW8a8Dynamic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.model_path = Path(self.temp_dir.name).joinpath("GLM-5.3-Flash-BF16")
        self.save_path = Path(self.temp_dir.name).joinpath("GLM-5.3-Flash-W8A8")
        _build_model(self.model_path)

    def _convert(self):
        convert_to_w8a8_dynamic(str(self.model_path), str(self.save_path))
        description = json.loads(self.save_path.joinpath(DESCRIPTION_FILE).read_text(encoding="utf-8"))
        index = json.loads(self.save_path.joinpath(WEIGHTS_INDEX_FILE).read_text(encoding="utf-8"))
        return description, index

    def test_convert_shouldMarkOnlyTheFfnProjectionsAsQuantized_when_converted(self):
        description, _ = self._convert()

        quantized = sorted(name for name, value in description.items() if value == "W8A8_DYNAMIC")
        self.assertEqual(
            quantized,
            [
                "model.language_model.layers.0.mlp.down_proj.weight",
                "model.language_model.layers.0.mlp.down_proj.weight_offset",
                "model.language_model.layers.0.mlp.down_proj.weight_scale",
                "model.language_model.layers.0.mlp.gate_proj.weight",
                "model.language_model.layers.0.mlp.gate_proj.weight_offset",
                "model.language_model.layers.0.mlp.gate_proj.weight_scale",
                "model.language_model.layers.0.mlp.up_proj.weight",
                "model.language_model.layers.0.mlp.up_proj.weight_offset",
                "model.language_model.layers.0.mlp.up_proj.weight_scale",
                "model.language_model.layers.1.mlp.experts.0.down_proj.weight",
                "model.language_model.layers.1.mlp.experts.0.down_proj.weight_offset",
                "model.language_model.layers.1.mlp.experts.0.down_proj.weight_scale",
                "model.language_model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.language_model.layers.1.mlp.experts.0.gate_proj.weight_offset",
                "model.language_model.layers.1.mlp.experts.0.gate_proj.weight_scale",
                "model.language_model.layers.1.mlp.experts.287.up_proj.weight",
                "model.language_model.layers.1.mlp.experts.287.up_proj.weight_offset",
                "model.language_model.layers.1.mlp.experts.287.up_proj.weight_scale",
                "model.language_model.layers.1.mlp.shared_experts.down_proj.weight",
                "model.language_model.layers.1.mlp.shared_experts.down_proj.weight_offset",
                "model.language_model.layers.1.mlp.shared_experts.down_proj.weight_scale",
            ],
        )
        for name in TENSORS:
            with self.subTest(name=name):
                expected = "W8A8_DYNAMIC" if should_quantize(name, NUM_QUANT_LAYERS) else "FLOAT"
                self.assertEqual(description.get(name), expected)

    def test_convert_shouldKeepMtpAndAttentionWeightsInBf16_when_converted(self):
        description, _ = self._convert()

        for name in (
            "model.language_model.layers.1.self_attn.q_a_proj.weight",
            "model.language_model.layers.1.self_attn.indexer.weights_proj.weight",
            "model.language_model.layers.0.self_attn.q_conv1d.weight",
            f"model.language_model.layers.{MTP_LAYER}.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.0.hc_attn_base.weight",
        ):
            with self.subTest(name=name):
                self.assertEqual(description[name], "FLOAT")
                self.assertNotIn(name.replace(".weight", ".weight_scale"), description)

    def test_convert_shouldExportQuantizedWeightsAndScales_when_converted(self):
        description, index = self._convert()

        self.assertEqual(description["group_size"], 0)
        self.assertEqual(description["metadata"], {})
        self.assertEqual(set(index["weight_map"]), set(description) - {"group_size", "metadata"})
        for name, file_name in index["weight_map"].items():
            with self.subTest(name=name):
                self.assertTrue(self.save_path.joinpath(file_name).is_file())

        with safe_open(
            str(self.save_path.joinpath(index["weight_map"]["model.language_model.layers.0.mlp.gate_proj.weight"])),
            framework="pt",
            device="cpu",
        ) as shard:
            quantized = shard.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight")
            scale = shard.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight_scale")
            offset = shard.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight_offset")

        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(torch.equal(offset, torch.zeros_like(scale)))

    def test_convert_shouldCopyAuxiliaryFiles_when_converted(self):
        self._convert()

        self.assertTrue(self.save_path.joinpath("config.json").is_file())
        self.assertTrue(self.save_path.joinpath("tokenizer_config.json").is_file())
        self.assertTrue(self.save_path.joinpath("configuration.json").is_file())

    def test_convert_shouldRaise_when_model_path_does_not_exist(self):
        with self.assertRaises(InvalidModelError):
            convert_to_w8a8_dynamic(str(self.model_path.joinpath("not_exist")), str(self.save_path))


if __name__ == '__main__':
    unittest.main()
