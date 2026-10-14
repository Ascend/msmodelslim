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

Unit tests for Qwen25OmniThinkerModelAdapter.
"""
# pylint: disable=consider-using-from-import,redefined-outer-name

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.infra.dataset_loader import VlmCalibSample
from msmodelslim.model.qwen2_5_omni_thinker.model_adapter import Qwen25OmniThinkerModelAdapter
from msmodelslim.utils.exception import InvalidDatasetError


class _DummyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.attention_type = "full_attention"
        self.self_attn = nn.Linear(8, 8)
        self.mlp = SimpleNamespace(up_proj=nn.Linear(8, 8), down_proj=nn.Linear(8, 8))
        self.post_attention_layernorm = nn.LayerNorm(8)

    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


def _build_mock_model(hidden_layers: int = 2, has_sliding_layers: bool = False):
    layers = nn.ModuleList([_DummyDecoderLayer() for _ in range(hidden_layers)])
    model = MagicMock()
    model.eval.return_value = model
    layer_types = ["sliding_attention"] * hidden_layers if has_sliding_layers else ["full_attention"] * hidden_layers
    sliding_window = 4096 if has_sliding_layers else None
    rope_params = {"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [8, 12, 12]}
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=hidden_layers,
            num_attention_heads=8,
            num_key_value_heads=8,
            hidden_size=64,
            intermediate_size=128,
            rms_norm_eps=1e-6,
            max_position_embeddings=32768,
            layer_types=layer_types,
            use_sliding_window=has_sliding_layers,
            sliding_window=sliding_window,
            attention_dropout=0.0,
            hidden_act="silu",
            rope_parameters=rope_params,
            rope_scaling=rope_params,
            rope_theta=1000000.0,
            _attn_implementation="eager",
        ),
        num_attention_heads=8,
        num_key_value_heads=8,
        use_cache=False,
        sliding_window=sliding_window,
        _attn_implementation="eager",
    )
    model.model = SimpleNamespace(
        config=model.config,
        has_sliding_layers=has_sliding_layers,
        rotary_emb=MagicMock(return_value=(torch.zeros(1), torch.zeros(1))),
        layers=layers,
    )
    model.audio_tower = MagicMock()
    model.audio_tower._get_feat_extract_output_lengths = MagicMock(return_value=(torch.tensor([3]), torch.tensor([3])))
    model.visual = MagicMock()
    model.load_state_dict = MagicMock()
    model.get_input_embeddings = MagicMock(
        return_value=lambda input_ids: torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
    )
    model.get_rope_index = MagicMock(return_value=(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 1)))

    def _placeholder_mask(input_ids, inputs_embeds=None, image_features=None, video_features=None):
        mask_shape = inputs_embeds.shape
        zero_mask = torch.zeros(mask_shape, dtype=torch.bool)
        return zero_mask, zero_mask, zero_mask

    def _get_submodule(name: str):
        if name.startswith("model.layers."):
            idx = int(name.split(".")[-1])
            return layers[idx]
        raise AttributeError(name)

    model.get_placeholder_mask = MagicMock(side_effect=_placeholder_mask)
    model.get_submodule = MagicMock(side_effect=_get_submodule)
    return model


def _run_forward_generator_to_end(gen):
    requests = []
    response = None
    while True:
        try:
            req = next(gen) if response is None else gen.send(response)
        except StopIteration:
            break
        requests.append(req)
        if req.name == "audio_tower":
            response = torch.zeros(1, 0, 8)
        elif req.name == "visual":
            response = torch.zeros(1, 0, 8)
        else:
            response = (req.args[0],)
    return requests


@pytest.fixture
def adapter(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({"model_type": "qwen2_5_omni_thinker"}), encoding="utf-8")
    mock_config = SimpleNamespace(
        model_type="qwen2_5_omni_thinker",
        use_cache=False,
        thinker_config=SimpleNamespace(
            text_config=SimpleNamespace(
                num_hidden_layers=2,
                num_attention_heads=8,
                num_key_value_heads=8,
                hidden_size=64,
                intermediate_size=128,
                rms_norm_eps=1e-6,
                max_position_embeddings=32768,
                layer_types=["full_attention", "full_attention"],
                use_sliding_window=False,
                sliding_window=None,
                attention_dropout=0.0,
                hidden_act="silu",
                rope_parameters={"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [8, 12, 12]},
                rope_scaling={"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [8, 12, 12]},
                rope_theta=1000000.0,
                _attn_implementation="eager",
            ),
            audio_config=SimpleNamespace(num_hidden_layers=1),
            vision_config=SimpleNamespace(depth=1),
        ),
    )
    with patch(
        "msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained",
        return_value=mock_config,
    ):
        return Qwen25OmniThinkerModelAdapter("Qwen2.5-Omni-7B", str(model_path))


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.apply_chat_template.return_value = "mock_prompt"
    processor.return_value = SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 3]]),
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=torch.tensor([[1, 1, 1]]),
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=torch.zeros(1, 3, dtype=torch.long),
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_audio_in_video=True,
        cache_position=None,
        video_second_per_grid=None,
    )
    with (
        patch(
            "transformers.Qwen2_5OmniProcessor.from_pretrained",
            return_value=processor,
        ),
        patch(
            "msmodelslim.model.qwen2_5_omni_thinker.model_adapter.process_mm_info",
            return_value=([], [], []),
        ),
        patch(
            "msmodelslim.model.qwen2_5_omni_thinker.model_adapter.get_valid_read_path",
            side_effect=lambda p, **kwargs: p,
        ),
    ):
        yield processor


@pytest.fixture
def mock_model_class():
    mock_cls = MagicMock()
    mock_cls.from_pretrained = MagicMock(return_value=_build_mock_model(hidden_layers=2))
    with patch("transformers.Qwen2_5OmniThinkerForConditionalGeneration", mock_cls, create=True):
        yield mock_cls


class TestQwen2_5OmniThinkerModelAdapter:
    def test_get_model_pedigree_return_expected_value_when_called(self, adapter):
        assert adapter.get_model_pedigree() == "qwen2_5_omni_thinker"

    def test_get_model_type_return_expected_value_when_called(self, adapter):
        assert adapter.get_model_type() == "Qwen2.5-Omni-7B"

    def test_handle_dataset_return_processed_list_when_with_all_modalities(self, adapter, mock_processor):
        samples = [VlmCalibSample(text="sample", image="image.jpg", audio="audio.wav", video="video.mp4")]
        result = adapter.handle_dataset(samples, torch.device("cpu"))
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert mock_processor.call_count == 1

    def test_handle_dataset_raise_value_error_when_missing_required_modalities(self, adapter, mock_processor):
        samples = [VlmCalibSample(text="sample", image="image.jpg", audio="audio.wav")]
        with pytest.raises(InvalidDatasetError, match="No valid multimodal samples found"):
            adapter.handle_dataset(samples, torch.device("cpu"))

    def test_init_model_return_model_when_from_pretrained_success(self, adapter, mock_model_class):
        with (
            patch.object(adapter, "_get_state_dict", return_value={}),
            patch(
                "msmodelslim.model.qwen2_5_omni_thinker.model_adapter.get_valid_read_path",
                side_effect=lambda p, **kwargs: p,
            ),
        ):
            model = adapter.init_model(torch.device("cpu"))
        assert model is not None
        assert mock_model_class.from_pretrained.called

    def test_generate_model_visit_return_requests_when_called(self, adapter):
        with patch(
            "msmodelslim.model.qwen2_5_omni_thinker.model_adapter.generated_decoder_layer_visit_func",
            return_value=iter([ProcessRequest(name="model.layers.0", module=MagicMock(), args=(), kwargs={})]),
        ):
            requests = list(adapter.generate_model_visit(_build_mock_model(hidden_layers=2)))
        names = [item.name for item in requests]
        assert names.count("audio_tower") == 1
        assert names.count("visual") == 2
        assert "model.layers.0" in names

    def test_generate_model_forward_return_audio_request_when_with_audio_inputs(self, adapter):
        adapter.config.thinker_config.text_config.num_hidden_layers = 1
        model = _build_mock_model(hidden_layers=1)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "input_features": torch.randn(1, 8, 10),
            "feature_attention_mask": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": torch.zeros(1, 3, dtype=torch.long),
        }
        gen = adapter.generate_model_forward(model, inputs)
        req = next(gen)
        assert isinstance(req, ProcessRequest)
        assert req.name == "audio_tower"

    def test_generate_model_forward_return_decoder_request_when_without_multimodal_inputs(self, adapter):
        adapter.config.thinker_config.text_config.num_hidden_layers = 1
        model = _build_mock_model(hidden_layers=1)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": {"full_attention": torch.ones(1, 1, 3, 3)},
            "position_ids": torch.zeros(1, 3, dtype=torch.long),
        }
        gen = adapter.generate_model_forward(model, inputs)
        req = next(gen)
        assert req.name == "model.layers.0"

    def test_generate_model_forward_raise_value_error_when_missing_audio_lengths(self, adapter):
        adapter.config.thinker_config.text_config.num_hidden_layers = 1
        model = _build_mock_model(hidden_layers=1)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "input_features": torch.randn(1, 8, 10),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": torch.zeros(1, 3, dtype=torch.long),
        }
        gen = adapter.generate_model_forward(model, inputs)
        with pytest.raises(ValueError, match="Either audio_feature_lengths or feature_attention_mask"):
            next(gen)

    def test_generate_model_forward_return_all_requests_when_with_all_modalities(self, adapter):
        adapter.config.thinker_config.text_config.num_hidden_layers = 1
        model = _build_mock_model(hidden_layers=1)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "input_features": torch.randn(1, 8, 10),
            "feature_attention_mask": torch.ones(1, 10, dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
            "pixel_values_videos": torch.randn(1, 2, 3, 2, 2),
            "video_grid_thw": torch.tensor([[2, 1, 1]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": torch.zeros(1, 3, dtype=torch.long),
        }
        with patch(
            "msmodelslim.model.qwen2_5_omni_thinker.model_adapter.create_causal_mask",
            return_value=torch.ones(1, 1, 3, 3),
            create=True,
        ):
            requests = _run_forward_generator_to_end(adapter.generate_model_forward(model, inputs))
        names = [r.name for r in requests]
        assert "audio_tower" in names
        assert names.count("visual") == 2
        assert "model.layers.0" in names

    def test_generate_model_forward_use_sliding_mask_when_layer_is_sliding_attention(self, adapter):
        adapter.config.thinker_config.text_config.num_hidden_layers = 1
        model = _build_mock_model(hidden_layers=1, has_sliding_layers=True)
        model.model.layers[0].attention_type = "sliding_attention"
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": torch.zeros(1, 3, dtype=torch.long),
        }
        with (
            patch(
                "msmodelslim.model.qwen2_5_omni_thinker.model_adapter.create_causal_mask",
                return_value=torch.ones(1, 1, 3, 3),
                create=True,
            ),
            patch(
                "msmodelslim.model.qwen2_5_omni_thinker.model_adapter.create_sliding_window_causal_mask",
                return_value=torch.ones(1, 1, 3, 3),
                create=True,
            ),
        ):
            requests = _run_forward_generator_to_end(adapter.generate_model_forward(model, inputs))
        assert any(r.name == "model.layers.0" for r in requests)

    def test_generate_model_forward_set_rope_deltas_when_position_ids_none(self, adapter):
        adapter.config.thinker_config.text_config.num_hidden_layers = 1
        model = _build_mock_model(hidden_layers=1)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": None,
        }
        with patch(
            "msmodelslim.model.qwen2_5_omni_thinker.model_adapter.create_causal_mask",
            return_value=torch.ones(1, 1, 3, 3),
            create=True,
        ):
            _run_forward_generator_to_end(adapter.generate_model_forward(model, inputs))
        assert hasattr(model, "rope_deltas")

    def test_generate_decoder_layer_return_named_layers_when_called(self, adapter):
        adapter.config.thinker_config.text_config.num_hidden_layers = 2
        model = _build_mock_model(hidden_layers=2)
        layers = list(adapter.generate_decoder_layer(model))
        assert len(layers) == 2
        assert layers[0][0] == "model.layers.0"
        assert layers[1][0] == "model.layers.1"

    def test_enable_kv_cache_set_use_cache_when_switch_true_and_false(self, adapter):
        model = _build_mock_model(hidden_layers=1)
        adapter.enable_kv_cache(model, True)
        assert model.config.use_cache is True
        adapter.enable_kv_cache(model, False)
        assert model.config.use_cache is False

    def test_get_adapter_config_for_subgraph_return_expected_size_when_configured(self, adapter):
        adapter.config.thinker_config.audio_config.num_hidden_layers = 1
        adapter.config.thinker_config.vision_config.depth = 1
        adapter.config.thinker_config.text_config.num_hidden_layers = 2
        result = adapter.get_adapter_config_for_subgraph()
        assert isinstance(result, list)
        assert len(result) == 1 + 3 + 2 * 3

    def test_get_weight_map_return_cached_result_when_called_twice(self, adapter):
        index_data = {"weight_map": {"thinker.model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors"}}
        index_path = Path(adapter.model_path) / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index_data), encoding="utf-8")
        adapter._get_weight_map.cache_clear()
        result_1 = adapter._get_weight_map()
        result_2 = adapter._get_weight_map()
        assert result_1 == result_2
        assert "thinker.model.layers.0.self_attn.q_proj.weight" in result_1

    def test_get_state_dict_return_grouped_tensors_when_weight_map_has_multiple_names(self, adapter):
        index_data = {
            "weight_map": {
                "decoder.weight": "model-00001.safetensors",
                "decoder.bias": "model-00001.safetensors",
            }
        }
        index_path = Path(adapter.model_path) / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index_data), encoding="utf-8")
        (Path(adapter.model_path) / "model-00001.safetensors").write_bytes(b"mock")
        module = nn.Linear(4, 4)
        with (
            patch(
                "msmodelslim.model.qwen2_5_omni_thinker.model_adapter.get_valid_read_path",
                side_effect=lambda p, **kwargs: p,
            ),
            patch("msmodelslim.model.qwen2_5_omni_thinker.model_adapter.safe_open") as mock_open,
        ):
            f = MagicMock()
            f.get_tensor.side_effect = lambda name: torch.zeros(4, 4) if name.endswith("weight") else torch.zeros(4)
            mock_open.return_value.__enter__.return_value = f
            state_dict = adapter._get_state_dict(module, prefix="decoder")
        assert "weight" in state_dict
        assert "bias" in state_dict

    def test_load_decoder_if_not_exist_return_existing_decoder_when_materialized(self, adapter):
        model = _build_mock_model(hidden_layers=1)
        result = adapter._load_decoder_if_not_exist(model=model, name="model.layers.0", idx=0)
        assert isinstance(result, _DummyDecoderLayer)

    def test_load_decoder_if_not_exist_create_decoder_when_layer_meta_runtime_error(self, adapter):
        model = _build_mock_model(hidden_layers=1)
        broken_layer = MagicMock()
        broken_layer.input_layernorm.weight = MagicMock()
        type(broken_layer.input_layernorm.weight).device = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("meta tensor"))
        )
        model.get_submodule = MagicMock(return_value=broken_layer)
        fake_decoder = _DummyDecoderLayer()
        with (
            patch(
                "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniDecoderLayer",
                return_value=fake_decoder,
                create=True,
            ),
            patch.object(adapter, "_get_state_dict", return_value=fake_decoder.state_dict()),
        ):
            result = adapter._load_decoder_if_not_exist(model=model, name="model.layers.0", idx=0)
        assert result is fake_decoder
