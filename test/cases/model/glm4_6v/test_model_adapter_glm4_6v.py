#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from msmodelslim.model.glm4_6v.model_adapter import GLM4_6VModelAdapter
from msmodelslim.utils.exception import InvalidDatasetError


class _Sample:
    def __init__(self, image, text):
        self.image = image
        self.text = text


class _FakeDecoderLayer(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Linear(8, 8, bias=False)


class _SafeOpenCtx:
    def __init__(self, tensor_map):
        self.tensor_map = tensor_map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_tensor(self, name):
        return self.tensor_map[name]


class TestGLM4_6VModelAdapter:
    def _build_adapter(self):
        with patch("msmodelslim.model.glm4_6v.model_adapter.VLMBaseModelAdapter.__init__", return_value=None):
            adapter = GLM4_6VModelAdapter.__new__(GLM4_6VModelAdapter)
        adapter.model_type = "glm4_6v"
        adapter.model_path = Path("fake-model")
        adapter.trust_remote_code = False
        adapter.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(
                num_hidden_layers=3,
                first_k_dense_replace=1,
                num_attention_heads=8,
                num_key_value_heads=2,
            ),
            vision_config=types.SimpleNamespace(depth=2),
            use_cache=False,
        )
        return adapter

    def test_get_model_pedigree_return_glm4_6v_when_called(self):
        adapter = self._build_adapter()
        assert adapter.get_model_pedigree() == "glm4_6v"

    def test_get_model_type_return_model_type_when_called(self):
        adapter = self._build_adapter()
        assert adapter.get_model_type() == "glm4_6v"

    def test_is_moe_layer_return_true_when_layer_index_ge_first_k_dense_replace(self):
        adapter = self._build_adapter()
        assert adapter._is_moe_layer(1) is True

    def test_is_moe_layer_return_false_when_layer_index_lt_first_k_dense_replace(self):
        adapter = self._build_adapter()
        assert adapter._is_moe_layer(0) is False

    def test_get_adapter_config_for_subgraph_return_expected_count_when_configured(self):
        adapter = self._build_adapter()
        result = adapter.get_adapter_config_for_subgraph()
        # 每层2个(text norm-linear + ov) + dense层(0层)再加2个(norm-linear + up-down)
        assert len(result) == (3 * 2 + 1 * 2)

    def test_get_adapter_config_for_subgraph_return_contains_up_down_when_dense_layer(self):
        adapter = self._build_adapter()
        result = adapter.get_adapter_config_for_subgraph()
        has_up_down = any(c.subgraph_type == "up-down" and "layers.0.mlp.up_proj" in c.mapping.source for c in result)
        assert has_up_down is True

    def test_handle_dataset_raise_invalid_dataset_error_when_image_or_text_missing(self):
        adapter = self._build_adapter()
        dataset = [_Sample(image=None, text="hello")]
        with patch("msmodelslim.model.glm4_6v.model_adapter.AutoProcessor.from_pretrained", return_value=MagicMock()):
            with pytest.raises(InvalidDatasetError):
                adapter.handle_dataset(dataset)

    def test_handle_dataset_return_processed_items_when_inputs_valid(self):
        adapter = self._build_adapter()
        dataset = [_Sample(image="a.jpg", text="hello")]
        fake_inputs = types.SimpleNamespace()
        fake_processor = MagicMock()
        fake_processor.apply_chat_template.return_value = fake_inputs
        with (
            patch("msmodelslim.model.glm4_6v.model_adapter.AutoProcessor.from_pretrained", return_value=fake_processor),
            patch("msmodelslim.model.glm4_6v.model_adapter.get_valid_read_path", return_value="a.jpg"),
            patch.object(
                adapter, "_collect_inputs_to_device", return_value={"input_ids": torch.ones(1, 2, dtype=torch.long)}
            ),
        ):
            result = adapter.handle_dataset(dataset)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "input_ids" in result[0]

    def test_get_state_dict_return_selected_tensors_when_weight_map_contains_keys(self):
        adapter = self._build_adapter()
        module = nn.Linear(4, 3, bias=False)
        weight_map = {"model.language_model.layers.0.weight": "model-00001.safetensors"}
        tensor_map = {"model.language_model.layers.0.weight": torch.ones_like(module.weight)}
        with (
            patch.object(adapter, "_get_weight_map", return_value=weight_map),
            patch("msmodelslim.model.glm4_6v.model_adapter.get_valid_read_path", side_effect=lambda p, **_: p),
            patch("msmodelslim.model.glm4_6v.model_adapter.safe_open", return_value=_SafeOpenCtx(tensor_map)),
        ):
            sd = adapter._get_state_dict(module, prefix="model.language_model.layers.0")
        assert "weight" in sd
        assert torch.equal(sd["weight"], tensor_map["model.language_model.layers.0.weight"])

    def test_load_decoder_if_not_exist_return_existing_layer_when_layer_already_loaded(self):
        adapter = self._build_adapter()
        loaded_layer = _FakeDecoderLayer()
        model = MagicMock()
        model.get_submodule.return_value = loaded_layer
        result = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)
        assert result is loaded_layer

    def test_load_decoder_if_not_exist_create_and_attach_layer_when_not_loaded(self):
        adapter = self._build_adapter()
        # 让 idx=0 走 dense 层，避免进入 MoE 替换分支，便于纯单元测试
        adapter.config.text_config.first_k_dense_replace = 10
        model = types.SimpleNamespace(
            get_submodule=lambda _name: (_ for _ in ()).throw(AttributeError("not found")),
            model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=nn.ModuleList())),
        )
        with (
            patch("msmodelslim.model.glm4_6v.model_adapter.Glm4vMoeTextDecoderLayer", _FakeDecoderLayer),
            patch.object(adapter, "_get_state_dict", return_value=_FakeDecoderLayer().state_dict()),
        ):
            layer = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)
        assert isinstance(layer, _FakeDecoderLayer)
        assert len(model.model.language_model.layers) == 1

    def test_generate_decoder_layer_yield_expected_names_when_num_layers_is_two(self):
        adapter = self._build_adapter()
        adapter.config.text_config.num_hidden_layers = 2
        with patch.object(adapter, "_load_decoder_if_not_exist", side_effect=["L0", "L1"]):
            result = list(adapter.generate_decoder_layer(model=MagicMock()))
        assert result == [("model.language_model.layers.0", "L0"), ("model.language_model.layers.1", "L1")]

    def test_init_sets_processor_to_none_when_called(self):
        with patch("msmodelslim.model.glm4_6v.model_adapter.VLMBaseModelAdapter.__init__", return_value=None):
            adapter = GLM4_6VModelAdapter("glm4_6v", Path("fake-model"), False)
        assert adapter._processor is None

    def test_create_model_instance_return_model_when_called(self):
        adapter = self._build_adapter()
        fake_model = MagicMock()
        fake_model.eval.return_value = fake_model  # eval() 返回 self
        fake_model_cls = MagicMock()
        fake_model_cls.from_pretrained.return_value = fake_model
        with patch("msmodelslim.model.glm4_6v.model_adapter.get_valid_read_path", return_value=adapter.model_path):
            result = adapter._create_model_instance(fake_model_cls)
        assert result is fake_model
        fake_model_cls.from_pretrained.assert_called_once()
        fake_model.eval.assert_called_once()

    def test_enable_kv_cache_sets_use_cache_when_called(self):
        adapter = self._build_adapter()
        model = types.SimpleNamespace(config=types.SimpleNamespace(use_cache=False))
        adapter.enable_kv_cache(model, True)
        assert model.config.use_cache is True
        adapter.enable_kv_cache(model, False)
        assert model.config.use_cache is False

    def test_get_weight_map_return_weight_map_when_index_exists(self):
        adapter = self._build_adapter()
        weight_map = {"layer.0.weight": "model-00001.safetensors"}
        index_data = {"weight_map": weight_map}
        with patch("msmodelslim.model.glm4_6v.model_adapter.json_safe_load", return_value=index_data):
            result = adapter._get_weight_map()
        assert result == weight_map

    def test_handle_dataset_raise_invalid_dataset_error_when_text_is_none(self):
        adapter = self._build_adapter()
        dataset = [_Sample(image="a.jpg", text=None)]
        with patch("msmodelslim.model.glm4_6v.model_adapter.AutoProcessor.from_pretrained", return_value=MagicMock()):
            with pytest.raises(InvalidDatasetError):
                adapter.handle_dataset(dataset)

    def test_handle_dataset_return_multiple_items_when_multiple_samples(self):
        adapter = self._build_adapter()
        dataset = [_Sample(image="a.jpg", text="hello"), _Sample(image="b.jpg", text="world")]
        fake_inputs = types.SimpleNamespace()
        fake_processor = MagicMock()
        fake_processor.apply_chat_template.return_value = fake_inputs
        with (
            patch("msmodelslim.model.glm4_6v.model_adapter.AutoProcessor.from_pretrained", return_value=fake_processor),
            patch("msmodelslim.model.glm4_6v.model_adapter.get_valid_read_path", side_effect=lambda p, **_: p),
            patch.object(
                adapter, "_collect_inputs_to_device", return_value={"input_ids": torch.ones(1, 2, dtype=torch.long)}
            ),
        ):
            result = adapter.handle_dataset(dataset)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_load_decoder_if_not_exist_create_moe_layer_when_layer_index_ge_first_k_dense_replace(self):
        adapter = self._build_adapter()
        model = types.SimpleNamespace(
            get_submodule=lambda _name: (_ for _ in ()).throw(AttributeError("not found")),
            model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=nn.ModuleList())),
        )
        fake_decoder = _FakeDecoderLayer()
        # 创建一个真正的nn.Module作为UnstackedGlm4vMoeTextMoE的返回值
        unstacked_moe = nn.Module()
        # 创建一个与替换后的decoder结构匹配的state_dict
        # _get_state_dict是在mlp被替换为unstacked_moe之后调用的，所以state_dict应该匹配替换后的结构
        # 由于unstacked_moe是空的，state_dict只包含decoder的基本参数
        expected_state_dict = {
            "input_layernorm.weight": fake_decoder.input_layernorm.weight.data.clone(),
            "input_layernorm.bias": fake_decoder.input_layernorm.bias.data.clone(),
            "self_attn.weight": fake_decoder.self_attn.weight.data.clone(),
        }
        with (
            patch("msmodelslim.model.glm4_6v.model_adapter.Glm4vMoeTextDecoderLayer", return_value=fake_decoder),
            patch("msmodelslim.model.glm4_6v.model_adapter.UnstackedGlm4vMoeTextMoE") as mock_unstacked,
            patch.object(adapter, "_get_state_dict", return_value=expected_state_dict),
        ):
            mock_unstacked.return_value = unstacked_moe
            layer = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.1", 1)
        assert layer is not None
        mock_unstacked.assert_called_once()

    def test_load_decoder_if_not_exist_handle_runtime_error_when_layer_on_meta_device(self):
        adapter = self._build_adapter()
        meta_layer = _FakeDecoderLayer()
        model = MagicMock()
        model.get_submodule.return_value = meta_layer

        # 创建一个会抛出RuntimeError的weight对象
        class MetaWeightParam(nn.Parameter):
            @property
            def device(self):
                raise RuntimeError("device is meta")

        meta_layer.input_layernorm.weight = MetaWeightParam(meta_layer.input_layernorm.weight.data)

        fake_decoder = _FakeDecoderLayer()
        model.model = types.SimpleNamespace(language_model=types.SimpleNamespace(layers=nn.ModuleList()))
        with (
            patch("msmodelslim.model.glm4_6v.model_adapter.Glm4vMoeTextDecoderLayer", return_value=fake_decoder),
            patch.object(adapter, "_get_state_dict", return_value=fake_decoder.state_dict()),
        ):
            adapter.config.text_config.first_k_dense_replace = 10
            layer = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)
        assert layer is not None

    def test_get_adapter_config_for_subgraph_return_no_up_down_when_moe_layer(self):
        adapter = self._build_adapter()
        # 设置所有层都是MoE层
        adapter.config.text_config.first_k_dense_replace = 0
        result = adapter.get_adapter_config_for_subgraph()
        # MoE层只有norm-linear和ov，没有up-down
        assert len(result) == 3 * 2  # 3层，每层2个(norm-linear + ov)
        has_up_down = any("up-down" in str(c.subgraph_type) for c in result)
        assert has_up_down is False

    def test_generate_model_visit_yield_vision_and_decoder_layers_when_called(self):
        adapter = self._build_adapter()
        from msmodelslim.core.base.protocol import ProcessRequest

        fake_layer = MagicMock()
        model = types.SimpleNamespace(model=types.SimpleNamespace(visual=MagicMock()))
        with (
            patch.object(adapter, "generate_decoder_layer", return_value=iter([("layers.0", fake_layer)])),
            patch("msmodelslim.model.glm4_6v.model_adapter.generated_decoder_layer_visit_func") as mock_visit,
        ):
            mock_visit.return_value = iter([ProcessRequest(name="layers.0", module=fake_layer, args=(), kwargs={})])
            requests = list(adapter.generate_model_visit(model))
        assert len(requests) >= 1
        assert requests[0].name == "model.visual"

    def test_get_state_dict_return_empty_dict_when_weight_map_does_not_contain_keys(self):
        adapter = self._build_adapter()
        module = nn.Linear(4, 3, bias=False)
        weight_map = {}
        with patch.object(adapter, "_get_weight_map", return_value=weight_map):
            sd = adapter._get_state_dict(module, prefix="model.language_model.layers.0")
        assert sd == {}

    def test_init_model_return_model_when_called(self):
        adapter = self._build_adapter()
        adapter.config.text_config.num_hidden_layers = 2
        fake_model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                text_config=types.SimpleNamespace(
                    num_attention_heads=8,
                    num_key_value_heads=2,
                ),
            ),
            eval=MagicMock(return_value=None),
        )
        with (
            patch.object(adapter, "_create_model_instance", return_value=fake_model),
            patch("msmodelslim.model.glm4_6v.model_adapter.get_valid_read_path", return_value=adapter.model_path),
            patch("msmodelslim.model.glm4_6v.model_adapter.Glm4vMoeForConditionalGeneration"),
        ):
            result = adapter.init_model()
        assert result is fake_model
        assert adapter.config.text_config.num_hidden_layers == 2  # 应该恢复原值
        assert adapter.config.text_config._attn_implementation == 'eager'
