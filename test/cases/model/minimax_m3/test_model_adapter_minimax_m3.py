#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn


# Mock transformers.models.minimax_m3_vl before importing model_adapter
# (UT 环境 transformers 版本较低，不包含该子模块)
class _MockMiniMaxM3VLDecoderLayer:
    def __init__(self, *args, **kwargs):
        pass


class _MockMiniMaxM3VLDenseMLP:
    def __init__(self, *args, **kwargs):
        pass


# 类名必须与 transformers 原始类名一致，否则 type(module).__name__ 断言会失败
class MiniMaxM3VLRMSNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _MockMiniMaxM3VLSparseMoeBlock:
    def __init__(self, *args, **kwargs):
        pass


class _MockMiniMaxM3VLVisionConfig:
    def __init__(self, *args, **kwargs):
        pass


class _MockMiniMaxM3VLTextConfig:
    def __init__(self, *args, **kwargs):
        pass


_mock_vl = MagicMock()
_mock_vl.modeling_minimax_m3_vl.MiniMaxM3VLDecoderLayer = _MockMiniMaxM3VLDecoderLayer
_mock_vl.modeling_minimax_m3_vl.MiniMaxM3VLDenseMLP = _MockMiniMaxM3VLDenseMLP
_mock_vl.modeling_minimax_m3_vl.MiniMaxM3VLRMSNorm = MiniMaxM3VLRMSNorm
_mock_vl.modeling_minimax_m3_vl.MiniMaxM3VLSparseMoeBlock = _MockMiniMaxM3VLSparseMoeBlock
_mock_vl.configuration_minimax_m3_vl.MiniMaxM3VLVisionConfig = _MockMiniMaxM3VLVisionConfig
_mock_vl.configuration_minimax_m3_vl.MiniMaxM3VLTextConfig = _MockMiniMaxM3VLTextConfig
sys.modules["transformers.models.minimax_m3_vl"] = _mock_vl
sys.modules["transformers.models.minimax_m3_vl.modeling_minimax_m3_vl"] = _mock_vl.modeling_minimax_m3_vl
sys.modules["transformers.models.minimax_m3_vl.configuration_minimax_m3_vl"] = _mock_vl.configuration_minimax_m3_vl

from msmodelslim.core.base.protocol import ProcessRequest  # noqa: E402
from msmodelslim.model.minimax_m3 import model_adapter as target  # noqa: E402
from msmodelslim.model.minimax_m3.model_adapter import MiniMaxM3ModelAdapter, _StandardRMSNorm  # noqa: E402


def _adapter(**kwargs):
    adapter = MiniMaxM3ModelAdapter.__new__(MiniMaxM3ModelAdapter)
    for key, value in kwargs.items():
        setattr(adapter, key, value)
    return adapter


def _make_text_config(**kwargs):
    defaults = dict(
        hidden_size=6144,
        intermediate_size=3072,
        num_hidden_layers=2,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        num_local_experts=4,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_vision_config(**kwargs):
    defaults = dict(
        hidden_size=1280,
        spatial_merge_size=2,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_config(**kwargs):
    text_config = _make_text_config()
    vision_config = _make_vision_config()
    defaults = dict(
        text_config=text_config,
        vision_config=vision_config,
        model_type="minimax_m3_vl",
        image_token_index=200025,
        video_token_index=200026,
        projector_hidden_size=6144,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Tests: _StandardRMSNorm
# ---------------------------------------------------------------------------


def test_standard_rms_norm_forward():
    norm = _StandardRMSNorm(dim=4, eps=1e-6)
    x = torch.arange(8, dtype=torch.float32).view(2, 4)
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


# ---------------------------------------------------------------------------
# Tests: __init__ and basic getters
# ---------------------------------------------------------------------------


def test_should_return_expected_values_when_basic_getters_called_given_adapter_fields():
    adapter = _adapter(
        model_type="minimax_m3_vl",
        config=_make_config(),
    )

    assert adapter.get_model_pedigree() == "minimax_m3"
    assert adapter.get_model_type() == "minimax_m3_vl"
    assert adapter.get_layer_wise_offload_device() == "meta"


# ---------------------------------------------------------------------------
# Tests: _is_moe_layer
# ---------------------------------------------------------------------------


def test_should_detect_moe_layer_when_mlp_layer_types_given_sparse():
    text_config = _make_text_config(mlp_layer_types=["sparse", "dense"])
    adapter = _adapter(config=_make_config(text_config=text_config))

    assert adapter._is_moe_layer(0) is True
    assert adapter._is_moe_layer(1) is False


def test_should_detect_moe_layer_when_mlp_layer_types_given_all_sparse():
    text_config = _make_text_config(mlp_layer_types=["sparse", "sparse"])
    adapter = _adapter(config=_make_config(text_config=text_config))

    assert adapter._is_moe_layer(0) is True
    assert adapter._is_moe_layer(1) is True


def test_should_detect_moe_layer_when_moe_layer_freq_given():
    text_config = _make_text_config(mlp_layer_types=None, moe_layer_freq=[True, False])
    adapter = _adapter(config=_make_config(text_config=text_config))

    assert adapter._is_moe_layer(0) is True
    assert adapter._is_moe_layer(1) is False


def test_should_detect_moe_layer_when_neither_mlp_types_nor_moe_freq_given():
    text_config = _make_text_config(
        mlp_layer_types=None,
        moe_layer_freq=None,
        first_k_dense_replace=2,
    )
    adapter = _adapter(config=_make_config(text_config=text_config))

    assert adapter._is_moe_layer(0) is False
    assert adapter._is_moe_layer(1) is False
    assert adapter._is_moe_layer(2) is True


# ---------------------------------------------------------------------------
# Tests: _get_num_experts
# ---------------------------------------------------------------------------


def test_should_return_num_local_experts_when_get_num_experts_called():
    text_config = _make_text_config(num_local_experts=128)
    adapter = _adapter(config=_make_config(text_config=text_config))
    assert adapter._get_num_experts() == 128


def test_should_fallback_to_n_routed_experts_when_num_local_experts_missing():
    text_config = SimpleNamespace(
        hidden_size=6144,
        intermediate_size=3072,
        num_hidden_layers=2,
        n_routed_experts=64,
    )
    adapter = _adapter(config=_make_config(text_config=text_config))
    assert adapter._get_num_experts() == 64


def test_should_return_default_experts_when_no_expert_attr_given():
    text_config = SimpleNamespace(
        hidden_size=6144,
        intermediate_size=3072,
        num_hidden_layers=2,
    )
    adapter = _adapter(config=_make_config(text_config=text_config))
    assert adapter._get_num_experts() == 128


# ---------------------------------------------------------------------------
# Tests: _is_sparse_layer
# ---------------------------------------------------------------------------


def test_should_return_false_when_sparse_freq_not_in_config(monkeypatch):
    monkeypatch.setattr(target, "json_safe_load", lambda _path: {"text_config": {}})
    adapter = _adapter(config=_make_config(), model_path="/tmp/model")
    assert adapter._is_sparse_layer(0) is False


def test_should_return_true_when_sparse_freq_has_true(monkeypatch):
    monkeypatch.setattr(
        target,
        "json_safe_load",
        lambda _path: {"text_config": {"sparse_attention_config": {"sparse_attention_freq": [True, False]}}},
    )
    adapter = _adapter(config=_make_config(), model_path="/tmp/model")
    assert adapter._is_sparse_layer(0) is True
    assert adapter._is_sparse_layer(1) is False


def test_should_return_false_on_exception(monkeypatch):
    monkeypatch.setattr(target, "json_safe_load", lambda _path: (_ for _ in ()).throw(Exception("file not found")))
    adapter = _adapter(config=_make_config(), model_path="/tmp/model")
    assert adapter._is_sparse_layer(0) is False


# ---------------------------------------------------------------------------
# Tests: get_adapter_config_for_subgraph (all dense layers)
# ---------------------------------------------------------------------------


def test_should_build_adapter_config_when_all_dense_layers():
    text_config = _make_text_config(
        num_hidden_layers=1,
        mlp_layer_types=["dense"],
        num_local_experts=4,
    )
    adapter = _adapter(config=_make_config(text_config=text_config), model_path="/tmp/model")

    out = adapter.get_adapter_config_for_subgraph()

    norm_linear = [c for c in out if c.subgraph_type == "norm-linear"]
    ov = [c for c in out if c.subgraph_type == "ov"]
    up_down = [c for c in out if c.subgraph_type == "up-down"]

    assert len(norm_linear) == 2  # attn + mlp
    assert len(ov) == 1
    assert len(up_down) == 1

    # attn norm-linear
    assert norm_linear[0].mapping.source == "model.language_model.layers.0.input_layernorm"
    assert norm_linear[0].mapping.targets == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.v_proj",
    ]

    # mlp norm-linear (dense)
    assert norm_linear[1].mapping.source == "model.language_model.layers.0.post_attention_layernorm"
    assert norm_linear[1].mapping.targets == [
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.mlp.up_proj",
    ]

    assert ov[0].mapping.source == "model.language_model.layers.0.self_attn.v_proj"
    assert ov[0].mapping.targets == ["model.language_model.layers.0.self_attn.o_proj"]

    assert up_down[0].mapping.source == "model.language_model.layers.0.mlp.up_proj"
    assert up_down[0].mapping.targets == ["model.language_model.layers.0.mlp.down_proj"]


def test_should_build_adapter_config_when_moe_layers():
    text_config = _make_text_config(
        num_hidden_layers=1,
        mlp_layer_types=["sparse"],
        num_local_experts=2,
    )
    adapter = _adapter(config=_make_config(text_config=text_config), model_path="/tmp/model")

    out = adapter.get_adapter_config_for_subgraph()

    norm_linear = [c for c in out if c.subgraph_type == "norm-linear"]
    assert len(norm_linear) == 2

    # MoE norm-linear targets include experts + shared_experts + gate
    assert norm_linear[1].mapping.targets == [
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
        "model.language_model.layers.0.mlp.experts.0.up_proj",
        "model.language_model.layers.0.mlp.experts.1.gate_proj",
        "model.language_model.layers.0.mlp.experts.1.up_proj",
        "model.language_model.layers.0.mlp.shared_experts.gate_proj",
        "model.language_model.layers.0.mlp.shared_experts.up_proj",
        "model.language_model.layers.0.mlp.gate",
    ]


def test_should_build_adapter_config_when_sparse_attention(monkeypatch):
    monkeypatch.setattr(
        target,
        "json_safe_load",
        lambda _path: {"text_config": {"sparse_attention_config": {"sparse_attention_freq": [True]}}},
    )
    text_config = _make_text_config(num_hidden_layers=1)
    adapter = _adapter(config=_make_config(text_config=text_config), model_path="/tmp/model")

    out = adapter.get_adapter_config_for_subgraph()

    norm_linear = [c for c in out if c.subgraph_type == "norm-linear"]
    # attn norm-linear should include indexer targets
    assert "model.language_model.layers.0.self_attn.indexer.q_proj" in norm_linear[0].mapping.targets
    assert "model.language_model.layers.0.self_attn.indexer.k_proj" in norm_linear[0].mapping.targets


# ---------------------------------------------------------------------------
# Tests: get_ln_fuse_map
# ---------------------------------------------------------------------------


def test_should_return_empty_ln_fuse_map_when_get_ln_fuse_map_called():
    adapter = _adapter()
    first_map, ln_fuse_map = adapter.get_ln_fuse_map()
    assert first_map == {}
    assert ln_fuse_map == {}


# ---------------------------------------------------------------------------
# Tests: get_bake_names
# ---------------------------------------------------------------------------


def test_should_return_empty_lists_when_get_bake_names_called():
    adapter = _adapter()
    first, second = adapter.get_bake_names()
    assert first == []
    assert second == []


# ---------------------------------------------------------------------------
# Tests: enable_kv_cache
# ---------------------------------------------------------------------------


def test_should_toggle_kv_cache_when_enable_kv_cache_called():
    model = SimpleNamespace(config=SimpleNamespace(use_cache=False))
    adapter = _adapter()
    adapter.enable_kv_cache(model, True)
    assert model.config.use_cache is True
    adapter.enable_kv_cache(model, False)
    assert model.config.use_cache is False


# ---------------------------------------------------------------------------
# Tests: ascendv1_save_postprocess
# ---------------------------------------------------------------------------


def test_should_delete_model_when_save_postprocess_called():
    model = MagicMock()
    adapter = _adapter()
    adapter.ascendv1_save_postprocess(model, "/tmp/save")
    # The method only calls `del model` — just verifies no exception
    assert True


# ---------------------------------------------------------------------------
# Tests: ascendv1_save_module_preprocess — prefix renaming
# ---------------------------------------------------------------------------


def test_should_expand_vit_prefix_when_save_preprocess_given_vit_layer():
    adapter = _adapter(config=_make_config())
    prefix, module = adapter.ascendv1_save_module_preprocess(
        "model.vision_tower.layers.0", nn.Linear(10, 10), MagicMock()
    )
    # model.vision_tower. → vision_tower. (catch-all rule), vision_tower.layers 规则已跳过
    assert prefix == "vision_tower.layers.0"


def test_should_expand_vit_patch_embedding_when_save_preprocess_given_vit_embeddings():
    adapter = _adapter(config=_make_config())
    prefix, module = adapter.ascendv1_save_module_preprocess(
        "vision_tower.embeddings.proj.0", nn.Linear(10, 10), MagicMock()
    )
    assert prefix == "vision_tower.vision_model.embeddings.patch_embedding.0"


def test_should_expand_vit_pre_layrnorm_when_save_preprocess_given_vit_pre_layrnorm():
    adapter = _adapter(config=_make_config())
    prefix, module = adapter.ascendv1_save_module_preprocess(
        "vision_tower.pre_layrnorm", nn.Linear(10, 10), MagicMock()
    )
    assert prefix == "vision_tower.vision_model.pre_layrnorm"


def test_should_rename_lm_head_when_save_preprocess_given_lm_head():
    adapter = _adapter(config=_make_config())
    prefix, module = adapter.ascendv1_save_module_preprocess("lm_head", nn.Linear(10, 10), MagicMock())
    assert prefix == "language_model.lm_head"


def test_should_rename_indexer_q_proj_when_save_preprocess_given_indexer():
    adapter = _adapter(config=_make_config())
    prefix, module = adapter.ascendv1_save_module_preprocess(
        "language_model.model.layers.0.self_attn.indexer.q_proj", nn.Linear(10, 10), MagicMock()
    )
    assert prefix == "language_model.model.layers.0.self_attn.index_q_proj"


def test_should_rename_indexer_k_proj_when_save_preprocess_given_indexer():
    adapter = _adapter(config=_make_config())
    prefix, module = adapter.ascendv1_save_module_preprocess(
        "language_model.model.layers.0.self_attn.indexer.k_proj", nn.Linear(10, 10), MagicMock()
    )
    assert prefix == "language_model.model.layers.0.self_attn.index_k_proj"


def test_should_rename_multi_modal_projector_merge_linear_1_when_save_preprocess():
    adapter = _adapter(config=_make_config())
    prefix, module = adapter.ascendv1_save_module_preprocess(
        "multi_modal_projector.merge_linear_1", nn.Linear(10, 10), MagicMock()
    )
    assert prefix == "patch_merge_mlp.linear_1"


def test_should_rename_multi_modal_projector_merge_linear_2_when_save_preprocess():
    adapter = _adapter(config=_make_config())
    prefix, module = adapter.ascendv1_save_module_preprocess(
        "multi_modal_projector.merge_linear_2", nn.Linear(10, 10), MagicMock()
    )
    assert prefix == "patch_merge_mlp.linear_2"


def test_should_rename_moe_expert_when_save_preprocess_given_moe_layer():
    text_config = _make_text_config(mlp_layer_types=["sparse", "dense"])
    adapter = _adapter(config=_make_config(text_config=text_config))

    prefix, module = adapter.ascendv1_save_module_preprocess(
        "language_model.model.layers.0.mlp.experts.0.gate_proj", nn.Linear(10, 10), MagicMock()
    )
    assert prefix == "language_model.model.layers.0.block_sparse_moe.experts.0.w1"

    prefix, module = adapter.ascendv1_save_module_preprocess(
        "language_model.model.layers.0.mlp.experts.0.up_proj", nn.Linear(10, 10), MagicMock()
    )
    assert prefix == "language_model.model.layers.0.block_sparse_moe.experts.0.w3"

    prefix, module = adapter.ascendv1_save_module_preprocess(
        "language_model.model.layers.0.mlp.experts.0.down_proj", nn.Linear(10, 10), MagicMock()
    )
    assert prefix == "language_model.model.layers.0.block_sparse_moe.experts.0.w2"


def test_should_skip_moe_rename_when_dense_layer():
    text_config = _make_text_config(mlp_layer_types=["sparse", "dense"])
    adapter = _adapter(config=_make_config(text_config=text_config))

    prefix, module = adapter.ascendv1_save_module_preprocess(
        "language_model.model.layers.1.mlp.gate_proj", nn.Linear(10, 10), MagicMock()
    )
    # Layer 1 is dense -> no rename
    assert prefix == "language_model.model.layers.1.mlp.gate_proj"


def test_should_rename_model_language_model_prefix():
    adapter = _adapter(config=_make_config())
    prefix, module = adapter.ascendv1_save_module_preprocess(
        "model.language_model.layers.0.self_attn.q_proj.weight", nn.Linear(10, 10), MagicMock()
    )
    assert prefix == "language_model.model.layers.0.self_attn.q_proj.weight"


# ---------------------------------------------------------------------------
# Tests: ascendv1_save_module_preprocess — RMSNorm conversion
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """命名与 transformers 的 RMSNorm 一致，触发 ascendv1_save_module_preprocess 中的类型名匹配"""

    def __init__(self, dim=8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * 2.0)
        self.variance_epsilon = 1e-6


class RMSNormBias(nn.Module):
    """同上，匹配 RMSNormBias 类型名"""

    def __init__(self, dim=8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * 2.0)
        self.variance_epsilon = 1e-6


def test_should_convert_rmsnorm_when_save_preprocess_given_rmsnorm():
    adapter = _adapter(config=_make_config())
    module = RMSNorm(dim=8)

    prefix, processed = adapter.ascendv1_save_module_preprocess("input_layernorm", module, MagicMock())

    assert type(processed).__name__ == "MiniMaxM3VLRMSNorm"
    # weight should be original - 1 (since _StandardRMSNorm stores w+1, save preprocess subtracts 1)
    assert torch.allclose(processed.weight.data.float(), torch.ones(8))


def test_should_convert_rmsnorm_bias_when_save_preprocess_given_rmsnorm_bias():
    adapter = _adapter(config=_make_config())
    module = RMSNormBias(dim=8)

    prefix, processed = adapter.ascendv1_save_module_preprocess("input_layernorm", module, MagicMock())

    assert type(processed).__name__ == "MiniMaxM3VLRMSNorm"
    assert torch.allclose(processed.weight.data.float(), torch.ones(8))


# ---------------------------------------------------------------------------
# Tests: _load_output_weight_map
# ---------------------------------------------------------------------------


def test_should_return_empty_map_when_no_save_directory(monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    adapter = _adapter()
    weight_map, index_path = adapter._load_output_weight_map("/tmp/save")
    assert weight_map["metadata"]["total_size"] == 0
    assert weight_map["weight_map"] == {}


def test_should_return_index_content_when_index_exists(monkeypatch):
    fake_data = {"metadata": {"total_size": 1024}, "weight_map": {"w1": "shard1"}}
    monkeypatch.setattr(os.path, "exists", lambda _path: True)
    monkeypatch.setattr(target, "json_safe_load", lambda _path: fake_data)
    adapter = _adapter()
    weight_map, _ = adapter._load_output_weight_map("/tmp/save")
    assert weight_map["metadata"]["total_size"] == 1024
    assert "w1" in weight_map["weight_map"]


def test_should_scan_single_file_when_no_index(monkeypatch):
    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return ["w1", "w2"]

    monkeypatch.setattr(os.path, "exists", lambda p: p.endswith("safetensors"))
    monkeypatch.setattr(target, "safe_open", lambda *a, **kw: FakeHandle())
    monkeypatch.setattr(os.path, "getsize", lambda _: 512)
    adapter = _adapter()
    weight_map, _ = adapter._load_output_weight_map("/tmp/save")
    assert weight_map["metadata"]["total_size"] == 512
    assert set(weight_map["weight_map"].keys()) == {"w1", "w2"}


# ---------------------------------------------------------------------------
# Tests: generate_model_visit
# ---------------------------------------------------------------------------


def _make_fake_model(num_layers=2):
    """Build a model-like SimpleNamespace mimic."""
    layers = nn.ModuleList()
    for i in range(num_layers):
        layer = nn.Linear(2, 2)
        layers.append(layer)
    language_model = SimpleNamespace(
        layers=layers,
        embed_tokens=nn.Linear(2, 2),
        rotary_emb=MagicMock(return_value=MagicMock()),
        norm=nn.Linear(2, 2),
    )
    return SimpleNamespace(
        model=SimpleNamespace(
            vision_tower=nn.Linear(2, 2),
            language_model=language_model,
        ),
        config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=2, num_hidden_layers=num_layers),
            image_token_id=200025,
        ),
    )


def test_should_yield_vision_tower_and_decoder_layers_when_generate_model_visit_called(monkeypatch):
    text_config = _make_text_config(num_hidden_layers=2)
    config = _make_config(text_config=text_config)
    adapter = _adapter(config=config, model_path="/tmp/model")

    model = _make_fake_model(num_layers=2)

    # Mock the internal loading to avoid checkpoint access
    monkeypatch.setattr(
        adapter, "_load_decoder_if_not_exist", lambda model, name, idx: model.model.language_model.layers[idx]
    )

    requests = list(adapter.generate_model_visit(model))

    assert len(requests) == 3
    assert requests[0].name == "vision_tower"
    assert requests[1].name == "model.language_model.layers.0"
    assert requests[2].name == "model.language_model.layers.1"
    assert all(isinstance(r, ProcessRequest) for r in requests)


# ---------------------------------------------------------------------------
# Tests: _remap_layer_weights
# ---------------------------------------------------------------------------


def _make_raw_weight(key, shape=(8, 8)):
    return {key: torch.randn(*shape)}


def _weights_for_moe_layer():
    weights = {}
    prefix = "language_model.model.layers.0.block_sparse_moe"
    # e_score_correction_bias
    weights[f"{prefix}.e_score_correction_bias"] = torch.randn(4)
    # shared_experts
    weights[f"{prefix}.shared_experts.gate_proj.weight"] = torch.randn(8, 8)
    weights[f"{prefix}.shared_experts.up_proj.weight"] = torch.randn(8, 8)
    # experts
    weights[f"{prefix}.experts.0.w1.weight"] = torch.randn(8, 8)
    weights[f"{prefix}.experts.0.w3.weight"] = torch.randn(8, 8)
    weights[f"{prefix}.experts.0.w2.weight"] = torch.randn(8, 8)
    return prefix, weights


def test_should_remap_e_score_correction_bias_when_remap_layer_weights_given_moe_layer():
    prefix, raw = _weights_for_moe_layer()
    state_dict = target.MiniMaxM3ModelAdapter._remap_layer_weights(raw, 0)
    assert "mlp.gate.e_score_correction_bias" in state_dict
    # mlp.e_score_correction_bias should NOT be in state_dict anymore
    assert "mlp.e_score_correction_bias" not in state_dict


def test_should_pack_shared_experts_when_remap_layer_weights_given_moe_layer():
    prefix, raw = _weights_for_moe_layer()
    state_dict = target.MiniMaxM3ModelAdapter._remap_layer_weights(raw, 0)
    assert "mlp.shared_experts.gate_up_proj.weight" in state_dict
    assert "mlp.shared_experts.gate_proj.weight" not in state_dict
    assert "mlp.shared_experts.up_proj.weight" not in state_dict


def test_should_pack_expert_gate_up_when_remap_layer_weights_given_moe_layer():
    prefix, raw = _weights_for_moe_layer()
    state_dict = target.MiniMaxM3ModelAdapter._remap_layer_weights(raw, 0)
    assert "mlp.experts.gate_up_proj" in state_dict
    assert state_dict["mlp.experts.gate_up_proj"].ndim == 3  # [num_experts, 2*inter, hidden]

    assert "mlp.experts.down_proj" in state_dict
    assert state_dict["mlp.experts.down_proj"].ndim == 3  # [num_experts, hidden, inter]


def test_should_remap_indexer_q_proj_when_remap_layer_weights_given_ckpt_prefix():
    raw = {"language_model.model.layers.0.self_attn.index_q_proj.weight": torch.randn(8, 8)}
    state_dict = target.MiniMaxM3ModelAdapter._remap_layer_weights(raw, 0)
    assert "self_attn.indexer.q_proj.weight" in state_dict
    assert "self_attn.index_q_proj.weight" not in state_dict


def test_should_remap_indexer_k_proj_when_remap_layer_weights_given_ckpt_prefix():
    raw = {"language_model.model.layers.0.self_attn.index_k_proj.weight": torch.randn(8, 8)}
    state_dict = target.MiniMaxM3ModelAdapter._remap_layer_weights(raw, 0)
    assert "self_attn.indexer.k_proj.weight" in state_dict


def test_should_remap_indexer_q_norm_when_remap_layer_weights_given_ckpt_prefix():
    raw = {"language_model.model.layers.0.self_attn.index_q_norm.weight": torch.randn(8)}
    state_dict = target.MiniMaxM3ModelAdapter._remap_layer_weights(raw, 0)
    assert "self_attn.indexer.q_norm.weight" in state_dict


def test_should_remap_indexer_k_norm_when_remap_layer_weights_given_ckpt_prefix():
    raw = {"language_model.model.layers.0.self_attn.index_k_norm.weight": torch.randn(8)}
    state_dict = target.MiniMaxM3ModelAdapter._remap_layer_weights(raw, 0)
    assert "self_attn.indexer.k_norm.weight" in state_dict
