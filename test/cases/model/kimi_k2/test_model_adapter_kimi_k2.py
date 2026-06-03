#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from msmodelslim.model.kimi_k2 import model_adapter as target
from msmodelslim.model.kimi_k2.model_adapter import KimiK2ModelAdapter, default_dtype
from msmodelslim.utils.exception import InvalidModelError


def _adapter(**kwargs):
    a = KimiK2ModelAdapter.__new__(KimiK2ModelAdapter)
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def _make_config(
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    qk_nope_head_dim=8,
    v_head_dim=8,
    qk_rope_head_dim=8,
    hidden_size=16,
    vocab_size=16,
    first_k_dense_replace=1,
    n_routed_experts=0,
    n_shared_experts=0,
    rms_norm_eps=1e-6,
    q_lora_rank=8,
    kv_lora_rank=8,
):
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        first_k_dense_replace=first_k_dense_replace,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        rms_norm_eps=rms_norm_eps,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
    )


def _make_model(weights=None):
    class _M:
        def __init__(self):
            self.model = SimpleNamespace(layers=[])
            self._state_dict = weights or {}

        def get_submodule(self, name):
            if name in self._state_dict:
                return self._state_dict[name]
            raise AttributeError(name)

        def load_state_dict(self, _state, **_):
            return None

        def eval(self):
            return None

        def to(self, *_args, **_kwargs):
            return self

    return _M()


def _make_fake_modules(num_layers=2):
    return [SimpleNamespace(weight=torch.ones((4, 4))) for _ in range(num_layers)]


def test_default_dtype_given_dtype_when_context_exit_then_restore_original():
    original = torch.get_default_dtype()
    with default_dtype(torch.bfloat16):
        assert torch.get_default_dtype() == torch.bfloat16
    assert torch.get_default_dtype() == original


def test_default_dtype_given_nested_contexts_when_inner_exit_then_inner_restored():
    original = torch.get_default_dtype()
    with default_dtype(torch.bfloat16):
        with default_dtype(torch.float16):
            assert torch.get_default_dtype() == torch.float16
        assert torch.get_default_dtype() == torch.bfloat16
    assert torch.get_default_dtype() == original


def test_default_dtype_given_exception_in_context_when_exit_then_restore_original():
    original = torch.get_default_dtype()
    with pytest.raises(RuntimeError):
        with default_dtype(torch.bfloat16):
            assert torch.get_default_dtype() == torch.bfloat16
            raise RuntimeError("boom")
    assert torch.get_default_dtype() == original


def test_get_model_type_given_model_type_when_called_then_return_model_type():
    assert _adapter(model_type="my_model_type").get_model_type() == "my_model_type"


def test_get_model_pedigree_given_default_when_called_then_return_kimi_k2():
    assert _adapter().get_model_pedigree() == "kimi_k2"


def test_handle_dataset_given_input_when_called_then_return_tokenized_data(monkeypatch):
    adapter = _adapter(model_path="/tmp/m", trust_remote_code=False)
    fake_dataset = ["tokenized_data"]
    monkeypatch.setattr(adapter, "_get_tokenized_data", lambda data, device: data)
    out = adapter.handle_dataset(fake_dataset, device="cpu")
    assert out == ["tokenized_data"]


def test_generate_model_visit_given_model_when_called_then_yield_from_generated_func(monkeypatch):
    adapter = _adapter()
    expected = [("block0", "layer0"), ("block1", "layer1")]

    def fake_func(model, transformer_blocks):
        assert transformer_blocks == expected
        for name, block in transformer_blocks:
            yield target.ProcessRequest(name=name, module=block, args=(), kwargs={})

    monkeypatch.setattr(target, "generated_decoder_layer_visit_func", fake_func)
    monkeypatch.setattr(adapter, "generate_decoder_layer", lambda _model: expected)
    requests = list(adapter.generate_model_visit(object()))
    assert [r.name for r in requests] == ["block0", "block1"]


def test_enable_kv_cache_given_need_true_when_called_then_set_use_cache_true(monkeypatch):
    class _InnerModel:
        def __init__(self):
            self.layers = []
            self.use_cache = None

        def register_forward_pre_hook(self, hook, with_kwargs=False):
            self.use_cache = True

    class _M:
        def __init__(self):
            self.model = _InnerModel()

    model = _M()
    _adapter().enable_kv_cache(model, True)
    assert model.model.use_cache is True


def test_enable_kv_cache_given_need_false_when_called_then_set_use_cache_false(monkeypatch):
    class _InnerModel:
        def __init__(self):
            self.layers = []
            self.use_cache = True

        def register_forward_pre_hook(self, hook, with_kwargs=False):
            self.use_cache = False

    class _M:
        def __init__(self):
            self.model = _InnerModel()

    model = _M()
    _adapter().enable_kv_cache(model, False)
    assert model.model.use_cache is False


def test_get_adapter_config_for_subgraph_given_three_layers_when_called_then_return_6_configs():
    adapter = _adapter(config=_make_config(num_hidden_layers=3))
    out = adapter.get_adapter_config_for_subgraph()
    assert len(out) == 6
    ov = [c for c in out if c.subgraph_type == "ov"]
    norm = [c for c in out if c.subgraph_type == "norm-linear"]
    assert len(ov) == 2
    assert len(norm) == 4


def test_get_adapter_config_for_subgraph_given_ov_config_when_called_then_contain_expected_fields():
    adapter = _adapter(config=_make_config(num_hidden_layers=2))
    out = adapter.get_adapter_config_for_subgraph()
    ov_configs = [c for c in out if c.subgraph_type == "ov"]
    assert ov_configs[0].mapping.source == "model.layers.0.self_attn.kv_b_proj"
    assert ov_configs[0].mapping.targets == ["model.layers.0.self_attn.o_proj"]
    assert ov_configs[0].extra_config == {"group_method": "max"}
    assert ov_configs[0].fusion.fusion_type == "kv"
    assert ov_configs[0].fusion.num_attention_heads == 8
    assert ov_configs[0].fusion.num_key_value_heads == 4
    assert ov_configs[0].fusion.custom_config["qk_nope_head_dim"] == 8
    assert ov_configs[0].fusion.custom_config["v_head_dim"] == 8


def test_get_adapter_config_for_subgraph_given_norm_linear_config_when_called_then_have_correct_sources():
    adapter = _adapter(config=_make_config(num_hidden_layers=2))
    out = adapter.get_adapter_config_for_subgraph()
    norm_configs = [c for c in out if c.subgraph_type == "norm-linear"]
    sources = [c.mapping.source for c in norm_configs]
    assert "model.layers.0.input_layernorm" in sources
    assert "model.layers.0.self_attn.q_a_layernorm" in sources


def test_get_adapter_config_for_subgraph_given_one_layer_when_called_then_return_3_configs():
    adapter = _adapter(config=_make_config(num_hidden_layers=1))
    out = adapter.get_adapter_config_for_subgraph()
    assert len(out) == 0


def test_get_ln_fuse_map_given_default_when_called_then_return_tuple_of_dicts():
    adapter = _adapter(config=_make_config(num_hidden_layers=2))
    with patch.object(target, "get_ln_fuse_map", wraps=target.get_ln_fuse_map):
        first, second = adapter.get_ln_fuse_map()
    assert first == {}
    assert isinstance(second, dict)
    assert "model.layers.0.input_layernorm" in second


def test_get_bake_names_given_default_when_called_then_return_two_empty_lists():
    adapter = _adapter()
    out = adapter.get_bake_names()
    assert out == ([], [])


def test_get_rotate_map_given_block_size_when_called_then_return_lists():
    adapter = _adapter(config=_make_config(num_hidden_layers=2))
    with patch.object(target, "get_rotate_map", wraps=target.get_rotate_map):
        pre_run, rot_pairs = adapter.get_rotate_map(block_size=4)
    assert isinstance(pre_run, list)
    assert isinstance(rot_pairs, list)
    assert len(pre_run) == 1
    assert len(rot_pairs) > 0


def test_get_weight_map_given_index_json_when_loaded_then_return_weight_map(monkeypatch):
    adapter = _adapter(model_path="/tmp/model")
    adapter.get_weight_map.cache_clear()
    monkeypatch.setattr(target, "json_safe_load", lambda p: {"weight_map": {"a": "f"}})
    result = adapter.get_weight_map()
    assert result == {"a": "f"}
    adapter.get_weight_map.cache_clear()


def test_get_state_dict_given_module_when_called_then_load_all_tensors(monkeypatch):
    adapter = _adapter(model_path="/tmp/model")
    adapter.get_weight_map = lambda: {"weight": "a.safetensors", "bias": "a.safetensors"}
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)

    class _FakeSafeOpen:
        def __init__(self, _p, framework, device):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_tensor(self, name):
            return torch.ones((4, 4)) if name == "weight" else torch.ones(4)

    monkeypatch.setattr(target, "safe_open", _FakeSafeOpen)
    module = nn.Linear(4, 4)
    out = adapter.get_state_dict(module)
    assert "weight" in out and "bias" in out


def test_get_state_dict_given_prefix_when_called_then_use_prefix_in_lookup(monkeypatch):
    adapter = _adapter(model_path="/tmp/model")
    adapter.get_weight_map = lambda: {"pre.weight": "a.safetensors", "pre.bias": "a.safetensors"}
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)

    captured = {}

    class _FakeSafeOpen:
        def __init__(self, _p, framework, device):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_tensor(self, name):
            captured["name"] = name
            return torch.ones((2, 2))

    monkeypatch.setattr(target, "safe_open", _FakeSafeOpen)
    module = nn.Linear(2, 2, bias=False)
    adapter.get_state_dict(module, prefix="pre")
    assert captured["name"] == "pre.weight"


def test_load_decoder_if_not_exist_given_submodule_exists_when_called_then_return_submodule():
    adapter = _adapter()
    layer = SimpleNamespace(weight=torch.ones(1))
    model = _make_model({"model.layers.0": layer})
    out = adapter.load_decoder_if_not_exist(model, "model.layers.0", 0)
    assert out is layer


def test_load_decoder_if_not_exist_given_submodule_missing_when_called_then_create_layer(monkeypatch):
    config = _make_config(num_hidden_layers=2)
    adapter = _adapter(model_path="/tmp/model", config=config, trust_remote_code=False)
    layer_list = _make_fake_modules(num_layers=1)

    class _M:
        def __init__(self):
            self.model = SimpleNamespace(layers=layer_list)

        def get_submodule(self, _name):
            raise AttributeError("missing")

    monkeypatch.setattr(adapter, "get_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(target, "auto_convert_module_fp8_to_bf16", lambda *args, **kwargs: None)

    class _DummyLayer:
        def __init__(self, layer_idx, config):
            self.layer_idx = layer_idx
            self.config = config
            self.created = True

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return None

    layer_list[0] = _DummyLayer(layer_idx=0, config=config)

    def fake_get_submodule(_self, _name):
        raise AttributeError("missing")

    with patch.object(_M, "get_submodule", fake_get_submodule):
        model = _M()
        out = adapter.load_decoder_if_not_exist(model, "model.layers.1", 1)

    assert out is not None
    assert len(layer_list) == 2


def test_load_mtp_if_not_load_given_shared_head_exists_when_called_then_skip(monkeypatch):
    adapter = _adapter()

    class _Decoder:
        def get_submodule(self, _name):
            return object()

    mtp_decoder = _Decoder()
    with patch.object(target, "get_mtp_layer") as mock_get_mtp, patch.object(target, "wrap_mtp_decoder") as mock_wrap:
        adapter.load_mtp_if_not_load(mtp_decoder)
    mock_get_mtp.assert_not_called()
    mock_wrap.assert_not_called()


def test_load_mtp_if_not_load_given_shared_head_missing_when_called_then_load_and_wrap(monkeypatch):
    adapter = _adapter(config=_make_config(), model_path="/tmp/m")

    class _Decoder:
        def get_submodule(self, _name):
            raise AttributeError("missing")

    mtp_decoder = _Decoder()
    fake_layer = SimpleNamespace()
    with (
        patch.object(target, "get_mtp_layer", return_value=fake_layer) as mock_get_mtp,
        patch.object(target, "wrap_mtp_decoder") as mock_wrap,
    ):
        adapter.load_mtp_if_not_load(mtp_decoder)
    mock_get_mtp.assert_called_once()
    mock_wrap.assert_called_once_with(mtp_decoder=mtp_decoder, mtp_layer=fake_layer)


def test_generate_decoder_layer_given_two_layers_when_called_then_yield_each(monkeypatch):
    config = _make_config(num_hidden_layers=2)
    adapter = _adapter(config=config)
    monkeypatch.setattr(adapter, "load_decoder_if_not_exist", lambda model, name, idx: f"layer_{idx}")
    out = list(adapter.generate_decoder_layer(object()))
    assert out == [("model.layers.0", "layer_0"), ("model.layers.1", "layer_1")]


def test_generate_decoder_layer_given_three_layers_when_called_then_yield_three(monkeypatch):
    config = _make_config(num_hidden_layers=3)
    adapter = _adapter(config=config)
    monkeypatch.setattr(adapter, "load_decoder_if_not_exist", lambda model, name, idx: f"layer_{idx}")
    out = list(adapter.generate_decoder_layer(object()))
    assert len(out) == 3


def test_ascendv1_save_postprocess_given_no_quant_modules_when_called_then_skip(monkeypatch, tmp_path):
    adapter = _adapter()
    with patch.object(target, "json_safe_load") as mock_load, patch.object(target, "json_safe_dump") as mock_dump:
        adapter.ascendv1_save_postprocess(nn.Linear(2, 2), str(tmp_path))
    mock_load.assert_not_called()
    mock_dump.assert_not_called()


def test_ascendv1_save_postprocess_given_w4a8_module_when_called_then_write_w8a8_dynamic_config(monkeypatch, tmp_path):
    adapter = _adapter()

    class _FakeW4A8(target.qir.W4A8DynamicFakeQuantLinear):
        def __init__(self):
            nn.Module.__init__(self)

    fake_w4a8_module = _FakeW4A8()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = fake_w4a8_module

    model = _Model()
    config_path = os.path.join(str(tmp_path), "config.json")

    def fake_load(p, check_user_stat=False):
        return {"existing": "value"}

    def fake_dump(data, p, indent=2, check_user_stat=False):
        with open(p, "w", encoding="utf-8") as f:
            import json

            json.dump(data, f)

    with (
        patch.object(target, "json_safe_load", side_effect=fake_load),
        patch.object(target, "json_safe_dump", side_effect=fake_dump),
    ):
        adapter.ascendv1_save_postprocess(model, str(tmp_path))

    import json

    with open(config_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["quantize"] == "w8a8_dynamic"
    assert saved["moe_quantize"] == "w4a8_dynamic"
    assert saved["mla_quantize"] == "w8a8_dynamic"


def test_ascendv1_save_postprocess_given_w4a8_and_c8_when_called_then_set_w8a8_for_mla(monkeypatch, tmp_path):
    adapter = _adapter()

    class _FakeW4A8(target.qir.W4A8DynamicFakeQuantLinear):
        def __init__(self):
            nn.Module.__init__(self)

    class _FakeC8(target.qir.FakeQuantActivationPerHead):
        def __init__(self):
            nn.Module.__init__(self)

    fake_w4a8_module = _FakeW4A8()
    fake_c8_module = _FakeC8()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = fake_w4a8_module
            self.b = fake_c8_module

    model = _Model()

    def fake_load(p, check_user_stat=False):
        return {}

    def fake_dump(data, p, indent=2, check_user_stat=False):
        with open(p, "w", encoding="utf-8") as f:
            import json

            json.dump(data, f)

    with (
        patch.object(target, "json_safe_load", side_effect=fake_load),
        patch.object(target, "json_safe_dump", side_effect=fake_dump),
    ):
        adapter.ascendv1_save_postprocess(model, str(tmp_path))

    import json

    config_path = os.path.join(str(tmp_path), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["mla_quantize"] == "w8a8"


def test_init_model_given_valid_path_when_called_then_return_model(monkeypatch):
    config = _make_config(num_hidden_layers=3)
    adapter = _adapter(model_path="/tmp/model", trust_remote_code=False, config=config)

    class _FakeModel:
        def __init__(self):
            self.model = SimpleNamespace(layers=[])

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return None

        def to(self, *_args, **_kwargs):
            return self

    fake_model = _FakeModel()

    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    monkeypatch.setattr(target.SafeGenerator, "get_model_from_pretrained", lambda **kwargs: fake_model)
    monkeypatch.setattr(adapter, "get_state_dict", lambda _model: {})
    monkeypatch.setattr(target, "auto_convert_module_fp8_to_bf16", lambda *args, **kwargs: None)

    out = adapter.init_model()
    assert out is fake_model
    assert config.num_hidden_layers == 3


def test_init_model_given_load_then_change_num_hidden_layers_to_one_then_restore(monkeypatch):
    config = _make_config(num_hidden_layers=5)
    adapter = _adapter(model_path="/tmp/model", trust_remote_code=False, config=config)

    seen_num_hidden_layers = []

    class _FakeModel:
        def load_state_dict(self, _state):
            return None

        def eval(self):
            return None

        def to(self, *_args, **_kwargs):
            return self

    fake_model = _FakeModel()

    def fake_get_model_from_pretrained(**kwargs):
        seen_num_hidden_layers.append(kwargs["config"].num_hidden_layers)
        return fake_model

    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    monkeypatch.setattr(target.SafeGenerator, "get_model_from_pretrained", fake_get_model_from_pretrained)
    monkeypatch.setattr(adapter, "get_state_dict", lambda _model: {})
    monkeypatch.setattr(target, "auto_convert_module_fp8_to_bf16", lambda *args, **kwargs: None)

    adapter.init_model()
    assert seen_num_hidden_layers[0] == 1
    assert config.num_hidden_layers == 5


def test_generate_model_forward_given_dict_inputs_when_called_then_yield_process_request(monkeypatch):
    adapter = _adapter()

    class _FirstBlock(nn.Module):
        def forward(self, *args, **kwargs):
            return None

    class _M:
        def __init__(self):
            self.model = SimpleNamespace(layers=[_FirstBlock()])

        def __call__(self, *args, **kwargs):
            return self.model.layers[0](*args, **kwargs)

    model = _M()

    class _Decoder(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (hidden_states + 1,)

    monkeypatch.setattr(adapter, "generate_decoder_layer", lambda _m: iter([("model.layers.0", _Decoder())]))
    monkeypatch.setattr(target.dist, "is_initialized", lambda: False)

    gen = adapter.generate_model_forward(model, {"input_ids": torch.tensor([[1, 2]])})
    req = next(gen)
    assert req.name == "model.layers.0"


def test_generate_model_forward_given_first_block_returns_none_when_called_then_raise_invalid_model_error(monkeypatch):
    adapter = _adapter()

    class _FirstBlock(nn.Module):
        def forward(self, *args, **kwargs):
            return None

    class _M:
        def __init__(self):
            self.model = SimpleNamespace(layers=[_FirstBlock()])

        def __call__(self, *args, **kwargs):
            return None

    model = _M()
    monkeypatch.setattr(adapter, "generate_decoder_layer", lambda _m: iter([]))
    monkeypatch.setattr(target.dist, "is_initialized", lambda: False)

    gen = adapter.generate_model_forward(model, torch.tensor([[1, 2]]))
    with pytest.raises(InvalidModelError):
        next(gen)


def test_generate_model_forward_given_list_inputs_when_called_then_pass_first_element(monkeypatch):
    adapter = _adapter()

    class _FirstBlock(nn.Module):
        def forward(self, *args, **kwargs):
            return None

    class _M:
        def __init__(self):
            self.model = SimpleNamespace(layers=[_FirstBlock()])
            self.received = {}

        def __call__(self, *args, **kwargs):
            self.received["args"] = args
            self.received["kwargs"] = kwargs
            return self.model.layers[0](*args, **kwargs)

    model = _M()

    class _Decoder(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (hidden_states,)

    monkeypatch.setattr(adapter, "generate_decoder_layer", lambda _m: iter([("model.layers.0", _Decoder())]))
    monkeypatch.setattr(target.dist, "is_initialized", lambda: False)

    inputs = [torch.tensor([[1, 2]]), "other"]
    gen = adapter.generate_model_forward(model, inputs)
    next(gen)
    assert torch.equal(model.received["args"][0], inputs[0])


def test_generate_model_forward_given_tensor_inputs_when_called_then_pass_tensor(monkeypatch):
    adapter = _adapter()

    class _FirstBlock(nn.Module):
        def forward(self, *args, **kwargs):
            return None

    class _M:
        def __init__(self):
            self.model = SimpleNamespace(layers=[_FirstBlock()])
            self.received = {}

        def __call__(self, *args, **kwargs):
            self.received["args"] = args
            self.received["kwargs"] = kwargs
            return self.model.layers[0](*args, **kwargs)

    model = _M()

    class _Decoder(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (hidden_states,)

    monkeypatch.setattr(adapter, "generate_decoder_layer", lambda _m: iter([("model.layers.0", _Decoder())]))
    monkeypatch.setattr(target.dist, "is_initialized", lambda: False)

    inputs = torch.tensor([[1, 2]])
    gen = adapter.generate_model_forward(model, inputs)
    next(gen)
    assert torch.equal(model.received["args"][0], inputs)
