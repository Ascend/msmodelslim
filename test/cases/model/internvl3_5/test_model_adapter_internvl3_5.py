from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.internvl3_5 import model_adapter as target
from msmodelslim.model.internvl3_5.model_adapter import InternVL3_5ModelAdapter
from msmodelslim.utils.exception import InvalidModelError


def _adapter(**kwargs):
    a = InternVL3_5ModelAdapter.__new__(InternVL3_5ModelAdapter)
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def _make_model(vision_model=None, language_model=None, downsample_ratio=0.5, mlp1=object()):
    embed_tokens = nn.Embedding(10, 4)

    if vision_model is None:
        vision_model = SimpleNamespace(
            embeddings=SimpleNamespace(patch_embedding=SimpleNamespace(weight=torch.ones(1, dtype=torch.float32)))
        )

    if language_model is None:
        language_model = SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=embed_tokens,
                rotary_emb=lambda hidden_states, position_ids: (hidden_states, position_ids),
                layers=nn.ModuleList([nn.Identity()]),
            ),
        )
    if not callable(getattr(language_model, "get_input_embeddings", None)):
        language_model.get_input_embeddings = lambda _lm=language_model: _lm.model.embed_tokens

    class M:
        def __init__(self):
            self.vision_model = vision_model
            self.language_model = language_model
            self.mlp1 = mlp1
            self.downsample_ratio = downsample_ratio
            self.config = SimpleNamespace(
                llm_config=SimpleNamespace(num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=4),
            )

        def pixel_shuffle(self, vit_embeds, scale_factor=0.5):
            return vit_embeds

    return M()


class _FakeSafeOpen:
    def __init__(self, collector=None):
        self.collector = collector

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_tensor(self, name):
        if self.collector is not None:
            self.collector.append(name)
        return (
            torch.ones((2, 2), dtype=torch.float32) if name.endswith("weight") else torch.ones(2, dtype=torch.float32)
        )


@pytest.mark.parametrize(
    "fn, expected",
    [
        (lambda a: a.get_model_pedigree(), "internvl3_5"),
        (lambda a: a.get_layer_wise_offload_device(), "meta"),
    ],
)
def test_basic_getters(fn, expected):
    assert fn(_adapter()) == expected


def test_get_model_type_given_model_type_when_called_then_return_model_type():
    assert _adapter(model_type="internvl").get_model_type() == "internvl"


def test_num_image_token_given_config_when_called_then_return_expected():
    config = SimpleNamespace(
        force_image_size=448,
        vision_config=SimpleNamespace(image_size=448, patch_size=14),
        downsample_ratio=0.5,
    )
    adapter = _adapter(config=config)
    expected = int((448 // 14) ** 2 * (0.5**2))
    assert adapter.num_image_token == expected


def test_num_image_token_given_no_force_image_size_when_called_then_use_vision_config():
    config = SimpleNamespace(
        force_image_size=None,
        vision_config=SimpleNamespace(image_size=336, patch_size=14),
        downsample_ratio=0.5,
    )
    adapter = _adapter(config=config)
    expected = int((336 // 14) ** 2 * (0.5**2))
    assert adapter.num_image_token == expected


def test_enable_kv_cache_given_need_flag_when_called_then_set_use_cache():
    model = SimpleNamespace(config=SimpleNamespace(llm_config=SimpleNamespace(use_cache=False)))
    _adapter().enable_kv_cache(model, True)
    assert model.config.llm_config.use_cache is True


def test_generate_decoder_layer_given_num_layers_when_called_then_return_all_layers():
    adapter = _adapter(config=SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=2)))
    adapter._load_decoder_if_not_exist = lambda model, name, idx: f"layer_{idx}"
    assert list(adapter.generate_decoder_layer(object())) == [
        ("language_model.model.layers.0", "layer_0"),
        ("language_model.model.layers.1", "layer_1"),
    ]


def test_get_weight_map_given_index_json_when_loaded_then_return_weight_map(monkeypatch):
    adapter = _adapter(model_path="/tmp/model")
    monkeypatch.setattr(target, "json_safe_load", lambda p: {"weight_map": {"a": "f"}})
    assert adapter._get_weight_map() == {"a": "f"}


def test_get_weight_map_given_lru_cache_when_called_twice_then_return_cached(monkeypatch):
    adapter = _adapter(model_path="/tmp/model")
    call_count = {"n": 0}

    def fake_load(p):
        call_count["n"] += 1
        return {"weight_map": {"a": "f"}}

    monkeypatch.setattr(target, "json_safe_load", fake_load)
    result1 = adapter._get_weight_map()
    result2 = adapter._get_weight_map()
    assert result1 == result2
    assert call_count["n"] == 1


@pytest.mark.parametrize(
    "weight_map,prefix,expected_names",
    [
        ({"weight": "a.safetensors", "bias": "a.safetensors"}, "", None),
        (
            {
                "language_model.model.layers.0.weight": "a.safetensors",
                "language_model.model.layers.0.bias": "a.safetensors",
            },
            "language_model.model.layers.0",
            {"language_model.model.layers.0.weight", "language_model.model.layers.0.bias"},
        ),
    ],
)
def test_get_state_dict_paths(monkeypatch, weight_map, prefix, expected_names):
    adapter = _adapter(model_path="/tmp/model")
    adapter._get_weight_map = lambda: weight_map
    module = nn.Linear(2, 2)
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)

    called = []
    monkeypatch.setattr(target, "safe_open", lambda *args, **kwargs: _FakeSafeOpen(called if expected_names else None))

    out = adapter._get_state_dict(module, prefix=prefix)
    assert "weight" in out and "bias" in out
    if expected_names is not None:
        assert expected_names.issubset(set(called))


def test_load_decoder_if_not_exist_given_loaded_decoder_when_access_ok_then_return_loaded():
    adapter = _adapter()

    class L(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = SimpleNamespace(weight=torch.ones(1))

    layer = L()

    class M:
        def get_submodule(self, _name):
            return layer

    assert adapter._load_decoder_if_not_exist(M(), "language_model.model.layers.0", 0) is layer


def test_load_decoder_if_not_exist_given_meta_like_decoder_when_called_then_replace_in_module_list(monkeypatch):
    adapter = _adapter(config=SimpleNamespace(llm_config=SimpleNamespace()), model_path="/tmp/model")
    adapter._model_torch_dtype = torch.float32

    class DummyLayer(nn.Module):
        def __init__(self, config=None, layer_idx=0):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(1)

    class MetaWeight:
        @property
        def device(self):
            raise RuntimeError("meta")

    loaded_decoder = SimpleNamespace(input_layernorm=SimpleNamespace(weight=MetaWeight()))
    module_list = [DummyLayer()]

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.other = nn.Linear(1, 1)

        def get_submodule(self, _name):
            return loaded_decoder

        language_model = SimpleNamespace(model=SimpleNamespace(layers=module_list))

    def fake_get_state_dict(module, prefix=""):
        sd = {}
        for name, param in module.named_parameters():
            sd[name] = torch.zeros_like(param.data)
        return sd

    monkeypatch.setattr(adapter, "_get_state_dict", fake_get_state_dict)
    monkeypatch.setattr(target, "Qwen3DecoderLayer", DummyLayer)

    with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
        out = adapter._load_decoder_if_not_exist(M(), "language_model.model.layers.0", 0)
        assert isinstance(out, DummyLayer)
        assert isinstance(module_list[0], DummyLayer)


def test_load_decoder_if_not_exist_given_missing_layer_when_called_then_create_new(monkeypatch):
    adapter = _adapter(config=SimpleNamespace(llm_config=SimpleNamespace()), model_path="/tmp/model")
    adapter._model_torch_dtype = torch.float32

    class DummyLayer(nn.Module):
        def __init__(self, config=None, layer_idx=0):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(1)

    module_list = nn.ModuleList()

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.other = nn.Linear(1, 1)

        def get_submodule(self, _name):
            raise AttributeError("missing")

        language_model = SimpleNamespace(model=SimpleNamespace(layers=module_list))

    def fake_get_state_dict(module, prefix=""):
        sd = {}
        for name, param in module.named_parameters():
            sd[name] = torch.zeros_like(param.data)
        return sd

    monkeypatch.setattr(adapter, "_get_state_dict", fake_get_state_dict)
    monkeypatch.setattr(target, "Qwen3DecoderLayer", DummyLayer)

    with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
        out = adapter._load_decoder_if_not_exist(M(), "language_model.model.layers.0", 0)
        assert isinstance(out, DummyLayer)


def test_generate_model_visit_given_model_when_called_then_yield_vision_and_decoder(monkeypatch):
    adapter = _adapter()
    adapter.generate_decoder_layer = lambda model: iter([("language_model.model.layers.0", nn.Identity())])

    def fake_generated_decoder_layer_visit_func(_model, transformer_blocks):
        for name, layer in transformer_blocks:
            yield ProcessRequest(name=name, module=layer, args=(), kwargs={})

    monkeypatch.setattr(target, "generated_decoder_layer_visit_func", fake_generated_decoder_layer_visit_func)
    requests = list(adapter.generate_model_visit(SimpleNamespace(vision_model=object())))

    assert requests[0].name == "vision_model"
    assert requests[1].name == "language_model.model.layers.0"


def test_generate_model_forward_given_no_pixel_values_when_called_then_skip_vision(monkeypatch):
    adapter = _adapter()
    adapter._img_context_token_id = None

    class EchoLayer(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    adapter.generate_decoder_layer = lambda model: iter([("language_model.model.layers.0", EchoLayer())])
    monkeypatch.setattr(target, "create_causal_mask", lambda *args, **kwargs: torch.ones((1, 1, 2, 2)))

    model = _make_model()
    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "pixel_values": None,
    }

    gen = adapter.generate_model_forward(model, inputs)
    req = next(gen)
    assert req.name == "language_model.model.layers.0"


def test_generate_model_forward_given_pixel_values_when_called_then_yield_vision_then_decoder(monkeypatch):
    adapter = _adapter()
    adapter._img_context_token_id = 3

    class EchoLayer(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    adapter.generate_decoder_layer = lambda model: iter([("language_model.model.layers.0", EchoLayer())])
    monkeypatch.setattr(target, "create_causal_mask", lambda *args, **kwargs: torch.ones((1, 1, 2, 2)))

    model = _make_model()

    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        "image_flags": torch.tensor([[1]], dtype=torch.long),
    }

    gen = adapter.generate_model_forward(model, inputs)
    req = next(gen)
    assert req.name == "vision_model"


def test_generate_model_forward_given_list_inputs_when_called_then_use_first_sample(monkeypatch):
    adapter = _adapter()
    adapter._img_context_token_id = None

    class EchoLayer(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    adapter.generate_decoder_layer = lambda model: iter([("language_model.model.layers.0", EchoLayer())])
    monkeypatch.setattr(target, "create_causal_mask", lambda *args, **kwargs: torch.ones((1, 1, 2, 2)))

    model = _make_model()
    inputs = [
        {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "pixel_values": None,
        }
    ]

    gen = adapter.generate_model_forward(model, inputs)
    req = next(gen)
    assert req.name == "language_model.model.layers.0"


def test_generate_model_forward_given_decoder_returns_tuple_when_called_then_unwrap_hidden_states(monkeypatch):
    adapter = _adapter()
    adapter._img_context_token_id = None

    class TupleLayer(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (hidden_states + 1,)

    adapter.generate_decoder_layer = lambda model: iter(
        [
            ("language_model.model.layers.0", TupleLayer()),
            ("language_model.model.layers.1", nn.Identity()),
        ]
    )
    monkeypatch.setattr(target, "create_causal_mask", lambda *args, **kwargs: torch.ones((1, 1, 2, 2)))

    model = _make_model()
    inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "pixel_values": None,
    }

    gen = adapter.generate_model_forward(model, inputs)
    assert next(gen).name == "language_model.model.layers.0"
    assert gen.send((torch.ones((1, 2, 4), dtype=torch.float32),)).name == "language_model.model.layers.1"


def test_get_adapter_config_for_subgraph_given_layers_when_called_then_return_configs():
    adapter = _adapter(
        config=SimpleNamespace(
            llm_config=SimpleNamespace(
                num_hidden_layers=2,
                num_attention_heads=8,
                num_key_value_heads=4,
            )
        )
    )

    out = adapter.get_adapter_config_for_subgraph()
    ov = [cfg for cfg in out if cfg.subgraph_type == "ov"]
    norm = [cfg for cfg in out if cfg.subgraph_type == "norm-linear"]
    up_down = [cfg for cfg in out if cfg.subgraph_type == "up-down"]

    assert len(out) == 8
    assert len(ov) == 2
    assert len(norm) == 4
    assert len(up_down) == 2
    assert ov[0].mapping.source == "language_model.model.layers.0.self_attn.v_proj"
    assert ov[0].mapping.targets == ["language_model.model.layers.0.self_attn.o_proj"]


def test_get_ln_fuse_map_given_layers_when_called_then_return_map():
    adapter = _adapter(config=SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=2)))

    first_map, ln_linear_map = adapter.get_ln_fuse_map()

    assert first_map == {}
    assert "language_model.model.layers.0.input_layernorm" in ln_linear_map
    assert "language_model.model.layers.0.post_attention_layernorm" in ln_linear_map
    assert "language_model.model.layers.1.input_layernorm" in ln_linear_map
    assert "language_model.model.norm" in ln_linear_map
    assert ln_linear_map["language_model.model.norm"] == ["language_model.lm_head"]


def test_get_bake_names_given_when_called_then_return_empty():
    adapter = _adapter()
    bake_in, bake_out = adapter.get_bake_names()
    assert bake_in == []
    assert bake_out == []


def test_get_rotate_map_given_block_size_when_called_then_return_pairs():
    adapter = _adapter(
        config=SimpleNamespace(
            llm_config=SimpleNamespace(num_hidden_layers=2, hidden_size=4096, head_dim=128),
        )
    )

    pre_run_list, rot_pairs = adapter.get_rotate_map(block_size=128)

    assert len(pre_run_list) == 1
    pre_run = pre_run_list[0]
    assert "language_model.model.embed_tokens" in pre_run.right_rot
    assert "mlp1.3" in pre_run.left_rot

    assert len(rot_pairs) == 2
    rot = rot_pairs[0]
    rot_uv = rot_pairs[1]
    assert "language_model.lm_head" in rot.right_rot
    assert "language_model.model.layers.0.self_attn.o_proj" in rot.left_rot
    assert "language_model.model.layers.0.self_attn.v_proj" in rot_uv.left_rot
    assert "language_model.model.layers.0.self_attn.o_proj" in rot_uv.right_rot


def test_init_model_given_load_raises_when_called_then_raise_invalid_model_error(monkeypatch):
    adapter = _adapter(
        model_path="/tmp/model",
        trust_remote_code=False,
        config=SimpleNamespace(
            llm_config=SimpleNamespace(num_hidden_layers=4, use_cache=False, _attn_implementation="eager"),
            vision_config=SimpleNamespace(num_hidden_layers=2, image_size=448),
            force_image_size=None,
            downsample_ratio=0.5,
        ),
    )
    adapter._model_torch_dtype = torch.float32

    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    monkeypatch.setattr(
        target.AutoModel, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(InvalidModelError):
        adapter.init_model(DeviceType.CPU)


def test_init_model_given_success_when_called_then_restore_layers_and_set_heads(monkeypatch):
    config = SimpleNamespace(
        llm_config=SimpleNamespace(
            num_hidden_layers=4,
            use_cache=True,
            _attn_implementation="eager",
            num_attention_heads=16,
            num_key_value_heads=8,
        ),
        vision_config=SimpleNamespace(num_hidden_layers=2, image_size=448),
        force_image_size=None,
        downsample_ratio=0.5,
    )
    adapter = _adapter(model_path="/tmp/model", trust_remote_code=False, config=config)
    adapter._model_torch_dtype = torch.float32

    class FakeModel:
        def __init__(self):
            self.vision_model = object()
            self.mlp1 = object()
            self.language_model = SimpleNamespace(lm_head=object())
            self.config = SimpleNamespace(
                llm_config=SimpleNamespace(
                    num_attention_heads=16,
                    num_key_value_heads=8,
                ),
            )

        def load_state_dict(self, _state_dict, **_kwargs):
            return [], []

        def eval(self):
            return self

    fake_model = FakeModel()
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    monkeypatch.setattr(target.AutoModel, "from_pretrained", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(adapter, "_get_state_dict", lambda model: {})

    out = adapter.init_model(DeviceType.CPU)
    assert out is fake_model
    assert adapter.config.llm_config.num_hidden_layers == 4
    assert adapter.config.llm_config.use_cache is False
    assert fake_model.config.num_attention_heads == 16
    assert fake_model.config.num_key_value_heads == 8


def test_handle_dataset_given_valid_item_with_image_when_called_then_return_processed_data(monkeypatch):
    adapter = _adapter(
        model_path=Path("."),
        trust_remote_code=False,
        config=SimpleNamespace(
            force_image_size=None,
            vision_config=SimpleNamespace(image_size=448, patch_size=14),
            downsample_ratio=0.5,
        ),
    )
    adapter._img_context_token_id = None

    class DummyTokenizer:
        padding_side = "right"

        def convert_tokens_to_ids(self, token):
            return 5

        def __call__(self, query, return_tensors="pt", padding=True):
            return {
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    monkeypatch.setattr(
        target, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyTokenizer())
    )
    monkeypatch.setattr(target, "load_image", lambda *args, **kwargs: torch.ones((1, 3, 448, 448)))
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    adapter._collect_inputs_to_device = lambda inputs, device, keys, defaults: {
        k: inputs[k] for k in keys if k in inputs
    }

    sample = SimpleNamespace(image="a.jpg", text="hello")
    out = adapter.handle_dataset([sample], DeviceType.CPU)
    assert isinstance(out, list)
    assert len(out) == 1


def test_handle_dataset_given_valid_item_without_image_when_called_then_return_processed_data(monkeypatch):
    adapter = _adapter(
        model_path=Path("."),
        trust_remote_code=False,
        config=SimpleNamespace(
            force_image_size=None,
            vision_config=SimpleNamespace(image_size=448, patch_size=14),
            downsample_ratio=0.5,
        ),
    )
    adapter._img_context_token_id = None

    class DummyTokenizer:
        padding_side = "right"

        def convert_tokens_to_ids(self, token):
            return 5

        def __call__(self, query, return_tensors="pt", padding=True):
            return {
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    monkeypatch.setattr(
        target, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyTokenizer())
    )
    adapter._collect_inputs_to_device = lambda inputs, device, keys, defaults: {
        k: inputs[k] for k in keys if k in inputs
    }

    sample = SimpleNamespace(image=None, text="hello")
    out = adapter.handle_dataset([sample], DeviceType.CPU)
    assert isinstance(out, list)
    assert len(out) == 1
