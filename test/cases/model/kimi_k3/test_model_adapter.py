# -*- coding: UTF-8 -*-
"""Unit tests for KimiK3ModelAdapter."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.common.utils import _get_expert_range
from msmodelslim.model.interface_hub import (
    AttentionAnalysisInterface,
    FA3QuantAdapterInterface,
    FA3QuantPlaceHolder,
)
from msmodelslim.model.kimi_k3 import model_adapter as target
from msmodelslim.model.kimi_k3.model_adapter import KimiK3ModelAdapter, default_dtype
from msmodelslim.processor.quant.fa3.processor import FA3QuantProcessor, FA3QuantProcessorConfig
from msmodelslim.utils.exception import UnsupportedError


def _adapter(**kwargs):
    a = KimiK3ModelAdapter.__new__(KimiK3ModelAdapter)
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def _text_config(**overrides):
    cfg = dict(
        num_hidden_layers=4,
        first_k_dense_replace=1,
        linear_attn_config={"kda_layers": [2], "use_full_rank_gate": False},
        num_attention_heads=8,
        num_key_value_heads=4,
        qk_nope_head_dim=128,
        v_head_dim=128,
        num_experts=4,
        routed_expert_hidden_size=256,
        latent_moe_use_norm=True,
        mla_use_output_gate=True,
        hidden_size=16,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _forward_model(mm_projector=object(), vision_dtype=torch.float32, use_attn_residuals=False):
    class Backbone:
        def __init__(self):
            self.embed_tokens = nn.Embedding(10, 4)
            self.config = SimpleNamespace()
            self.use_attn_residuals = use_attn_residuals

        def _update_linear_attn_mask(self, attention_mask, cache_position):
            return torch.ones((1, 1, 2, 2))

    class M:
        def __init__(self):
            self.language_model = SimpleNamespace(model=Backbone())
            self.vision_tower = SimpleNamespace(
                patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=torch.ones(1, dtype=vision_dtype)))
            )
            self.mm_projector = mm_projector

        def _merge_input_ids_with_image_features(self, **kwargs):
            ids = kwargs["input_ids"]
            return kwargs["inputs_embeds"], kwargs["attention_mask"], None, torch.arange(ids.shape[1]).unsqueeze(0)

    return M()


def _init_fake_model(with_heads=True):
    text_config = SimpleNamespace(num_attention_heads=16, num_key_value_heads=8) if with_heads else SimpleNamespace()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = nn.Identity()
            self.mm_projector = None
            self.language_model = SimpleNamespace()
            self.config = SimpleNamespace(text_config=text_config)

        def eval(self):
            return self

    return FakeModel()


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
        (lambda a: a.get_model_pedigree(), "kimi_k3"),
        (lambda a: a.get_layer_wise_offload_device(), "meta"),
    ],
)
def test_basic_getters(fn, expected):
    assert fn(_adapter()) == expected


def test_get_model_type_given_model_type_when_called_then_return_model_type():
    assert _adapter(model_type="kimi").get_model_type() == "kimi"


def test_default_dtype_given_dtype_when_context_exit_then_restore_original():
    original = torch.get_default_dtype()
    with default_dtype(torch.bfloat16):
        assert torch.get_default_dtype() == torch.bfloat16
    assert torch.get_default_dtype() == original


@pytest.mark.parametrize(
    "sample",
    [SimpleNamespace(image=None, text="hi"), SimpleNamespace(image="a.jpg", text=None)],
)
def test_handle_dataset_given_item_missing_modality_then_raise_unsupported_error(monkeypatch, sample):
    adapter = _adapter(model_path=Path("."), trust_remote_code=False)
    adapter._collect_inputs_to_device = lambda *args, **kwargs: {}

    class DummyProcessor:
        def __call__(self, messages=None, return_tensors="pt"):
            return {"input_ids": torch.ones((1, 2), dtype=torch.long)}

    monkeypatch.setattr(target.AutoProcessor, "from_pretrained", lambda *args, **kwargs: DummyProcessor())
    with pytest.raises(UnsupportedError):
        adapter.handle_dataset([sample], DeviceType.CPU)


@pytest.mark.parametrize(
    "tokenizer_factory",
    [
        lambda: (_ for _ in ()).throw(RuntimeError("t")),
        object,
    ],
)
def test_handle_dataset_given_processor_fails_then_raise_unsupported_error(monkeypatch, tokenizer_factory):
    adapter = _adapter(model_path=Path("."), trust_remote_code=False)
    monkeypatch.setattr(
        target.AutoProcessor, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("p"))
    )
    monkeypatch.setattr(target.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer_factory())
    with pytest.raises(UnsupportedError):
        adapter.handle_dataset([], DeviceType.CPU)


def test_handle_dataset_given_valid_item_when_called_then_return_processed_data(monkeypatch):
    adapter = _adapter(model_path=Path("."), trust_remote_code=False)

    class DummyProcessor:
        def __call__(self, messages=None, return_tensors="pt"):
            return {
                "input_ids": torch.ones((1, 2), dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
                "grid_thws": torch.ones((1, 3), dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    monkeypatch.setattr(target.AutoProcessor, "from_pretrained", lambda *args, **kwargs: DummyProcessor())
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    adapter._collect_inputs_to_device = lambda inputs, device, keys, defaults: {
        k: inputs[k] for k in keys if k in inputs
    }
    out = adapter.handle_dataset([SimpleNamespace(image="a.jpg", text="hello")], DeviceType.CPU)
    assert isinstance(out, list)
    assert len(out) == 1
    assert "input_ids" in out[0]
    assert "pixel_values" in out[0]


def test_strip_quantization_config_with_and_without_attr():
    text = SimpleNamespace(quantization_config={"bits": 4})
    cfg = SimpleNamespace(quantization_config={"bits": 8}, text_config=text)
    adapter = _adapter(config=cfg)
    adapter._strip_quantization_config()
    assert not hasattr(cfg, "quantization_config")
    assert not hasattr(text, "quantization_config")

    adapter2 = _adapter(config=SimpleNamespace(text_config=SimpleNamespace()))
    adapter2._strip_quantization_config()  # no-op when attrs missing


def test_generate_decoder_layer_given_num_layers_when_called_then_return_all_layers():
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2)))
    adapter._load_decoder_if_not_exist = lambda model, name, idx: f"layer_{idx}"
    assert list(adapter.generate_decoder_layer(object())) == [
        ("language_model.model.layers.0", "layer_0"),
        ("language_model.model.layers.1", "layer_1"),
    ]


def test_generate_model_visit_given_model_when_called_then_yield_vision_mm_and_decoder(monkeypatch):
    adapter = _adapter()
    adapter.generate_decoder_layer = lambda model: iter([("language_model.model.layers.0", nn.Identity())])

    def fake_generated_decoder_layer_visit_func(_model, transformer_blocks):
        for name, layer in transformer_blocks:
            yield target.ProcessRequest(name=name, module=layer, args=(), kwargs={})

    monkeypatch.setattr(target, "generated_decoder_layer_visit_func", fake_generated_decoder_layer_visit_func)
    requests = list(adapter.generate_model_visit(SimpleNamespace(vision_tower=object(), mm_projector=object())))

    assert requests[0].name == "vision_tower"
    assert requests[1].name == "mm_projector"
    assert requests[2].name == "language_model.model.layers.0"


def test_enable_kv_cache_given_need_flag_when_called_then_set_use_cache():
    model = SimpleNamespace(
        config=SimpleNamespace(use_cache=False),
        language_model=SimpleNamespace(config=SimpleNamespace(use_cache=False)),
    )
    _adapter().enable_kv_cache(model, True)
    assert model.config.use_cache is True
    assert model.language_model.config.use_cache is True


def test_patch_mm_projector_rot_proj_adds_identity_and_wraps_forward():
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4)))

    class MM(nn.Module):
        def forward(self, x):
            return x

    class ListMM(nn.Module):
        def forward(self, x):
            return [x, x + 1]

    model = SimpleNamespace(mm_projector=MM())
    adapter._patch_mm_projector_rot_proj(model)
    assert hasattr(model.mm_projector, "rot_proj")
    assert isinstance(model.mm_projector.rot_proj, nn.Linear)
    assert torch.allclose(model.mm_projector.rot_proj.weight, torch.eye(4))
    x = torch.randn(2, 4)
    assert torch.allclose(model.mm_projector(x), x)

    # already has rot_proj → early return
    adapter._patch_mm_projector_rot_proj(model)

    model_list = SimpleNamespace(mm_projector=ListMM())
    adapter._patch_mm_projector_rot_proj(model_list)
    outs = model_list.mm_projector(x)
    assert isinstance(outs, list) and len(outs) == 2

    model_none = SimpleNamespace(mm_projector=None)
    adapter._patch_mm_projector_rot_proj(model_none)


class TestQuaRotMaps:
    def test_get_ln_fuse_map_patches_builder(self, monkeypatch):
        config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2))
        adapter = _adapter(config=config)
        fake = ({"pre": ["run"]}, {"k": ["v"]})
        monkeypatch.setattr(target, "build_ln_fuse_map", lambda c, num_hidden_layers=None: fake)
        assert adapter.get_ln_fuse_map() is fake

    def test_get_rotate_map_patches_builder(self, monkeypatch):
        captured = {}
        config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=3))
        adapter = _adapter(config=config)
        adapter.enable_rot = True
        adapter.enable_rot_b_proj = False
        adapter.enable_rot_kv_b_proj = True
        adapter.enable_rot_latent = False

        def fake_build(c, block_size, num_hidden_layers=None, **kwargs):
            captured["block_size"] = block_size
            captured["num_hidden_layers"] = num_hidden_layers
            captured.update(kwargs)
            return ["pre"], ["rot"]

        monkeypatch.setattr(target, "build_rotate_map", fake_build)
        assert adapter.get_rotate_map(64) == (["pre"], ["rot"])
        assert captured["block_size"] == 64
        assert captured["num_hidden_layers"] == 3
        assert captured["enable_rot_b_proj"] is False
        assert captured["enable_rot_latent"] is False

    def test_get_bake_names_returns_empty_lists(self):
        assert _adapter().get_bake_names() == ([], [])


class TestSubgraphConfigs:
    def test_mla_kda_ffn_and_get_adapter_config(self, monkeypatch):
        # layer0: MLA + dense; layer1: KDA + MoE; layers 2-3: MLA + MoE
        text = _text_config()
        adapter = _adapter(config=SimpleNamespace(text_config=text))
        monkeypatch.setattr(target, "_get_expert_range", lambda cfg: (0, cfg.num_experts))

        mla = adapter._mla_subgraph_configs(0)
        assert [c.subgraph_type for c in mla] == ["ov", "norm-linear", "norm-linear"]
        assert mla[0].mapping.source == "language_model.model.layers.0.self_attn.kv_b_proj"
        assert mla[0].mapping.targets == ["language_model.model.layers.0.self_attn.o_proj"]
        assert mla[0].fusion.num_attention_heads == 8
        assert mla[0].fusion.custom_config["qk_nope_head_dim"] == 128
        assert any(t.endswith(".g_proj") for t in mla[1].mapping.targets)  # mla_use_output_gate

        kda = adapter._kda_subgraph_configs(1)
        assert [c.subgraph_type for c in kda] == ["norm-linear", "linear-linear", "linear-linear"]
        assert kda[1].mapping.source.endswith(".f_a_proj")
        assert kda[1].mapping.targets == [kda[1].mapping.source.replace("f_a_proj", "f_b_proj")]
        assert kda[2].mapping.source.endswith(".g_a_proj")

        # full-rank gate skips g_a→g_b
        text_full = _text_config(linear_attn_config={"kda_layers": [2], "use_full_rank_gate": True})
        adapter_full = _adapter(config=SimpleNamespace(text_config=text_full))
        kda_full = adapter_full._kda_subgraph_configs(1)
        assert len(kda_full) == 2

        dense_ffn = adapter._ffn_subgraph_configs(0)
        assert [c.subgraph_type for c in dense_ffn] == ["norm-linear", "up-down"]
        assert dense_ffn[1].mapping.source == "language_model.model.layers.0.mlp.up_proj"

        moe_ffn = adapter._ffn_subgraph_configs(2)
        types = [c.subgraph_type for c in moe_ffn]
        assert types[0] == "norm-linear"
        assert types[1] == "up-down"
        assert moe_ffn[1].mapping.source.endswith("block_sparse_moe.shared_experts.up_proj")
        assert types[2] == "norm-linear"  # routed_expert_norm
        assert moe_ffn[2].mapping.source.endswith("routed_expert_norm")
        expert_cfgs = [c for c in moe_ffn if c.subgraph_type == "up-down" and ".experts." in c.mapping.source]
        assert len(expert_cfgs) == 4
        assert expert_cfgs[0].mapping.source.endswith(".w3")
        assert expert_cfgs[0].mapping.targets[0].endswith(".w2")

        out = adapter.get_adapter_config_for_subgraph()
        # layer0: 3 MLA + 2 dense FFN = 5
        # layer1: 3 KDA + (1 post + 1 shared + 1 routed_norm + 4 experts) = 10
        # layer2/3: 3 MLA + 7 MoE = 10 each
        assert len(out) == 5 + 10 + 10 + 10

    def test_ffn_moe_without_latent_norm_skips_routed_norm(self, monkeypatch):
        text = _text_config(first_k_dense_replace=0, latent_moe_use_norm=False, num_hidden_layers=1)
        adapter = _adapter(config=SimpleNamespace(text_config=text))
        monkeypatch.setattr(target, "_get_expert_range", lambda cfg: (0, 2))
        ffn = adapter._ffn_subgraph_configs(0)
        assert not any("routed_expert_norm" in (c.mapping.source or "") for c in ffn)


def test_init_model_main_path(monkeypatch):
    config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=8),
        vision_config=SimpleNamespace(vt_num_hidden_layers=2),
        use_cache=True,
    )
    adapter = _adapter(model_path="/tmp/model", trust_remote_code=False, config=config)
    fake_model = _init_fake_model(with_heads=True)

    calls = {"ep": 0, "runtime": 0, "dequant": 0, "rot": 0, "load": 0}

    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)
    monkeypatch.setattr(
        target,
        "apply_kimi_k3_ep_patches",
        lambda **kwargs: calls.__setitem__("ep", calls["ep"] + 1),
    )
    monkeypatch.setattr(
        target,
        "apply_kimi_k3_runtime_patches",
        lambda m: calls.__setitem__("runtime", calls["runtime"] + 1),
    )
    monkeypatch.setattr(
        target.AutoModelForCausalLM,
        "from_config",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(adapter, "_get_state_dict", lambda model: {"w": torch.ones(1)})
    monkeypatch.setattr(
        adapter,
        "_load_state_dict_compatible",
        lambda m, sd: calls.__setitem__("load", calls["load"] + 1),
    )
    monkeypatch.setattr(
        target,
        "dequant_subtree_mxfp4_to_bf16",
        lambda *a, **k: calls.__setitem__("dequant", calls["dequant"] + 1),
    )
    monkeypatch.setattr(
        adapter,
        "_patch_mm_projector_rot_proj",
        lambda m: calls.__setitem__("rot", calls["rot"] + 1),
    )

    out = adapter.init_model(DeviceType.CPU)
    assert out is fake_model
    assert adapter.config.text_config.num_hidden_layers == 4
    assert adapter.config.use_cache is False
    assert calls["ep"] == 2
    assert calls["runtime"] == 1
    assert calls["dequant"] == 1
    assert calls["rot"] == 1
    assert calls["load"] == 1
    assert fake_model.config.num_attention_heads == 16
    assert fake_model.config.num_key_value_heads == 8


@pytest.mark.parametrize(
    "inputs, mm_projector, vision_dtype, use_attn_residuals, expected",
    [
        (
            {"input_ids": torch.tensor([[1, 2]], dtype=torch.long), "pixel_values": None},
            object(),
            torch.float32,
            False,
            ["language_model.model.layers.0"],
        ),
        (
            {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
                "grid_thws": torch.ones((1, 3), dtype=torch.long),
            },
            object(),
            torch.float32,
            False,
            ["vision_tower", "mm_projector", "language_model.model.layers.0"],
        ),
        (
            [
                {
                    "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                    "attention_mask": None,
                    "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
                    "grid_thws": None,
                }
            ],
            None,
            torch.float16,
            False,
            ["vision_tower", "language_model.model.layers.0"],
        ),
        (
            {
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
            },
            object(),
            torch.float32,
            False,
            ["language_model.model.layers.0"],
        ),
        (
            {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "pixel_values": torch.empty((0, 3, 2, 2), dtype=torch.float32),
            },
            object(),
            torch.float32,
            True,
            ["language_model.model.layers.0"],
        ),
    ],
)
def test_generate_model_forward_paths(monkeypatch, inputs, mm_projector, vision_dtype, use_attn_residuals, expected):
    adapter = _adapter()

    class EchoLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_linear_attn = False

        def forward(self, hidden_states, **kwargs):
            if use_attn_residuals:
                return hidden_states, kwargs.get("block_residual")
            return hidden_states

    adapter.generate_decoder_layer = lambda model: iter([("language_model.model.layers.0", EchoLayer())])
    monkeypatch.setattr(target, "create_causal_mask", lambda **kwargs: torch.ones((1, 1, 2, 2)))
    monkeypatch.setattr(target.dist, "is_initialized", lambda: False)

    gen = adapter.generate_model_forward(
        _forward_model(mm_projector, vision_dtype, use_attn_residuals=use_attn_residuals), inputs
    )

    req = next(gen)
    got = [req.name]
    if req.name == "vision_tower":
        if len(expected) == 3:
            req = gen.send([torch.ones((1, 2, 4), dtype=torch.float32)])
            got.append(req.name)
            req = gen.send([torch.ones((1, 2, 4), dtype=torch.float32)])
            got.append(req.name)
        else:
            req = gen.send([torch.ones((1, 2, 4), dtype=torch.float16)])
            got.append(req.name)
    assert got == expected

    if use_attn_residuals:
        assert "block_residual" in req.kwargs


def test_generate_model_forward_unwraps_tuple_when_no_attn_residuals(monkeypatch):
    adapter = _adapter()

    class TupleLayer(nn.Module):
        is_linear_attn = False

        def forward(self, hidden_states, **kwargs):
            return (hidden_states + 1,)

    adapter.generate_decoder_layer = lambda model: iter(
        [
            ("language_model.model.layers.0", TupleLayer()),
            ("language_model.model.layers.1", nn.Identity()),
        ]
    )
    monkeypatch.setattr(target, "create_causal_mask", lambda **kwargs: torch.ones((1, 1, 2, 2)))
    monkeypatch.setattr(target.dist, "is_initialized", lambda: False)

    gen = adapter.generate_model_forward(
        _forward_model(use_attn_residuals=False),
        {"input_ids": torch.tensor([[1, 2]], dtype=torch.long), "pixel_values": None},
    )
    assert next(gen).name == "language_model.model.layers.0"
    assert gen.send((torch.ones((1, 2, 4), dtype=torch.float32),)).name == "language_model.model.layers.1"


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
    module = nn.Linear(2, 2)
    monkeypatch.setattr(target, "get_full_weight_map", lambda p: weight_map)
    monkeypatch.setattr(target, "get_valid_read_path", lambda p, **kwargs: p)

    called = []
    monkeypatch.setattr(target, "safe_open", lambda *args, **kwargs: _FakeSafeOpen(called if expected_names else None))

    out = adapter._get_state_dict(module, prefix=prefix)
    assert "weight" in out and "bias" in out
    if expected_names is not None:
        assert expected_names.issubset(set(called))


def test_load_state_dict_compatible_slices_a_log_and_skips_mismatch():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.A_log = nn.Parameter(torch.zeros(4), requires_grad=False)
            self.other = nn.Parameter(torch.zeros(2, 2))
            self.short_A_log = nn.Parameter(torch.zeros(8), requires_grad=False)

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    root = Root()
    state = {
        "self_attn.A_log": torch.arange(8, dtype=torch.float32),
        "self_attn.other": torch.ones(3, 3),  # mismatch → skip
        "self_attn.short_A_log": torch.ones(3),  # shorter A_log → skip
        "missing.extra": torch.ones(1),
    }
    _adapter()._load_state_dict_compatible(root, state)
    assert torch.allclose(root.self_attn.A_log, torch.arange(4, dtype=torch.float32))
    assert torch.allclose(root.self_attn.other, torch.zeros(2, 2))
    assert torch.allclose(root.self_attn.short_A_log, torch.zeros(8))


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


def test_load_decoder_if_not_exist_given_missing_layer_cls_when_called_then_raise_unsupported_error():
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace()), model_path="/tmp/model")

    class M:
        def get_submodule(self, _name):
            raise AttributeError("missing")

        language_model = SimpleNamespace(model=SimpleNamespace(layers=[]))

    with pytest.raises(UnsupportedError):
        adapter._load_decoder_if_not_exist(M(), "language_model.model.layers.1", 1)


def test_load_decoder_if_not_exist_given_meta_like_decoder_when_called_then_replace_in_module_list(monkeypatch):
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace()), model_path="/tmp/model")

    class DummyLayer(nn.Module):
        def __init__(self, config=None, layer_idx=0):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(1)

    class MetaWeight:
        @property
        def device(self):
            raise RuntimeError("meta")

    loaded_decoder = SimpleNamespace(input_layernorm=SimpleNamespace(weight=MetaWeight()))
    module_list = nn.ModuleList([DummyLayer()])

    class M:
        def get_submodule(self, _name):
            return loaded_decoder

        language_model = SimpleNamespace(model=SimpleNamespace(layers=module_list))

    monkeypatch.setattr(adapter, "_get_state_dict", lambda decoder, prefix="": {})
    monkeypatch.setattr(adapter, "_load_state_dict_compatible", lambda m, sd: None)
    monkeypatch.setattr(target, "dequant_subtree_mxfp4_to_bf16", lambda *args, **kwargs: None)

    out = adapter._load_decoder_if_not_exist(M(), "language_model.model.layers.0", 0)
    assert isinstance(out, DummyLayer)
    assert isinstance(module_list[0], DummyLayer)


def test_load_decoder_if_not_exist_appends_when_index_beyond_list(monkeypatch):
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace()), model_path="/tmp/model")

    class DummyLayer(nn.Module):
        def __init__(self, config=None, layer_idx=0):
            super().__init__()
            self.layer_idx = layer_idx
            self.input_layernorm = nn.LayerNorm(1)

    module_list = nn.ModuleList([DummyLayer(layer_idx=0)])

    class M:
        def get_submodule(self, _name):
            raise AttributeError("missing")

        language_model = SimpleNamespace(model=SimpleNamespace(layers=module_list))

    monkeypatch.setattr(adapter, "_get_state_dict", lambda decoder, prefix="": {})
    monkeypatch.setattr(adapter, "_load_state_dict_compatible", lambda m, sd: None)
    monkeypatch.setattr(target, "dequant_subtree_mxfp4_to_bf16", lambda *args, **kwargs: None)

    out = adapter._load_decoder_if_not_exist(M(), "language_model.model.layers.1", 1)
    assert isinstance(out, DummyLayer)
    assert len(module_list) == 2
    assert module_list[1] is out


def test_ascendv1_save_module_preprocess_pads_short_a_log():
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace(linear_attn_config={"head_dim": 8})))

    class Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.A_log = nn.Parameter(torch.arange(3, dtype=torch.float32), requires_grad=False)
            self.head_dim = 8

    mod = Mod()
    prefix, out = adapter.ascendv1_save_module_preprocess("layer.0.self_attn", mod, nn.Identity())
    assert prefix == "layer.0.self_attn"
    assert out.A_log.numel() == 8
    assert torch.allclose(out.A_log[:3], torch.arange(3, dtype=torch.float32))
    assert torch.allclose(out.A_log[3:], torch.zeros(5))

    # already long enough → no change
    mod2 = Mod()
    mod2.A_log = nn.Parameter(torch.ones(8), requires_grad=False)
    _, out2 = adapter.ascendv1_save_module_preprocess("x", mod2, nn.Identity())
    assert out2.A_log.numel() == 8

    # no A_log → passthrough
    plain = nn.Linear(2, 2)
    p, m = adapter.ascendv1_save_module_preprocess("y", plain, nn.Identity())
    assert m is plain and p == "y"


def test_ascendv1_save_module_preprocess_reads_head_dim_from_config_when_missing_on_module():
    adapter = _adapter(config=SimpleNamespace(text_config=SimpleNamespace(linear_attn_config={"head_dim": 6})))

    class Mod(nn.Module):
        def __init__(self):
            super().__init__()
            self.A_log = nn.Parameter(torch.ones(2), requires_grad=False)

    mod = Mod()
    _, out = adapter.ascendv1_save_module_preprocess("attn", mod, nn.Identity())
    assert out.A_log.numel() == 6


def test_ascendv1_save_postprocess_given_tiktoken_exists_when_called_then_copy_and_chmod(monkeypatch, tmp_path):
    adapter = _adapter(model_path=str(tmp_path))
    (tmp_path / "tiktoken.model").write_text("x", encoding="utf-8")

    called = {"copy": 0, "chmod": 0}
    monkeypatch.setattr(target, "safe_copy_file", lambda src_path, dest_path: called.__setitem__("copy", 1))
    monkeypatch.setattr(target.os, "chmod", lambda p, m: called.__setitem__("chmod", 1))

    adapter.ascendv1_save_postprocess(nn.Linear(2, 2), str(tmp_path / "save"))
    assert called["copy"] == 1
    assert called["chmod"] == 1


def test_ascendv1_save_postprocess_skips_when_tiktoken_missing(tmp_path):
    adapter = _adapter(model_path=str(tmp_path))
    adapter.ascendv1_save_postprocess(nn.Linear(2, 2), str(tmp_path / "save"))


# ---------------------------------------------------------------------------
# FA3 injection (from former test_fa3_inject.py)
# ---------------------------------------------------------------------------


class KimiMLAAttention(nn.Module):
    """Minimal MLA module with attrs required by absorb FA3 wrap."""

    def __init__(self, hidden=32, heads=4, nope=4, rope=4, kv_lora=16, v_dim=4):
        super().__init__()
        q_head_dim = nope + rope
        self.q_lora_rank = 16
        self.q_a_proj = nn.Linear(hidden, 16, bias=False)
        self.q_a_layernorm = nn.LayerNorm(16)
        self.q_b_proj = nn.Linear(16, heads * q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden, kv_lora + rope, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(kv_lora)
        self.kv_b_proj = nn.Linear(kv_lora, heads * (nope + v_dim), bias=False)
        self.o_proj = nn.Linear(heads * v_dim, hidden, bias=False)
        self.num_heads = heads
        self.q_head_dim = q_head_dim
        self.qk_nope_head_dim = nope
        self.qk_rope_head_dim = rope
        self.kv_lora_rank = kv_lora
        self.v_head_dim = v_dim
        self.scaling = q_head_dim ** (-0.5)
        self.attention_dropout = 0.0
        self.layer_idx = 1
        self.rotary_emb = None
        self.use_output_gate = False

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class KimiDeltaAttention(nn.Module):
    """KDA stand-in: must NOT receive FA3 placeholders."""

    def __init__(self, hidden=32):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.q_proj(hidden_states)


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = KimiMLAAttention()
        self.linear_attn = KimiDeltaAttention()


def test_adapter_implements_fa3_interface():
    assert issubclass(KimiK3ModelAdapter, FA3QuantAdapterInterface)


def test_adapter_implements_attention_analysis_interface():
    assert issubclass(KimiK3ModelAdapter, AttentionAnalysisInterface)


def test_attention_mse_interface_methods():
    adapter = _adapter()
    assert adapter.get_attention_module_cls() == "KimiMLAAttention"
    extractor = adapter.get_attention_output_extractor()
    tensor = torch.randn(2, 4)
    assert torch.equal(extractor(tensor), tensor)
    wrapped = (tensor, None)
    assert torch.equal(extractor(wrapped), tensor)


def test_inject_fa3_on_mla_skips_kda():
    adapter = _adapter()
    root = _DecoderLayer()
    called = []

    adapter.inject_fa3_placeholders(
        "language_model.model.layers.3",
        root,
        lambda name: (called.append(name), True)[1],
    )

    assert called == ["language_model.model.layers.3.self_attn"]

    mla = root.self_attn
    assert hasattr(mla, "fa_q") and isinstance(mla.fa_q, FA3QuantPlaceHolder)
    assert hasattr(mla, "fa_k") and isinstance(mla.fa_k, FA3QuantPlaceHolder)
    assert hasattr(mla, "fa_v") and isinstance(mla.fa_v, FA3QuantPlaceHolder)
    assert mla.fa_q.get_ratio() == 0.9999
    assert mla.fa_k.get_ratio() == 0.9999
    assert mla.fa_v.get_ratio() == 1.0

    assert not hasattr(root.linear_attn, "fa_q")
    assert not hasattr(root.linear_attn, "fa_k")
    assert not hasattr(root.linear_attn, "fa_v")


def test_inject_fa3_respects_should_inject_false():
    adapter = _adapter()
    root = _DecoderLayer()
    adapter.inject_fa3_placeholders("layer", root, lambda _name: False)
    assert not hasattr(root.self_attn, "fa_q")


def test_wrapped_mla_forward_runs_and_hits_placeholders():
    adapter = _adapter()
    root = _DecoderLayer()
    hits = {"q": 0, "k": 0, "v": 0}

    adapter.inject_fa3_placeholders("layer", root, lambda _n: True)

    class _Probe(nn.Module):
        def __init__(self, key, ratio=1.0):
            super().__init__()
            self.key = key
            self.ratio = ratio

        def forward(self, x):
            hits[self.key] += 1
            return x

        def get_ratio(self):
            return self.ratio

    root.self_attn.add_module("fa_q", _Probe("q", 0.9999))
    root.self_attn.add_module("fa_k", _Probe("k", 0.9999))
    root.self_attn.add_module("fa_v", _Probe("v", 1.0))

    hs = torch.randn(2, 5, 32)
    mask = torch.zeros(2, 1, 5, 5)
    out = root.self_attn(hs, attention_mask=mask)
    assert out.shape == (2, 5, 32)
    assert hits == {"q": 1, "k": 1, "v": 1}


def test_fa3_processor_preprocess_injects_mla_only():
    """FA3QuantProcessor.preprocess should inject MLA and leave KDA untouched."""
    root = _DecoderLayer()
    model = nn.Sequential()
    model.add_module("layer", root)

    adapter = _adapter()
    processor = FA3QuantProcessor(model, FA3QuantProcessorConfig(), adapter=adapter)
    processor.preprocess(BatchProcessRequest(name="layer", module=root, datas=None, outputs=None))

    # Placeholders are replaced by per-head observers on MLA path.
    assert hasattr(root.self_attn, "fa_q")
    assert hasattr(root.self_attn, "fa_k")
    assert hasattr(root.self_attn, "fa_v")
    assert not isinstance(root.self_attn.fa_q, FA3QuantPlaceHolder)

    assert not hasattr(root.linear_attn, "fa_q")
    assert not hasattr(root.linear_attn, "fa_k")
    assert not hasattr(root.linear_attn, "fa_v")


# ---------------------------------------------------------------------------
# IterSmooth EP range (from former test_adapter_ep_range.py)
# ---------------------------------------------------------------------------


def _ffn_expert_sources(layer_idx: int, text_cfg) -> list:
    """Mirror of KimiK3ModelAdapter._ffn_subgraph_configs routed-expert loop."""
    prefix = f"language_model.model.layers.{layer_idx}"
    expert_start, expert_end = _get_expert_range(text_cfg)
    return [f"{prefix}.block_sparse_moe.experts.{expert}.w3" for expert in range(expert_start, expert_end)]


class TestFfnSubgraphEpRange:
    def test_single_process_maps_all_experts(self):
        cfg = SimpleNamespace(num_experts=8)
        sources = _ffn_expert_sources(1, cfg)
        assert sources == [f"language_model.model.layers.1.block_sparse_moe.experts.{i}.w3" for i in range(8)]

    def test_distributed_maps_local_experts_only(self):
        cfg = SimpleNamespace(num_experts=8)
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 4
            mock_dist.get_rank.return_value = 1  # [2, 4)
            sources = _ffn_expert_sources(1, cfg)
        assert sources == [
            "language_model.model.layers.1.block_sparse_moe.experts.2.w3",
            "language_model.model.layers.1.block_sparse_moe.experts.3.w3",
        ]

    def test_kimi_896_world8_rank0(self):
        cfg = SimpleNamespace(num_experts=896)
        with patch("msmodelslim.model.common.utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_world_size.return_value = 8
            mock_dist.get_rank.return_value = 0
            sources = _ffn_expert_sources(3, cfg)
        assert len(sources) == 112
        assert sources[0].endswith("experts.0.w3")
        assert sources[-1].endswith("experts.111.w3")
