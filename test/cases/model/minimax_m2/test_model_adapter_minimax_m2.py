from types import SimpleNamespace

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.minimax_m2 import model_adapter as target
from msmodelslim.model.minimax_m2.model_adapter import MiniMaxM2ModelAdapter
from msmodelslim.utils.exception import InvalidModelError


def _adapter(**kwargs):
    adapter = MiniMaxM2ModelAdapter.__new__(MiniMaxM2ModelAdapter)
    for key, value in kwargs.items():
        setattr(adapter, key, value)
    return adapter


class MiniMaxM2SparseMoeBlock(nn.Module):
    def __init__(self, gate_dtype=torch.bfloat16, bias_dtype=torch.bfloat16, bias_as_parameter=False):
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False, dtype=gate_dtype)
        bias = torch.ones(2, dtype=bias_dtype)
        if bias_as_parameter:
            self.e_score_correction_bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_buffer("e_score_correction_bias", bias)


class _ModelBody(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        self.layers = nn.ModuleList(layers or [nn.Identity()])


class _ForwardModel(nn.Module):
    def __init__(self, layer=None):
        super().__init__()
        self.model = _ModelBody([layer or nn.Identity()])

    def forward(self, inputs=None, **kwargs):
        return self.model.layers[0](inputs, **kwargs)


class _FakeDecoder(nn.Module):
    def __init__(self, config=None, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.linear = nn.Linear(2, 2)


class _FakeMiniMaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ModelBody([_FakeDecoder(layer_idx=0)])


class _FakeSafeOpen:
    tensors = {}

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    @classmethod
    def get_tensor(cls, key):
        return cls.tensors[key]


def _install_fake_checkpoint(monkeypatch, model_path="/tmp/minimax"):
    weight_map = {}
    tensors = {}
    for layer_idx in range(2):
        prefix = f"model.layers.{layer_idx}.linear"
        weight_key = f"{prefix}.weight"
        scale_key = f"{prefix}.weight_scale_inv"
        bias_key = f"{prefix}.bias"
        weight_map[weight_key] = "weights.safetensors"
        weight_map[scale_key] = "weights.safetensors"
        weight_map[bias_key] = "weights.safetensors"
        tensors[weight_key] = torch.ones(2, 2)
        tensors[scale_key] = torch.full((1, 1), float(layer_idx + 2))
        tensors[bias_key] = torch.full((2,), float(layer_idx + 1))

    def fake_json_safe_load(path):
        if path == f"{model_path}/model.safetensors.index.json":
            return {"weight_map": weight_map}
        if path == f"{model_path}/config.json":
            return {"quantization_config": {"weight_block_size": [2, 2]}}
        return {}

    _FakeSafeOpen.tensors = tensors
    monkeypatch.setattr(target, "json_safe_load", fake_json_safe_load)
    monkeypatch.setattr(target, "get_valid_read_path", lambda path, **_kwargs: path)
    monkeypatch.setattr(target, "safe_open", _FakeSafeOpen)


def test_should_return_expected_values_when_basic_getters_called_given_adapter_fields():
    # given
    adapter = _adapter(model_type="MiniMax-M2")

    # when
    pedigree = adapter.get_model_pedigree()
    model_type = adapter.get_model_type()

    # then
    assert pedigree == "minimax_m2"
    assert model_type == "MiniMax-M2"


def test_should_delegate_to_tokenizer_helper_when_handle_dataset_called_given_dataset_and_device():
    # given
    adapter = _adapter()
    adapter._get_tokenized_data = lambda dataset, device, padding=False: [dataset, device, padding]

    # when
    out = adapter.handle_dataset("sample", device=DeviceType.CPU)

    # then
    assert out == ["sample", DeviceType.CPU, False]


def test_should_build_expected_layer_and_expert_mappings_when_get_adapter_config_for_subgraph_called_given_config():
    # given
    adapter = _adapter(config=SimpleNamespace(num_hidden_layers=2, num_local_experts=3))

    # when
    out = adapter.get_adapter_config_for_subgraph()

    # then
    norm_linear = [cfg for cfg in out if cfg.subgraph_type == "norm-linear"]
    ov = [cfg for cfg in out if cfg.subgraph_type == "ov"]
    up_down = [cfg for cfg in out if cfg.subgraph_type == "up-down"]

    assert len(out) == 10
    assert len(norm_linear) == 2
    assert len(ov) == 2
    assert len(up_down) == 6
    assert norm_linear[0].mapping.source == "model.layers.0.input_layernorm"
    assert norm_linear[0].mapping.targets == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    assert ov[0].mapping.source == "model.layers.0.self_attn.v_proj"
    assert ov[0].mapping.targets == ["model.layers.0.self_attn.o_proj"]
    assert ov[0].extra_config["group_method"] == "max"
    assert up_down[0].mapping.source == "model.layers.0.block_sparse_moe.experts.0.w3"
    assert up_down[0].mapping.targets == ["model.layers.0.block_sparse_moe.experts.0.w2"]


def test_should_build_quarot_maps_when_public_quarot_apis_called_given_small_config(monkeypatch):
    # given
    calls = []

    def fake_rotate_command(size, block_size, mode):
        calls.append((size, block_size, mode))
        return f"rot-{size}-{block_size}"

    monkeypatch.setattr(target.QuaRotInterface, "get_rotate_command", staticmethod(fake_rotate_command))
    adapter = _adapter(
        config=SimpleNamespace(
            hidden_size=16,
            head_dim=4,
            num_hidden_layers=2,
            num_local_experts=2,
        )
    )

    # when
    pre_run, rotate_pairs = adapter.get_rotate_map(block_size=8)
    first_map, ln_fuse_map = adapter.get_ln_fuse_map()
    bake_names = adapter.get_bake_names()

    # then
    assert len(calls) == 2
    assert len(pre_run) == 1
    assert len(rotate_pairs) == 2
    assert not first_map
    assert bake_names == ([], [])
    assert pre_run[0].right_rot["model.embed_tokens"] == "rot-16-8"
    assert rotate_pairs[0].right_rot["lm_head"] == "rot-16-8"
    assert rotate_pairs[0].right_rot["model.layers.0.block_sparse_moe.experts.1.w3"] == "rot-16-8"
    assert rotate_pairs[0].left_rot["model.layers.1.block_sparse_moe.experts.1.w2"] == "rot-16-8"
    assert rotate_pairs[1].left_rot["model.layers.0.self_attn.v_proj"] == "rot-4-8"
    assert rotate_pairs[1].right_rot["model.layers.1.self_attn.o_proj"] == "rot-4-8"
    assert ln_fuse_map["model.layers.0.post_attention_layernorm"] == [
        "model.layers.0.block_sparse_moe.gate",
        "model.layers.0.block_sparse_moe.experts.0.w1",
        "model.layers.0.block_sparse_moe.experts.0.w3",
        "model.layers.0.block_sparse_moe.experts.1.w1",
        "model.layers.0.block_sparse_moe.experts.1.w3",
    ]


def test_should_register_hook_with_use_cache_when_enable_kv_cache_called_given_forward_model():
    # given
    model = _ForwardModel()
    adapter = _adapter()

    # when
    adapter.enable_kv_cache(model, True)

    # then
    _, kwargs = model.model._forward_pre_hooks[next(iter(model.model._forward_pre_hooks))](model.model, (), {})
    assert kwargs["use_cache"] is True


def test_should_initialize_template_and_lazy_layers_when_init_model_called_given_fp8_checkpoint(monkeypatch):
    # given
    model_path = "/tmp/minimax"
    model = _FakeMiniMaxModel()
    config = SimpleNamespace(num_hidden_layers=2, _attn_implementation="eager", quantization_config={"fp8": True})
    adapter = _adapter(config=config, model_path=model_path, trust_remote_code=True)
    _install_fake_checkpoint(monkeypatch, model_path=model_path)

    def fake_get_model_from_pretrained(**kwargs):
        assert kwargs["model_path"] == model_path
        assert kwargs["device_map"] == "cpu"
        assert kwargs["trust_remote_code"] is True
        return model

    monkeypatch.setattr(target.SafeGenerator, "get_model_from_pretrained", fake_get_model_from_pretrained)

    # when
    initialized = adapter.init_model()

    # then
    assert initialized is model
    assert config.num_hidden_layers == 2
    assert config.use_cache is False
    assert config._attn_implementation == "sdpa"
    assert not hasattr(config, "quantization_config")
    assert len(model.model.layers) == 2
    assert model.model.layers[0].linear.weight.dtype == torch.bfloat16
    assert torch.equal(model.model.layers[0].linear.weight, torch.full((2, 2), 2.0, dtype=torch.bfloat16))
    assert torch.equal(model.model.layers[0].linear.bias, torch.ones(2))


def test_should_raise_invalid_model_error_when_init_model_called_given_missing_layer_count():
    # given
    adapter = _adapter(config=SimpleNamespace())

    # when / then
    with pytest.raises(InvalidModelError):
        adapter.init_model()


def test_should_visit_lazy_layers_and_load_weights_when_generate_model_visit_called(monkeypatch):
    # given
    model_path = "/tmp/minimax"
    model = _FakeMiniMaxModel()
    model.model.layers.append(_FakeDecoder(layer_idx=1))
    setattr(model.model.layers[1], target.LAZY_LOAD_ATTR, False)
    adapter = _adapter(config=SimpleNamespace(num_hidden_layers=2), model_path=model_path)
    _install_fake_checkpoint(monkeypatch, model_path=model_path)

    # when
    requests = list(adapter.generate_model_visit(model))

    # then
    assert [request.name for request in requests] == ["model.layers.0", "model.layers.1"]
    assert all(isinstance(request, ProcessRequest) for request in requests)
    assert getattr(model.model.layers[1], target.LAZY_LOAD_ATTR) is True
    assert model.model.layers[1].linear.weight.dtype == torch.bfloat16
    assert torch.equal(model.model.layers[1].linear.weight, torch.full((2, 2), 3.0, dtype=torch.bfloat16))
    assert torch.equal(model.model.layers[1].linear.bias, torch.full((2,), 2.0))


def test_should_generate_forward_requests_with_kv_cache_when_public_forward_api_called():
    # given
    model = _ForwardModel()
    adapter = _adapter(config=SimpleNamespace(num_hidden_layers=1))

    # when
    generator = adapter.generate_model_forward(model, {"inputs": torch.ones(1), "use_cache": True})
    request = next(generator)

    # then
    assert isinstance(request, ProcessRequest)
    assert request.name == "model.layers.0"
    assert request.module is model.model.layers[0]
    assert "past_key_values" in request.kwargs

    with pytest.raises(StopIteration):
        generator.send(torch.ones(1))


def test_should_raise_original_exception_when_public_forward_api_model_forward_fails():
    # given
    class FailingModel(_ForwardModel):
        def forward(self, inputs=None, **kwargs):
            raise RuntimeError("forward failed")

    adapter = _adapter()

    # when / then
    with pytest.raises(RuntimeError, match="forward failed"):
        next(adapter.generate_model_forward(FailingModel(), torch.ones(1)))


def test_should_keep_router_pieces_in_fp32_and_promote_bias_when_save_preprocess_called_given_moe_block():
    # given
    adapter = _adapter()
    moe = MiniMaxM2SparseMoeBlock()

    # when
    prefix, processed = adapter.ascendv1_save_module_preprocess("model.layers.0.block_sparse_moe", moe, nn.Identity())

    # then
    assert prefix == "model.layers.0.block_sparse_moe"
    assert processed.gate.weight.dtype == torch.float32
    assert isinstance(processed.e_score_correction_bias, nn.Parameter)
    assert processed.e_score_correction_bias.dtype == torch.float32


def test_should_leave_module_unchanged_when_save_preprocess_called_given_already_fp32_router():
    # given
    adapter = _adapter()
    moe = MiniMaxM2SparseMoeBlock(gate_dtype=torch.float32, bias_dtype=torch.float32, bias_as_parameter=True)

    # when
    prefix, processed = adapter.ascendv1_save_module_preprocess("model.layers.0.block_sparse_moe", moe, nn.Identity())

    # then
    assert prefix == "model.layers.0.block_sparse_moe"
    assert processed is moe
