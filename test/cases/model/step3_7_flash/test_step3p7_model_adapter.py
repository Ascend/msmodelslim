#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.step3_7_flash.model_adapter import Step3_7FlashModelAdapter


ADAPTER_PATH = "msmodelslim.model.step3_7_flash.model_adapter"


def _ensure_masking_utils_shim():
    """Inject a stub ``transformers.masking_utils`` for test environments
    where the installed transformers release predates 4.55 (the module was
    added in 4.55). The model adapter's ``generate_model_forward`` imports
    ``create_causal_mask`` / ``create_sliding_window_causal_mask`` from it;
    the test only asserts on the yielded ``ProcessRequest`` structure, so
    MagicMock callables returning ``None`` are sufficient.
    """
    try:
        import transformers.masking_utils  # noqa: F401

        return
    except ImportError:
        pass

    stub = types.ModuleType("transformers.masking_utils")
    stub.create_causal_mask = MagicMock(return_value=None)
    stub.create_sliding_window_causal_mask = MagicMock(return_value=None)
    sys.modules["transformers.masking_utils"] = stub


_ensure_masking_utils_shim()


class _SafeOpenCtx:
    """Minimal stand-in for safetensors.safe_open used in lazy-load tests."""

    def __init__(self, tensor_map):
        self.tensor_map = tensor_map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def get_tensor(self, name):
        return self.tensor_map[name]


class _FakeDecoderLayer(nn.Module):
    """Fake Step3p7DecoderLayer that mirrors the real one's attribute layout."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = nn.Linear(4, 4)
        self.post_attention_layernorm = nn.Linear(4, 4)
        self._moe = None

    def load_state_dict(self, state_dict, strict=True):
        return [], list(state_dict)


class _FakeMTPLayer(nn.Module):
    """Fake Step3p7MTPModule for tests."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.eh_proj = nn.Linear(8, 4)
        self._loaded_keys = []

    def load_state_dict(self, state_dict, strict=True):
        self._loaded_keys = list(state_dict.keys())
        return [], []


class _FakeTextModel(nn.Module):
    """Container that mimics Step3p7TextModel — has a ModuleList of decoder layers."""

    def __init__(self, config, decoder_cls=_FakeDecoderLayer, n_initial_layers=1):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(8, 4)
        initial = nn.ModuleList([decoder_cls(config, layer_idx=i) for i in range(n_initial_layers)])
        self.layers = initial
        self.norm = nn.Linear(4, 4)


class _FakeModelRoot(nn.Module):
    """Root container mimicking Step3p7Model — exposes .model.language_model.layers."""

    def __init__(self, config, decoder_cls=_FakeDecoderLayer, n_initial_layers=1):
        super().__init__()
        self.config = config
        text_model = _FakeTextModel(config, decoder_cls, n_initial_layers)
        self.model = nn.Module()
        self.model.language_model = text_model


def _config(num_hidden_layers=4, moe_layers_enum=None, num_experts=4, moe_intermediate_size=8):
    """Build a text_config-shaped SimpleNamespace."""
    cfg = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=8,
        # Step3p7 uses ``num_attention_groups`` for the KV-head count; mirror
        # the real config field name so iter_smooth mirroring in init_model
        # can read it.
        num_attention_groups=2,
        num_key_value_heads=2,
        use_cache=True,
        vocab_size=100,
        moe_num_experts=num_experts,
        moe_intermediate_size=moe_intermediate_size,
        moe_top_k=2,
    )
    if moe_layers_enum is not None:
        cfg.moe_layers_enum = moe_layers_enum
    return SimpleNamespace(text_config=cfg)


def _filename(path: str) -> str:
    """Return the basename regardless of OS path separator."""
    return os.path.basename(path.rstrip("/\\").rstrip(os.sep))


class TestStep3_7FlashModelAdapter(unittest.TestCase):
    def setUp(self):
        self.model_type = "step3_7_flash"
        self.model_path = Path(tempfile.mkdtemp())

    def _create_adapter(self, config=None):
        """Construct an adapter without invoking the real VLMBaseModelAdapter.__init__."""

        def fake_default_init(adapter, model_type, model_path, trust_remote_code=False):
            adapter.model_type = model_type
            adapter.model_path = model_path
            adapter.trust_remote_code = trust_remote_code
            adapter.config = config or _config()

        with patch(
            f"{ADAPTER_PATH}.VLMBaseModelAdapter.__init__",
            autospec=True,
            side_effect=fake_default_init,
        ):
            return Step3_7FlashModelAdapter(self.model_type, self.model_path, trust_remote_code=True)

    # ===== __init__ / basic delegation =====

    def test_init_sets_mtp_layer_range_and_moe_indices(self):
        adapter = self._create_adapter(_config(num_hidden_layers=6, moe_layers_enum="1,2,3,4,5"))

        self.assertEqual(adapter.mtp_start_layer, 6)
        self.assertEqual(adapter.mtp_layer_num, 3)
        self.assertEqual(adapter._moe_layers_idx, [1, 2, 3, 4, 5])
        self.assertIsNone(adapter._processor)
        self.assertIsNone(adapter._tokenizer)
        self.assertIsNone(adapter._decoder_layer_cls)  # set later in init_model

    def test_init_when_no_moe_layers_enum_then_defaults_to_layer_1_through_last(self):
        adapter = self._create_adapter(_config(num_hidden_layers=4))

        self.assertEqual(adapter._moe_layers_idx, [1, 2, 3])

    def test_basic_methods_delegate_to_default_helpers(self):
        adapter = self._create_adapter()
        model = nn.Linear(2, 2)
        dataset_result = [{"input_ids": torch.ones(1, dtype=torch.long)}]
        # load_model delegates to init_model in our refactor.
        adapter.init_model = MagicMock(return_value=model)
        # handle_dataset is overridden; mock it directly.
        adapter.handle_dataset = MagicMock(return_value=dataset_result)
        adapter._enable_kv_cache = MagicMock(return_value=None)

        self.assertEqual(adapter.get_model_type(), self.model_type)
        self.assertEqual(adapter.get_model_pedigree(), "step_3_7_flash")
        self.assertIs(adapter.load_model(DeviceType.CPU), model)
        self.assertIs(adapter.handle_dataset("dataset", DeviceType.NPU), dataset_result)
        self.assertIsNone(adapter.enable_kv_cache(model, False))
        adapter.init_model.assert_called_once_with(DeviceType.CPU)
        adapter.handle_dataset.assert_called_once_with("dataset", DeviceType.NPU)
        adapter._enable_kv_cache.assert_called_once_with(model, False)

    def test_generate_model_visit_passes_custom_transformer_blocks(self):
        """generate_model_visit must hand generate_decoder_layer to the framework helper."""
        adapter = self._create_adapter()
        expected = [object(), object()]
        captured_kwargs = {}

        def fake_visit_func(model, transformer_blocks):
            captured_kwargs["model"] = model
            captured_kwargs["blocks"] = transformer_blocks
            yield from expected

        # Stub out the generator so it doesn't try to materialise layers
        stub_blocks = iter(["fake_block"])
        with (
            patch(f"{ADAPTER_PATH}.generated_decoder_layer_visit_func", side_effect=fake_visit_func),
            patch.object(adapter, "generate_decoder_layer", return_value=stub_blocks),
        ):
            result = list(adapter.generate_model_visit("model"))

        self.assertEqual(result, expected)
        # The framework helper was called with the (model, transformer_blocks=...) kw
        self.assertIs(captured_kwargs["model"], "model")
        self.assertIs(captured_kwargs["blocks"], stub_blocks)

    def test_generate_model_forward_yields_one_request_per_decoder_layer_plus_norm(self):
        """Forward generator must yield exactly num_layers decoder requests + 1 final norm request.

        MTP slots are deliberately skipped — ``Step3p7MTPModule`` ships weights for
        save / MoE-unpack only and has no ``forward`` method, so the framework's
        ``module(*args, attention_mask=...)`` call would raise ``TypeError``. The
        official Step-3.7 forward (``modeling_step3p7.py:997``) also iterates
        ``self.layers[:num_hidden_layers]`` for the same reason.
        """
        # moe_layers_enum omitted → falls back to range(1, num_hidden_layers).
        config = _config(num_hidden_layers=3)
        adapter = self._create_adapter(config)

        # Patch _load_decoder_if_not_exist so the generator doesn't actually try
        # to materialise real Step3p7DecoderLayer instances.
        def fake_load(model, layer_idx):
            return nn.Linear(2, 2)

        adapter._load_decoder_if_not_exist = MagicMock(side_effect=fake_load)

        # Build a minimal model mock
        model = MagicMock()
        # embed_tokens is called directly (not yielded)
        model.model.language_model.embed_tokens = MagicMock(
            return_value=torch.zeros(1, 4, 8)  # (B=1, T=4, hidden=8)
        )
        # Norm is yielded at the end
        model.model.language_model.norm = nn.Linear(8, 8)
        model.model.language_model.rotary_emb = MagicMock(return_value=(torch.zeros(1, 4, 8), torch.zeros(1, 4, 8)))

        inputs = [{"input_ids": torch.zeros(1, 4, dtype=torch.long)}]
        gen = adapter.generate_model_forward(model, inputs)
        # Send None for each yielded ProcessRequest to drive the generator
        requests = []
        try:
            while True:
                req = gen.send(None)
                requests.append(req)
        except StopIteration:
            pass

        # Expect num_layers decoder layers + 1 final norm = 4 requests.
        # MTP layers (3, 4, 5) are intentionally NOT yielded.
        self.assertEqual(len(requests), config.text_config.num_hidden_layers + 1)
        expected_names = [f"model.language_model.layers.{i}" for i in range(config.text_config.num_hidden_layers)] + [
            "model.language_model.norm"
        ]
        actual_names = [req.name for req in requests]
        self.assertEqual(actual_names, expected_names)
        # Confirm MTP slots did not leak through the generator
        for name in actual_names:
            self.assertNotIn(
                name,
                ["model.language_model.layers.3", "model.language_model.layers.4", "model.language_model.layers.5"],
                f"MTP slot '{name}' should not be yielded by generate_model_forward",
            )

    def test_handle_dataset_returns_list_of_dicts_with_input_ids_and_attention_mask(self):
        """handle_dataset must return list[dict] (not list[list]) so that
        ``generate_model_forward``'s ``sample["input_ids"]`` string lookup works.
        Regression: the previous implementation returned ``list[[ids, attn]]``
        and produced ``IndexError: too many indices for tensor of dimension 2``
        during calibration.
        """
        adapter = self._create_adapter()
        fake_tokenizer = MagicMock()
        # The tokenizer call returns a BatchEncoding-like object that supports
        # ``__getitem__`` for "input_ids" and "attention_mask".
        fake_batch = MagicMock()
        fake_batch.__getitem__.side_effect = lambda key: {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }[key]
        fake_tokenizer.return_value = fake_batch
        adapter._tokenizer = fake_tokenizer

        result = adapter.handle_dataset(["hello world", {"text": "another"}, "third"], DeviceType.CPU)

        # Each item must be a dict, not a list/tuple
        self.assertEqual(len(result), 3)
        for item in result:
            self.assertIsInstance(item, dict)
            self.assertIn("input_ids", item)
            self.assertIn("attention_mask", item)
        # Tokenizer was called once per sample
        self.assertEqual(fake_tokenizer.call_count, 3)

    def test_ascendv1_save_module_preprocess_rewrites_prefixes(self):
        adapter = self._create_adapter()

        prefix, module = adapter.ascendv1_save_module_preprocess(
            "model.language_model.layers.3.self_attn.q_proj",
            nn.Linear(1, 1),
            MagicMock(),
        )
        self.assertEqual(prefix, "model.layers.3.self_attn.q_proj")

        prefix, _ = adapter.ascendv1_save_module_preprocess(
            "model.vision_model.encoder.layers.0",
            nn.Linear(1, 1),
            MagicMock(),
        )
        self.assertEqual(prefix, "vision_model.encoder.layers.0")

        prefix, _ = adapter.ascendv1_save_module_preprocess(
            "model.vit_large_projector.weight",
            nn.Linear(1, 1),
            MagicMock(),
        )
        self.assertEqual(prefix, "vit_large_projector.weight")

    # ===== _get_weight_map =====

    def test_get_weight_map_returns_index_weight_map_and_uses_cache(self):
        adapter = self._create_adapter()
        index_path = self.model_path / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps({"weight_map": {"a": "model.safetensors"}}),
            encoding="utf-8",
        )
        adapter._get_weight_map.cache_clear()

        first = adapter._get_weight_map()
        index_path.write_text(
            json.dumps({"weight_map": {"b": "model.safetensors"}}),
            encoding="utf-8",
        )
        second = adapter._get_weight_map()

        self.assertEqual(first, {"a": "model.safetensors"})
        self.assertIs(first, second)  # cache hit

    # ===== _resolve_moe_layers_idx =====

    def test_resolve_moe_layers_idx_when_enum_present_then_parses_csv(self):
        adapter = self._create_adapter(_config(num_hidden_layers=10, moe_layers_enum="3,5,7"))

        self.assertEqual(adapter._resolve_moe_layers_idx(), [3, 5, 7])

    def test_resolve_moe_layers_idx_when_enum_missing_then_skips_layer_zero(self):
        adapter = self._create_adapter(_config(num_hidden_layers=4))

        self.assertEqual(adapter._resolve_moe_layers_idx(), [1, 2, 3])

    # ===== _remap_to_model_keys =====

    def test_remap_to_model_keys_rewrites_safetensors_prefix(self):
        raw = {
            "model.layers.3.self_attn.q_proj.weight": torch.zeros(1),
            "model.layers.3.moe.gate.weight": torch.zeros(1),
            "model.layers.45.enorm.weight": torch.zeros(1),
        }

        remapped = Step3_7FlashModelAdapter._remap_to_model_keys(raw, 3)

        self.assertIn("model.language_model.layers.3.self_attn.q_proj.weight", remapped)
        self.assertIn("model.language_model.layers.3.moe.gate.weight", remapped)
        # Keys for other layers should NOT leak into the result
        self.assertNotIn("model.language_model.layers.45.enorm.weight", remapped)

    def test_remap_to_model_keys_when_key_does_not_match_prefix_then_skips(self):
        raw = {"model.layers.OTHER.x": torch.zeros(1)}

        remapped = Step3_7FlashModelAdapter._remap_to_model_keys(raw, 3)

        self.assertEqual(remapped, {})

    # ===== _load_raw_weights_for_layer =====

    def test_load_raw_weights_for_layer_groups_keys_by_safetensors_file(self):
        adapter = self._create_adapter()
        weight_map = {
            "model.layers.3.self_attn.q_proj.weight": "model-00001.safetensors",
            "model.layers.3.self_attn.k_proj.weight": "model-00001.safetensors",
            "model.layers.3.moe.gate.weight": "model-00002.safetensors",
            "model.layers.4.self_attn.q_proj.weight": "model-00001.safetensors",
        }
        q_proj_tensor = torch.ones(2, 2)
        k_proj_tensor = torch.ones(2, 2) + 1
        gate_tensor = torch.zeros(3, 3)
        file_tensor_map = {
            "model-00001.safetensors": {
                "model.layers.3.self_attn.q_proj.weight": q_proj_tensor,
                "model.layers.3.self_attn.k_proj.weight": k_proj_tensor,
            },
            "model-00002.safetensors": {
                "model.layers.3.moe.gate.weight": gate_tensor,
            },
        }

        def open_side_effect(path, **kwargs):
            return _SafeOpenCtx(file_tensor_map[_filename(path)])

        with (
            patch.object(adapter, "_get_weight_map", return_value=weight_map),
            patch(f"{ADAPTER_PATH}.get_valid_read_path", side_effect=lambda p, **_: p),
            patch(f"{ADAPTER_PATH}.safe_open", side_effect=open_side_effect) as safe_open_mock,
            patch(f"{ADAPTER_PATH}.tqdm", side_effect=lambda items, **_: items),
        ):
            result = adapter._load_raw_weights_for_layer(3)

        # Only layer 3 keys are returned
        self.assertEqual(
            set(result),
            {
                "model.layers.3.self_attn.q_proj.weight",
                "model.layers.3.self_attn.k_proj.weight",
                "model.layers.3.moe.gate.weight",
            },
        )
        self.assertTrue(torch.equal(result["model.layers.3.moe.gate.weight"], gate_tensor))
        # Two distinct files → two safe_open calls
        self.assertEqual(safe_open_mock.call_count, 2)

    def test_load_raw_weights_for_layer_when_layer_not_in_index_then_returns_empty(self):
        adapter = self._create_adapter()
        weight_map = {"model.layers.0.x": "model-00001.safetensors"}

        with (
            patch.object(adapter, "_get_weight_map", return_value=weight_map),
            patch(f"{ADAPTER_PATH}.safe_open"),
            patch(f"{ADAPTER_PATH}.tqdm", side_effect=lambda items, **_: items),
        ):
            result = adapter._load_raw_weights_for_layer(7)

        self.assertEqual(result, {})

    # ===== _load_decoder_if_not_exist =====

    def test_load_decoder_if_not_exist_when_layer_already_loaded_then_reuses_it(self):
        """Layer 0 is already loaded by init_model — should be reused, not reloaded."""
        config = _config(num_hidden_layers=3)
        adapter = self._create_adapter(config)
        model = _FakeModelRoot(config, n_initial_layers=1)
        existing_layer = model.model.language_model.layers[0]

        result = adapter._load_decoder_if_not_exist(model, 0)

        self.assertIs(result, existing_layer)

    def test_load_decoder_if_not_exist_when_layer_missing_then_loads_and_returns(self):
        """Layer 1 not in model.layers → construct fresh + load weights from disk."""
        config = _config(num_hidden_layers=3)
        adapter = self._create_adapter(config)
        model = _FakeModelRoot(config, n_initial_layers=1)
        adapter._decoder_layer_cls = _FakeDecoderLayer

        with (
            patch.object(
                adapter,
                "_load_raw_weights_for_layer",
                return_value={"model.layers.1.self_attn.q_proj.weight": torch.zeros(1)},
            ) as load_raw,
            patch(f"{ADAPTER_PATH}.convert_step35_moe_to_unpacked") as convert_mock,
        ):
            result = adapter._load_decoder_if_not_exist(model, 1)

        self.assertIsInstance(result, _FakeDecoderLayer)
        self.assertEqual(result.layer_idx, 1)
        load_raw.assert_called_once_with(1)
        convert_mock.assert_not_called()  # layer 1 not in moe_layers_idx
        # Layer 1 must now be in the ModuleList at index 1
        self.assertIs(model.model.language_model.layers[1], result)
        self.assertEqual(len(model.model.language_model.layers), 2)

    def test_load_decoder_if_not_exist_when_moe_layer_then_unpacks_moe(self):
        """When the layer is a MoE layer and decoder.moe exists, convert it."""
        config = _config(num_hidden_layers=3, moe_layers_enum="1,2")
        adapter = self._create_adapter(config)
        model = _FakeModelRoot(config, n_initial_layers=1)

        class _MoEDecoder(_FakeDecoderLayer):
            def __init__(self, cfg, layer_idx):
                super().__init__(cfg, layer_idx)
                self.moe = f"fused-moe-{layer_idx}"

        adapter._decoder_layer_cls = _MoEDecoder

        with (
            patch.object(adapter, "_load_raw_weights_for_layer", return_value={}),
            patch(
                f"{ADAPTER_PATH}.convert_step35_moe_to_unpacked",
                side_effect=lambda moe, _cfg: f"unpacked-{moe}",
            ) as convert_mock,
        ):
            result = adapter._load_decoder_if_not_exist(model, 1)

        self.assertEqual(result.moe, "unpacked-fused-moe-1")
        convert_mock.assert_called_once()

    def test_load_decoder_if_not_exist_when_moe_convert_fails_then_propagates(self):
        config = _config(num_hidden_layers=2, moe_layers_enum="1")
        adapter = self._create_adapter(config)
        model = _FakeModelRoot(config, n_initial_layers=1)

        class _MoEDecoder(_FakeDecoderLayer):
            def __init__(self, cfg, layer_idx):
                super().__init__(cfg, layer_idx)
                self.moe = "fused"

        adapter._decoder_layer_cls = _MoEDecoder

        with (
            patch.object(adapter, "_load_raw_weights_for_layer", return_value={}),
            patch(
                f"{ADAPTER_PATH}.convert_step35_moe_to_unpacked",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                adapter._load_decoder_if_not_exist(model, 1)

    def test_load_decoder_if_not_exist_strips_parent_prefix_before_load_state_dict(self):
        """Regression: passing a full-path state dict to a sub-module's
        ``load_state_dict(strict=False)`` silently drops every tensor, leaving the
        decoder at random init. The lazy load must strip the
        ``model.language_model.layers.X.`` prefix first.
        """
        config = _config(num_hidden_layers=3)
        adapter = self._create_adapter(config)
        model = _FakeModelRoot(config, n_initial_layers=1)
        adapter._decoder_layer_cls = _FakeDecoderLayer

        captured_state_dict = {}

        real_load_sd = _FakeDecoderLayer.load_state_dict

        def spy_load_state_dict(self_, state_dict, strict=True):
            # Snapshot what the decoder actually sees
            captured_state_dict.clear()
            captured_state_dict.update(state_dict)
            return [], []

        _FakeDecoderLayer.load_state_dict = spy_load_state_dict
        try:
            with (
                patch.object(
                    adapter,
                    "_load_raw_weights_for_layer",
                    return_value={
                        "model.layers.1.self_attn.q_proj.weight": torch.ones(1),
                        "model.layers.1.mlp.gate_proj.weight": torch.ones(1),
                    },
                ),
                patch(f"{ADAPTER_PATH}.convert_step35_moe_to_unpacked"),
            ):
                adapter._load_decoder_if_not_exist(model, 1)
        finally:
            _FakeDecoderLayer.load_state_dict = real_load_sd

        # The keys passed to load_state_dict must be RELATIVE to the decoder,
        # not absolute ("model.language_model.layers.1.self_attn.q_proj.weight").
        self.assertIn("self_attn.q_proj.weight", captured_state_dict)
        self.assertIn("mlp.gate_proj.weight", captured_state_dict)
        for key in captured_state_dict:
            self.assertFalse(
                key.startswith("model.language_model.layers."),
                f"key '{key}' still has the parent prefix; load_state_dict would silently drop it under strict=False",
            )

    def test_load_mtp_layer_if_not_exist_strips_parent_prefix_before_load_state_dict(self):
        """Same regression as decoder: MTP state dict must be relative to the
        bare ``Step3p7MTPModule`` instance, not the full ``model.language_model…`` path.
        """
        config = _config(num_hidden_layers=2)
        adapter = self._create_adapter(config)
        model = _FakeModelRoot(config, n_initial_layers=1)

        captured_state_dict = {}

        def spy_load_state_dict(self_, state_dict, strict=True):
            captured_state_dict.clear()
            captured_state_dict.update(state_dict)
            return [], []

        _FakeMTPLayer.load_state_dict = spy_load_state_dict
        try:
            with (
                patch(f"{ADAPTER_PATH}.Step3p7MTPModule", _FakeMTPLayer),
                patch.object(
                    adapter,
                    "_load_raw_weights_for_layer",
                    return_value={
                        "model.layers.45.enorm.weight": torch.ones(1),
                        "model.layers.45.eh_proj.weight": torch.ones(1),
                    },
                ),
            ):
                adapter._load_mtp_layer_if_not_exist(model, 45)
        finally:
            del _FakeMTPLayer.load_state_dict

        self.assertIn("enorm.weight", captured_state_dict)
        self.assertIn("eh_proj.weight", captured_state_dict)
        for key in captured_state_dict:
            self.assertFalse(key.startswith("model.language_model.layers."))

    # ===== _load_mtp_layer_if_not_exist =====

    def test_load_mtp_layer_if_not_exist_when_missing_then_appends_mtp_module(self):
        config = _config(num_hidden_layers=2)
        adapter = self._create_adapter(config)
        model = _FakeModelRoot(config, n_initial_layers=1)
        with (
            patch(f"{ADAPTER_PATH}.Step3p7MTPModule", _FakeMTPLayer),
            patch.object(adapter, "_load_raw_weights_for_layer", return_value={}),
        ):
            result = adapter._load_mtp_layer_if_not_exist(model, 45)

        self.assertIsInstance(result, _FakeMTPLayer)
        self.assertEqual(result.layer_idx, 45)
        self.assertEqual(len(model.model.language_model.layers), 2)
        self.assertIs(model.model.language_model.layers[1], result)

    def test_load_mtp_layer_if_not_exist_when_already_loaded_then_reuses(self):
        """If the MTP slot already exists in the ModuleList, reuse it without loading."""
        config = _config(num_hidden_layers=2)
        adapter = self._create_adapter(config)
        # Build a model whose layers ModuleList already contains an MTP at index 45
        existing_mtp = _FakeMTPLayer(config.text_config, layer_idx=45)
        layers = nn.ModuleList([_FakeDecoderLayer(config.text_config, layer_idx=0)])
        # Pad up to index 45 with placeholder decoder layers
        while len(layers) < 45:
            layers.append(_FakeDecoderLayer(config.text_config, layer_idx=len(layers)))
        layers.append(existing_mtp)  # index 45
        model = _FakeModelRoot(config, n_initial_layers=0)
        model.model.language_model.layers = layers

        with patch.object(adapter, "_load_raw_weights_for_layer") as load_raw:
            result = adapter._load_mtp_layer_if_not_exist(model, 45)

        self.assertIs(result, existing_mtp)
        load_raw.assert_not_called()

    # ===== generate_decoder_layer =====

    def test_generate_decoder_layer_yields_all_decoder_then_mtp_layers(self):
        """Generator must yield layers 0..N-1 then MTP layers (offset by num_hidden_layers)."""
        num_layers = 2
        config = _config(num_hidden_layers=num_layers)
        adapter = self._create_adapter(config)
        model = _FakeModelRoot(config, n_initial_layers=1)
        adapter._decoder_layer_cls = _FakeDecoderLayer

        # Pre-build layer 1 so the generator reuses via get_submodule
        layer_1 = _FakeDecoderLayer(config.text_config, layer_idx=1)
        model.model.language_model.layers.append(layer_1)

        with (
            patch.object(adapter, "_load_raw_weights_for_layer", return_value={}),
            patch(f"{ADAPTER_PATH}.Step3p7MTPModule", _FakeMTPLayer),
        ):
            generated_layers = list(adapter.generate_decoder_layer(model))

        # num_layers decoder layers + 3 MTP layers
        self.assertEqual(len(generated_layers), num_layers + 3)
        names = [n for n, _ in generated_layers]
        mtp_start = num_layers  # MTP begins where the decoder layers end
        self.assertEqual(
            names,
            [
                "model.language_model.layers.0",
                "model.language_model.layers.1",
                f"model.language_model.layers.{mtp_start}",
                f"model.language_model.layers.{mtp_start + 1}",
                f"model.language_model.layers.{mtp_start + 2}",
            ],
        )

    # ===== init_model =====

    def test_init_model_uses_num_hidden_layers_1_trick(self):
        """init_model must temporarily set num_hidden_layers=1 to keep CPU footprint small."""
        config = _config(num_hidden_layers=10, moe_layers_enum="1,2")
        adapter = self._create_adapter(config)
        fake_model = _FakeModelRoot(config, n_initial_layers=1)

        captured_num_layers_at_call_time = {}

        def fake_from_pretrained(*args, **kwargs):
            # Capture the num_hidden_layers value as it was when transformers was called
            captured_num_layers_at_call_time["value"] = kwargs["config"].text_config.num_hidden_layers
            return fake_model

        fake_auto_model = MagicMock()
        fake_auto_model.from_pretrained = MagicMock(side_effect=fake_from_pretrained)

        with (
            patch(f"{ADAPTER_PATH}.get_valid_read_path", return_value=str(self.model_path)),
            patch(f"{ADAPTER_PATH}.AutoModelForCausalLM", fake_auto_model),
        ):
            result = adapter.init_model(DeviceType.CPU)

        # During from_pretrained we must have asked transformers to build only 1 layer
        self.assertEqual(captured_num_layers_at_call_time["value"], 1)
        # After init_model returns, the config has been restored
        self.assertEqual(config.text_config.num_hidden_layers, 10)
        self.assertEqual(config.text_config.use_cache, False)
        # decoder layer class captured from the loaded model
        self.assertIs(adapter._decoder_layer_cls, _FakeDecoderLayer)
        self.assertIs(result, fake_model)

    def test_init_model_does_not_eagerly_convert_moe_or_load_mtp(self):
        """Lazy mode: init_model must NOT call _convert_moe_layers_to_unpacked or load MTP."""
        config = _config(num_hidden_layers=2, moe_layers_enum="1")
        adapter = self._create_adapter(config)
        fake_model = _FakeModelRoot(config, n_initial_layers=1)
        fake_auto_model = MagicMock()
        fake_auto_model.from_pretrained = MagicMock(return_value=fake_model)

        with (
            patch(f"{ADAPTER_PATH}.get_valid_read_path", return_value=str(self.model_path)),
            patch(f"{ADAPTER_PATH}.AutoModelForCausalLM", fake_auto_model),
            patch(f"{ADAPTER_PATH}.Step3p7MTPModule") as mtp_mock,
            patch(f"{ADAPTER_PATH}.convert_step35_moe_to_unpacked") as convert_mock,
        ):
            adapter.init_model(DeviceType.CPU)

        mtp_mock.assert_not_called()
        convert_mock.assert_not_called()

    # ===== IterSmooth / get_adapter_config_for_subgraph =====
    # IterSmooth integration was reverted from the Step3_7Flash adapter in
    # commit ``cfe9af60 移除smooth和quarot``; the adapter no longer subclasses
    # ``IterSmoothInterface`` nor exposes ``get_adapter_config_for_subgraph``.
    # Tests for those entry points are intentionally absent here.

    def test_init_model_mirrors_attention_head_counts_onto_model_config(self):
        """iter_smooth reads num_attention_heads / num_key_value_heads from model.config;
        init_model must populate them (Step3.7 keeps them only on text_config).
        """
        config = _config(num_hidden_layers=2, moe_layers_enum="0,1")
        adapter = self._create_adapter(config)
        fake_model = _FakeModelRoot(config, n_initial_layers=1)
        fake_auto_model = MagicMock()
        fake_auto_model.from_pretrained = MagicMock(return_value=fake_model)

        with (
            patch(f"{ADAPTER_PATH}.get_valid_read_path", return_value=str(self.model_path)),
            patch(f"{ADAPTER_PATH}.AutoModelForCausalLM", fake_auto_model),
        ):
            adapter.init_model(DeviceType.CPU)

        self.assertEqual(
            fake_model.config.num_attention_heads,
            config.text_config.num_attention_heads,
        )
        self.assertEqual(
            fake_model.config.num_key_value_heads,
            config.text_config.num_attention_groups,
        )


if __name__ == "__main__":
    unittest.main()
