#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.step3_5_flash.model_adapter import Step3_5FlashModelAdapter


ADAPTER_PATH = "msmodelslim.model.step3_5_flash.model_adapter"


class _SafeOpenCtx:
    def __init__(self, tensor_map):
        self.tensor_map = tensor_map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def get_tensor(self, name):
        return self.tensor_map[name]


class _FakeCausalModel:
    def __init__(self, config):
        self.config = config
        self.model = SimpleNamespace(layers=[])
        self.load_state_dict = MagicMock()

    def eval(self):
        return self


class TestStep3_5FlashModelAdapter(unittest.TestCase):
    def setUp(self):
        self.model_type = "step3_5_flash"
        self.model_path = Path(tempfile.mkdtemp())

    @staticmethod
    def _config(num_hidden_layers=4, moe_layers_enum=None):
        config = SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=8,
            num_key_value_heads=2,
            use_cache=True,
        )
        if moe_layers_enum is not None:
            config.moe_layers_enum = moe_layers_enum
        return config

    def _create_adapter(self, config=None):
        def fake_default_init(adapter, model_type, model_path, trust_remote_code=False):
            adapter.model_type = model_type
            adapter.model_path = model_path
            adapter.trust_remote_code = trust_remote_code
            adapter.config = config or self._config()

        with patch(f"{ADAPTER_PATH}.DefaultModelAdapter.__init__", autospec=True, side_effect=fake_default_init):
            return Step3_5FlashModelAdapter(self.model_type, self.model_path, trust_remote_code=True)

    def test_init_sets_mtp_layer_range(self):
        adapter = self._create_adapter(self._config(num_hidden_layers=6))

        self.assertEqual(adapter.mtp_start_layer, 7)
        self.assertEqual(adapter.mtp_layer_num, 3)
        self.assertIsNone(adapter._processor)
        self.assertIsNone(adapter._tokenizer)

    def test_basic_methods_delegate_to_default_helpers(self):
        adapter = self._create_adapter()
        model = nn.Linear(2, 2)
        dataset_result = [{"input_ids": torch.ones(1, dtype=torch.long)}]
        adapter._load_model = MagicMock(return_value=model)
        adapter._get_tokenized_data = MagicMock(return_value=dataset_result)
        adapter._enable_kv_cache = MagicMock(return_value=None)

        self.assertEqual(adapter.get_model_type(), self.model_type)
        self.assertEqual(adapter.get_model_pedigree(), "step3_5_flash")
        self.assertIs(adapter.load_model(DeviceType.CPU), model)
        self.assertIs(adapter.handle_dataset("dataset", DeviceType.NPU), dataset_result)
        self.assertIsNone(adapter.enable_kv_cache(model, False))
        adapter._load_model.assert_called_once_with(DeviceType.CPU)
        adapter._get_tokenized_data.assert_called_once_with("dataset", DeviceType.NPU)
        adapter._enable_kv_cache.assert_called_once_with(model, False)

    def test_generate_model_visit_delegates_to_decoder_visit_func(self):
        adapter = self._create_adapter()
        expected_requests = [object(), object()]

        with patch(
            f"{ADAPTER_PATH}.generated_decoder_layer_visit_func", return_value=iter(expected_requests)
        ) as mocked:
            result = list(adapter.generate_model_visit("model"))

        self.assertEqual(result, expected_requests)
        mocked.assert_called_once_with("model")

    def test_generate_model_forward_raises_not_implemented(self):
        adapter = self._create_adapter()

        with self.assertRaisesRegex(NotImplementedError, "only supports dynamic quantization"):
            adapter.generate_model_forward(MagicMock(), {})

    def test_get_weight_map_returns_index_weight_map_and_uses_cache(self):
        adapter = self._create_adapter()
        index_path = self.model_path / "model.safetensors.index.json"
        index_path.write_text(json.dumps({"weight_map": {"a": "model.safetensors"}}), encoding="utf-8")
        adapter._get_weight_map.cache_clear()

        first = adapter._get_weight_map()
        index_path.write_text(json.dumps({"weight_map": {"b": "model.safetensors"}}), encoding="utf-8")
        second = adapter._get_weight_map()

        self.assertEqual(first, {"a": "model.safetensors"})
        self.assertIs(first, second)

    def test_get_state_dict_groups_parameters_by_safetensors_file(self):
        adapter = self._create_adapter()
        module = nn.Linear(3, 2)
        weight_tensor = torch.ones_like(module.weight)
        bias_tensor = torch.ones_like(module.bias)
        tensor_map = {
            "model.layers.0.weight": weight_tensor,
            "model.layers.0.bias": bias_tensor,
        }
        weight_map = {
            "model.layers.0.weight": "model-00001.safetensors",
            "model.layers.0.bias": "model-00002.safetensors",
            "model.layers.0.ignored": "model-00002.safetensors",
        }

        with (
            patch.object(adapter, "_get_weight_map", return_value=weight_map),
            patch(f"{ADAPTER_PATH}.get_valid_read_path", side_effect=lambda path, **_: path),
            patch(f"{ADAPTER_PATH}.safe_open", return_value=_SafeOpenCtx(tensor_map)) as safe_open_mock,
            patch(f"{ADAPTER_PATH}.tqdm", side_effect=lambda items, **_: items),
        ):
            state_dict = adapter._get_state_dict(module, prefix="model.layers.0")

        self.assertTrue(torch.equal(state_dict["weight"], weight_tensor))
        self.assertTrue(torch.equal(state_dict["bias"], bias_tensor))
        self.assertEqual(set(state_dict), {"weight", "bias"})
        self.assertEqual(safe_open_mock.call_count, 2)

    def test_convert_moe_layers_to_unpacked_uses_configured_moe_layer_enum(self):
        adapter = self._create_adapter(self._config(num_hidden_layers=4, moe_layers_enum="1,3"))
        layers = [SimpleNamespace(moe=f"moe-{idx}") for idx in range(4)]
        model = SimpleNamespace(model=SimpleNamespace(layers=layers))

        with patch(f"{ADAPTER_PATH}.convert_step35_moe_to_unpacked", side_effect=lambda moe, _: f"new-{moe}") as mocked:
            adapter._convert_moe_layers_to_unpacked(model)

        self.assertEqual(layers[0].moe, "moe-0")
        self.assertEqual(layers[1].moe, "new-moe-1")
        self.assertEqual(layers[2].moe, "moe-2")
        self.assertEqual(layers[3].moe, "new-moe-3")
        self.assertEqual(mocked.call_count, 2)

    def test_convert_moe_layers_to_unpacked_raises_when_conversion_fails(self):
        adapter = self._create_adapter(self._config(num_hidden_layers=2))
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(), SimpleNamespace(moe="bad")]))

        with patch(f"{ADAPTER_PATH}.convert_step35_moe_to_unpacked", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                adapter._convert_moe_layers_to_unpacked(model)

    def test_init_model_loads_base_model_appends_mtp_layers_and_converts_moe(self):
        config = self._config(num_hidden_layers=2)
        adapter = self._create_adapter(config)
        fake_model = _FakeCausalModel(config)
        fake_state_dict = {"weight": torch.ones(1)}
        fake_auto_model = SimpleNamespace(from_pretrained=MagicMock(return_value=fake_model))

        with (
            patch(f"{ADAPTER_PATH}.get_valid_read_path", return_value=str(self.model_path)) as valid_path_mock,
            patch(f"{ADAPTER_PATH}.AutoModelForCausalLM", fake_auto_model),
            patch(f"{ADAPTER_PATH}.Step3p5MTPModule", side_effect=lambda _, layer_idx: f"mtp-{layer_idx}") as mtp_mock,
            patch.object(adapter, "_get_state_dict", return_value=fake_state_dict) as state_dict_mock,
            patch.object(adapter, "_convert_moe_layers_to_unpacked") as convert_mock,
        ):
            result = adapter.init_model(DeviceType.CPU)

        self.assertIs(result, fake_model)
        self.assertFalse(config.use_cache)
        self.assertEqual(fake_model.model.layers, ["mtp-3", "mtp-4", "mtp-5"])
        fake_model.load_state_dict.assert_called_once_with(fake_state_dict)
        valid_path_mock.assert_called_once_with(str(self.model_path), is_dir=True, check_user_stat=True)
        fake_auto_model.from_pretrained.assert_called_once()
        self.assertEqual(mtp_mock.call_count, 3)
        state_dict_mock.assert_called_once_with(fake_model)
        convert_mock.assert_called_once_with(fake_model)


if __name__ == "__main__":
    unittest.main()
