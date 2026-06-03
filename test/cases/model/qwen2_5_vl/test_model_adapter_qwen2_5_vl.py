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

import builtins
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import torch
except ModuleNotFoundError as e:
    raise unittest.SkipTest("torch is not installed; skip qwen2_5_vl adapter unit tests") from e

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig, MappingConfig
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.utils.exception import InvalidModelError, UnsupportedError

try:
    from msmodelslim.model.qwen2_5_vl.model_adapter import (
        Qwen25VLModelAdapter,
        _qwen2_5_vl_get_ln_fuse_map,
        _qwen2_5_vl_get_rotate_map,
    )

    _QWEN25_VL_IMPORT_OK = True
except Exception:
    Qwen25VLModelAdapter = None
    _qwen2_5_vl_get_ln_fuse_map = None
    _qwen2_5_vl_get_rotate_map = None
    _QWEN25_VL_IMPORT_OK = False


def _make_adapter(model_type="Qwen2.5-VL-7B-Instruct", model_path="."):
    """Create adapter with config loading mocked."""
    with patch(
        "msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()
    ):
        return Qwen25VLModelAdapter(model_type, Path(model_path), trust_remote_code=False)


class DummyVisionConfig:
    def __init__(self, depth: int = 2):
        self.depth = depth


class DummyConfig:
    """Minimal config stub for Qwen25VLModelAdapter UT."""

    def __init__(
        self,
        num_hidden_layers: int = 3,
        output_attentions: bool = False,
        vision_depth: int = 2,
        hidden_size: int = 128,
        num_attention_heads: int = 8,
        image_token_id: int = 151655,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.output_attentions = output_attentions
        self.vision_config = DummyVisionConfig(depth=vision_depth)
        self.use_cache = True
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.image_token_id = image_token_id


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGetModelType(unittest.TestCase):
    """测试Qwen25VLModelAdapter的get_model_type方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_get_model_type_with_valid_type_when_called_then_return_model_type(self):
        """正常：应返回初始化时的model_type"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.model_type = self.model_type

        result = adapter.get_model_type()

        self.assertEqual(result, self.model_type)


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGetModelPedigree(unittest.TestCase):
    """测试Qwen25VLModelAdapter的get_model_pedigree方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_get_model_pedigree_when_called_then_return_qwen25_vl(self):
        """正常：应返回qwen25_vl谱系标识"""
        adapter = _make_adapter(self.model_type, self.model_path)

        result = adapter.get_model_pedigree()

        self.assertEqual(result, "qwen25_vl")


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterEnableKvCache(unittest.TestCase):
    """测试Qwen25VLModelAdapter的enable_kv_cache方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_enable_kv_cache_with_need_cache_true_when_called_then_set_use_cache(self):
        """正常：need_kv_cache为True时应启用缓存"""
        adapter = _make_adapter(self.model_type, self.model_path)
        model = SimpleNamespace(config=SimpleNamespace(use_cache=None))

        adapter.enable_kv_cache(model, True)

        self.assertTrue(model.config.use_cache)

    def test_enable_kv_cache_with_need_cache_false_when_called_then_disable_cache(self):
        """边界：need_kv_cache为False时应禁用缓存"""
        adapter = _make_adapter(self.model_type, self.model_path)
        model = SimpleNamespace(config=SimpleNamespace(use_cache=True))

        adapter.enable_kv_cache(model, False)

        self.assertFalse(model.config.use_cache)


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterHandleDataset(unittest.TestCase):
    """测试Qwen25VLModelAdapter的handle_dataset方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_handle_dataset_with_image_and_text_when_called_then_return_processed_data(self):
        """正常：图文样本应返回processor处理后的数据"""
        adapter = _make_adapter(self.model_type, self.model_path)
        dataset = [VlmCalibSample(text="describe", image="a.jpg")]

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "TEMPLATE_TEXT"
        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        adapter._collect_inputs_to_device = MagicMock(return_value={"input_ids": "ok"})

        with (
            patch(
                "msmodelslim.model.qwen2_5_vl.model_adapter.AutoProcessor.from_pretrained", return_value=mock_processor
            ),
            patch("msmodelslim.model.qwen2_5_vl.model_adapter.process_vision_info", return_value=(["img"], None)),
            patch("msmodelslim.model.qwen2_5_vl.model_adapter.get_valid_read_path", side_effect=lambda p, *a, **k: p),
        ):
            result = adapter.handle_dataset(dataset, device=DeviceType.CPU)

        self.assertEqual(result, [{"input_ids": "ok"}])
        mock_processor.apply_chat_template.assert_called_once()
        mock_processor.assert_called_once()
        args, kwargs = adapter._collect_inputs_to_device.call_args
        self.assertIs(args[0], mock_inputs)
        self.assertEqual(args[1], DeviceType.CPU)
        self.assertIn("input_ids", kwargs["keys"])
        self.assertIn("pixel_values", kwargs["keys"])
        self.assertEqual(kwargs["defaults"].get("logits_to_keep"), 0)

    def test_handle_dataset_with_empty_dataset_when_called_then_return_empty_list(self):
        """边界：空数据集应返回空列表"""
        adapter = _make_adapter(self.model_type, self.model_path)

        with patch(
            "msmodelslim.model.qwen2_5_vl.model_adapter.AutoProcessor.from_pretrained", return_value=MagicMock()
        ):
            result = adapter.handle_dataset([], device=DeviceType.CPU)

        self.assertEqual(result, [])

    def test_handle_dataset_with_missing_image_when_called_then_raise_unsupported_error(self):
        """异常：缺少image时应抛出UnsupportedError"""
        adapter = _make_adapter(self.model_type, self.model_path)

        with patch(
            "msmodelslim.model.qwen2_5_vl.model_adapter.AutoProcessor.from_pretrained", return_value=MagicMock()
        ):
            with self.assertRaises(UnsupportedError) as context:
                adapter.handle_dataset([VlmCalibSample(text="hi", image=None)], device=DeviceType.CPU)

        self.assertIn("image and text", str(context.exception))

    def test_handle_dataset_with_missing_text_when_called_then_raise_unsupported_error(self):
        """异常：缺少text时应抛出UnsupportedError"""
        adapter = _make_adapter(self.model_type, self.model_path)

        with patch(
            "msmodelslim.model.qwen2_5_vl.model_adapter.AutoProcessor.from_pretrained", return_value=MagicMock()
        ):
            with self.assertRaises(UnsupportedError) as context:
                adapter.handle_dataset([VlmCalibSample(text=None, image="a.jpg")], device=DeviceType.CPU)

        self.assertIn("image and text", str(context.exception))


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterInitModel(unittest.TestCase):
    """测试Qwen25VLModelAdapter的init_model方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_init_model_with_valid_env_when_called_then_load_and_restore_layers(self):
        """正常：应加载模型并恢复num_hidden_layers"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=5, vision_depth=2)
        origin_layers = adapter.config.num_hidden_layers

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        with (
            patch("msmodelslim.model.qwen2_5_vl.model_adapter.get_valid_read_path", side_effect=lambda p, *a, **k: p),
            patch("transformers.Qwen2_5_VLForConditionalGeneration") as mock_cls,
            patch.object(adapter, "_get_state_dict", return_value={}),
        ):
            mock_cls.from_pretrained.return_value = mock_model
            model = adapter.init_model(device=DeviceType.CPU)

        self.assertIs(model, mock_model)
        self.assertEqual(adapter.config.num_hidden_layers, origin_layers)
        self.assertEqual(getattr(adapter.config, "_attn_implementation", None), "eager")
        mock_model.load_state_dict.assert_called_once()
        _, call_kwargs = mock_cls.from_pretrained.call_args
        self.assertEqual(call_kwargs.get("local_files_only"), True)
        self.assertEqual(call_kwargs.get("device_map"), "cpu")
        self.assertEqual(call_kwargs.get("attn_implementation"), "eager")

    def test_init_model_when_import_fails_then_raise_invalid_model_error(self):
        """异常：无法导入Qwen2_5_VLForConditionalGeneration时应抛出InvalidModelError"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig()
        real_import = builtins.__import__

        def import_mock(name, global_vars=None, local_vars=None, fromlist=(), level=0):
            if name == "transformers" and fromlist and "Qwen2_5_VLForConditionalGeneration" in fromlist:
                raise ImportError("cannot import Qwen2_5_VLForConditionalGeneration")
            return real_import(name, global_vars, local_vars, fromlist, level)

        with patch("builtins.__import__", side_effect=import_mock):
            with self.assertRaises(InvalidModelError) as context:
                adapter.init_model(device=DeviceType.CPU)

        self.assertIn("Qwen2_5_VLForConditionalGeneration", str(context.exception))


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGenerateModelVisit(unittest.TestCase):
    """测试Qwen25VLModelAdapter的generate_model_visit方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_generate_model_visit_with_valid_model_when_called_then_yield_visual_then_layers(self):
        """正常：应先yield visual再yield decoder layers"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        mock_visual = MagicMock()
        mock_layer0 = MagicMock()
        mock_layer1 = MagicMock()
        model = MagicMock()
        model.visual = mock_visual

        def mock_generate_decoder_layer(_m):
            yield "model.layers.0", mock_layer0
            yield "model.layers.1", mock_layer1

        adapter.generate_decoder_layer = MagicMock(side_effect=mock_generate_decoder_layer)

        def mock_visit_func(_m, transformer_blocks=None):
            for name, layer in transformer_blocks:
                yield ProcessRequest(name=name, module=layer, args=(), kwargs={})

        with patch(
            "msmodelslim.model.qwen2_5_vl.model_adapter.generated_decoder_layer_visit_func", side_effect=mock_visit_func
        ):
            requests = list(adapter.generate_model_visit(model))

        self.assertGreaterEqual(len(requests), 1)
        self.assertEqual(requests[0].name, "visual")
        self.assertIs(requests[0].module, mock_visual)
        decoder_requests = [r for r in requests[1:] if r.name.startswith("model.layers")]
        self.assertEqual(len(decoder_requests), 2)


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGenerateModelForward(unittest.TestCase):
    """测试Qwen25VLModelAdapter的generate_model_forward方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def _build_model_and_adapter(self, num_hidden_layers=2):
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=num_hidden_layers, output_attentions=False)

        mock_visual = MagicMock()
        mock_layer0 = MagicMock()
        mock_layer1 = MagicMock()

        model = MagicMock()
        model.visual = mock_visual
        model.config = SimpleNamespace(image_token_id=151655, output_attentions=False)
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=torch.randn(1, 10, 128))
        model.model._update_causal_mask = MagicMock(return_value=torch.ones(1, 1, 10, 10))
        model.model.rotary_emb = MagicMock(return_value=torch.randn(1, 10, 128))
        model.get_rope_index = MagicMock(return_value=(torch.arange(10, dtype=torch.long).unsqueeze(0), None))

        def mock_generate_decoder_layer(_m):
            yield "model.layers.0", mock_layer0
            yield "model.layers.1", mock_layer1

        adapter.generate_decoder_layer = MagicMock(side_effect=mock_generate_decoder_layer)
        return adapter, model, mock_visual, mock_layer0, mock_layer1

    def _make_sample(self):
        sample = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }
        sample["input_ids"][0, 0] = 151655
        return sample

    def test_generate_model_forward_with_list_inputs_when_called_then_yield_visual_then_layers(self):
        """正常：list输入时应先yield visual再yield decoder layers"""
        adapter, model, mock_visual, mock_layer0, mock_layer1 = self._build_model_and_adapter()
        gen = adapter.generate_model_forward(model, [self._make_sample()])

        first_req = next(gen)
        self.assertEqual(first_req.name, "visual")
        self.assertIs(first_req.module, mock_visual)

        second_req = gen.send(torch.randn(1, 10, 128))
        self.assertEqual(second_req.name, "model.layers.0")
        self.assertIs(second_req.module, mock_layer0)

        third_req = gen.send((torch.randn(1, 10, 128),))
        self.assertEqual(third_req.name, "model.layers.1")

        with self.assertRaises(StopIteration):
            gen.send((torch.randn(1, 10, 128),))

    def test_generate_model_forward_with_dict_inputs_when_called_then_use_single_sample(self):
        """边界：非list输入时应直接使用单个sample"""
        adapter, model, mock_visual, _, _ = self._build_model_and_adapter(num_hidden_layers=1)
        adapter.generate_decoder_layer = MagicMock(side_effect=lambda _m: iter([("model.layers.0", MagicMock())]))
        gen = adapter.generate_model_forward(model, self._make_sample())

        first_req = next(gen)
        self.assertEqual(first_req.name, "visual")

    def test_generate_model_forward_with_non_tuple_layer_output_when_called_then_use_output_directly(self):
        """边界：layer输出非tuple时应直接使用返回值作为hidden_states"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=1, output_attentions=False)

        mock_layer = MagicMock()
        model = MagicMock()
        model.visual = MagicMock()
        model.config = SimpleNamespace(image_token_id=151655)
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=torch.randn(1, 10, 128))
        model.model._update_causal_mask = MagicMock(return_value=torch.ones(1, 1, 10, 10))
        model.model.rotary_emb = MagicMock(return_value=torch.randn(1, 10, 128))
        model.get_rope_index = MagicMock(return_value=(torch.arange(10, dtype=torch.long).unsqueeze(0), None))
        adapter.generate_decoder_layer = MagicMock(side_effect=lambda _m: iter([("model.layers.0", mock_layer)]))

        sample = self._make_sample()
        gen = adapter.generate_model_forward(model, sample)
        next(gen)
        layer_req = gen.send(torch.randn(1, 10, 128))
        self.assertEqual(layer_req.name, "model.layers.0")

        with self.assertRaises(StopIteration):
            gen.send(torch.randn(1, 10, 128))


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGenerateDecoderLayer(unittest.TestCase):
    """测试Qwen25VLModelAdapter的generate_decoder_layer方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_generate_decoder_layer_with_valid_model_when_called_then_yield_layer_names(self):
        """正常：应按层索引yield layer名称与模块"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=3)
        dummy_model = MagicMock()
        adapter._load_decoder_if_not_exist = MagicMock(side_effect=lambda _m, _n, i: f"layer-{i}")

        items = list(adapter.generate_decoder_layer(dummy_model))

        self.assertEqual(
            items,
            [
                ("model.layers.0", "layer-0"),
                ("model.layers.1", "layer-1"),
                ("model.layers.2", "layer-2"),
            ],
        )
        self.assertEqual(adapter._load_decoder_if_not_exist.call_count, 3)

    def test_generate_decoder_layer_with_zero_layers_when_called_then_yield_nothing(self):
        """边界：num_hidden_layers为0时不应yield任何层"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=0)
        adapter._load_decoder_if_not_exist = MagicMock()

        items = list(adapter.generate_decoder_layer(MagicMock()))

        self.assertEqual(items, [])
        adapter._load_decoder_if_not_exist.assert_not_called()


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGetWeightMap(unittest.TestCase):
    """测试Qwen25VLModelAdapter的_get_weight_map方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_get_weight_map_with_valid_index_when_called_then_return_weight_map(self):
        """正常：应从index.json加载weight_map"""
        adapter = _make_adapter(self.model_type, self.model_path)
        index_data = {
            "weight_map": {
                "model.layers.0.weight": "model-00001.safetensors",
                "model.layers.1.weight": "model-00002.safetensors",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter.model_path = tmpdir
            with patch("msmodelslim.model.qwen2_5_vl.model_adapter.json_safe_load", return_value=index_data):
                adapter._get_weight_map.cache_clear()
                result = adapter._get_weight_map()

        self.assertEqual(result["model.layers.0.weight"], "model-00001.safetensors")
        self.assertEqual(result["model.layers.1.weight"], "model-00002.safetensors")


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGetStateDict(unittest.TestCase):
    """测试Qwen25VLModelAdapter的_get_state_dict方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_get_state_dict_with_valid_weights_when_called_then_load_tensors(self):
        """正常：应从safetensors加载权重"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.model_path = "."
        linear = torch.nn.Linear(4, 8)
        weight_map = {"weight": "model.safetensors", "bias": "model.safetensors"}
        mock_tensor = torch.randn(8, 4)
        mock_bias = torch.randn(8)

        with patch.object(adapter, "_get_weight_map", return_value=weight_map):
            with patch("msmodelslim.model.qwen2_5_vl.model_adapter.safe_open") as mock_safe_open:
                mock_f = MagicMock()
                mock_f.get_tensor = lambda name: mock_tensor if name == "weight" else mock_bias
                mock_safe_open.return_value.__enter__ = MagicMock(return_value=mock_f)
                mock_safe_open.return_value.__exit__ = MagicMock(return_value=False)
                with patch(
                    "msmodelslim.model.qwen2_5_vl.model_adapter.get_valid_read_path", side_effect=lambda p, *a, **k: p
                ):
                    result = adapter._get_state_dict(linear, prefix="")

        self.assertIn("weight", result)
        self.assertIn("bias", result)
        self.assertEqual(result["weight"].shape, (8, 4))

    def test_get_state_dict_with_param_not_in_weight_map_when_called_then_skip_param(self):
        """边界：weight_map中不存在的参数应被跳过"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.model_path = "."
        linear = torch.nn.Linear(4, 8)

        with patch.object(adapter, "_get_weight_map", return_value={}):
            result = adapter._get_state_dict(linear, prefix="")

        self.assertEqual(result, {})


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterLoadDecoderIfNotExist(unittest.TestCase):
    """测试Qwen25VLModelAdapter的_load_decoder_if_not_exist方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_load_decoder_if_not_exist_when_layer_already_loaded_then_return_existing(self):
        """正常：层已加载时应直接返回现有层"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        existing_layer = MagicMock()
        existing_layer.input_layernorm = MagicMock()
        existing_layer.input_layernorm.weight = MagicMock()
        existing_layer.input_layernorm.weight.device = torch.device("cpu")

        model = MagicMock()
        model.get_submodule = MagicMock(return_value=existing_layer)

        result = adapter._load_decoder_if_not_exist(model, "model.layers.0", 0)

        self.assertIs(result, existing_layer)

    def test_load_decoder_if_not_exist_when_layer_on_meta_then_create_and_load(self):
        """边界：层在meta设备上时应重新创建并加载"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        meta_layer = MagicMock()
        weight_mock = MagicMock()
        type(weight_mock).device = property(lambda self: (_ for _ in ()).throw(RuntimeError("meta device")))
        meta_layer.input_layernorm.weight = weight_mock

        model = MagicMock()
        model.get_submodule = MagicMock(return_value=meta_layer)
        mock_module_list = []
        model.model = MagicMock()
        model.model.layers = mock_module_list

        mock_decoder = MagicMock()
        mock_decoder.eval = MagicMock(return_value=mock_decoder)

        with patch("msmodelslim.model.qwen2_5_vl.model_adapter.Qwen2_5_VLDecoderLayer", return_value=mock_decoder):
            with patch.object(adapter, "_get_state_dict", return_value={"weight": torch.randn(1)}):
                with patch.object(torch.nn.Linear, "reset_parameters", lambda self: None):
                    result = adapter._load_decoder_if_not_exist(model, "model.layers.0", 0)

        self.assertIs(result, mock_decoder)
        mock_decoder.load_state_dict.assert_called_once()

    def test_load_decoder_if_not_exist_when_layer_missing_then_append_to_module_list(self):
        """异常：层不存在时应创建并追加到module_list"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        model = MagicMock()
        model.get_submodule = MagicMock(side_effect=AttributeError("no such module"))
        mock_module_list = []
        model.model = MagicMock()
        model.model.layers = mock_module_list

        mock_decoder = MagicMock()
        mock_decoder.eval = MagicMock(return_value=mock_decoder)

        with patch("msmodelslim.model.qwen2_5_vl.model_adapter.Qwen2_5_VLDecoderLayer", return_value=mock_decoder):
            with patch.object(adapter, "_get_state_dict", return_value={"weight": torch.randn(1)}):
                with patch.object(torch.nn.Linear, "reset_parameters", lambda self: None):
                    result = adapter._load_decoder_if_not_exist(model, "model.layers.0", 0)

        self.assertIs(result, mock_decoder)
        self.assertIn(mock_decoder, mock_module_list)

    def test_load_decoder_if_not_exist_when_index_exists_then_replace_layer(self):
        """边界：module_list已有该索引时应替换层"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        model = MagicMock()
        model.get_submodule = MagicMock(side_effect=AttributeError("no such module"))
        old_layer = MagicMock()
        mock_module_list = [old_layer]
        model.model = MagicMock()
        model.model.layers = mock_module_list

        mock_decoder = MagicMock()
        mock_decoder.eval = MagicMock(return_value=mock_decoder)

        with patch("msmodelslim.model.qwen2_5_vl.model_adapter.Qwen2_5_VLDecoderLayer", return_value=mock_decoder):
            with patch.object(adapter, "_get_state_dict", return_value={"weight": torch.randn(1)}):
                with patch.object(torch.nn.Linear, "reset_parameters", lambda self: None):
                    result = adapter._load_decoder_if_not_exist(model, "model.layers.0", 0)

        self.assertIs(result, mock_decoder)
        self.assertEqual(mock_module_list[0], mock_decoder)


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGetAdapterConfigForSubgraph(unittest.TestCase):
    """测试Qwen25VLModelAdapter的get_adapter_config_for_subgraph方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_get_adapter_config_for_subgraph_with_valid_layers_when_called_then_return_configs(self):
        """正常：每层应生成3个AdapterConfig"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        result = adapter.get_adapter_config_for_subgraph()

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 6)

        norm_attn = result[0]
        self.assertIsInstance(norm_attn, AdapterConfig)
        self.assertEqual(norm_attn.subgraph_type, "norm-linear")
        self.assertIsInstance(norm_attn.mapping, MappingConfig)
        self.assertEqual(norm_attn.mapping.source, "model.layers.0.input_layernorm")

        up_down = result[2]
        self.assertEqual(up_down.subgraph_type, "up-down")
        self.assertEqual(up_down.mapping.source, "model.layers.0.mlp.up_proj")

    def test_get_adapter_config_for_subgraph_with_zero_layers_when_called_then_return_empty_list(self):
        """边界：num_hidden_layers为0时应返回空列表"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=0)

        result = adapter.get_adapter_config_for_subgraph()

        self.assertEqual(result, [])


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGetLnFuseMap(unittest.TestCase):
    """测试Qwen25VLModelAdapter的get_ln_fuse_map方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_get_ln_fuse_map_with_valid_config_when_called_then_return_fused_map(self):
        """正常：应返回空pre_run和包含layer映射的fused_map"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        pre_run, fused_map = adapter.get_ln_fuse_map()

        self.assertEqual(pre_run, {})
        self.assertIn("model.layers.0.input_layernorm", fused_map)
        self.assertIn("model.layers.0.post_attention_layernorm", fused_map)
        self.assertIn("model.norm", fused_map)
        self.assertEqual(fused_map["model.norm"], ["lm_head"])


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGetBakeNames(unittest.TestCase):
    """测试Qwen25VLModelAdapter的get_bake_names方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_get_bake_names_when_called_then_return_empty_lists(self):
        """正常：RMSNorm模型应返回空bake列表"""
        adapter = _make_adapter(self.model_type, self.model_path)

        pre_run_bake, bake_names = adapter.get_bake_names()

        self.assertEqual(pre_run_bake, [])
        self.assertEqual(bake_names, [])


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VLModelAdapterGetRotateMap(unittest.TestCase):
    """测试Qwen25VLModelAdapter的get_rotate_map方法"""

    def setUp(self):
        self.model_type = "Qwen2.5-VL-7B-Instruct"
        self.model_path = Path(".")

    def test_get_rotate_map_with_valid_config_when_called_then_return_pre_run_and_pairs(self):
        """正常：应返回pre_run列表和rotate_pairs列表"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2, hidden_size=128, num_attention_heads=8)

        pre_run_list, rot_pairs_list = adapter.get_rotate_map(block_size=8)

        self.assertIsInstance(pre_run_list, list)
        self.assertEqual(len(pre_run_list), 1)
        self.assertTrue(hasattr(pre_run_list[0], "left_rot") or hasattr(pre_run_list[0], "right_rot"))
        self.assertIsInstance(rot_pairs_list, list)
        self.assertGreater(len(rot_pairs_list), 0)


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VlGetLnFuseMapHelper(unittest.TestCase):
    """测试_qwen2_5_vl_get_ln_fuse_map辅助函数"""

    def test_get_ln_fuse_map_helper_with_valid_config_when_called_then_return_mapping(self):
        """正常：应返回各层LayerNorm到Linear的映射"""
        config = DummyConfig(num_hidden_layers=1)

        result = _qwen2_5_vl_get_ln_fuse_map(config)

        self.assertIn("model.layers.0.input_layernorm", result)
        self.assertIn("model.layers.0.post_attention_layernorm", result)
        self.assertIn("model.norm", result)
        self.assertIn("model.layers.0.self_attn.q_proj", result["model.layers.0.input_layernorm"])

    def test_get_ln_fuse_map_helper_with_zero_layers_when_called_then_return_norm_only(self):
        """边界：0层时仅包含model.norm映射"""
        config = DummyConfig(num_hidden_layers=0)

        result = _qwen2_5_vl_get_ln_fuse_map(config)

        self.assertEqual(list(result.keys()), ["model.norm"])


@unittest.skipUnless(_QWEN25_VL_IMPORT_OK, "Qwen2.5-VL dependencies are not available for import")
class TestQwen25VlGetRotateMapHelper(unittest.TestCase):
    """测试_qwen2_5_vl_get_rotate_map辅助函数"""

    def test_get_rotate_map_helper_with_valid_config_when_called_then_return_pairs(self):
        """正常：应返回pre_run和rot_pairs字典"""
        config = DummyConfig(num_hidden_layers=2, hidden_size=128, num_attention_heads=8)

        pre_run, rot_pairs = _qwen2_5_vl_get_rotate_map(config, block_size=8)

        self.assertIsNotNone(pre_run)
        self.assertIn("rot", rot_pairs)
        self.assertIn("rot_uv", rot_pairs)

    def test_get_rotate_map_helper_with_single_layer_when_called_then_contain_layer_rotations(self):
        """边界：单层模型应包含该层的rotation配置"""
        config = DummyConfig(num_hidden_layers=1, hidden_size=64, num_attention_heads=4)

        _, rot_pairs = _qwen2_5_vl_get_rotate_map(config, block_size=4)

        rot_pair = rot_pairs["rot"]
        self.assertIn("model.layers.0.self_attn.q_proj", rot_pair.right_rot)
