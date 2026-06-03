#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import torch
except ModuleNotFoundError as e:
    raise unittest.SkipTest("torch is not installed; skip qwen3_vl adapter unit tests") from e

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.utils.exception import InvalidModelError, UnsupportedError


def _setup_transformers_mocks():
    """Inject minimal transformers mocks so model_adapter can be imported."""
    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
        except ImportError:
            sys.modules["transformers"] = types.ModuleType("transformers")

    transformers_mod = sys.modules["transformers"]
    if "transformers.masking_utils" not in sys.modules:
        masking_utils = types.ModuleType("transformers.masking_utils")
        masking_utils.create_causal_mask = MagicMock(return_value=torch.ones(1, 1, 10, 10))
        sys.modules["transformers.masking_utils"] = masking_utils
    setattr(transformers_mod, "masking_utils", sys.modules["transformers.masking_utils"])

    if not hasattr(transformers_mod, "Qwen3VLForConditionalGeneration"):
        transformers_mod.Qwen3VLForConditionalGeneration = MagicMock(name="Qwen3VLForConditionalGeneration")

    if "transformers.models.qwen3_vl.modeling_qwen3_vl" not in sys.modules:
        qwen3_vl_pkg = sys.modules.get("transformers.models.qwen3_vl")
        if qwen3_vl_pkg is None:
            qwen3_vl_pkg = types.ModuleType("transformers.models.qwen3_vl")
            sys.modules["transformers.models.qwen3_vl"] = qwen3_vl_pkg
        modeling = types.ModuleType("transformers.models.qwen3_vl.modeling_qwen3_vl")
        modeling.Qwen3VLTextDecoderLayer = MagicMock
        sys.modules["transformers.models.qwen3_vl.modeling_qwen3_vl"] = modeling


_setup_transformers_mocks()

try:
    from msmodelslim.model.qwen3_vl.model_adapter import (
        Qwen3VLModelAdapter,
        _qwen3_vl_get_ln_fuse_map,
        _qwen3_vl_get_rotate_map,
    )

    _QWEN3_VL_IMPORT_OK = True
except Exception:
    Qwen3VLModelAdapter = None
    _qwen3_vl_get_ln_fuse_map = None
    _qwen3_vl_get_rotate_map = None
    _QWEN3_VL_IMPORT_OK = False


class DummyTextConfig:
    def __init__(
        self,
        num_hidden_layers: int = 3,
        hidden_size: int = 128,
        num_attention_heads: int = 8,
        tie_word_embeddings: bool = False,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = 4
        self.head_dim = hidden_size // num_attention_heads
        self.tie_word_embeddings = tie_word_embeddings


class DummyVisionConfig:
    def __init__(self, depth: int = 2, deepstack_indexes=(0, 1)):
        self.depth = depth
        self.deepstack_visual_indexes = list(deepstack_indexes)


class DummyConfig:
    """Minimal config stub for Qwen3VLModelAdapter UT."""

    def __init__(
        self,
        num_hidden_layers: int = 3,
        vision_depth: int = 2,
        hidden_size: int = 128,
        num_attention_heads: int = 8,
        image_token_id: int = 151655,
        tie_word_embeddings: bool = False,
        deepstack_indexes=(0, 1),
    ):
        self.text_config = DummyTextConfig(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            tie_word_embeddings=tie_word_embeddings,
        )
        self.vision_config = DummyVisionConfig(
            depth=vision_depth,
            deepstack_indexes=deepstack_indexes,
        )
        self.use_cache = True
        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings


def _make_adapter(model_type="Qwen3-VL-8B-Instruct", model_path="."):
    with patch(
        "msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()
    ):
        return Qwen3VLModelAdapter(model_type, Path(model_path), trust_remote_code=False)


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGetModelType(unittest.TestCase):
    """测试Qwen3VLModelAdapter的get_model_type方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_get_model_type_with_valid_type_when_called_then_return_model_type(self):
        """正常：应返回初始化时的model_type"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.model_type = self.model_type

        result = adapter.get_model_type()

        self.assertEqual(result, self.model_type)


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGetModelPedigree(unittest.TestCase):
    """测试Qwen3VLModelAdapter的get_model_pedigree方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_get_model_pedigree_when_called_then_return_qwen3_vl(self):
        """正常：应返回qwen3_vl谱系标识"""
        adapter = _make_adapter(self.model_type, self.model_path)

        result = adapter.get_model_pedigree()

        self.assertEqual(result, "qwen3_vl")


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterEnableKvCache(unittest.TestCase):
    """测试Qwen3VLModelAdapter的enable_kv_cache方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
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


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterHandleDataset(unittest.TestCase):
    """测试Qwen3VLModelAdapter的handle_dataset方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_handle_dataset_with_image_and_text_when_called_then_return_processed_data(self):
        """正常：图文样本应返回processor处理后的数据"""
        adapter = _make_adapter(self.model_type, self.model_path)
        dataset = [VlmCalibSample(text="describe", image="a.jpg")]

        mock_processor = MagicMock()
        mock_inputs = {"input_ids": torch.tensor([[1, 2]])}
        mock_processor.apply_chat_template.return_value = mock_inputs
        adapter._collect_inputs_to_device = MagicMock(return_value={"input_ids": "ok"})

        with (
            patch(
                "msmodelslim.model.qwen3_vl.model_adapter.AutoProcessor.from_pretrained", return_value=mock_processor
            ),
            patch("msmodelslim.model.qwen3_vl.model_adapter.get_valid_read_path", side_effect=lambda p, *a, **k: p),
        ):
            result = adapter.handle_dataset(dataset, device=DeviceType.CPU)

        self.assertEqual(result, [{"input_ids": "ok"}])
        mock_processor.apply_chat_template.assert_called_once()
        args, kwargs = adapter._collect_inputs_to_device.call_args
        self.assertIs(args[0], mock_inputs)
        self.assertEqual(args[1], DeviceType.CPU)
        self.assertIn("input_ids", kwargs["keys"])
        self.assertIn("pixel_values", kwargs["keys"])

    def test_handle_dataset_with_empty_dataset_when_called_then_return_empty_list(self):
        """边界：空数据集应返回空列表"""
        adapter = _make_adapter(self.model_type, self.model_path)

        with patch("msmodelslim.model.qwen3_vl.model_adapter.AutoProcessor.from_pretrained", return_value=MagicMock()):
            result = adapter.handle_dataset([], device=DeviceType.CPU)

        self.assertEqual(result, [])

    def test_handle_dataset_with_missing_image_when_called_then_raise_unsupported_error(self):
        """异常：缺少image时应抛出UnsupportedError"""
        adapter = _make_adapter(self.model_type, self.model_path)

        with patch("msmodelslim.model.qwen3_vl.model_adapter.AutoProcessor.from_pretrained", return_value=MagicMock()):
            with self.assertRaises(UnsupportedError) as context:
                adapter.handle_dataset([VlmCalibSample(text="hi", image=None)], device=DeviceType.CPU)

        self.assertIn("image and text", str(context.exception))

    def test_handle_dataset_with_missing_text_when_called_then_raise_unsupported_error(self):
        """异常：缺少text时应抛出UnsupportedError"""
        adapter = _make_adapter(self.model_type, self.model_path)

        with patch("msmodelslim.model.qwen3_vl.model_adapter.AutoProcessor.from_pretrained", return_value=MagicMock()):
            with self.assertRaises(UnsupportedError) as context:
                adapter.handle_dataset([VlmCalibSample(text=None, image="a.jpg")], device=DeviceType.CPU)

        self.assertIn("image and text", str(context.exception))


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterInitModel(unittest.TestCase):
    """测试Qwen3VLModelAdapter的init_model方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_init_model_with_valid_env_when_called_then_load_and_restore_layers(self):
        """正常：应加载模型并恢复text_config.num_hidden_layers"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=5, vision_depth=2)
        origin_layers = adapter.config.text_config.num_hidden_layers

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.config = SimpleNamespace(
            text_config=SimpleNamespace(
                num_attention_heads=8,
                num_key_value_heads=4,
            ),
            num_attention_heads=None,
            num_key_value_heads=None,
        )
        mock_model.load_state_dict.return_value = ([], [])

        mock_generation_cls = MagicMock()
        mock_generation_cls.from_pretrained.return_value = mock_model

        with (
            patch("msmodelslim.model.qwen3_vl.model_adapter.get_valid_read_path", side_effect=lambda p, *a, **k: p),
            patch.object(
                sys.modules["transformers"],
                "Qwen3VLForConditionalGeneration",
                mock_generation_cls,
            ),
            patch.object(adapter, "_get_state_dict", return_value={"w": torch.randn(2)}),
            patch.object(adapter, "get_global_model_torch_dtype", return_value=torch.float32),
        ):
            model = adapter.init_model(device=DeviceType.CPU)

        self.assertIs(model, mock_model)
        self.assertEqual(adapter.config.text_config.num_hidden_layers, origin_layers)
        self.assertEqual(getattr(adapter.config.text_config, "_attn_implementation", None), "eager")
        self.assertEqual(mock_model.config.num_attention_heads, 8)
        self.assertEqual(mock_model.config.num_key_value_heads, 4)

    def test_init_model_when_import_fails_then_raise_invalid_model_error(self):
        """异常：无法导入Qwen3VLForConditionalGeneration时应抛出InvalidModelError"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig()
        real_import = builtins.__import__

        def import_mock(name, global_vars=None, local_vars=None, fromlist=(), level=0):
            if name == "transformers" and fromlist and "Qwen3VLForConditionalGeneration" in fromlist:
                raise ImportError("cannot import Qwen3VLForConditionalGeneration")
            return real_import(name, global_vars, local_vars, fromlist, level)

        with patch("builtins.__import__", side_effect=import_mock):
            with self.assertRaises(InvalidModelError) as context:
                adapter.init_model(device=DeviceType.CPU)

        self.assertIn("Qwen3VLForConditionalGeneration", str(context.exception))


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGenerateModelVisit(unittest.TestCase):
    """测试Qwen3VLModelAdapter的generate_model_visit方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_generate_model_visit_with_valid_model_when_called_then_yield_visual_then_layers(self):
        """正常：应先yield model.visual再yield decoder layers"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        mock_visual = MagicMock()
        mock_layer0 = MagicMock()
        mock_layer1 = MagicMock()
        model = MagicMock()
        model.model = MagicMock()
        model.model.visual = mock_visual

        def mock_generate_decoder_layer(_m):
            yield "model.language_model.layers.0", mock_layer0
            yield "model.language_model.layers.1", mock_layer1

        adapter.generate_decoder_layer = MagicMock(side_effect=mock_generate_decoder_layer)

        def mock_visit_func(_m, transformer_blocks=None):
            for name, layer in transformer_blocks:
                yield ProcessRequest(name=name, module=layer, args=(), kwargs={})

        with patch(
            "msmodelslim.model.qwen3_vl.model_adapter.generated_decoder_layer_visit_func", side_effect=mock_visit_func
        ):
            requests = list(adapter.generate_model_visit(model))

        self.assertEqual(requests[0].name, "model.visual")
        self.assertIs(requests[0].module, mock_visual)
        decoder_requests = [r for r in requests[1:] if "language_model.layers" in r.name]
        self.assertEqual(len(decoder_requests), 2)


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGenerateModelForward(unittest.TestCase):
    """测试Qwen3VLModelAdapter的generate_model_forward方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def _build_model_and_adapter(self, num_hidden_layers=2):
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=num_hidden_layers)

        mock_visual = MagicMock()
        mock_layer0 = MagicMock()
        mock_layer1 = MagicMock()

        model = MagicMock()
        model.model = MagicMock()
        model.model.visual = mock_visual
        model.config = SimpleNamespace(image_token_id=151655, text_config=DummyTextConfig())
        model.model.language_model = MagicMock()
        model.model.language_model.embed_tokens = MagicMock(return_value=torch.randn(1, 10, 128))
        model.model.language_model.rotary_emb = MagicMock(return_value=torch.randn(1, 10, 128))
        model.model.get_rope_index = MagicMock(return_value=(torch.arange(10, dtype=torch.long).unsqueeze(0), None))

        def mock_generate_decoder_layer(_m):
            yield "model.language_model.layers.0", mock_layer0
            if num_hidden_layers > 1:
                yield "model.language_model.layers.1", mock_layer1

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
        adapter, model, mock_visual, mock_layer0, _ = self._build_model_and_adapter()
        gen = adapter.generate_model_forward(model, [self._make_sample()])

        first_req = next(gen)
        self.assertEqual(first_req.name, "model.visual")
        self.assertIs(first_req.module, mock_visual)

        image_embeds = torch.randn(1, 10, 128)
        deepstack = [torch.randn(1, 128)]
        second_req = gen.send((image_embeds, deepstack))
        self.assertEqual(second_req.name, "model.language_model.layers.0")
        self.assertIs(second_req.module, mock_layer0)

        third_req = gen.send(torch.randn(1, 10, 128))
        self.assertEqual(third_req.name, "model.language_model.layers.1")

        with self.assertRaises(StopIteration):
            gen.send(torch.randn(1, 10, 128))

    def test_generate_model_forward_with_dict_inputs_when_called_then_use_single_sample(self):
        """边界：非list输入时应直接使用单个sample"""
        adapter, model, _, _, _ = self._build_model_and_adapter(num_hidden_layers=1)
        gen = adapter.generate_model_forward(model, self._make_sample())

        first_req = next(gen)
        self.assertEqual(first_req.name, "model.visual")

    def test_generate_model_forward_with_list_image_embeds_when_called_then_concat_embeds(self):
        """边界：image_embeds为list时应concat后再融合"""
        adapter, model, _, mock_layer0, _ = self._build_model_and_adapter(num_hidden_layers=1)
        gen = adapter.generate_model_forward(model, self._make_sample())

        next(gen)
        layer_req = gen.send(([torch.randn(5, 128), torch.randn(5, 128)], None))
        self.assertEqual(layer_req.name, "model.language_model.layers.0")
        self.assertIs(layer_req.module, mock_layer0)

        with self.assertRaises(StopIteration):
            gen.send(torch.randn(1, 10, 128))


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGenerateDecoderLayer(unittest.TestCase):
    """测试Qwen3VLModelAdapter的generate_decoder_layer方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_generate_decoder_layer_with_valid_model_when_called_then_yield_layer_names(self):
        """正常：应按层索引yield language_model层名称"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=3)
        dummy_model = MagicMock()
        adapter._load_decoder_if_not_exist = MagicMock(side_effect=lambda _m, _n, i: f"layer-{i}")

        items = list(adapter.generate_decoder_layer(dummy_model))

        self.assertEqual(
            items,
            [
                ("model.language_model.layers.0", "layer-0"),
                ("model.language_model.layers.1", "layer-1"),
                ("model.language_model.layers.2", "layer-2"),
            ],
        )

    def test_generate_decoder_layer_with_zero_layers_when_called_then_yield_nothing(self):
        """边界：num_hidden_layers为0时不应yield任何层"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=0)
        adapter._load_decoder_if_not_exist = MagicMock()

        items = list(adapter.generate_decoder_layer(MagicMock()))

        self.assertEqual(items, [])


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGetAdapterConfigForSubgraph(unittest.TestCase):
    """测试Qwen3VLModelAdapter的get_adapter_config_for_subgraph方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_get_adapter_config_for_subgraph_with_valid_layers_when_called_then_return_configs(self):
        """正常：每层应生成4个AdapterConfig（含ov）"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        result = adapter.get_adapter_config_for_subgraph()

        self.assertEqual(len(result), 8)
        self.assertIsInstance(result[0], AdapterConfig)
        self.assertEqual(result[0].subgraph_type, "norm-linear")
        self.assertEqual(
            result[0].mapping.source,
            "model.language_model.layers.0.input_layernorm",
        )
        ov_config = result[2]
        self.assertEqual(ov_config.subgraph_type, "ov")
        self.assertEqual(ov_config.extra_config, {"group_method": "max"})

    def test_get_adapter_config_for_subgraph_with_zero_layers_when_called_then_return_empty_list(self):
        """边界：num_hidden_layers为0时应返回空列表"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=0)

        result = adapter.get_adapter_config_for_subgraph()

        self.assertEqual(result, [])


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGetLnFuseMap(unittest.TestCase):
    """测试Qwen3VLModelAdapter的get_ln_fuse_map方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_get_ln_fuse_map_with_valid_config_when_called_then_return_fused_map(self):
        """正常：应返回空pre_run和language_model层映射"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        pre_run, fused_map = adapter.get_ln_fuse_map()

        self.assertEqual(pre_run, {})
        self.assertIn("model.language_model.layers.0.input_layernorm", fused_map)
        self.assertIn("model.language_model.norm", fused_map)
        self.assertEqual(fused_map["model.language_model.norm"], ["lm_head"])


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGetBakeNames(unittest.TestCase):
    """测试Qwen3VLModelAdapter的get_bake_names方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_get_bake_names_when_called_then_return_empty_lists(self):
        """正常：RMSNorm模型应返回空bake列表"""
        adapter = _make_adapter(self.model_type, self.model_path)

        pre_run_bake, bake_names = adapter.get_bake_names()

        self.assertEqual(pre_run_bake, [])
        self.assertEqual(bake_names, [])


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGetRotateMap(unittest.TestCase):
    """测试Qwen3VLModelAdapter的get_rotate_map方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_get_rotate_map_with_valid_config_when_called_then_return_pre_run_and_pairs(self):
        """正常：tie_word_embeddings为False时应返回rotation配置"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2, tie_word_embeddings=False)

        pre_run_list, rot_pairs_list = adapter.get_rotate_map(block_size=8)

        self.assertEqual(len(pre_run_list), 1)
        self.assertGreater(len(rot_pairs_list), 0)

    def test_get_rotate_map_with_tie_word_embeddings_on_config_when_called_then_raise_error(self):
        """异常：config.tie_word_embeddings为True时应抛出UnsupportedError"""
        with patch(
            "msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()
        ):
            adapter = Qwen3VLModelAdapter.__new__(Qwen3VLModelAdapter)
            adapter.config = MagicMock()
            adapter.config.tie_word_embeddings = True
            adapter.config.text_config = MagicMock(tie_word_embeddings=False)

        with self.assertRaises(UnsupportedError) as context:
            adapter.get_rotate_map(block_size=64)

        self.assertIn("tie_word_embeddings", str(context.exception))
        self.assertIn("QuaRot", str(context.exception))

    def test_get_rotate_map_with_tie_word_embeddings_on_text_config_when_called_then_raise_error(self):
        """异常：text_config.tie_word_embeddings为True时应抛出UnsupportedError"""
        with patch(
            "msmodelslim.model.common.vlm_base.SafeGenerator.get_config_from_pretrained", return_value=DummyConfig()
        ):
            adapter = Qwen3VLModelAdapter.__new__(Qwen3VLModelAdapter)
            adapter.config = MagicMock()
            adapter.config.tie_word_embeddings = False
            adapter.config.text_config = MagicMock(tie_word_embeddings=True)

        with self.assertRaises(UnsupportedError) as context:
            adapter.get_rotate_map(block_size=64)

        self.assertIn("tie_word_embeddings", str(context.exception))


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGetWeightMap(unittest.TestCase):
    """测试Qwen3VLModelAdapter的_get_weight_map方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_get_weight_map_with_valid_index_when_called_then_return_weight_map(self):
        """正常：应从index.json加载weight_map"""
        adapter = _make_adapter(self.model_type, self.model_path)
        index_data = {
            "weight_map": {
                "model.language_model.layers.0.weight": "model-00001.safetensors",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter.model_path = tmpdir
            with patch("msmodelslim.model.qwen3_vl.model_adapter.json_safe_load", return_value=index_data):
                adapter._get_weight_map.cache_clear()
                result = adapter._get_weight_map()

        self.assertEqual(result["model.language_model.layers.0.weight"], "model-00001.safetensors")


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterGetStateDict(unittest.TestCase):
    """测试Qwen3VLModelAdapter的_get_state_dict方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
        self.model_path = Path(".")

    def test_get_state_dict_with_valid_weights_when_called_then_load_tensors(self):
        """正常：应从safetensors加载权重"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.model_path = "."
        linear = torch.nn.Linear(4, 8)
        weight_map = {"weight": "model.safetensors", "bias": "model.safetensors"}

        with patch.object(adapter, "_get_weight_map", return_value=weight_map):
            with patch("msmodelslim.model.qwen3_vl.model_adapter.safe_open") as mock_safe_open:
                mock_f = MagicMock()
                mock_f.get_tensor = lambda name: torch.randn(8, 4) if name == "weight" else torch.randn(8)
                mock_safe_open.return_value.__enter__ = MagicMock(return_value=mock_f)
                mock_safe_open.return_value.__exit__ = MagicMock(return_value=False)
                with patch(
                    "msmodelslim.model.qwen3_vl.model_adapter.get_valid_read_path", side_effect=lambda p, *a, **k: p
                ):
                    result = adapter._get_state_dict(linear, prefix="")

        self.assertIn("weight", result)
        self.assertIn("bias", result)

    def test_get_state_dict_with_param_not_in_weight_map_when_called_then_skip_param(self):
        """边界：weight_map中不存在的参数应被跳过"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.model_path = "."
        linear = torch.nn.Linear(4, 8)

        with patch.object(adapter, "_get_weight_map", return_value={}):
            result = adapter._get_state_dict(linear, prefix="")

        self.assertEqual(result, {})


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VLModelAdapterLoadDecoderIfNotExist(unittest.TestCase):
    """测试Qwen3VLModelAdapter的_load_decoder_if_not_exist方法"""

    def setUp(self):
        self.model_type = "Qwen3-VL-8B-Instruct"
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

        result = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)

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
        model.parameters = MagicMock(return_value=iter([torch.randn(1, dtype=torch.float32)]))
        mock_module_list = []
        model.model = MagicMock()
        model.model.language_model = MagicMock()
        model.model.language_model.layers = mock_module_list

        mock_decoder = MagicMock()
        mock_decoder.eval = MagicMock(return_value=mock_decoder)

        with patch("msmodelslim.model.qwen3_vl.model_adapter.Qwen3VLTextDecoderLayer", return_value=mock_decoder):
            with patch.object(adapter, "_get_state_dict", return_value={"weight": torch.randn(1)}):
                with patch.object(torch.nn.Linear, "reset_parameters", lambda self: None):
                    result = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)

        self.assertIs(result, mock_decoder)
        mock_decoder.load_state_dict.assert_called_once()

    def test_load_decoder_if_not_exist_when_layer_missing_then_append_to_module_list(self):
        """异常：层不存在时应创建并追加到module_list"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        model = MagicMock()
        model.get_submodule = MagicMock(side_effect=AttributeError("no such module"))
        model.parameters = MagicMock(return_value=iter([torch.randn(1, dtype=torch.float32)]))
        mock_module_list = []
        model.model = MagicMock()
        model.model.language_model = MagicMock()
        model.model.language_model.layers = mock_module_list

        mock_decoder = MagicMock()
        mock_decoder.eval = MagicMock(return_value=mock_decoder)

        with patch("msmodelslim.model.qwen3_vl.model_adapter.Qwen3VLTextDecoderLayer", return_value=mock_decoder):
            with patch.object(adapter, "_get_state_dict", return_value={"weight": torch.randn(1)}):
                with patch.object(torch.nn.Linear, "reset_parameters", lambda self: None):
                    result = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)

        self.assertIs(result, mock_decoder)
        self.assertIn(mock_decoder, mock_module_list)

    def test_load_decoder_if_not_exist_when_index_exists_then_replace_layer(self):
        """边界：module_list已有该索引时应替换层"""
        adapter = _make_adapter(self.model_type, self.model_path)
        adapter.config = DummyConfig(num_hidden_layers=2)

        model = MagicMock()
        model.get_submodule = MagicMock(side_effect=AttributeError("no such module"))
        model.parameters = MagicMock(return_value=iter([torch.randn(1, dtype=torch.float32)]))
        old_layer = MagicMock()
        mock_module_list = [old_layer]
        model.model = MagicMock()
        model.model.language_model = MagicMock()
        model.model.language_model.layers = mock_module_list

        mock_decoder = MagicMock()
        mock_decoder.eval = MagicMock(return_value=mock_decoder)

        with patch("msmodelslim.model.qwen3_vl.model_adapter.Qwen3VLTextDecoderLayer", return_value=mock_decoder):
            with patch.object(adapter, "_get_state_dict", return_value={"weight": torch.randn(1)}):
                with patch.object(torch.nn.Linear, "reset_parameters", lambda self: None):
                    result = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)

        self.assertIs(result, mock_decoder)
        self.assertEqual(mock_module_list[0], mock_decoder)


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VlGetLnFuseMapHelper(unittest.TestCase):
    """测试_qwen3_vl_get_ln_fuse_map辅助函数"""

    def test_get_ln_fuse_map_helper_with_valid_config_when_called_then_return_mapping(self):
        """正常：应返回language_model层LayerNorm映射"""
        config = DummyTextConfig(num_hidden_layers=1)

        result = _qwen3_vl_get_ln_fuse_map(config)

        self.assertIn("model.language_model.layers.0.input_layernorm", result)
        self.assertIn("model.language_model.norm", result)

    def test_get_ln_fuse_map_helper_with_zero_layers_when_called_then_return_norm_only(self):
        """边界：0层时仅包含model.language_model.norm映射"""
        config = DummyTextConfig(num_hidden_layers=0)

        result = _qwen3_vl_get_ln_fuse_map(config)

        self.assertEqual(list(result.keys()), ["model.language_model.norm"])


@unittest.skipUnless(_QWEN3_VL_IMPORT_OK, "Qwen3-VL dependencies are not available for import")
class TestQwen3VlGetRotateMapHelper(unittest.TestCase):
    """测试_qwen3_vl_get_rotate_map辅助函数"""

    def test_get_rotate_map_helper_with_deepstack_indexes_when_called_then_include_deepstack(self):
        """正常：含deepstack_visual_indexes时应生成deepstack rotation"""
        config = DummyConfig(num_hidden_layers=1, deepstack_indexes=(0, 1))

        pre_run, rot_pairs = _qwen3_vl_get_rotate_map(config, block_size=8)

        self.assertIn("model.visual.deepstack_merger_list.0.linear_fc2", pre_run.left_rot)
        self.assertIn("model.visual.deepstack_merger_list.1.linear_fc2", pre_run.left_rot)
        self.assertIn("rot", rot_pairs)
        self.assertIn("rot_uv", rot_pairs)

    def test_get_rotate_map_helper_without_deepstack_attr_when_called_then_skip_deepstack(self):
        """边界：vision_config无deepstack_visual_indexes时不生成deepstack rotation"""
        config = DummyConfig(num_hidden_layers=1)
        config.vision_config = SimpleNamespace(depth=2)

        pre_run, rot_pairs = _qwen3_vl_get_rotate_map(config, block_size=8)

        deepstack_keys = [k for k in pre_run.left_rot if "deepstack_merger_list" in k]
        self.assertEqual(deepstack_keys, [])
        self.assertIn("rot", rot_pairs)
