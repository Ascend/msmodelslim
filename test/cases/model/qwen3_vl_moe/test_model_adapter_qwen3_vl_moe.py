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
msmodelslim.model.qwen3_vl_moe.model_adapter 模块的单元测试
"""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.graph import AdapterConfig
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.model.qwen3_vl_moe.model_adapter import (
    Qwen3VLMoeModelAdapter,
    _qwen3_vl_moe_get_ln_fuse_map,
    _qwen3_vl_moe_get_rotate_map,
)
from msmodelslim.utils.exception import InvalidModelError, UnsupportedError


class DummyTextConfig:
    """模拟 text_config"""

    def __init__(
        self,
        num_hidden_layers=2,
        hidden_size=128,
        head_dim=64,
        num_experts=4,
        decoder_sparse_step=2,
        mlp_only_layers=None,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = mlp_only_layers or []
        self.num_attention_heads = 8
        self.num_key_value_heads = 2
        self._attn_implementation = "eager"


class DummyVisionConfig:
    def __init__(self, depth=4, deepstack_visual_indexes=None):
        self.depth = depth
        self.deepstack_visual_indexes = deepstack_visual_indexes or [0, 1]


class DummyFullConfig:
    """模拟完整 Qwen3VLMoe 配置"""

    def __init__(self, num_hidden_layers=2, num_experts=4, decoder_sparse_step=2):
        self.text_config = DummyTextConfig(
            num_hidden_layers=num_hidden_layers,
            num_experts=num_experts,
            decoder_sparse_step=decoder_sparse_step,
        )
        self.vision_config = DummyVisionConfig(deepstack_visual_indexes=[0, 1])
        self.use_cache = True
        self.image_token_id = 151655


def create_adapter(model_type="Qwen3-VL-MoE-30B", model_path=None):
    """构造跳过基类初始化的适配器实例"""
    adapter = Qwen3VLMoeModelAdapter.__new__(Qwen3VLMoeModelAdapter)
    adapter.model_type = model_type
    adapter.model_path = model_path or Path("/tmp/qwen3_vl_moe")
    adapter.trust_remote_code = False
    adapter.config = DummyFullConfig()
    adapter._processor = None
    adapter._tokenizer = None
    return adapter


class _FakeDecoderLayer(nn.Module):
    """模拟 Qwen3VLMoeTextDecoderLayer"""

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Linear(8, 8, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(8)

    def forward(self, hidden_states, **_kwargs):
        return hidden_states


class _SafeOpenCtx:
    def __init__(self, tensor_map):
        self.tensor_map = tensor_map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_tensor(self, name):
        return self.tensor_map[name]


def _build_calibration_sample(image_token_id=151655, seq_len=3, hidden_size=8):
    input_ids = torch.tensor([[image_token_id, 1, 2]])
    return {
        "pixel_values": torch.randn(1, 3, 4, 4),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "input_ids": input_ids,
        "attention_mask": torch.ones(1, seq_len),
        "hidden_size": hidden_size,
    }


def _build_mock_forward_model(adapter, sample, hidden_size=8):
    inputs_embeds = torch.randn(1, sample["input_ids"].shape[1], hidden_size)
    mock_model = MagicMock()
    mock_model.config = adapter.config
    mock_model.model.visual = MagicMock()
    mock_model.model.language_model.embed_tokens.return_value = inputs_embeds
    mock_model.model.get_rope_index.return_value = (
        torch.arange(sample["input_ids"].shape[1]).unsqueeze(0),
        torch.zeros(1),
    )
    mock_model.model.language_model.rotary_emb.return_value = (
        torch.randn(1, sample["input_ids"].shape[1], hidden_size),
    )
    return mock_model, inputs_embeds


class TestQwen3VLMoeModelAdapter(unittest.TestCase):
    """测试 Qwen3VLMoeModelAdapter 类"""

    def setUp(self):
        self.model_type = "Qwen3-VL-MoE-30B"
        self.model_path = Path("/tmp/qwen3_vl_moe")

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_get_model_pedigree_return_qwen3_vl_moe_when_called(self, _mock_init):
        """正常调用时应返回 qwen3_vl_moe 谱系标识"""
        adapter = create_adapter()
        self.assertEqual(adapter.get_model_pedigree(), "qwen3_vl_moe")

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_get_model_type_return_model_type_when_initialized(self, _mock_init):
        """初始化后 get_model_type 应返回 model_type"""
        adapter = create_adapter(model_type=self.model_type)
        self.assertEqual(adapter.get_model_type(), self.model_type)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_handle_dataset_raise_unsupported_when_image_missing(self, _mock_init):
        """校准样本缺少 image 时应抛出 UnsupportedError"""
        adapter = create_adapter()
        dataset = [VlmCalibSample(text="hello", image=None)]

        with patch("msmodelslim.model.qwen3_vl_moe.model_adapter.AutoProcessor.from_pretrained"):
            with self.assertRaises(UnsupportedError) as ctx:
                adapter.handle_dataset(dataset)

        self.assertIn("image", str(ctx.exception).lower())

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_handle_dataset_raise_unsupported_when_text_missing(self, _mock_init):
        """校准样本缺少 text 时应抛出 UnsupportedError"""
        adapter = create_adapter()
        dataset = [{"text": None, "image": "/tmp/a.jpg"}]

        with patch("msmodelslim.model.qwen3_vl_moe.model_adapter.AutoProcessor.from_pretrained"):
            with self.assertRaises(UnsupportedError) as ctx:
                adapter.handle_dataset(dataset)

        self.assertIn("text", str(ctx.exception).lower())

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.tqdm", side_effect=lambda x, **_: x)
    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.get_valid_read_path", side_effect=lambda p: p)
    def test_handle_dataset_return_processed_list_when_valid_vlm_sample(self, _mock_path, _mock_tqdm, _mock_init):
        """image+text 齐全的 VlmCalibSample 应返回 processor 处理后的列表"""
        adapter = create_adapter()
        sample = VlmCalibSample(text="describe", image="/tmp/img.jpg")
        processor_out = SimpleNamespace(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[1, 1]]),
        )
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = processor_out
        expected_item = {"input_ids": processor_out.input_ids}

        with patch(
            "msmodelslim.model.qwen3_vl_moe.model_adapter.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ):
            with patch.object(adapter, "_collect_inputs_to_device", return_value=expected_item):
                result = adapter.handle_dataset([sample], device=DeviceType.CPU)

        self.assertEqual(len(result), 1)  # 校验返回一条样本
        self.assertEqual(result[0], expected_item)  # 校验内容与 _collect_inputs_to_device 一致
        mock_processor.apply_chat_template.assert_called_once()

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_enable_kv_cache_set_use_cache_true_when_need_kv_cache_true(self, _mock_init):
        """need_kv_cache=True 时应将 model.config.use_cache 设为 True"""
        adapter = create_adapter()
        model = MagicMock()
        model.config = SimpleNamespace(use_cache=False)

        adapter.enable_kv_cache(model, need_kv_cache=True)

        self.assertTrue(model.config.use_cache)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_enable_kv_cache_set_use_cache_false_when_need_kv_cache_false(self, _mock_init):
        """need_kv_cache=False 时应将 model.config.use_cache 设为 False"""
        adapter = create_adapter()
        model = MagicMock()
        model.config = SimpleNamespace(use_cache=True)

        adapter.enable_kv_cache(model, need_kv_cache=False)

        self.assertFalse(model.config.use_cache)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_get_adapter_config_for_subgraph_return_configs_when_layers_exist(self, _mock_init):
        """存在 decoder 层时应返回 norm-linear 与 ov 等 AdapterConfig"""
        adapter = create_adapter()
        adapter.config = DummyFullConfig(num_hidden_layers=2, decoder_sparse_step=2)

        result = adapter.get_adapter_config_for_subgraph()

        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(c, AdapterConfig) for c in result))
        # 每层至少 norm-linear + ov
        self.assertGreaterEqual(len(result), 2 * adapter.config.text_config.num_hidden_layers)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_get_adapter_config_for_subgraph_return_empty_list_when_zero_layers(self, _mock_init):
        """num_hidden_layers=0 时应返回空列表"""
        adapter = create_adapter()
        adapter.config.text_config.num_hidden_layers = 0

        result = adapter.get_adapter_config_for_subgraph()

        self.assertEqual(result, [])

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_get_ln_fuse_map_delegate_to_helper_when_called(self, _mock_init):
        """get_ln_fuse_map 应委托 _qwen3_vl_moe_get_ln_fuse_map 并返回空 pre_run"""
        adapter = create_adapter()
        pre_run, fused_map = adapter.get_ln_fuse_map()

        self.assertEqual(pre_run, {})
        self.assertIn("model.language_model.norm", fused_map)
        self.assertIn("model.language_model.layers.0.input_layernorm", fused_map)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_get_bake_names_return_empty_lists_when_called(self, _mock_init):
        """RMSNorm 模型不需要 bake mean，应返回两个空列表"""
        adapter = create_adapter()
        pre_run, bake_names = adapter.get_bake_names()

        self.assertEqual(pre_run, [])
        self.assertEqual(bake_names, [])

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_get_rotate_map_return_pre_run_and_pairs_when_called(self, _mock_init):
        """get_rotate_map 应返回 pre_run 列表与 rot_pairs 列表"""
        adapter = create_adapter()
        from msmodelslim.processor.quarot import QuaRotInterface

        pre_run_list, rot_pairs_list = adapter.get_rotate_map(block_size=64)

        self.assertEqual(len(pre_run_list), 1)
        self.assertIsInstance(pre_run_list[0], QuaRotInterface.RotatePair)
        self.assertTrue(len(rot_pairs_list) >= 1)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_is_moe_layer_return_false_when_in_mlp_only_layers(self, _mock_init):
        """层索引在 mlp_only_layers 中时不视为 MoE 层"""
        adapter = create_adapter()
        adapter.config.text_config.mlp_only_layers = [1]

        self.assertFalse(adapter._is_moe_layer(1))

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_is_moe_layer_return_true_when_sparse_step_matches(self, _mock_init):
        """(layer_idx+1) 整除 decoder_sparse_step 时应视为 MoE 层"""
        adapter = create_adapter()
        adapter.config.text_config.decoder_sparse_step = 2

        self.assertTrue(adapter._is_moe_layer(1))

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_is_moe_layer_return_false_when_not_sparse_step_layer(self, _mock_init):
        """非 sparse step 层且不在 mlp_only_layers 时应返回 False"""
        adapter = create_adapter()
        adapter.config.text_config.decoder_sparse_step = 2

        self.assertFalse(adapter._is_moe_layer(0))

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_init_model_raise_invalid_model_error_when_transformers_import_fails(self, _mock_init):
        """无法导入 Qwen3VLMoeForConditionalGeneration 时应抛出 InvalidModelError"""
        import builtins

        adapter = create_adapter()
        real_import = builtins.__import__

        def selective_import(name, globalns=None, localns=None, fromlist=(), level=0):
            if name == "transformers" and fromlist and "Qwen3VLMoeForConditionalGeneration" in fromlist:
                raise ImportError("no qwen3 vl moe")
            return real_import(name, globalns, localns, fromlist, level)

        with patch("builtins.__import__", side_effect=selective_import):
            with patch(
                "msmodelslim.model.qwen3_vl_moe.model_adapter.get_valid_read_path",
                return_value=str(adapter.model_path),
            ):
                with self.assertRaises(InvalidModelError) as ctx:
                    adapter.init_model()

        self.assertIn("Qwen3VLMoeForConditionalGeneration", str(ctx.exception))

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.gc.collect")
    def test_convert_single_moe_layer_skip_when_mlp_not_sparse_block(self, _mock_gc, _mock_init):
        """mlp 非 Qwen3VLMoeTextSparseMoeBlock 时应跳过转换并保留原 mlp"""
        adapter = create_adapter()
        layer = nn.Module()
        layer.mlp = nn.Linear(4, 4)
        original_mlp = layer.mlp

        adapter._convert_single_moe_layer(layer, layer_idx=0)

        self.assertIs(layer.mlp, original_mlp)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.get_valid_read_path", side_effect=lambda p, **_: p)
    def test_get_weight_map_return_weight_map_when_index_json_valid(self, _mock_path, _mock_init):
        """index.json 合法时应返回 weight_map 字典"""
        adapter = create_adapter()
        weight_map = {"model.embed_tokens.weight": "model-00001.safetensors"}

        with patch(
            "msmodelslim.model.qwen3_vl_moe.model_adapter.json_safe_load",
            return_value={"weight_map": weight_map},
        ):
            result = adapter._get_weight_map()

        self.assertEqual(result, weight_map)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_init_sets_processor_none_when_called(self, _mock_init):
        """__init__ 应将 _processor/_tokenizer 初始化为 None 并调用基类构造"""
        with patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__") as mock_super_init:
            adapter = Qwen3VLMoeModelAdapter(self.model_type, self.model_path, False)

        self.assertIsNone(adapter._processor)
        self.assertIsNone(adapter._tokenizer)
        mock_super_init.assert_called_once()

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_create_model_instance_return_eval_model_when_from_pretrained_succeeds(self, _mock_init):
        """_create_model_instance 应调用 from_pretrained 并返回 eval 后的模型"""
        adapter = create_adapter()
        fake_model = MagicMock()
        fake_model.eval.return_value = fake_model
        model_cls = MagicMock()
        model_cls.from_pretrained.return_value = fake_model

        result = adapter._create_model_instance(model_cls)

        self.assertIs(result, fake_model)
        model_cls.from_pretrained.assert_called_once()
        fake_model.eval.assert_called_once()

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.get_valid_read_path", side_effect=lambda p, **_: p)
    def test_init_model_return_model_when_import_and_load_succeed(self, _mock_path, _mock_init):
        """transformers 可导入且加载成功时应恢复层数并返回模型"""
        adapter = create_adapter()
        adapter.config.text_config.num_hidden_layers = 3
        adapter.config.text_config.decoder_sparse_step = 2

        fake_model = MagicMock()
        fake_model.config = adapter.config
        fake_model.load_state_dict = MagicMock()
        fake_model.model.language_model.layers = [MagicMock()]

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.Qwen3VLMoeForConditionalGeneration = MagicMock()

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            with patch.object(adapter, "_create_model_instance", return_value=fake_model):
                with patch.object(adapter, "_get_state_dict", return_value={}):
                    with patch.object(adapter, "_is_moe_layer", return_value=False):
                        result = adapter.init_model()

        self.assertIs(result, fake_model)
        self.assertEqual(adapter.config.text_config.num_hidden_layers, 3)
        self.assertEqual(adapter.config.text_config._attn_implementation, "eager")
        fake_model.load_state_dict.assert_called_once()

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.get_valid_read_path", side_effect=lambda p, **_: p)
    def test_init_model_convert_layer0_when_layer0_is_moe(self, _mock_path, _mock_init):
        """layer 0 为 MoE 层时 init_model 应调用 _convert_single_moe_layer"""
        adapter = create_adapter()
        adapter.config.text_config.decoder_sparse_step = 1

        layer0 = MagicMock()
        fake_model = MagicMock()
        fake_model.config = adapter.config
        fake_model.model.language_model.layers = [layer0]

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.Qwen3VLMoeForConditionalGeneration = MagicMock()

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            with patch.object(adapter, "_create_model_instance", return_value=fake_model):
                with patch.object(adapter, "_get_state_dict", return_value={}):
                    with patch.object(adapter, "_convert_single_moe_layer") as mock_convert:
                        adapter.init_model()

        mock_convert.assert_called_once_with(layer0, 0)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.get_valid_read_path", side_effect=lambda p, **_: p)
    def test_get_state_dict_return_tensors_when_weight_map_contains_keys(self, _mock_path, _mock_init):
        """weight_map 命中参数时应从 safetensors 加载对应张量"""
        adapter = create_adapter()
        module = nn.Linear(4, 3, bias=False)
        prefix = "model.language_model.layers.0"
        full_name = f"{prefix}.weight"
        weight_map = {full_name: "model-00001.safetensors"}
        tensor_map = {full_name: torch.ones_like(module.weight)}

        with (
            patch.object(adapter, "_get_weight_map", return_value=weight_map),
            patch("msmodelslim.model.qwen3_vl_moe.model_adapter.safe_open", return_value=_SafeOpenCtx(tensor_map)),
        ):
            state_dict = adapter._get_state_dict(module, prefix=prefix)

        self.assertIn("weight", state_dict)
        self.assertTrue(torch.equal(state_dict["weight"], tensor_map[full_name]))

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_get_state_dict_return_empty_dict_when_weight_map_has_no_match(self, _mock_init):
        """weight_map 无匹配键时应返回空 state_dict"""
        adapter = create_adapter()
        module = nn.Linear(4, 3, bias=False)

        with patch.object(adapter, "_get_weight_map", return_value={}):
            state_dict = adapter._get_state_dict(module, prefix="model.language_model.layers.0")

        self.assertEqual(state_dict, {})

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_load_decoder_if_not_exist_return_existing_layer_when_already_loaded(self, _mock_init):
        """层已加载且 weight.device 可访问时应直接返回已有层"""
        adapter = create_adapter()
        loaded_layer = _FakeDecoderLayer()
        model = MagicMock()
        model.get_submodule.return_value = loaded_layer

        result = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)

        self.assertIs(result, loaded_layer)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_load_decoder_if_not_exist_reload_when_weight_on_meta_device(self, _mock_init):
        """层在 meta device 上时应重新创建并加载"""
        adapter = create_adapter()
        adapter.config.text_config.decoder_sparse_step = 10

        meta_layer = _FakeDecoderLayer()

        class MetaWeightParam(nn.Parameter):
            @property
            def device(self):
                raise RuntimeError("device is meta")

        meta_layer.input_layernorm.weight = MetaWeightParam(meta_layer.input_layernorm.weight.data)

        model = MagicMock()
        model.get_submodule.return_value = meta_layer
        model.model = SimpleNamespace(language_model=SimpleNamespace(layers=nn.ModuleList()))

        fake_decoder = _FakeDecoderLayer()
        with (
            patch(
                "msmodelslim.model.qwen3_vl_moe.model_adapter.Qwen3VLMoeTextDecoderLayer",
                return_value=fake_decoder,
            ),
            patch.object(adapter, "_get_state_dict", return_value=fake_decoder.state_dict()),
        ):
            result = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)

        self.assertIsInstance(result, _FakeDecoderLayer)
        self.assertEqual(len(model.model.language_model.layers), 1)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_load_decoder_if_not_exist_create_and_append_when_layer_missing(self, _mock_init):
        """层不存在时应创建并 append 到 ModuleList"""
        adapter = create_adapter()
        adapter.config.text_config.decoder_sparse_step = 10

        model = SimpleNamespace(
            get_submodule=lambda _name: (_ for _ in ()).throw(AttributeError("not found")),
            model=SimpleNamespace(language_model=SimpleNamespace(layers=nn.ModuleList())),
        )

        fake_decoder = _FakeDecoderLayer()
        with (
            patch(
                "msmodelslim.model.qwen3_vl_moe.model_adapter.Qwen3VLMoeTextDecoderLayer",
                return_value=fake_decoder,
            ),
            patch.object(adapter, "_get_state_dict", return_value=fake_decoder.state_dict()),
        ):
            layer = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)

        self.assertIsInstance(layer, _FakeDecoderLayer)
        self.assertEqual(len(model.model.language_model.layers), 1)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_load_decoder_if_not_exist_replace_slot_when_index_exists(self, _mock_init):
        """ModuleList 已有占位槽位时应按索引替换"""
        adapter = create_adapter()
        adapter.config.text_config.decoder_sparse_step = 10

        placeholder = nn.Module()
        layers = nn.ModuleList([placeholder])
        model = SimpleNamespace(
            get_submodule=lambda _name: (_ for _ in ()).throw(AttributeError("not found")),
            model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
        )

        fake_decoder = _FakeDecoderLayer()
        with (
            patch(
                "msmodelslim.model.qwen3_vl_moe.model_adapter.Qwen3VLMoeTextDecoderLayer",
                return_value=fake_decoder,
            ),
            patch.object(adapter, "_get_state_dict", return_value=fake_decoder.state_dict()),
        ):
            layer = adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)

        self.assertIs(layer, layers[0])
        self.assertIsNot(layer, placeholder)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_load_decoder_if_not_exist_call_convert_when_moe_layer(self, _mock_init):
        """MoE 层加载后应触发 _convert_single_moe_layer"""
        adapter = create_adapter()
        adapter.config.text_config.decoder_sparse_step = 1

        model = SimpleNamespace(
            get_submodule=lambda _name: (_ for _ in ()).throw(AttributeError("not found")),
            model=SimpleNamespace(language_model=SimpleNamespace(layers=nn.ModuleList())),
        )

        fake_decoder = _FakeDecoderLayer()
        with (
            patch(
                "msmodelslim.model.qwen3_vl_moe.model_adapter.Qwen3VLMoeTextDecoderLayer",
                return_value=fake_decoder,
            ),
            patch.object(adapter, "_get_state_dict", return_value=fake_decoder.state_dict()),
            patch.object(adapter, "_convert_single_moe_layer") as mock_convert,
        ):
            adapter._load_decoder_if_not_exist(model, "model.language_model.layers.0", 0)

        mock_convert.assert_called_once()

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_generate_decoder_layer_yield_expected_pairs_when_two_layers(self, _mock_init):
        """应按 num_hidden_layers 依次 yield 层名与模块"""
        adapter = create_adapter()
        adapter.config.text_config.num_hidden_layers = 2

        with patch.object(adapter, "_load_decoder_if_not_exist", side_effect=["L0", "L1"]):
            result = list(adapter.generate_decoder_layer(model=MagicMock()))

        self.assertEqual(
            result,
            [
                ("model.language_model.layers.0", "L0"),
                ("model.language_model.layers.1", "L1"),
            ],
        )

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_generate_model_visit_yield_vision_and_decoder_when_called(self, _mock_init):
        """应先 yield vision，再 yield decoder 层访问请求"""
        adapter = create_adapter()
        fake_layer = MagicMock()
        model = SimpleNamespace(model=SimpleNamespace(visual=MagicMock()))

        decoder_req = ProcessRequest(name="model.language_model.layers.0", module=fake_layer, args=(), kwargs={})
        with (
            patch.object(
                adapter, "generate_decoder_layer", return_value=iter([("model.language_model.layers.0", fake_layer)])
            ),
            patch(
                "msmodelslim.model.qwen3_vl_moe.model_adapter.generated_decoder_layer_visit_func",
                return_value=iter([decoder_req]),
            ),
        ):
            requests = list(adapter.generate_model_visit(model))

        self.assertGreaterEqual(len(requests), 2)
        self.assertEqual(requests[0].name, "model.visual")
        self.assertEqual(requests[1].name, "model.language_model.layers.0")

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_get_adapter_config_include_up_down_when_regular_mlp_layer(self, _mock_init):
        """非 MoE 层应包含 norm-linear 与 up-down 配置"""
        adapter = create_adapter()
        adapter.config = DummyFullConfig(num_hidden_layers=1, decoder_sparse_step=2)

        result = adapter.get_adapter_config_for_subgraph()
        has_up_down = any(c.subgraph_type == "up-down" for c in result)

        self.assertTrue(has_up_down)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.gc.collect")
    def test_convert_single_moe_layer_replace_mlp_when_sparse_moe_block(self, _mock_gc, _mock_init):
        """MoE block 类型正确时应替换为 UnstackedQwen3VLMoeSparseMoeBlock"""
        adapter = create_adapter()
        layer = nn.Module()
        layer.mlp = MagicMock()

        mock_unstacked = MagicMock()
        mock_unstacked.eval.return_value = mock_unstacked

        with (
            patch(
                "msmodelslim.model.qwen3_vl_moe.model_adapter.isinstance",
                return_value=True,
            ),
            patch(
                "msmodelslim.model.qwen3_vl_moe.model_adapter.UnstackedQwen3VLMoeSparseMoeBlock",
                return_value=mock_unstacked,
            ),
        ):
            adapter._convert_single_moe_layer(layer, layer_idx=1)

        self.assertIs(layer.mlp, mock_unstacked)
        mock_unstacked._transform_weights_from_original.assert_called_once()
        mock_unstacked.eval.assert_called_once()

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.tqdm", side_effect=lambda x, **_: x)
    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.get_valid_read_path", side_effect=lambda p, **_: p)
    def test_handle_dataset_return_two_items_when_two_valid_samples(self, _mock_path, _mock_tqdm, _mock_init):
        """两个有效样本时应返回长度为 2 的列表"""
        adapter = create_adapter()
        dataset = [
            VlmCalibSample(text="a", image="/tmp/a.jpg"),
            VlmCalibSample(text="b", image="/tmp/b.jpg"),
        ]
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = SimpleNamespace()

        with (
            patch(
                "msmodelslim.model.qwen3_vl_moe.model_adapter.AutoProcessor.from_pretrained",
                return_value=mock_processor,
            ),
            patch.object(adapter, "_collect_inputs_to_device", return_value={"input_ids": torch.tensor([1])}),
        ):
            result = adapter.handle_dataset(dataset)

        self.assertEqual(len(result), 2)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_generate_model_forward_use_first_sample_when_inputs_is_list(self, _mock_init):
        """inputs 为 list 时应取首个样本进行 forward"""
        adapter = create_adapter()
        adapter.config.text_config.num_hidden_layers = 0
        sample = _build_calibration_sample()
        mock_model, _ = _build_mock_forward_model(adapter, sample)

        gen = adapter.generate_model_forward(mock_model, [sample, {"ignored": True}])
        req = next(gen)

        self.assertEqual(req.name, "model.visual")
        self.assertIs(req.module, mock_model.model.visual)

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_generate_model_forward_use_sample_directly_when_inputs_not_list(self, _mock_init):
        """inputs 非 list 时应直接使用该 dict"""
        adapter = create_adapter()
        adapter.config.text_config.num_hidden_layers = 0
        sample = _build_calibration_sample()
        mock_model, _ = _build_mock_forward_model(adapter, sample)

        gen = adapter.generate_model_forward(mock_model, sample)
        req = next(gen)

        self.assertEqual(req.name, "model.visual")

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_generate_model_forward_concat_list_image_embeds_when_tuple_returned(self, _mock_init):
        """image_embeds 为 list 时应 torch.cat 后融合"""
        adapter = create_adapter()
        adapter.config.text_config.num_hidden_layers = 0
        sample = _build_calibration_sample()
        mock_model, inputs_embeds = _build_mock_forward_model(adapter, sample)

        gen = adapter.generate_model_forward(mock_model, sample)
        next(gen)
        image_embeds = [torch.randn(2, sample["hidden_size"]), torch.randn(1, sample["hidden_size"])]
        try:
            gen.send((image_embeds, None))
        except StopIteration:
            pass

        mock_model.model.language_model.embed_tokens.assert_called_once()

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_generate_model_forward_use_tensor_image_embeds_when_not_list(self, _mock_init):
        """image_embeds 为 Tensor 时应直接 to(device,dtype) 融合"""
        adapter = create_adapter()
        adapter.config.text_config.num_hidden_layers = 0
        sample = _build_calibration_sample()
        mock_model, _ = _build_mock_forward_model(adapter, sample)

        gen = adapter.generate_model_forward(mock_model, sample)
        next(gen)
        image_embeds = torch.randn(2, sample["hidden_size"])
        try:
            gen.send((image_embeds, None))
        except StopIteration:
            pass

        mock_model.model.language_model.embed_tokens.assert_called_once()

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_generate_model_forward_expand_position_ids_when_2d(self, _mock_init):
        """position_ids 为 2D 时应 expand 为 3D mROPE 格式"""
        adapter = create_adapter()
        adapter.config.text_config.num_hidden_layers = 0
        sample = _build_calibration_sample()
        mock_model, _ = _build_mock_forward_model(adapter, sample)
        mock_model.model.get_rope_index.return_value = (
            torch.arange(sample["input_ids"].shape[1]).unsqueeze(0),
            torch.zeros(1),
        )

        with patch(
            "msmodelslim.model.qwen3_vl_moe.model_adapter.create_causal_mask",
            return_value=torch.ones(1, 1, 3, 3),
        ):
            gen = adapter.generate_model_forward(mock_model, sample)
            next(gen)
            try:
                gen.send((torch.randn(2, sample["hidden_size"]), None))
            except StopIteration:
                pass

        mock_model.model.get_rope_index.assert_called_once()

    @patch("msmodelslim.model.qwen3_vl_moe.model_adapter.VLMBaseModelAdapter.__init__", return_value=None)
    def test_generate_model_forward_run_decoder_and_deepstack_when_one_layer(self, _mock_init):
        """存在 decoder 层且 deepstack 特征时应 yield 层并注入 visual"""
        adapter = create_adapter()
        adapter.config.text_config.num_hidden_layers = 1
        sample = _build_calibration_sample()
        mock_model, inputs_embeds = _build_mock_forward_model(adapter, sample)

        fake_layer = MagicMock()
        layer_out = inputs_embeds.clone()
        deepstack = [torch.randn(1, sample["hidden_size"])]

        mock_model.model.get_rope_index.return_value = (
            torch.arange(3).view(1, 3).expand(3, 1, 3),
            torch.zeros(1),
        )

        with (
            patch.object(
                adapter,
                "generate_decoder_layer",
                return_value=iter([("model.language_model.layers.0", fake_layer)]),
            ),
            patch(
                "msmodelslim.model.qwen3_vl_moe.model_adapter.create_causal_mask",
                return_value=torch.ones(1, 1, 3, 3),
            ),
        ):
            gen = adapter.generate_model_forward(mock_model, sample)
            vis_req = next(gen)
            self.assertEqual(vis_req.name, "model.visual")

            layer_req = gen.send((torch.randn(2, sample["hidden_size"]), deepstack))
            self.assertEqual(layer_req.name, "model.language_model.layers.0")

            try:
                gen.send(layer_out)
            except StopIteration:
                pass


class TestQwen3VlMoeModuleFunctions(unittest.TestCase):
    """测试 model_adapter 模块级辅助函数"""

    def setUp(self):
        self.text_config = DummyTextConfig(num_hidden_layers=2, num_experts=2)
        self.full_config = DummyFullConfig(num_hidden_layers=2, num_experts=2)

    def test_qwen3_vl_moe_get_ln_fuse_map_include_language_model_prefix_when_called(self):
        """融合映射应使用 model.language_model 前缀"""
        result = _qwen3_vl_moe_get_ln_fuse_map(self.text_config)

        self.assertIn("model.language_model.norm", result)
        self.assertIn("model.language_model.layers.0.input_layernorm", result)
        self.assertIn(
            "model.language_model.layers.0.mlp.gate", result["model.language_model.layers.0.post_attention_layernorm"]
        )

    def test_qwen3_vl_moe_get_rotate_map_include_visual_and_text_rotations_when_called(self):
        """旋转映射应包含视觉 merger 与文本层"""
        from msmodelslim.processor.quarot import QuaRotInterface

        pre_run, rot_pairs = _qwen3_vl_moe_get_rotate_map(self.full_config, block_size=64)

        self.assertIsInstance(pre_run, QuaRotInterface.RotatePair)
        self.assertIn("model.language_model.embed_tokens", pre_run.right_rot)
        self.assertIn("model.visual.merger.linear_fc2", pre_run.left_rot)
        self.assertIn("rot", rot_pairs)
        self.assertIn("rot_uv", rot_pairs)
        self.assertIn("lm_head", rot_pairs["rot"].right_rot)


if __name__ == "__main__":
    unittest.main()
