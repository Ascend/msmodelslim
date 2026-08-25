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

import unittest
from pathlib import Path
from typing import List
from unittest.mock import ANY, patch, Mock, MagicMock

import torch
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.glm_5_2.model_adapter import GLM52ModelAdapter
from msmodelslim.utils.exception import InvalidModelError


class DummyModelArgs:
    """模拟ModelArgs配置类"""

    def __init__(self):
        self.num_hidden_layers = 2
        self.hidden_size = 128
        self.vocab_size = 1000
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.qk_nope_head_dim = 192
        self.v_head_dim = 256
        self.index_head_dim = 128
        self.first_k_dense_replace = 3  # 前k层使用Dense FFN，之后使用MoE
        self.n_routed_experts = 8  # MoE路由专家数量
        self.n_shared_experts = 1  # MoE共享专家数量
        self.hidden_num_hidden_layers = 62
        self.indexer_types = None


class DummyRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return hidden_states * self.weight


class DummySharedHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = DummyRMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.head(self.norm(hidden_states))


class DummyMTPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enorm = DummyRMSNorm(config.hidden_size)
        self.hnorm = DummyRMSNorm(config.hidden_size)
        self.shared_head = DummySharedHead(config)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class DummyDecoderLayer(nn.Module):
    """模拟解码器层（含MTP head与forward pre-hook支持）"""

    def __init__(self, layer_id=0, args=None):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.shared_head = None
        self.hook_id = 0
        self._forward_pre_hooks_with_kwargs = {}

    def get_submodule(self, name):
        if name == "shared_head" and hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"No submodule named {name}")

    def register_forward_pre_hook(self, hook, with_kwargs=True, prepend=True):
        current_id = self.hook_id
        self.hook_id += 1
        self._forward_pre_hooks_with_kwargs[current_id] = hook

        def remove(*args, **kwargs):
            if current_id in self._forward_pre_hooks_with_kwargs:
                del self._forward_pre_hooks_with_kwargs[current_id]

        return type('', (), {'remove': remove})()

    def forward(self, hidden_states, **kwargs):
        for _, hook in self._forward_pre_hooks_with_kwargs.items():
            args_kwargs_result = hook(self, (hidden_states,), kwargs)
            if args_kwargs_result is not None:
                if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                    (hidden_states,), kwargs = args_kwargs_result
                else:
                    raise RuntimeError(
                        "forward pre-hook must return None or a tuple "
                        f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                    )
        return hidden_states


class DummyModelInner(nn.Module):
    def __init__(self, num_layers=2, config=None):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer(layer_id=i, args=config) for i in range(num_layers)])
        self.norm = DummyRMSNorm(config.hidden_size if config else 128)
        self.freqs_cis = torch.randn(100, 128)

    def forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return self.norm(hidden_states)

    def get_all_param_names(self) -> List[str]:
        return [name for name, _ in self.named_parameters()]


class DummyModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.model = DummyModelInner(num_layers=config.num_hidden_layers if config else 2, config=config)
        self.lm_head = nn.Linear(
            config.hidden_size if config else 128,
            config.vocab_size if config else 1000,
            bias=True,  # 匹配lm_head.bias参数
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden_states = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        return self.lm_head(hidden_states)

    def generate_full_state_dict(self):
        """生成完整state_dict，避免加载缺失键"""
        state_dict = {}
        for name, param in self.model.named_parameters():
            state_dict[f"model.{name}"] = param.data.clone()
        for name, param in self.lm_head.named_parameters():
            state_dict[f"lm_head.{name}"] = param.data.clone()
        return state_dict


class TestGLM52ModelAdapter(unittest.TestCase):
    def setUp(self):
        self.model_path = Path(".")
        self.model_type = "GLM-5.2"
        self.dummy_config = DummyModelArgs()
        self.dummy_config.num_hidden_layers = 62
        self.test_device = "cpu"
        self.dummy_full_state_dict = DummyModel(config=self.dummy_config).generate_full_state_dict()
        self.adapter_patcher = patch.object(GLM52ModelAdapter, "__init__", lambda x, model_path, model_type: None)

    def create_adapter(self, **kwargs):
        with self.adapter_patcher:
            adapter = GLM52ModelAdapter(model_path=self.model_path, model_type=self.model_type)
            for key, value in kwargs.items():
                setattr(adapter, key, value)
            if 'config' not in kwargs:
                adapter.config = self.dummy_config
            if 'model_path' not in kwargs:
                adapter.model_path = self.model_path
            return adapter

    # ---- 基础方法 ----

    def test_getModelPedigree_shouldReturnFixedValue_when_called(self):
        self.assertEqual(self.create_adapter().get_model_pedigree(), "glm_5_2")

    def test_getModelType_shouldReturnInitType_when_called(self):
        self.assertEqual(self.create_adapter(model_type=self.model_type).get_model_type(), self.model_type)

    def test_enableKvCache_shouldRunWithoutError_when_called(self):
        self.create_adapter(model_type=self.model_type).enable_kv_cache(Mock(), True)
        assert True

    def test_handleDataset_shouldCallGetTokenizedData_when_called(self):
        adapter = self.create_adapter()
        mock_tokenized_data = [{"input_ids": torch.tensor([1, 2, 3])}]
        adapter._get_tokenized_data = Mock(return_value=mock_tokenized_data)

        result = adapter.handle_dataset(Mock())
        adapter._get_tokenized_data.assert_called_once_with(ANY, DeviceType.NPU)
        self.assertEqual(result, mock_tokenized_data)

    # ---- _load_config ----

    @patch("msmodelslim.model.glm_5_2.model_adapter.GLM52ModelAdapter._load_model_config_json")
    def test_loadConfig_shouldReturnModelArgs_when_configJsonLoaded(self, mock_load_config_json):
        mock_load_config_json.return_value = {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "indexer_types": ["full", "shared", "full", "shared"],
        }
        result = self.create_adapter()._load_config()
        self.assertEqual(result.hidden_size, 256)
        self.assertEqual(result.num_hidden_layers, 4)
        self.assertEqual(result.indexer_types, ["full", "shared", "full", "shared"])
        self.assertEqual(result.hidden_num_hidden_layers, 4)

    @patch("msmodelslim.model.glm_5_2.model_adapter.GLM52ModelAdapter._load_model_config_json")
    def test_loadConfig_shouldReturnDefault_when_loadFails(self, mock_load_config_json):
        mock_load_config_json.side_effect = Exception("Config not found")
        result = self.create_adapter()._load_config()
        self.assertEqual(result.num_hidden_layers, 78)

    # ---- init_model ----

    @patch("msmodelslim.model.glm_5_2.model_adapter.Transformer", new=DummyModel)
    @patch("msmodelslim.model.glm_5_2.model_adapter.auto_convert_module_fp8_to_bf16")
    @patch("msmodelslim.model.glm_5_2.model_adapter.get_logger")
    @patch("torch.set_default_dtype")
    def test_initModel_shouldReturnModel_when_called(self, mock_set_dtype, mock_get_logger, mock_auto_convert):
        adapter = self.create_adapter()
        adapter.get_state_dict = Mock(return_value=self.dummy_full_state_dict)
        adapter._sync_indexer_types_from_config = Mock()

        result_model = adapter.init_model(device=self.test_device)
        result_model.load_state_dict = MagicMock()
        result_model.load_state_dict(self.dummy_full_state_dict)

        self.assertIsInstance(result_model, DummyModel)
        self.assertEqual(adapter.config.num_hidden_layers, 63)
        result_model.load_state_dict.assert_called_once_with(self.dummy_full_state_dict)
        mock_auto_convert.assert_called_once_with("", result_model, str(self.model_path))
        mock_set_dtype.assert_called_with(torch.bfloat16)

    # ---- get_weight_map ----

    @patch("msmodelslim.model.glm_5_2.model_adapter.json_safe_load")
    @patch("msmodelslim.model.glm_5_2.model_adapter.os.path.join")
    def test_getWeightMap_shouldReturnWeightMap_when_called(self, mock_path_join, mock_json_load):
        adapter = self.create_adapter()
        mock_json_load.return_value = {
            "weight_map": {"model.layers.0.self_attn.q_a_proj.weight": "model-00001.safetensors"}
        }
        mock_path_join.return_value = self.model_path / "model.safetensors.index.json"

        weight_map = adapter.get_weight_map()
        self.assertEqual(weight_map["model.layers.0.self_attn.q_a_proj.weight"], "model-00001.safetensors")

        adapter.get_weight_map()  # lru_cache
        self.assertEqual(mock_json_load.call_count, 1)

    # ---- load_mtp_if_not_load ----

    @patch(
        "msmodelslim.model.glm_5_2.model_adapter.GLM52ModelAdapter.get_mtp_layer",
        return_value=DummyMTPLayer(DummyModelArgs()),
    )
    @patch("msmodelslim.model.glm_5_2.model_adapter.wrap_mtp_decoder")
    @patch("msmodelslim.model.glm_5_2.model_adapter.get_logger")
    def test_loadMtpIfNotLoad_shouldCreateLayer_when_missing(self, mock_get_logger, mock_wrap_mtp, mock_get_mtp):
        adapter = self.create_adapter()
        dummy_decoder = DummyDecoderLayer()
        if hasattr(dummy_decoder, 'shared_head'):
            del dummy_decoder.shared_head

        adapter.load_mtp_if_not_load(mtp_decoder=dummy_decoder)
        mock_get_mtp.assert_called_once()
        mock_wrap_mtp.assert_called_once_with(mtp_decoder=dummy_decoder, mtp_layer=mock_get_mtp.return_value)

    @patch("msmodelslim.model.glm_5_2.model_adapter.GLM52ModelAdapter.get_mtp_layer")
    def test_loadMtpIfNotLoad_shouldSkip_when_layerExists(self, mock_get_mtp):
        adapter = self.create_adapter()
        dummy_decoder = DummyDecoderLayer()
        dummy_decoder.shared_head = DummySharedHead(DummyModelArgs())

        adapter.load_mtp_if_not_load(mtp_decoder=dummy_decoder)
        mock_get_mtp.assert_not_called()

    # ---- load_decoder_if_not_exist ----

    def test_loadDecoderIfNotExist_shouldCreate_when_missing(self):
        dummy_decoder = DummyDecoderLayer(layer_id=1, args=self.dummy_config)
        actual_param_names = [name for name, _ in dummy_decoder.named_parameters()]
        mock_state_dict = {name: torch.ones(1) for name in actual_param_names if "input_layernorm.weight" not in name}
        adapter = self.create_adapter(get_state_dict=Mock(return_value=mock_state_dict))

        dummy_model = DummyModel(config=self.dummy_config)
        dummy_model.model.layers = nn.ModuleList([DummyDecoderLayer(layer_id=0)])

        with patch("msmodelslim.model.glm_5_2.model_adapter.auto_convert_module_fp8_to_bf16") as mock_auto_convert:
            result_decoder = adapter.load_decoder_if_not_exist(model=dummy_model, name="model.layers.1", idx=1)
        self.assertIsInstance(result_decoder, DummyDecoderLayer)
        self.assertEqual(len(dummy_model.model.layers), 2)
        mock_auto_convert.assert_called_once_with("model.layers.1", result_decoder, str(self.model_path))

    def test_loadDecoderIfNotExist_shouldReturnExisting_when_found(self):
        adapter = self.create_adapter()
        dummy_model = DummyModel(config=self.dummy_config)
        existing_decoder = dummy_model.model.layers[0]

        with patch("msmodelslim.model.glm_5_2.model_adapter.auto_convert_module_fp8_to_bf16") as mock_auto_convert:
            result_decoder = adapter.load_decoder_if_not_exist(model=dummy_model, name="model.layers.0", idx=0)
        self.assertEqual(result_decoder, existing_decoder)
        mock_auto_convert.assert_not_called()

    # ---- generate_decoder_layer ----

    def test_generateDecoderLayer_shouldGenerateAllLayers_when_called(self):
        mock_decoders = [DummyDecoderLayer(0), DummyDecoderLayer(1), DummyDecoderLayer(2)]
        adapter = self.create_adapter(
            config=Mock(num_hidden_layers=3), load_decoder_if_not_exist=Mock(side_effect=mock_decoders)
        )

        with patch("msmodelslim.model.glm_5_2.model_adapter.GLM52ModelAdapter.load_mtp_if_not_load") as mock_load_mtp:
            layers = list(adapter.generate_decoder_layer(model=DummyModel(config=self.dummy_config)))

        self.assertEqual([name for name, _ in layers], ["model.layers.0", "model.layers.1", "model.layers.2"])
        mock_load_mtp.assert_called_once_with(mock_decoders[2])

    # ---- generate_model_forward ----

    def test_generateModelForward_shouldRaiseError_when_firstBlockInputMissing(self):
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)
        dummy_model = DummyModel(config=self.dummy_config)
        first_layer = dummy_model.model.layers[0]

        def no_op_register_forward_pre_hook(self, *args, **kwargs):
            class DummyRemove:
                @staticmethod
                def remove():
                    pass

            return DummyRemove()

        first_layer.register_forward_pre_hook = no_op_register_forward_pre_hook

        with self.assertRaises(InvalidModelError) as cm:
            gen = adapter.generate_model_forward(model=dummy_model, inputs=torch.randint(0, 1000, (1, 128)).float())
            next(gen)
        self.assertIn("Can't get first block input", str(cm.exception))

    @patch("msmodelslim.model.glm_5_2.model_adapter.dist")
    def test_generateModelForward_shouldCallBarrier_when_distInitialized(self, mock_dist):
        mock_dist.is_initialized.return_value = True

        adapter = self.create_adapter()
        adapter.generate_decoder_layer = Mock(return_value=[])

        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 10)).float()

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        try:
            next(gen)
        except StopIteration:
            pass

        mock_dist.barrier.assert_called_once()

    def test_generateModelForward_shouldCallMtpPreprocess_when_lastLayer(self):
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)
        adapter.mtp_preprocess = Mock(return_value=((torch.tensor([1]), torch.tensor([2])), {}))

        adapter.config.num_hidden_layers = 2
        adapter.generate_decoder_layer = Mock(return_value=[('model.layers.0', Mock()), ('model.layers.1', Mock())])
        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 10))

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        request = next(gen)
        self.assertEqual(request.name, 'model.layers.0')
        try:
            gen.send((torch.tensor([1]), torch.tensor([2])))
        except StopIteration:
            pass
        adapter.mtp_preprocess.assert_called_once()

    # ---- get_state_dict ----

    def test_get_state_dict_with_prefix(self):
        adapter = self.create_adapter()
        adapter.get_weight_map = Mock(return_value={"prefix.layer.weight": "file.safetensors"})
        mock_module = Mock(spec=nn.Module)
        mock_module.named_parameters.return_value = [("layer.weight", Mock())]

        with (
            patch(
                'msmodelslim.model.glm_5_2.model_adapter.get_valid_read_path',
                side_effect=lambda x, **kwargs: x,
            ),
            patch('msmodelslim.model.glm_5_2.model_adapter.safe_open') as mock_safe_open,
        ):
            mock_file = MagicMock()
            mock_file.get_tensor.return_value = "dummy_tensor"
            mock_safe_open.return_value.__enter__.return_value = mock_file

            result = adapter.get_state_dict(mock_module, prefix="prefix")
            self.assertEqual(len(result), 1)
            self.assertIn("layer.weight", result)

    # ---- 融合/旋转相关映射 ----

    def test_get_ln_fuse_map(self):
        """测试get_ln_fuse_map：仅full层添加indexer映射"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2
        adapter.config.indexer_types = ["full", "shared"]

        with patch('msmodelslim.model.glm_5_2.model_adapter.get_ln_fuse_map') as mock_get_ln_fuse:
            mock_get_ln_fuse.return_value = {
                "model.layers.0.input_layernorm": ["model.layers.0.self_attn.q_a_proj"],
                "model.layers.0.self_attn.q_a_layernorm": [],
                "model.layers.1.input_layernorm": ["model.layers.1.self_attn.q_a_proj"],
                "model.layers.1.self_attn.q_a_layernorm": [],
            }
            empty_dict, ln_linear_map = adapter.get_ln_fuse_map()

        self.assertEqual(empty_dict, {})
        self.assertIn(
            "model.layers.0.self_attn.indexer.wk",
            ln_linear_map["model.layers.0.input_layernorm"],
        )
        self.assertIn(
            "model.layers.0.self_attn.indexer.wq_b",
            ln_linear_map["model.layers.0.self_attn.q_a_layernorm"],
        )
        self.assertNotIn(
            "model.layers.1.self_attn.indexer.wk",
            ln_linear_map["model.layers.1.input_layernorm"],
        )

    def test_get_bake_names(self):
        adapter = self.create_adapter()

        result1, result2 = adapter.get_bake_names()

        self.assertEqual(result1, [])
        self.assertEqual(result2, [])

    def test_get_rotate_map(self):
        """测试get_rotate_map：仅full层添加indexer旋转映射"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2
        adapter.config.indexer_types = ["full", "shared"]

        with patch('msmodelslim.model.glm_5_2.model_adapter.get_rotate_map') as mock_get_rotate:
            mock_rot_pair = Mock()
            mock_rot_pair.right_rot = {}
            mock_rot_b_proj_pair = Mock()
            mock_rot_b_proj_pair.right_rot = {}
            mock_get_rotate.return_value = (
                Mock(),
                {'rot': mock_rot_pair, 'rot_b_proj': mock_rot_b_proj_pair},
                {'rot': torch.randn(128, 128), 'rot_b_proj': torch.randn(128, 128)},
            )

            pre_run_list, rot_pairs_list = adapter.get_rotate_map(128)

        mock_get_rotate.assert_called_once_with(adapter.config, 128, num_hidden_layers=2)
        self.assertEqual(pre_run_list, [mock_get_rotate.return_value[0]])
        self.assertEqual(len(rot_pairs_list), 2)
        self.assertIn("model.layers.0.self_attn.indexer.wk", mock_rot_pair.right_rot)
        self.assertNotIn("model.layers.1.self_attn.indexer.wk", mock_rot_pair.right_rot)

    def test_extractLayerIndex_shouldHandle_when_variousNames(self):
        adapter = self.create_adapter()
        self.assertEqual(adapter._extract_layer_index("model.layers.0.self_attn"), 0)
        self.assertEqual(adapter._extract_layer_index("model.layers.12.self_attn"), 12)
        self.assertIsNone(adapter._extract_layer_index("model.embed_tokens"))
        self.assertIsNone(adapter._extract_layer_index("model.layers.abc.self_attn"))

    def test_getAttentionModuleClsAndExtractor_shouldReturn_when_called(self):
        adapter = self.create_adapter()
        self.assertEqual(adapter.get_attention_module_cls(), "MLA")
        x = torch.tensor([1, 2, 3])
        self.assertEqual(adapter.get_attention_output_extractor()(x).tolist(), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
