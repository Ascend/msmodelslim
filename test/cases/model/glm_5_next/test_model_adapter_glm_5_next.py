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
from unittest.mock import ANY, MagicMock, Mock, patch

import torch
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.glm_5_next.model_adapter import (
    Glm5NextModelAdapter,
    _get_full_expert_range,
)
from msmodelslim.utils.exception import InvalidModelError


class DummyTextConfig:
    """模拟 text_config（Glm5NextConfig.text_config）。"""

    def __init__(self):
        self.num_hidden_layers = 2
        self.hidden_size = 128
        self.vocab_size = 1000
        self.layer_types = ["deepseek_sparse_attention", "linear_attention"]
        self.mlp_layer_types = ["dense", "sparse"]
        self.indexer_types = ["full", "shared"]
        self.num_nextn_predict_layers = 1
        self.n_routed_experts = 4
        self.q_lora_rank = 64
        self.v_head_dim = 64
        self.kv_lora_rank = 64
        self.qk_nope_head_dim = 64
        self.qk_rope_head_dim = 32


class DummyMultimodalConfig:
    """模拟多模态 config（含 text_config 包装）。"""

    def __init__(self):
        self.text_config = DummyTextConfig()


class DummyRMSNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return hidden_states * self.weight


class DummySharedHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = DummyRMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class DummyMTPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enorm = DummyRMSNorm(config.hidden_size)
        self.hnorm = DummyRMSNorm(config.hidden_size)
        self.shared_head = DummySharedHead(config)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class DummyDecoderLayer(nn.Module):
    """模拟解码器层，支持 forward pre-hook。"""

    def __init__(self, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.shared_head = None

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class DummyLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer(i) for i in range(config.num_hidden_layers)])
        self.norm = DummyRMSNorm(config.hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class DummyModel(nn.Module):
    """模拟骨架模型：model.language_model.layers 结构 + lm_head。"""

    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = DummyLanguageModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, **kwargs):
        hidden_states = self.model.language_model.embed_tokens(input_ids)
        for layer in self.model.language_model.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(self.model.language_model.norm(hidden_states))


class TestGetFullExpertRange(unittest.TestCase):
    def test_shouldReturnFullRange_when_routedExpertsIsInt(self):
        config = Mock(n_routed_experts=8)

        self.assertEqual(_get_full_expert_range(config), (0, 8))

    def test_shouldReturnZeroRange_when_noRoutedExperts(self):
        config = Mock(spec=[])

        self.assertEqual(_get_full_expert_range(config), (0, 0))

    def test_shouldReturnZeroRange_when_routedExpertsNotInt(self):
        config = Mock()
        config.n_routed_experts = None

        self.assertEqual(_get_full_expert_range(config), (0, 0))


class TestGlm5NextModelAdapter(unittest.TestCase):
    def setUp(self):
        self.model_path = Path(".")
        self.model_type = "GLM-5.3-Flash"
        self.dummy_config = DummyTextConfig()
        self.adapter_patcher = patch.object(Glm5NextModelAdapter, "__init__", lambda x, model_path, model_type: None)

    def create_adapter(self, **kwargs):
        with self.adapter_patcher:
            adapter = Glm5NextModelAdapter(model_path=self.model_path, model_type=self.model_type)
            for key, value in kwargs.items():
                setattr(adapter, key, value)
            if 'config' not in kwargs:
                adapter.config = self.dummy_config
            if 'model_path' not in kwargs:
                adapter.model_path = self.model_path
            if 'model_type' not in kwargs:
                adapter.model_type = self.model_type
            return adapter

    # ---- 基础方法 ----

    def test_getModelPedigree_shouldReturnFixedValue_when_called(self):
        self.assertEqual(self.create_adapter().get_model_pedigree(), 'glm_5_next')

    def test_getModelType_shouldReturnInitType_when_called(self):
        self.assertEqual(self.create_adapter().get_model_type(), self.model_type)

    def test_getTextConfig_shouldReturnTextConfig_when_multimodal(self):
        config = DummyMultimodalConfig()
        adapter = self.create_adapter(config=config)

        self.assertIs(adapter._get_text_config(), config.text_config)

    def test_getTextConfig_shouldReturnSelf_when_noTextConfig(self):
        adapter = self.create_adapter()

        self.assertIs(adapter._get_text_config(), self.dummy_config)

    def test_enableKvCache_shouldRunWithoutError_when_called(self):
        self.create_adapter().enable_kv_cache(Mock(), True)

    def test_handleDataset_shouldCallGetTokenizedData_when_called(self):
        adapter = self.create_adapter()
        mock_tokenized_data = [{"input_ids": torch.tensor([1, 2, 3])}]
        adapter._get_tokenized_data = Mock(return_value=mock_tokenized_data)

        result = adapter.handle_dataset(Mock())

        adapter._get_tokenized_data.assert_called_once_with(ANY, DeviceType.NPU)
        self.assertEqual(result, mock_tokenized_data)

    def test_getBakeNames_shouldReturnEmpty_when_called(self):
        result1, result2 = self.create_adapter().get_bake_names()

        self.assertEqual(result1, [])
        self.assertEqual(result2, [])

    def test_getAttentionModuleCls_shouldReturnGlm5Next_when_called(self):
        self.assertEqual(self.create_adapter().get_attention_module_cls(), "Glm5NextTextAttention")

    def test_getAttentionOutputExtractor_shouldReturnIdentity_when_called(self):
        x = torch.tensor([1.0, 2.0, 3.0])

        self.assertTrue(torch.equal(self.create_adapter().get_attention_output_extractor()(x), x))

    # ---- _layer_has_indexer ----

    def test_layerHasIndexer_shouldReturnTrue_when_mtpLayer(self):
        adapter = self.create_adapter()

        # MTP 层（idx >= num_hidden_layers）总是 DSA + 有 indexer
        self.assertTrue(adapter._layer_has_indexer(self.dummy_config.num_hidden_layers))

    def test_layerHasIndexer_shouldFollowIndexerTypes_when_dsaLayer(self):
        adapter = self.create_adapter()

        # layer 0: DSA + indexer_types=["full", "shared"] -> True
        self.assertTrue(adapter._layer_has_indexer(0))

    def test_layerHasIndexer_shouldReturnFalse_when_sharedIndexer(self):
        adapter = self.create_adapter()

        # layer 1: linear_attention（非 DSA）-> False
        self.assertFalse(adapter._layer_has_indexer(1))

    def test_layerHasIndexer_shouldReturnFalse_when_outsideIndexerTypes(self):
        adapter = self.create_adapter()
        adapter.config.layer_types = ["deepseek_sparse_attention"] * 2
        adapter.config.indexer_types = None

        self.assertFalse(adapter._layer_has_indexer(0))

    # ---- _attach_visual_module / _has_visual_module ----

    @patch("msmodelslim.model.glm_5_next.model_adapter.safe_open")
    @patch("msmodelslim.model.glm_5_next.model_adapter.os.listdir")
    def test_attachVisualModule_shouldBuildParameterTree_when_visualWeightsExist(self, mock_listdir, mock_safe_open):
        mock_listdir.return_value = ["model-00001.safetensors"]
        visual_tensor = torch.randn(4, 4)
        mock_file = MagicMock()
        mock_file.keys.return_value = ["model.visual.merger.down_proj.weight"]
        mock_file.get_tensor.return_value = visual_tensor
        mock_safe_open.return_value.__enter__.return_value = mock_file

        model = DummyModel(self.dummy_config)
        adapter = self.create_adapter()
        adapter._attach_visual_module(model)

        # 挂载点：model.model.visual
        self.assertTrue(hasattr(model.model, 'visual'))
        # 参数树：visual.merger.down_proj.weight
        param_names = [name for name, _ in model.model.visual.named_parameters()]
        self.assertEqual(param_names, ["merger.down_proj.weight"])
        attached = dict(model.model.visual.named_parameters())["merger.down_proj.weight"]
        self.assertTrue(torch.equal(attached.data, visual_tensor))
        self.assertFalse(attached.requires_grad)

    @patch("msmodelslim.model.glm_5_next.model_adapter.safe_open")
    @patch("msmodelslim.model.glm_5_next.model_adapter.os.listdir")
    def test_attachVisualModule_shouldSkip_when_noVisualWeights(self, mock_listdir, mock_safe_open):
        mock_listdir.return_value = ["model-00001.safetensors"]
        mock_file = MagicMock()
        mock_file.keys.return_value = ["model.language_model.embed_tokens.weight"]
        mock_safe_open.return_value.__enter__.return_value = mock_file

        model = DummyModel(self.dummy_config)
        adapter = self.create_adapter()
        adapter._attach_visual_module(model)

        self.assertFalse(hasattr(model.model, 'visual'))

    @patch("msmodelslim.model.glm_5_next.model_adapter.safe_open")
    @patch("msmodelslim.model.glm_5_next.model_adapter.os.listdir")
    def test_attachVisualModule_shouldSkipNonSafetensors_when_scanning(self, mock_listdir, mock_safe_open):
        mock_listdir.return_value = ["config.json", "tokenizer.json"]

        model = DummyModel(self.dummy_config)
        adapter = self.create_adapter()
        adapter._attach_visual_module(model)

        mock_safe_open.assert_not_called()
        self.assertFalse(hasattr(model.model, 'visual'))

    @patch("msmodelslim.model.glm_5_next.model_adapter.safe_open")
    @patch("msmodelslim.model.glm_5_next.model_adapter.os.listdir")
    def test_attachVisualModule_shouldContinueOnError_when_fileUnreadable(self, mock_listdir, mock_safe_open):
        mock_listdir.return_value = ["bad-00001.safetensors", "good-00001.safetensors"]
        visual_tensor = torch.randn(2, 2)
        good_file = MagicMock()
        good_file.keys.return_value = ["model.visual.patch.weight"]
        good_file.get_tensor.return_value = visual_tensor

        def safe_open_side_effect(path, **kwargs):
            ctx = MagicMock()
            if "bad" in str(path):
                ctx.__enter__.side_effect = Exception("corrupted file")
            else:
                ctx.__enter__.return_value = good_file
            return ctx

        mock_safe_open.side_effect = safe_open_side_effect

        model = DummyModel(self.dummy_config)
        adapter = self.create_adapter()
        adapter._attach_visual_module(model)

        # 坏文件被跳过，好文件的 visual 权重仍被挂载
        self.assertTrue(hasattr(model.model, 'visual'))
        self.assertEqual(mock_safe_open.call_count, 2)

    @patch("msmodelslim.model.glm_5_next.model_adapter.safe_open")
    @patch("msmodelslim.model.glm_5_next.model_adapter.os.listdir")
    def test_hasVisualModule_shouldReturnTrue_when_visualWeightsExist(self, mock_listdir, mock_safe_open):
        mock_listdir.return_value = ["model-00001.safetensors"]
        mock_file = MagicMock()
        mock_file.keys.return_value = ["model.visual.merger.down_proj.weight"]
        mock_safe_open.return_value.__enter__.return_value = mock_file

        self.assertTrue(self.create_adapter()._has_visual_module())

    @patch("msmodelslim.model.glm_5_next.model_adapter.safe_open")
    @patch("msmodelslim.model.glm_5_next.model_adapter.os.listdir")
    def test_hasVisualModule_shouldReturnFalse_when_noVisualWeights(self, mock_listdir, mock_safe_open):
        mock_listdir.return_value = ["model-00001.safetensors"]
        mock_file = MagicMock()
        mock_file.keys.return_value = ["model.language_model.norm.weight"]
        mock_safe_open.return_value.__enter__.return_value = mock_file

        self.assertFalse(self.create_adapter()._has_visual_module())

    @patch("msmodelslim.model.glm_5_next.model_adapter.os.listdir")
    def test_hasVisualModule_shouldReturnFalse_when_noSafetensorsFiles(self, mock_listdir):
        mock_listdir.return_value = ["config.json"]

        self.assertFalse(self.create_adapter()._has_visual_module())

    # ---- load_mtp_if_not_load ----

    def test_loadMtpIfNotLoad_shouldSkip_when_layerExists(self):
        adapter = self.create_adapter()
        adapter.get_state_dict = Mock()
        dummy_decoder = DummyDecoderLayer()
        dummy_decoder.shared_head = DummySharedHead(self.dummy_config)

        adapter.load_mtp_if_not_load(mtp_decoder=dummy_decoder)

        # shared_head 已存在，直接跳过 MTP 组件构建
        adapter.get_state_dict.assert_not_called()

    # ---- generate_decoder_layer ----

    def test_generateDecoderLayer_shouldYieldAllAndMtpLayers_when_called(self):
        adapter = self.create_adapter()
        # 2 常规层 + 1 MTP 层
        decoders = [DummyDecoderLayer(i) for i in range(3)]
        adapter.load_decoder_if_not_exist = Mock(side_effect=decoders)
        adapter.load_mtp_if_not_load = Mock()

        layers = list(adapter.generate_decoder_layer(model=DummyModel(self.dummy_config)))

        self.assertEqual(
            [name for name, _ in layers],
            ["model.language_model.layers.0", "model.language_model.layers.1", "model.language_model.layers.2"],
        )
        # MTP 层（idx=2 >= num_hidden_layers=2）应触发 load_mtp_if_not_load
        adapter.load_mtp_if_not_load.assert_called_once_with(decoders[2])

    def test_generateDecoderLayer_shouldNotLoadMtp_when_noNextnLayers(self):
        adapter = self.create_adapter()
        adapter.config.num_nextn_predict_layers = 0
        decoders = [DummyDecoderLayer(i) for i in range(2)]
        adapter.load_decoder_if_not_exist = Mock(side_effect=decoders)
        adapter.load_mtp_if_not_load = Mock()

        layers = list(adapter.generate_decoder_layer(model=DummyModel(self.dummy_config)))

        self.assertEqual(len(layers), 2)
        adapter.load_mtp_if_not_load.assert_not_called()

    # ---- generate_model_forward ----

    def test_generateModelForward_shouldRaiseError_when_firstBlockInputMissing(self):
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)
        dummy_model = DummyModel(self.dummy_config)
        first_layer = dummy_model.model.language_model.layers[0]

        def no_op_register_forward_pre_hook(*args, **kwargs):
            class DummyRemove:
                @staticmethod
                def remove():
                    pass

            return DummyRemove()

        first_layer.register_forward_pre_hook = no_op_register_forward_pre_hook

        with self.assertRaises(InvalidModelError) as cm:
            gen = adapter.generate_model_forward(model=dummy_model, inputs=torch.randint(0, 1000, (1, 8)))
            next(gen)
        self.assertIn("Can't get first block input", str(cm.exception))

    @patch("msmodelslim.model.glm_5_next.model_adapter.dist")
    def test_generateModelForward_shouldCallBarrier_when_distInitialized(self, mock_dist):
        mock_dist.is_initialized.return_value = True

        adapter = self.create_adapter()
        adapter.generate_decoder_layer = Mock(return_value=[])
        dummy_model = DummyModel(self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 8))

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        try:
            next(gen)
        except StopIteration:
            pass

        mock_dist.barrier.assert_called_once()

    def test_generateModelForward_shouldCallMtpPreprocess_when_mtpLayer(self):
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)
        adapter.mtp_preprocess = Mock(return_value=torch.randn(1, 8, 128))
        # 2 常规层 + 1 MTP 层
        adapter.generate_decoder_layer = Mock(
            return_value=[
                ('model.language_model.layers.0', Mock()),
                ('model.language_model.layers.1', Mock()),
                ('model.language_model.layers.2', Mock()),
            ]
        )
        dummy_model = DummyModel(self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 8))

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        request = next(gen)
        self.assertEqual(request.name, 'model.language_model.layers.0')
        # 持续 send 推进全部层（含 MTP 层 idx=2），触发 mtp_preprocess
        try:
            while True:
                gen.send(torch.randn(1, 8, 128))
        except StopIteration:
            pass
        adapter.mtp_preprocess.assert_called_once()

    # ---- mtp_preprocess ----

    def test_mtpPreprocess_shouldCollapse_when_hiddenStatesAre4D(self):
        adapter = self.create_adapter()
        config = self.dummy_config
        dummy_model = DummyModel(config)
        mtp_layer = DummyMTPLayer(config)

        # 4D 输入：[batch, seq, hc, hidden] -> collapse(dim=2) -> [batch, seq, hidden]
        hidden_states = torch.randn(1, 4, 3, config.hidden_size)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))

        with patch.object(adapter, '_get_text_config', return_value=config):
            result = adapter.mtp_preprocess(dummy_model, mtp_layer, hidden_states, {'input_ids': input_ids})

        self.assertEqual(result.shape, (1, 4, config.hidden_size))

    def test_mtpPreprocess_shouldSkipLogitsPath_when_inputIdsUnavailable(self):
        adapter = self.create_adapter()
        config = self.dummy_config
        dummy_model = DummyModel(config)
        mtp_layer = DummyMTPLayer(config)

        hidden_states = torch.randn(1, 4, config.hidden_size)

        with patch.object(adapter, '_get_text_config', return_value=config):
            # inputs 无法解析出 input_ids -> 直接返回 collapse 后的 hidden_states
            result = adapter.mtp_preprocess(dummy_model, mtp_layer, hidden_states, None)

        self.assertTrue(torch.equal(result, hidden_states))

    # ---- get_rotate_map（含 merger 主阶段旋转） ----

    def test_getRotateMap_shouldPutMergerInMainPhase_when_visualExists(self):
        """visual merger 左旋应在主阶段 rot_pairs（而非 pre_run），不改公共 QuarotProcessor。"""
        adapter = self.create_adapter()
        adapter._has_visual_module = Mock(return_value=True)
        adapter._layer_has_indexer = Mock(return_value=False)

        pre_run_list, rot_pairs_list = adapter.get_rotate_map(block_size=128)

        # pre_run 仅含 embed_tokens 右旋（可挂 hook 的模块路径），无 merger
        self.assertEqual(len(pre_run_list), 1)
        pre_run = pre_run_list[0]
        self.assertEqual(list(pre_run.left_rot.keys()), [])
        self.assertEqual(list(pre_run.right_rot.keys()), ["model.language_model.embed_tokens"])

        # merger 左旋位于主阶段 'rot' pair
        rot_pair = rot_pairs_list[0]
        self.assertIn("model.visual.merger.down_proj.weight", rot_pair.left_rot)

    def test_getRotateMap_shouldOmitMerger_when_noVisual(self):
        """纯文本 checkpoint 不注册 merger 旋转路径。"""
        adapter = self.create_adapter()
        adapter._has_visual_module = Mock(return_value=False)
        adapter._layer_has_indexer = Mock(return_value=False)

        pre_run_list, rot_pairs_list = adapter.get_rotate_map(block_size=128)

        for pair in rot_pairs_list:
            self.assertNotIn("model.visual.merger.down_proj.weight", pair.left_rot)
        for pre_run in pre_run_list:
            self.assertNotIn("model.visual.merger.down_proj.weight", pre_run.left_rot)

    def test_getRotateMap_shouldAddIndexerPaths_when_layerHasIndexer(self):
        adapter = self.create_adapter()
        adapter._has_visual_module = Mock(return_value=False)
        adapter._layer_has_indexer = Mock(return_value=True)

        _, rot_pairs_list = adapter.get_rotate_map(block_size=128)

        rot_pair = rot_pairs_list[0]
        self.assertIn("model.language_model.layers.0.self_attn.indexer.wk", rot_pair.right_rot)
        self.assertIn("model.language_model.layers.0.self_attn.indexer.weights_proj", rot_pair.right_rot)
        self.assertIn("model.language_model.layers.0.self_attn.indexer.index_kpool_compress_gate", rot_pair.right_rot)

    def test_getRotateMap_shouldRemapLayerPrefix_when_called(self):
        """层名应从 model.layers.* 重映射为 model.language_model.layers.*。"""
        adapter = self.create_adapter()
        adapter._has_visual_module = Mock(return_value=False)
        adapter._layer_has_indexer = Mock(return_value=False)

        _, rot_pairs_list = adapter.get_rotate_map(block_size=128)

        rot_pair = rot_pairs_list[0]
        for name in list(rot_pair.right_rot.keys()) + list(rot_pair.left_rot.keys()):
            if "layers" in name:
                self.assertIn("model.language_model.layers.", name)
        self.assertIn("lm_head", rot_pair.right_rot)

    # ---- get_ln_fuse_map ----

    def test_getLnFuseMap_shouldRemapToLanguageModelPrefix_when_called(self):
        adapter = self.create_adapter()
        adapter._layer_has_indexer = Mock(return_value=False)

        _, ln_linear_map = adapter.get_ln_fuse_map()

        # 顶层 norm -> lm_head 应重映射为 model.language_model.norm
        self.assertIn("model.language_model.norm", ln_linear_map)
        self.assertEqual(ln_linear_map["model.language_model.norm"], ["lm_head"])
        # 层名应带 language_model 前缀
        self.assertIn("model.language_model.layers.0.input_layernorm", ln_linear_map)

    def test_getLnFuseMap_shouldAddMtpLayerEntries_when_nextnExists(self):
        adapter = self.create_adapter()
        adapter._layer_has_indexer = Mock(return_value=False)

        _, ln_linear_map = adapter.get_ln_fuse_map()

        # MTP 层（idx=2）：enorm+hnorm -> eh_proj；shared_head.norm -> shared_head.head
        mtp_prefix = "model.language_model.layers.2"
        self.assertIn((f"{mtp_prefix}.enorm", f"{mtp_prefix}.hnorm"), ln_linear_map)
        self.assertIn(f"{mtp_prefix}.shared_head.norm", ln_linear_map)

    def test_getLnFuseMap_shouldAppendIndexerTargets_when_layerHasIndexer(self):
        # 全 DSA 层 + full indexer：input_layernorm/q_a_layernorm 键都存在
        adapter = self.create_adapter()
        adapter.config.layer_types = ["deepseek_sparse_attention"] * 2
        adapter.config.indexer_types = ["full", "full"]

        _, ln_linear_map = adapter.get_ln_fuse_map()

        targets = ln_linear_map["model.language_model.layers.0.input_layernorm"]
        self.assertIn("model.language_model.layers.0.self_attn.indexer.wk", targets)
        self.assertIn("model.language_model.layers.0.self_attn.indexer.weights_proj", targets)
        qa_targets = ln_linear_map["model.language_model.layers.0.self_attn.q_a_layernorm"]
        self.assertIn("model.language_model.layers.0.self_attn.indexer.wq_b", qa_targets)

    # ---- _remap_rot_pairs ----

    def test_remapRotPairs_shouldRewritePrefixes_when_called(self):
        adapter = self.create_adapter()
        pair = Mock()
        pair.left_rot = {"model.layers.1.self_attn.o_proj": "rot_l"}
        pair.right_rot = {"model.layers.1.self_attn.q_proj": "rot_r"}
        pre_run = Mock()
        pre_run.left_rot = {}
        pre_run.right_rot = {"model.embed_tokens": "rot"}

        adapter._remap_rot_pairs({'rot': pair}, {'rot': 'matrix'}, pre_run)

        self.assertEqual(list(pair.left_rot.keys()), ["model.language_model.layers.1.self_attn.o_proj"])
        self.assertEqual(list(pair.right_rot.keys()), ["model.language_model.layers.1.self_attn.q_proj"])
        self.assertEqual(list(pre_run.right_rot.keys()), ["model.language_model.embed_tokens"])


if __name__ == '__main__':
    unittest.main()
