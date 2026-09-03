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

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


class _FakeLoader:
    pass


def _add_fake_module(name, attrs=None):
    spec = importlib.util.spec_from_loader(name, _FakeLoader())
    mod = types.ModuleType(name)
    mod.__spec__ = spec
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod


_add_fake_module("librosa")
_add_fake_module("qwen_vl_utils", {"process_vision_info": MagicMock()})
_add_fake_module("qwen_omni_utils", {"process_mm_info": MagicMock()})

# Mock transformers components when not available (e.g. older transformers without Qwen3-Omni-Moe)
import transformers  # noqa: E402

if not hasattr(transformers, "Qwen3OmniMoeProcessor"):
    transformers.Qwen3OmniMoeProcessor = MagicMock()
if not hasattr(transformers, "Qwen3OmniMoeThinkerForConditionalGeneration"):
    transformers.Qwen3OmniMoeThinkerForConditionalGeneration = MagicMock()


def _ensure_module_chain(module_name: str):
    """Ensure module exists in sys.modules and is attached to its parent module."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    parent_name, _, child_name = module_name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], child_name, module)
    return module


# Ensure qwen3 omni moe module chain exists without relying on transformers version details.
if hasattr(transformers, "models") and "transformers.models" not in sys.modules:
    sys.modules["transformers.models"] = transformers.models  # pylint: disable=c-extension-no-member
qwen3_mod = _ensure_module_chain("transformers.models.qwen3_omni_moe")
modeling_mod = _ensure_module_chain("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")
if not hasattr(qwen3_mod, "modeling_qwen3_omni_moe"):
    setattr(qwen3_mod, "modeling_qwen3_omni_moe", modeling_mod)
if not hasattr(modeling_mod, "Qwen3OmniMoeThinkerTextDecoderLayer"):
    modeling_mod.Qwen3OmniMoeThinkerTextDecoderLayer = MagicMock()
# masking_utils: add mocks if missing
_masking = sys.modules.get("transformers.masking_utils")
if _masking is None:
    _masking = types.ModuleType("transformers.masking_utils")
    sys.modules["transformers.masking_utils"] = _masking
if not hasattr(_masking, "create_causal_mask"):
    _masking.create_causal_mask = MagicMock()
if not hasattr(_masking, "create_sliding_window_causal_mask"):
    _masking.create_sliding_window_causal_mask = MagicMock()

# Ensure test env mocks are applied before any msmodelslim import (e.g. when run without pytest conftest)
_test_dir = Path(__file__).resolve().parents[3]
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))
try:
    from testing_utils.mock import mock_init_config

    mock_init_config()
except ImportError:
    pass

import torch  # noqa: E402
from torch import nn  # noqa: E402

from msmodelslim.core.base.protocol import ProcessRequest  # noqa: E402

# Ensure model_adapter is loaded so patch target "....model_adapter.VLMBaseModelAdapter" resolves
import msmodelslim.model.qwen3_omni_moe.model_adapter  # noqa: F401, E402
from msmodelslim.core.const import DeviceType  # noqa: E402
from msmodelslim.core.graph import AdapterConfig, MappingConfig  # noqa: E402
from msmodelslim.utils.exception import InvalidDatasetError  # noqa: E402


def _make_thinker_config(
    num_audio_layers=1,
    vision_depth=1,
    num_text_layers=2,
    mlp_only_layers=None,
    decoder_sparse_step=2,
    num_experts=2,
):
    """Build dummy thinker_config for adapter.config.thinker_config."""
    _mlp_only = list(mlp_only_layers) if mlp_only_layers is not None else []
    _step = decoder_sparse_step
    _experts = num_experts
    _text_layers = num_text_layers
    _audio_layers = num_audio_layers
    _vision_d = vision_depth

    class TextConfig:
        num_hidden_layers = _text_layers
        mlp_only_layers = _mlp_only
        decoder_sparse_step = _step
        num_experts = _experts
        num_attention_heads = 8
        num_key_value_heads = 8
        hidden_size = 64
        intermediate_size = 128
        rms_norm_eps = 1e-6
        vocab_size = 1000
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2
        attention_bias = False
        attention_dropout = 0.0
        hidden_act = "silu"

    class AudioConfig:
        num_hidden_layers = _audio_layers

    class VisionConfig:
        depth = _vision_d

    class ThinkerConfig:
        text_config = TextConfig()
        audio_config = AudioConfig()
        vision_config = VisionConfig()

    return ThinkerConfig()


class TestQwen3OmniMoeThinkerModelAdapter(unittest.TestCase):
    """Unit tests for Qwen3OmniMoeThinkerModelAdapter."""

    def setUp(self):
        self.model_type = "Qwen3-Omni-30B-A3B-Thinking"
        self.model_path = Path(tempfile.mkdtemp())
        self.trust_remote_code = False
        self._librosa_patcher = patch.dict(sys.modules, {"librosa": MagicMock()})
        self._librosa_patcher.start()
        self._adapter_patcher = patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.VLMBaseModelAdapter.__init__",
            return_value=None,
        )

    def tearDown(self):
        self._librosa_patcher.stop()

    def _create_adapter(self, thinker_config=None):
        if thinker_config is None:
            thinker_config = _make_thinker_config()
        with self._adapter_patcher:
            from msmodelslim.model.qwen3_omni_moe.model_adapter import (
                Qwen3OmniMoeThinkerModelAdapter,
            )

            adapter = Qwen3OmniMoeThinkerModelAdapter(
                model_type=self.model_type,
                model_path=str(self.model_path),
                trust_remote_code=self.trust_remote_code,
            )
        adapter.config = MagicMock()
        adapter.config.thinker_config = thinker_config
        adapter.config.use_cache = False
        adapter.config.dtype = torch.bfloat16
        adapter.model_path = str(self.model_path)
        adapter._model_path = str(self.model_path)
        adapter.trust_remote_code = self.trust_remote_code
        return adapter

    def test_get_model_pedigree(self):
        """get_model_pedigree 应返回 'qwen3_omni_moe'"""
        adapter = self._create_adapter()
        self.assertEqual(adapter.get_model_pedigree(), "qwen3_omni_moe")

    def test_get_model_type(self):
        """get_model_type 应返回传入的 model_type"""
        adapter = self._create_adapter()
        adapter.model_type = self.model_type
        self.assertEqual(adapter.get_model_type(), self.model_type)

    def test_handle_dataset_text_only(self):
        """handle_dataset 纯文本样本：应调用 processor 并返回 _collect_inputs_to_device 结果"""
        adapter = self._create_adapter()
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "hello"
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_processor.return_value = mock_inputs

        with (
            patch(
                "transformers.Qwen3OmniMoeProcessor.from_pretrained",
                return_value=mock_processor,
            ),
            patch(
                "msmodelslim.model.qwen3_omni_moe.model_adapter.process_mm_info",
                return_value=([], [], []),
            ),
        ):
            adapter._collect_inputs_to_device = MagicMock(return_value={"input_ids": mock_inputs["input_ids"]})
            sample = MagicMock()
            sample.text = "hello"
            # Use "" instead of None for optional modalities (compatible with old model_adapter skip logic)
            sample.image = ""
            sample.audio = ""
            sample.video = ""
            result = adapter.handle_dataset(dataset=[sample], device=DeviceType.CPU)
        self.assertEqual(len(result), 1)
        adapter._collect_inputs_to_device.assert_called_once()
        mock_processor.apply_chat_template.assert_called_once()

    def test_handle_dataset_with_multimodal(self):
        """handle_dataset 多模态样本：应构建 content 并调用 process_mm_info"""
        adapter = self._create_adapter()
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "text"
        mock_processor.return_value = {"input_ids": torch.tensor([[1]])}
        with (
            patch(
                "transformers.Qwen3OmniMoeProcessor.from_pretrained",
                return_value=mock_processor,
            ),
            patch(
                "msmodelslim.model.qwen3_omni_moe.model_adapter.process_mm_info",
                return_value=([], [], []),
            ),
            patch(
                "msmodelslim.model.qwen3_omni_moe.model_adapter.get_valid_read_path",
                side_effect=lambda x: x,
            ),
        ):
            adapter._collect_inputs_to_device = MagicMock(return_value={})
            sample = MagicMock()
            sample.text = "hi"
            sample.image = "/tmp/img.jpg"
            # Use "" for optional modalities (compatible with old model_adapter skip logic)
            sample.audio = ""
            sample.video = ""
            result = adapter.handle_dataset(dataset=[sample], device=DeviceType.CPU)
        self.assertEqual(len(result), 1)
        mock_processor.assert_called_once()

    def _run_handle_dataset_with_patches(self, adapter, dataset, extra_patches=()):
        from contextlib import ExitStack

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "text"
        mock_processor.return_value = {"input_ids": torch.tensor([[1]])}
        with ExitStack() as stack:
            stack.enter_context(
                patch("transformers.Qwen3OmniMoeProcessor.from_pretrained", return_value=mock_processor)
            )
            stack.enter_context(
                patch("msmodelslim.model.qwen3_omni_moe.model_adapter.process_mm_info", return_value=([], [], []))
            )
            stack.enter_context(
                patch(
                    "msmodelslim.model.qwen3_omni_moe.model_adapter.get_valid_read_path",
                    side_effect=lambda x: x,
                )
            )
            for extra_patch in extra_patches:
                stack.enter_context(extra_patch)
            adapter._collect_inputs_to_device = MagicMock(return_value={})
            return adapter.handle_dataset(dataset=dataset, device=DeviceType.CPU)

    def test_handle_dataset_rejects_mixed_modalities(self):
        """handle_dataset 异构模态样本应抛出 InvalidDatasetError"""
        adapter = self._create_adapter()
        text_only = MagicMock(text="hello", image="", audio="", video="")
        with_image = MagicMock(text="hi", image="/tmp/img.jpg", audio="", video="")
        with self.assertRaises(InvalidDatasetError) as ctx:
            self._run_handle_dataset_with_patches(adapter, [text_only, with_image])
        self.assertIn("Inconsistent calibration sample modalities", str(ctx.exception))

    def test_handle_dataset_accepts_homogeneous_modalities(self):
        """handle_dataset 同质模态多样本应全部处理"""
        adapter = self._create_adapter()
        samples = [MagicMock(text=t, image="", audio="", video="") for t in ("hello", "world")]
        result = self._run_handle_dataset_with_patches(adapter, samples)
        self.assertEqual(len(result), 2)
        self.assertEqual(adapter._collect_inputs_to_device.call_count, 2)

    def test_handle_dataset_rejects_mixed_video_audio_tracks(self):
        """text+video 混用有音轨/无声视频应抛出 InvalidDatasetError"""
        adapter = self._create_adapter()
        with_audio = MagicMock(text="with audio", image="", audio="", video="/tmp/with_audio.mp4")
        silent = MagicMock(text="silent", image="", audio="", video="/tmp/silent.mp4")
        track_patch = patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.Qwen3OmniMoeThinkerModelAdapter._video_has_audio_track",
            side_effect=lambda path: str(path).endswith("with_audio.mp4"),
        )
        with self.assertRaises(InvalidDatasetError):
            self._run_handle_dataset_with_patches(adapter, [with_audio, silent], extra_patches=(track_patch,))

    def test_handle_dataset_accepts_homogeneous_video_audio_tracks(self):
        """text+video 均为无声视频时应全部处理"""
        adapter = self._create_adapter()
        samples = [MagicMock(text=t, image="", audio="", video=f"/tmp/{t}.mp4") for t in ("clip-a", "clip-b")]
        track_patch = patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.Qwen3OmniMoeThinkerModelAdapter._video_has_audio_track",
            return_value=False,
        )
        result = self._run_handle_dataset_with_patches(adapter, samples, extra_patches=(track_patch,))
        self.assertEqual(len(result), 2)

    def test_init_model(self):
        """init_model 应加载 skeleton、state_dict 并设置 config"""
        adapter = self._create_adapter()
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        mock_model.config.num_attention_heads = None
        mock_model.config.num_key_value_heads = None
        from_pretrained_return = MagicMock()
        from_pretrained_return.eval.return_value = mock_model
        mock_model.from_pretrained.return_value = from_pretrained_return
        mock_model.load_state_dict = MagicMock()
        with (
            patch(
                "msmodelslim.model.qwen3_omni_moe.model_adapter.get_valid_read_path",
                return_value=str(self.model_path),
            ),
            patch(
                "transformers.Qwen3OmniMoeThinkerForConditionalGeneration",
                return_value=mock_model,
            ) as mock_cls,
        ):
            adapter._get_state_dict = MagicMock(return_value={})
            result = adapter.init_model(device=DeviceType.CPU)
        self.assertIsNotNone(result)
        mock_cls.from_pretrained.assert_called_once()
        adapter._get_state_dict.assert_called_once()
        self.assertEqual(
            result.config.num_attention_heads,
            adapter.config.thinker_config.text_config.num_attention_heads,
        )

    def test_generate_model_visit(self):
        """generate_model_visit 应固定 yield audio/visual/decoder tower"""
        adapter = self._create_adapter()
        mock_audio = MagicMock()
        mock_visual = MagicMock()
        mock_model = MagicMock()
        mock_model.audio_tower = mock_audio
        mock_model.visual = mock_visual
        mock_decoder_reqs = [
            ProcessRequest(name="model.layers.0", module=MagicMock(), args=(), kwargs={}),
        ]
        with patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.generated_decoder_layer_visit_func",
            return_value=iter(mock_decoder_reqs),
        ):
            gen = adapter.generate_model_visit(model=mock_model)
            reqs = list(gen)
        self.assertGreaterEqual(len(reqs), 3)
        self.assertEqual(reqs[0].name, "audio_tower")
        self.assertIs(reqs[0].module, mock_audio)
        self.assertEqual(reqs[1].name, "visual")
        self.assertEqual(reqs[2].name, "visual")
        self.assertEqual(reqs[3].name, "model.layers.0")

    def test_generate_model_forward_text_only(self):
        """generate_model_forward 仅文本：无 audio/visual yield，仅 decoder layers"""
        adapter = self._create_adapter()
        thinker_config = _make_thinker_config(num_text_layers=1)
        adapter.config.thinker_config = thinker_config
        sample = {
            "input_ids": torch.randint(0, 100, (1, 4)),
            "input_features": None,
            "pixel_values": None,
            "pixel_values_videos": None,
            "image_grid_thw": None,
            "video_grid_thw": None,
            "attention_mask": torch.ones(1, 4),
            "feature_attention_mask": None,
            "audio_feature_lengths": None,
            "position_ids": None,
            "use_audio_in_video": False,
            "video_second_per_grid": None,
        }
        mock_emb = torch.randn(1, 4, 64)
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = MagicMock(return_value=mock_emb)
        mock_model.get_placeholder_mask.return_value = (None, None, None)
        mock_model.get_rope_index.return_value = (
            torch.arange(4).unsqueeze(0).expand(3, 1, 4),
            torch.zeros(1, 4),
        )
        mock_model.model.rotary_emb.return_value = (torch.randn(1, 4, 64),)
        mock_model.config.dtype = torch.float32
        mock_layer = MagicMock()
        mock_layer.return_value = (torch.randn(1, 4, 64),)
        adapter._load_decoder_if_not_exist = MagicMock(return_value=mock_layer)
        with patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.create_causal_mask",
            return_value=torch.ones(1, 1, 4, 4),
            create=True,
        ):
            gen = adapter.generate_model_forward(model=mock_model, inputs=sample)
            collected = []
            try:
                req = next(gen)
                while True:
                    collected.append(req)
                    if req.name.startswith("model.layers"):
                        out = (
                            (torch.randn(1, 4, 64),)
                            if isinstance(req.module.return_value, tuple)
                            else req.module.return_value
                        )
                        if isinstance(out, tuple):
                            req = gen.send(out[0])
                        else:
                            req = gen.send(out)
                    else:
                        req = next(gen)
            except StopIteration:
                pass
        self.assertGreater(len(collected), 0)
        self.assertTrue(any(r.name.startswith("model.layers") for r in collected))

    def test_generate_model_forward_with_audio(self):
        """generate_model_forward 含 input_features 时应先 yield audio_tower"""
        adapter = self._create_adapter()
        thinker_config = _make_thinker_config(num_text_layers=0)
        adapter.config.thinker_config = thinker_config
        sample = {
            "input_ids": torch.randint(0, 10, (1, 2)),
            "input_features": torch.randn(1, 80, 10),
            "pixel_values": None,
            "pixel_values_videos": None,
            "image_grid_thw": None,
            "video_grid_thw": None,
            "attention_mask": torch.ones(1, 2),
            "feature_attention_mask": torch.ones(1, 10),
            "audio_feature_lengths": None,
            "position_ids": None,
            "use_audio_in_video": False,
            "video_second_per_grid": None,
        }
        mock_emb = torch.randn(1, 2, 64)
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = MagicMock(return_value=mock_emb)
        mock_model.get_placeholder_mask.return_value = (
            torch.zeros(1, 2, 64).bool(),
            torch.zeros(1, 2, 64).bool(),
            torch.zeros(1, 2, 64).bool(),
        )
        mock_model.get_rope_index.return_value = (
            torch.arange(2).unsqueeze(0).expand(3, 1, 2),
            torch.zeros(1, 2),
        )
        mock_model.model.rotary_emb.return_value = (torch.randn(1, 2, 64),)
        mock_model.config.dtype = torch.float32
        audio_out = MagicMock()
        audio_out.last_hidden_state = torch.randn(1, 5, 64)
        with patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.create_causal_mask",
            return_value=torch.ones(1, 1, 2, 2),
            create=True,
        ):
            gen = adapter.generate_model_forward(model=mock_model, inputs=sample)
            req = next(gen)
            self.assertEqual(req.name, "audio_tower")
            try:
                gen.send(audio_out)
            except StopIteration:
                pass
        mock_model.get_placeholder_mask.assert_called()

    def test_generate_model_forward_with_image(self):
        """generate_model_forward 含 pixel_values 时应 yield visual 并 merge image"""
        adapter = self._create_adapter()
        thinker_config = _make_thinker_config(num_text_layers=0)
        adapter.config.thinker_config = thinker_config
        sample = {
            "input_ids": torch.randint(0, 10, (1, 3)),
            "input_features": None,
            "pixel_values": torch.randn(1, 3, 14, 14),
            "pixel_values_videos": None,
            "image_grid_thw": None,
            "video_grid_thw": None,
            "attention_mask": torch.ones(1, 3),
            "feature_attention_mask": None,
            "audio_feature_lengths": None,
            "position_ids": None,
            "use_audio_in_video": False,
            "video_second_per_grid": None,
        }
        mock_emb = torch.randn(1, 3, 64)
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = MagicMock(return_value=mock_emb)
        image_emb = torch.randn(2, 64)
        image_multiscale = (torch.randn(2, 64),)
        mock_model.get_placeholder_mask.return_value = (
            torch.tensor([[[True], [True], [False]]]),
            torch.zeros(1, 3, 1).bool(),
            torch.zeros(1, 3, 1).bool(),
        )
        mock_model.get_rope_index.return_value = (
            torch.arange(3).unsqueeze(0).expand(3, 1, 3),
            torch.zeros(1, 3),
        )
        mock_model.model.rotary_emb.return_value = (torch.randn(1, 3, 64),)
        with patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.create_causal_mask",
            return_value=torch.ones(1, 1, 3, 3),
            create=True,
        ):
            gen = adapter.generate_model_forward(model=mock_model, inputs=sample)
            req = next(gen)
            if req.name == "visual":
                try:
                    gen.send((image_emb, image_multiscale))
                except StopIteration:
                    pass
            else:
                try:
                    next(gen)
                except StopIteration:
                    pass
        mock_model.get_placeholder_mask.assert_called()

    def test_generate_model_forward_with_video(self):
        """generate_model_forward 含 pixel_values_videos 时应 yield visual 并 merge video"""
        adapter = self._create_adapter()
        thinker_config = _make_thinker_config(num_text_layers=0)
        adapter.config.thinker_config = thinker_config
        sample = {
            "input_ids": torch.randint(0, 10, (1, 4)),
            "input_features": None,
            "pixel_values": None,
            "pixel_values_videos": torch.randn(1, 3, 4, 14, 14),
            "image_grid_thw": None,
            "video_grid_thw": None,
            "attention_mask": torch.ones(1, 4),
            "feature_attention_mask": None,
            "audio_feature_lengths": None,
            "position_ids": None,
            "use_audio_in_video": False,
            "video_second_per_grid": None,
        }
        mock_emb = torch.randn(1, 4, 64)
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = MagicMock(return_value=mock_emb)
        mock_model.get_placeholder_mask.return_value = (
            torch.zeros(1, 4, 1).bool(),
            torch.tensor([[[False], [True], [True], [False]]]),
            torch.zeros(1, 4, 1).bool(),
        )
        mock_model.get_rope_index.return_value = (
            torch.arange(4).unsqueeze(0).expand(3, 1, 4),
            torch.zeros(1, 4),
        )
        mock_model.model.rotary_emb.return_value = (torch.randn(1, 4, 64),)
        with patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.create_causal_mask",
            return_value=torch.ones(1, 1, 4, 4),
            create=True,
        ):
            gen = adapter.generate_model_forward(model=mock_model, inputs=sample)
            req = next(gen)
            if req.name == "visual":
                video_emb = torch.randn(2, 64)
                video_multiscale = (torch.randn(2, 64),)
                try:
                    gen.send((video_emb, video_multiscale))
                except StopIteration:
                    pass
        mock_model.get_placeholder_mask.assert_called()

    def test_generate_model_forward_position_ids_4d(self):
        """generate_model_forward position_ids 4D 时正确取 text_position_ids"""
        adapter = self._create_adapter()
        thinker_config = _make_thinker_config(num_text_layers=0)
        adapter.config.thinker_config = thinker_config
        pos_4d = torch.arange(12).reshape(4, 1, 3)
        sample = {
            "input_ids": torch.randint(0, 10, (1, 3)),
            "input_features": None,
            "pixel_values": None,
            "pixel_values_videos": None,
            "image_grid_thw": None,
            "video_grid_thw": None,
            "attention_mask": torch.ones(1, 3),
            "feature_attention_mask": None,
            "audio_feature_lengths": None,
            "position_ids": pos_4d,
            "use_audio_in_video": False,
            "video_second_per_grid": None,
        }
        mock_emb = torch.randn(1, 3, 64)
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = MagicMock(return_value=mock_emb)
        mock_model.get_placeholder_mask.return_value = (None, None, None)
        mock_model.get_rope_index.return_value = (pos_4d, torch.zeros(1, 3))
        mock_model.model.rotary_emb.return_value = (torch.randn(1, 3, 64),)
        with patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.create_causal_mask",
            return_value=torch.ones(1, 1, 3, 3),
            create=True,
        ):
            gen = adapter.generate_model_forward(model=mock_model, inputs=sample)
            list(gen)
        mock_model.model.rotary_emb.assert_called_once()

    def test_generate_decoder_layer(self):
        """generate_decoder_layer 应按 num_hidden_layers  yield 并调用 _load_decoder_if_not_exist"""
        adapter = self._create_adapter()
        mock_model = MagicMock()
        mock_layer = MagicMock()
        adapter._load_decoder_if_not_exist = MagicMock(return_value=mock_layer)
        layers = list(adapter.generate_decoder_layer(model=mock_model))
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0][0], "model.layers.0")
        self.assertEqual(layers[1][0], "model.layers.1")
        self.assertEqual(adapter._load_decoder_if_not_exist.call_count, 2)

    def test_enable_kv_cache(self):
        """enable_kv_cache 应设置 model.config.use_cache"""
        adapter = self._create_adapter()
        mock_model = MagicMock()
        mock_model.config.use_cache = False
        adapter.enable_kv_cache(model=mock_model, need_kv_cache=True)
        self.assertTrue(mock_model.config.use_cache)
        adapter.enable_kv_cache(model=mock_model, need_kv_cache=False)
        self.assertFalse(mock_model.config.use_cache)

    def test_get_adapter_config_for_subgraph(self):
        """get_adapter_config_for_subgraph 应返回 audio + vision + text 的 AdapterConfig 列表"""
        adapter = self._create_adapter()
        result = adapter.get_adapter_config_for_subgraph()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for cfg in result:
            self.assertIsInstance(cfg, AdapterConfig)
            self.assertIn(cfg.subgraph_type, ("norm-linear", "ov", "up-down"))
            self.assertIsInstance(cfg.mapping, MappingConfig)
        sources = [c.mapping.source for c in result]
        self.assertIn("audio_tower.layers.0.self_attn_layer_norm", sources)
        self.assertIn("model.layers.0.input_layernorm", sources)

    def test_get_adapter_config_for_subgraph_with_moe(self):
        """get_adapter_config_for_subgraph 含 MoE 时应包含 experts up-down"""
        thinker_config = _make_thinker_config(
            num_text_layers=2,
            decoder_sparse_step=2,
            num_experts=2,
        )
        thinker_config.text_config.mlp_only_layers = []
        adapter = self._create_adapter(thinker_config=thinker_config)
        result = adapter.get_adapter_config_for_subgraph()
        expert_sources = [
            c.mapping.source for c in result if "experts" in c.mapping.source and c.subgraph_type == "up-down"
        ]
        self.assertGreater(len(expert_sources), 0)

    def test_get_adapter_config_for_subgraph_mlp_only_layers(self):
        """get_adapter_config_for_subgraph 当 layer 在 mlp_only_layers 时不加 MLP/up-down"""
        thinker_config = _make_thinker_config(num_text_layers=2, num_experts=0)
        thinker_config.text_config.mlp_only_layers = [0]
        thinker_config.text_config.decoder_sparse_step = 4
        adapter = self._create_adapter(thinker_config=thinker_config)
        result = adapter.get_adapter_config_for_subgraph()
        post_ln_sources = [c.mapping.source for c in result if "post_attention_layernorm" in c.mapping.source]
        self.assertGreaterEqual(len(post_ln_sources), 1)

    def test_get_weight_map(self):
        """_get_weight_map 应从 model.safetensors.index.json 读取 weight_map"""
        adapter = self._create_adapter()
        index_path = self.model_path / "model.safetensors.index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        weight_map = {"thinker.model.layers.0.weight": "model-00001.safetensors"}
        index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
        adapter._get_weight_map.cache_clear()
        result = adapter._get_weight_map()
        self.assertEqual(result, weight_map)

    def test_get_weight_map_cached(self):
        """_get_weight_map 应使用 lru_cache 缓存"""
        adapter = self._create_adapter()
        index_path = self.model_path / "model.safetensors.index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps({"weight_map": {}}), encoding="utf-8")
        adapter._get_weight_map.cache_clear()
        r1 = adapter._get_weight_map()
        r2 = adapter._get_weight_map()
        self.assertIs(r1, r2)

    def test_get_state_dict(self):
        """_get_state_dict 应根据 weight_map 和 prefix 返回参数字典"""
        adapter = self._create_adapter()
        index_path = self.model_path / "model.safetensors.index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps({"weight_map": {"thinker.model.layers.0.input_layernorm.weight": "model.safetensors"}}),
            encoding="utf-8",
        )
        adapter._get_weight_map.cache_clear()
        mock_module = MagicMock()
        mock_module.named_parameters.return_value = [("input_layernorm.weight", torch.nn.Parameter(torch.zeros(64)))]
        with (
            patch(
                "msmodelslim.model.qwen3_omni_moe.model_adapter.get_valid_read_path",
                return_value=str(self.model_path / "model.safetensors"),
            ),
            patch(
                "msmodelslim.model.qwen3_omni_moe.model_adapter.safe_open",
            ) as mock_open,
        ):
            mock_f = MagicMock()
            mock_f.get_tensor.return_value = torch.zeros(64)
            mock_open.return_value.__enter__.return_value = mock_f
            mock_open.return_value.__exit__.return_value = None
            with patch("msmodelslim.model.qwen3_omni_moe.model_adapter.tqdm", side_effect=lambda x, **kw: x):
                result = adapter._get_state_dict(mock_module, prefix="thinker.model.layers.0")
        self.assertIn("input_layernorm.weight", result)
        self.assertEqual(result["input_layernorm.weight"].shape, (64,))

    def test_load_decoder_if_not_exist_already_loaded(self):
        """_load_decoder_if_not_exist 当层已加载且 materialize 时直接返回"""
        adapter = self._create_adapter()
        mock_decoder = MagicMock()
        mock_decoder.input_layernorm.weight.device = "cpu"
        mock_model = MagicMock()
        mock_model.get_submodule.return_value = mock_decoder
        result = adapter._load_decoder_if_not_exist(model=mock_model, name="model.layers.0", idx=0)
        self.assertIs(result, mock_decoder)
        mock_model.get_submodule.assert_called_once_with("model.layers.0")

    def test_load_decoder_if_not_exist_submodule_missing(self):
        """_load_decoder_if_not_exist 当 get_submodule 抛 AttributeError 时走加载逻辑"""
        adapter = self._create_adapter()
        mock_model = MagicMock()
        mock_model.get_submodule.side_effect = AttributeError("no layers")
        mock_model.config.text_config = adapter.config.thinker_config.text_config
        mock_model.model = types.SimpleNamespace(layers=nn.ModuleList())

        class _DummyDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = nn.LayerNorm(64)
                self.self_attn = nn.Module()
                self.self_attn.q_proj = nn.Linear(64, 64)
                self.self_attn.k_proj = nn.Linear(64, 64)
                self.self_attn.v_proj = nn.Linear(64, 64)
                self.self_attn.o_proj = nn.Linear(64, 64, bias=False)
                self.mlp = nn.Module()
                self.mlp.gate_proj = nn.Linear(64, 128, bias=False)
                self.mlp.up_proj = nn.Linear(64, 128, bias=False)
                self.mlp.down_proj = nn.Linear(128, 64, bias=False)
                self.post_attention_layernorm = nn.LayerNorm(64)

        real_decoder = _DummyDecoder()
        mock_state_dict = {name: torch.randn_like(param) for name, param in real_decoder.named_parameters()}
        with (
            patch.object(real_decoder, "load_state_dict", MagicMock(return_value=None)),
            patch.object(
                sys.modules["transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe"],
                "Qwen3OmniMoeThinkerTextDecoderLayer",
                return_value=real_decoder,
                create=True,
            ),
            patch.object(nn.Linear, "reset_parameters", lambda self: None),
        ):
            adapter._get_state_dict = MagicMock(return_value=mock_state_dict)
            result = adapter._load_decoder_if_not_exist(model=mock_model, name="model.layers.0", idx=0)
        self.assertIsInstance(result, nn.Module)
        self.assertEqual(len(mock_model.model.layers), 1)

    def test_load_decoder_if_not_exist_meta_device_continues_load(self):
        """_load_decoder_if_not_exist 当已存在但 weight 在 meta 上时重新加载"""

        class MetaWeight:
            @property
            def device(self):
                raise RuntimeError("meta")

        adapter = self._create_adapter()
        mock_decoder = MagicMock()
        mock_decoder.input_layernorm.weight = MetaWeight()
        mock_model = MagicMock()
        mock_model.get_submodule.return_value = mock_decoder
        mock_model.config.text_config = adapter.config.thinker_config.text_config
        mock_model.model = types.SimpleNamespace(layers=nn.ModuleList([nn.Linear(1, 1)]))

        class _DummyDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = nn.LayerNorm(64)
                self.self_attn = nn.Module()
                self.self_attn.q_proj = nn.Linear(64, 64)
                self.self_attn.k_proj = nn.Linear(64, 64)
                self.self_attn.v_proj = nn.Linear(64, 64)
                self.self_attn.o_proj = nn.Linear(64, 64, bias=False)
                self.mlp = nn.Module()
                self.mlp.gate_proj = nn.Linear(64, 128, bias=False)
                self.mlp.up_proj = nn.Linear(64, 128, bias=False)
                self.mlp.down_proj = nn.Linear(128, 64, bias=False)
                self.post_attention_layernorm = nn.LayerNorm(64)

        real_decoder = _DummyDecoder()
        mock_state_dict = {name: torch.randn_like(param) for name, param in real_decoder.named_parameters()}
        with (
            patch.object(real_decoder, "load_state_dict", MagicMock(return_value=None)),
            patch.object(
                sys.modules["transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe"],
                "Qwen3OmniMoeThinkerTextDecoderLayer",
                return_value=real_decoder,
                create=True,
            ),
            patch.object(nn.Linear, "reset_parameters", lambda self: None),
        ):
            adapter._get_state_dict = MagicMock(return_value=mock_state_dict)
            result = adapter._load_decoder_if_not_exist(model=mock_model, name="model.layers.0", idx=0)
        self.assertIsInstance(result, nn.Module)
        adapter._get_state_dict.assert_called_once()

    @unittest.skip("ImportError path is hard to trigger once module has already imported the class")
    def test_init_model_import_error(self):
        """init_model 在无法导入 Qwen3OmniMoeThinkerForConditionalGeneration 时抛出 ImportError"""
        adapter = self._create_adapter()

        with patch.object(
            transformers,
            "Qwen3OmniMoeThinkerForConditionalGeneration",
            PropertyMock(side_effect=ImportError("Please install transformers with Qwen3-Omni-Moe support.")),
        ):
            with self.assertRaises(ImportError) as ctx:
                adapter.init_model(device=DeviceType.CPU)
            self.assertIn("Qwen3-Omni-Moe", str(ctx.exception))

    def test_handle_dataset_empty(self):
        """handle_dataset 空数据集应返回空列表"""
        adapter = self._create_adapter()
        result = adapter.handle_dataset(dataset=[], device=DeviceType.CPU)
        self.assertEqual(result, [])

    def test_generate_model_forward_inputs_not_list(self):
        """generate_model_forward 当 inputs 非 list 时当作单样本"""
        adapter = self._create_adapter()
        thinker_config = _make_thinker_config(num_text_layers=0)
        adapter.config.thinker_config = thinker_config
        sample = {
            "input_ids": torch.randint(0, 10, (1, 2)),
            "input_features": None,
            "pixel_values": None,
            "pixel_values_videos": None,
            "image_grid_thw": None,
            "video_grid_thw": None,
            "attention_mask": torch.ones(1, 2),
            "feature_attention_mask": None,
            "audio_feature_lengths": None,
            "position_ids": torch.arange(2).unsqueeze(0).expand(3, 1, 2),
            "use_audio_in_video": False,
            "video_second_per_grid": None,
        }
        mock_emb = torch.randn(1, 2, 64)
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = MagicMock(return_value=mock_emb)
        mock_model.get_placeholder_mask.return_value = (None, None, None)
        mock_model.model.rotary_emb.return_value = (torch.randn(1, 2, 64),)
        with patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.create_causal_mask",
            return_value=torch.ones(1, 1, 2, 2),
            create=True,
        ):
            gen = adapter.generate_model_forward(model=mock_model, inputs=sample)
            list(gen)
        mock_model.get_input_embeddings.return_value.assert_called_once()

    def test_generate_model_forward_audio_feature_lengths_none_raises(self):
        """generate_model_forward 仅有 input_features 但无 feature_attention_mask/audio_feature_lengths 时抛 ValueError"""
        adapter = self._create_adapter()
        thinker_config = _make_thinker_config(num_text_layers=0)
        adapter.config.thinker_config = thinker_config
        sample = {
            "input_ids": torch.randint(0, 10, (1, 2)),
            "input_features": torch.randn(1, 80, 5),
            "pixel_values": None,
            "pixel_values_videos": None,
            "image_grid_thw": None,
            "video_grid_thw": None,
            "attention_mask": None,
            "feature_attention_mask": None,
            "audio_feature_lengths": None,
            "position_ids": None,
            "use_audio_in_video": False,
            "video_second_per_grid": None,
        }
        mock_emb = torch.randn(1, 2, 64)
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = MagicMock(return_value=mock_emb)
        gen = adapter.generate_model_forward(model=mock_model, inputs=sample)
        with self.assertRaises(ValueError) as ctx:
            next(gen)
        self.assertTrue("audio_feature_lengths" in str(ctx.exception) or "feature_attention_mask" in str(ctx.exception))

    def test_position_embeddings_not_tuple_warning(self):
        """generate_model_forward 当 rotary_emb 返回非 tuple 时打 warning 仍继续"""
        adapter = self._create_adapter()
        thinker_config = _make_thinker_config(num_text_layers=1)
        adapter.config.thinker_config = thinker_config
        sample = {
            "input_ids": torch.randint(0, 10, (1, 2)),
            "input_features": None,
            "pixel_values": None,
            "pixel_values_videos": None,
            "image_grid_thw": None,
            "video_grid_thw": None,
            "attention_mask": torch.ones(1, 2),
            "feature_attention_mask": None,
            "audio_feature_lengths": None,
            "position_ids": torch.arange(2).unsqueeze(0).expand(3, 1, 2),
            "use_audio_in_video": False,
            "video_second_per_grid": None,
        }
        mock_emb = torch.randn(1, 2, 64)
        mock_model = MagicMock()
        mock_model.get_input_embeddings.return_value = MagicMock(return_value=mock_emb)
        mock_model.get_placeholder_mask.return_value = (None, None, None)
        mock_model.model.rotary_emb.return_value = torch.randn(1, 2, 64)
        mock_layer = MagicMock()
        mock_layer.return_value = torch.randn(1, 2, 64)
        adapter._load_decoder_if_not_exist = MagicMock(return_value=mock_layer)
        with patch(
            "msmodelslim.model.qwen3_omni_moe.model_adapter.create_causal_mask",
            return_value=torch.ones(1, 1, 2, 2),
            create=True,
        ):
            gen = adapter.generate_model_forward(model=mock_model, inputs=sample)
            list(gen)
        mock_model.model.rotary_emb.assert_called_once()


if __name__ == "__main__":
    unittest.main()
