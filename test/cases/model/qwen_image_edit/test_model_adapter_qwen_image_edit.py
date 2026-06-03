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
msmodelslim.model.qwen_image_edit.model_adapter 模块的单元测试
"""

import argparse
import builtins
import os
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.common.layer_wise_forward import TransformersForwardBreak
from msmodelslim.model.qwen_image_edit.model_adapter import (
    QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS,
    QwenImageEditModelAdapter,
)
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError


def _make_selective_import(real_import, should_block):
    """构造可 patch 的 __import__，避免遮蔽 builtins.globals/locals。"""

    def _selective_import(name, globalns=None, localns=None, fromlist=(), level=0):
        if should_block(name, fromlist):
            raise ImportError(f"blocked import: {name}")
        return real_import(name, globalns, localns, fromlist, level)

    return _selective_import


class _RotationTestBlock(nn.Module):
    """get_online_rotation_configs 测试用的最小 block。"""

    def __init__(self, head_dim, *, with_rot=False, register_raises=False):
        super().__init__()
        self.attention_head_dim = head_dim
        if with_rot:
            self.q_rot = nn.Identity()
            self.k_rot = nn.Identity()
        if register_raises:
            self.register_module = Mock(side_effect=RuntimeError("fail"))

    def forward(self, x):
        return x


class FakeTransformerBlock(nn.Module):
    """类名含 transformerblock，用于 generate_model_forward 发现块。"""

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        return hidden_states, encoder_hidden_states


def _make_block():
    block = FakeTransformerBlock()
    return block


def _adapter(model_path=None):
    path = model_path or Path(tempfile.mkdtemp())
    return QwenImageEditModelAdapter("qwen-image-edit", path, trust_remote_code=False)


class _TestTransformerModel(nn.Module):
    """用于 generate_model_forward 的真实 Module 容器。"""

    def __init__(self, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([_make_block() for _ in range(num_blocks)])

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        return self.blocks[0](hidden_states, encoder_hidden_states, **kwargs)


def _model_with_blocks(num_blocks=2):
    model = _TestTransformerModel(num_blocks=num_blocks)
    return model, list(model.blocks)


class TestQwenImageEditModelAdapterBasic:
    """基础接口测试"""

    def test_get_model_type_return_model_type_when_initialized(self):
        adapter = _adapter()
        assert adapter.get_model_type() == "qwen-image-edit"

    def test_get_model_pedigree_return_qwen_image_edit_when_called(self):
        adapter = _adapter()
        assert adapter.get_model_pedigree() == "qwen_image_edit"

    def test_handle_dataset_return_same_iterable_when_called(self):
        adapter = _adapter()
        dataset = [1, 2, 3]
        assert list(adapter.handle_dataset(dataset)) == dataset

    def test_init_model_return_transformer_dict_when_transformer_set(self):
        adapter = _adapter()
        adapter.transformer = nn.Linear(4, 4)
        result = adapter.init_model()
        assert result == {"": adapter.transformer}

    def test_enable_kv_cache_no_op_when_called(self):
        adapter = _adapter()
        adapter.enable_kv_cache(nn.Linear(2, 2), True)

    def test_run_calib_inference_no_op_when_called(self):
        adapter = _adapter()
        adapter.run_calib_inference()

    def test_load_pipeline_delegate_to_internal_when_called(self):
        adapter = _adapter()
        with patch.object(adapter, "_load_pipeline") as mock_load:
            adapter.load_pipeline()
        mock_load.assert_called_once()


class TestQwenImageEditGenerateModelForward:
    """generate_model_forward 各输入/输出分支"""

    def test_generate_model_forward_raise_invalid_model_when_no_transformer_block(self):
        adapter = _adapter()
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("linear", nn.Linear(2, 2))]

        with pytest.raises(InvalidModelError, match="transformerblock"):
            list(adapter.generate_model_forward(mock_model, {"x": 1}))

    def test_generate_model_forward_raise_invalid_model_when_hook_not_captured(self):
        adapter = _adapter()
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("blocks.0", _make_block())]
        mock_model.side_effect = lambda *a, **k: None

        with patch(
            "msmodelslim.model.qwen_image_edit.model_adapter.TransformersForwardBreak",
            TransformersForwardBreak,
        ):
            with pytest.raises(InvalidModelError, match="Could not capture"):
                list(adapter.generate_model_forward(mock_model, {}))

    @pytest.mark.parametrize(
        "inputs",
        [
            {"hidden_states": torch.randn(1, 2, 8), "encoder_hidden_states": torch.randn(1, 2, 8)},
            (torch.randn(1, 2, 8),),
            torch.randn(1, 2, 8),
        ],
        ids=["dict_input", "tuple_input", "tensor_input"],
    )
    def test_generate_model_forward_capture_input_when_various_input_types(self, inputs):
        adapter = _adapter()
        mock_model, _ = _model_with_blocks(1)

        with (
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.to_device",
                side_effect=lambda x, _: x,
            ),
            patch("msmodelslim.model.qwen_image_edit.model_adapter.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            gen = adapter.generate_model_forward(mock_model, inputs)
            with pytest.raises(StopIteration):
                while True:
                    next(gen)

    def test_generate_model_forward_yield_and_update_dual_stream_when_tuple_outputs(self):
        adapter = _adapter()
        mock_model, _ = _model_with_blocks(1)
        inputs = {
            "hidden_states": torch.randn(1, 2, 8),
            "encoder_hidden_states": torch.randn(1, 2, 8),
        }
        hs = torch.randn(1, 2, 8)
        enc = torch.randn(1, 2, 8)

        with (
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.to_device",
                side_effect=lambda x, _: x,
            ),
            patch("msmodelslim.model.qwen_image_edit.model_adapter.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            gen = adapter.generate_model_forward(mock_model, inputs)
            req = next(gen)
            assert isinstance(req, ProcessRequest)
            assert "blocks.0" in req.name
            with pytest.raises(StopIteration):
                gen.send((hs, enc))

    def test_generate_model_forward_update_tensor_output_when_single_tensor_returned(self):
        adapter = _adapter()
        mock_model, blocks = _model_with_blocks(2)
        inputs = {"hidden_states": torch.randn(1, 2, 8)}

        with (
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.to_device",
                side_effect=lambda x, _: x,
            ),
            patch("msmodelslim.model.qwen_image_edit.model_adapter.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            gen = adapter.generate_model_forward(mock_model, inputs)
            first = next(gen)
            assert "blocks.0" in first.name
            second = gen.send(torch.randn(1, 2, 8))
            assert "blocks.1" in second.name
            with pytest.raises(StopIteration):
                gen.send(torch.randn(1, 2, 8))

    def test_generate_model_forward_update_first_element_when_list_outputs(self):
        adapter = _adapter()
        mock_model, _ = _model_with_blocks(1)
        inputs = {"hidden_states": torch.randn(1, 2, 8)}

        with (
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.to_device",
                side_effect=lambda x, _: x,
            ),
            patch("msmodelslim.model.qwen_image_edit.model_adapter.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            gen = adapter.generate_model_forward(mock_model, inputs)
            next(gen)
            with pytest.raises(StopIteration):
                gen.send([torch.randn(1, 2, 8)])

    def test_generate_model_forward_reraise_when_model_forward_raises(self):
        adapter = _adapter()
        mock_model, _ = _model_with_blocks(1)
        mock_model.forward = MagicMock(side_effect=ValueError("boom"))

        with patch(
            "msmodelslim.model.qwen_image_edit.model_adapter.TransformersForwardBreak",
            TransformersForwardBreak,
        ):
            with pytest.raises(ValueError, match="boom"):
                list(adapter.generate_model_forward(mock_model, {}))

    def test_generate_model_forward_use_else_branch_when_outputs_unknown(self):
        adapter = _adapter()
        mock_model, _ = _model_with_blocks(1)
        inputs = {"hidden_states": torch.randn(1, 2, 8)}

        with (
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.to_device",
                side_effect=lambda x, _: x,
            ),
            patch("msmodelslim.model.qwen_image_edit.model_adapter.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = False
            gen = adapter.generate_model_forward(mock_model, inputs)
            next(gen)
            with pytest.raises(StopIteration):
                gen.send(object())

    def test_generate_model_forward_call_barrier_when_dist_initialized(self):
        adapter = _adapter()
        mock_model, _ = _model_with_blocks(1)
        inputs = {"hidden_states": torch.randn(1, 2, 8)}

        with (
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.to_device",
                side_effect=lambda x, _: x,
            ),
            patch("msmodelslim.model.qwen_image_edit.model_adapter.dist") as mock_dist,
        ):
            mock_dist.is_initialized.return_value = True
            gen = adapter.generate_model_forward(mock_model, inputs)
            with pytest.raises(StopIteration):
                while True:
                    next(gen)
            mock_dist.barrier.assert_called_once()


class TestQwenImageEditGenerateModelVisit:
    def test_generate_model_visit_delegate_keyword_when_called(self):
        adapter = _adapter()
        mock_model = MagicMock()
        sentinel = iter([ProcessRequest("blocks.0", MagicMock(), (), {})])
        with patch(
            "msmodelslim.model.qwen_image_edit.model_adapter.generated_decoder_layer_visit_func_with_keyword",
            return_value=sentinel,
        ) as mock_visit:
            result = adapter.generate_model_visit(mock_model)
        assert result is sentinel
        mock_visit.assert_called_once_with(mock_model, keyword="transformerblock")


class TestQwenImageEditLoadPipeline:
    def test_load_pipeline_load_transformer_and_pipeline_when_import_ok(self, mock_qwenimage_edit_modules):
        adapter = _adapter()
        mock_transformer = MagicMock()
        mock_pipeline = MagicMock()
        mock_qwenimage_edit_modules["transformer_cls"].from_pretrained.return_value = mock_transformer
        mock_qwenimage_edit_modules["pipeline_cls"].from_pretrained.return_value = mock_pipeline

        with patch(
            "msmodelslim.model.qwen_image_edit.model_adapter.get_valid_read_path",
            return_value=str(adapter.model_path),
        ):
            adapter._load_pipeline()

        assert adapter.transformer is mock_transformer
        assert adapter.model is mock_pipeline

    def test_load_pipeline_use_float32_when_torch_dtype_float32(self, mock_qwenimage_edit_modules):
        adapter = _adapter()
        adapter.model_args.torch_dtype = "float32"
        with patch(
            "msmodelslim.model.qwen_image_edit.model_adapter.get_valid_read_path",
            return_value=str(adapter.model_path),
        ):
            adapter._load_pipeline()
        call_kwargs = mock_qwenimage_edit_modules["transformer_cls"].from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.float32

    def test_load_pipeline_raise_invalid_model_when_import_fails(self):
        adapter = _adapter()
        real_import = builtins.__import__
        selective_import = _make_selective_import(
            real_import,
            lambda name, _fromlist: name == "qwenimage_edit" or (name and name.startswith("qwenimage_edit")),
        )

        with (
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.get_valid_read_path",
                return_value=str(adapter.model_path),
            ),
            patch("builtins.__import__", side_effect=selective_import),
        ):
            with pytest.raises(InvalidModelError, match="qwenimage_edit"):
                adapter._load_pipeline()


class TestQwenImageEditSetModelArgs:
    def test_set_model_args_update_and_parse_when_valid_keys(self):
        adapter = _adapter()
        adapter.model_path = Path("/tmp/model")

        with patch.object(adapter, "_validate_args"):
            adapter.set_model_args({"num_inference_steps": 30, "seed": 42})

        assert adapter.model_args.num_inference_steps == 30
        assert adapter.model_args.seed == 42

    def test_set_model_args_raise_schema_error_when_invalid_keys(self):
        adapter = _adapter()
        with pytest.raises(SchemaValidateError, match="Invalid config attributes"):
            adapter.set_model_args({"not_a_real_arg": 1})

    def test_set_model_args_skip_none_values_when_building_argv(self):
        adapter = _adapter()
        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = adapter.model_args

        with patch.object(adapter, "_get_parser", return_value=mock_parser), patch.object(adapter, "_validate_args"):
            adapter.set_model_args({"img_paths": None, "seed": 7})

        argv = mock_parser.parse_args.call_args[0][0]
        assert "--img_paths" not in argv
        assert "--seed" in argv

    def test_set_model_args_append_bool_flag_when_bool_true(self):
        adapter = _adapter()
        adapter.model_path = Path("/tmp/model")
        parser = adapter._get_parser()
        parser.add_argument("--dry_run", action="store_true", default=False)
        adapter.model_args = parser.parse_args([])

        with patch.object(adapter, "_get_parser", return_value=parser), patch.object(adapter, "_validate_args"):
            adapter.set_model_args({"dry_run": True})

        assert adapter.model_args.dry_run is True


class TestQwenImageEditValidateArgs:
    def test_validate_args_set_default_steps_when_steps_none(self):
        adapter = _adapter()
        args = adapter.model_args
        args.num_inference_steps = None
        with patch("os.makedirs"):
            adapter._validate_args(args)
        assert args.num_inference_steps == 40

    def test_validate_args_set_task_config_when_called(self):
        adapter = _adapter()
        args = adapter.model_args
        with patch("os.makedirs"):
            adapter._validate_args(args)
        assert args.task_config == "qwen_image_edit"

    def test_validate_args_raise_schema_error_when_steps_not_positive(self):
        adapter = _adapter()
        args = adapter.model_args
        args.num_inference_steps = 0
        with patch("os.makedirs"):
            with pytest.raises(SchemaValidateError, match="num_inference_steps"):
                adapter._validate_args(args)

    def test_validate_args_raise_file_not_found_when_quant_path_missing(self):
        adapter = _adapter()
        args = adapter.model_args
        args.quant_desc_path = "/nonexistent/quant_model_description_x.json"
        with patch("os.makedirs"), patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                adapter._validate_args(args)

    def test_validate_args_raise_schema_error_when_quant_path_invalid_format(self):
        adapter = _adapter()
        args = adapter.model_args
        args.quant_desc_path = "/tmp/bad.json"
        with patch("os.makedirs"), patch("os.path.exists", return_value=True):
            with pytest.raises(SchemaValidateError, match="quant_desc_path"):
                adapter._validate_args(args)

    def test_validate_args_pass_when_quant_path_valid(self):
        adapter = _adapter()
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="quant_model_description_", delete=False) as f:
            quant_path = f.name
        try:
            args = adapter.model_args
            args.quant_desc_path = quant_path
            with patch("os.makedirs"):
                adapter._validate_args(args)
        finally:
            os.unlink(quant_path)


class TestQwenImageEditApplyQuantization:
    def test_apply_quantization_call_func_when_no_sync_missing(self):
        adapter = _adapter()
        called = []

        def quant_func():
            called.append(True)

        with patch("torch.cuda.amp.autocast"), patch("torch.no_grad"):
            adapter.apply_quantization(quant_func)

        assert called == [True]

    def test_apply_quantization_use_no_sync_when_present(self):
        adapter = _adapter()
        cm = MagicMock()
        adapter.no_sync = MagicMock(return_value=cm)
        quant_func = MagicMock()

        with patch("torch.cuda.amp.autocast"), patch("torch.no_grad"):
            adapter.apply_quantization(quant_func)

        adapter.no_sync.assert_called_once()
        quant_func.assert_called_once()


class TestQwenImageEditOnlineRotation:
    def test_get_online_rotation_configs_return_empty_when_model_none_and_no_transformer(self):
        adapter = _adapter()
        adapter.transformer = None
        assert not adapter.get_online_rotation_configs(None)

    def test_get_online_rotation_configs_return_empty_when_head_dim_unknown(self):
        adapter = _adapter()
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("blocks.0", SimpleNamespace(__class__=type("Other", (), {})))]
        assert not adapter.get_online_rotation_configs(mock_model)

    def test_get_online_rotation_configs_register_and_return_when_block_present(self):
        adapter = _adapter()

        block = _RotationTestBlock(64)
        block.__class__.__name__ = QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("blocks.0", block)]

        configs = adapter.get_online_rotation_configs(mock_model)

        assert hasattr(block, "q_rot")
        assert hasattr(block, "k_rot")
        assert "blocks.0.q_rot" in configs
        assert "blocks.0.k_rot" in configs

    def test_get_online_rotation_configs_keep_existing_rot_when_already_present(self):
        adapter = _adapter()

        block = _RotationTestBlock(64, with_rot=True)
        block.__class__.__name__ = QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("blocks.0", block)]

        configs = adapter.get_online_rotation_configs(mock_model)

        assert "blocks.0.q_rot" in configs
        assert block.q_rot is not None

    def test_get_online_rotation_configs_continue_when_register_raises(self):
        adapter = _adapter()

        block = _RotationTestBlock(32, register_raises=True)
        block.__class__.__name__ = QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS

        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("b0", block)]
        configs = adapter.get_online_rotation_configs(mock_model)
        assert "b0.q_rot" in configs

    def test_get_online_rotation_configs_use_root_paths_when_block_name_empty(self):
        adapter = _adapter()

        block = _RotationTestBlock(16)
        block.__class__.__name__ = QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("", block)]

        configs = adapter.get_online_rotation_configs(mock_model)

        assert "q_rot" in configs
        assert "k_rot" in configs


class TestQwenImageEditInjectFa3:
    HEADS = 2
    HEAD_DIM = 4
    HIDDEN = 8

    @staticmethod
    def _fake_block_module(adaln_fuse=False):
        mod = types.ModuleType("fake_qwen_block")
        mod.ADALN_FUSE = adaln_fuse
        mod.apply_rotary_emb_qwen = lambda q, f, use_real=False: q + 0
        return mod

    @classmethod
    def _build_attention_block(cls, adaln_fuse=False, with_norms=False, with_rotary=False, extra_to_out=False):
        """构造可执行 new_forward 的最小 QwenImageTransformerBlock。"""

        class _Proj(nn.Module):
            def forward(self, x):
                b, s, _ = x.shape
                return torch.zeros(
                    b,
                    s,
                    cls.HEADS * cls.HEAD_DIM,
                    device=x.device,
                    dtype=x.dtype,
                )

        class _OutProj(nn.Module):
            def forward(self, x):
                return x

        class _Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = cls.HEADS
                self.scale = None
                self.norm_q = nn.Identity() if with_norms else None
                self.norm_k = nn.Identity() if with_norms else None
                self.norm_added_q = nn.Identity() if with_norms else None
                self.norm_added_k = nn.Identity() if with_norms else None
                self.to_out = nn.ModuleList([_OutProj()])
                if extra_to_out:
                    self.to_out.append(nn.Identity())
                self.to_add_out = _OutProj()
                for proj_name in ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"]:
                    setattr(self, proj_name, _Proj())

            def forward(self, *args, **kwargs):
                raise NotImplementedError("_Attn is a stub container")

        class QwenImageTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = _Attn()
                self.img_mlp = nn.Identity()
                self.txt_mlp = nn.Identity()
                self.q_rot = nn.Identity()
                self.k_rot = nn.Identity()

            def img_mod(self, temb):
                return torch.zeros(1, cls.HIDDEN * 2, device=temb.device, dtype=temb.dtype)

            def txt_mod(self, temb):
                return torch.zeros(1, cls.HIDDEN * 2, device=temb.device, dtype=temb.dtype)

            def _modulate(self, x, mod):
                gate = torch.ones(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)
                return x, gate

            def img_norm1(self, x, mod=None):
                if mod is None:
                    return x
                return x, torch.ones(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)

            def txt_norm1(self, x, mod=None):
                if mod is None:
                    return x
                return x, torch.ones(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)

            def img_norm2(self, x, mod=None):
                if mod is None:
                    return x
                return x, torch.ones(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)

            def txt_norm2(self, x, mod=None):
                if mod is None:
                    return x
                return x, torch.ones(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)

            def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
                return hidden_states, encoder_hidden_states

        QwenImageTransformerBlock.__name__ = QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS
        QwenImageTransformerBlock.forward.__module__ = "fake_qwen_block"
        block = QwenImageTransformerBlock()
        return block, TestQwenImageEditInjectFa3._fake_block_module(adaln_fuse)

    @classmethod
    def _inject_block(cls, block, mod, use_mindiesd=False, inject_block=True):
        adapter = _adapter()
        wrapper = nn.Module()
        wrapper.add_module("blocks_0", block)
        real_import = builtins.__import__

        def _block_mindiesd(name, _fromlist):
            if name != "mindiesd":
                return False
            if use_mindiesd:
                return False
            return True

        def _mindiesd_import(name, globalns=None, localns=None, fromlist=(), level=0):
            if name == "mindiesd" and use_mindiesd:
                fake = types.ModuleType("mindiesd")
                fake.attention_forward = lambda q, k, v, **kwargs: torch.zeros_like(q)
                return fake
            return real_import(name, globalns, localns, fromlist, level)

        selective_import = _mindiesd_import if use_mindiesd else _make_selective_import(real_import, _block_mindiesd)

        def _should_inject(name):
            return inject_block and name == "blocks_0"

        with (
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.import_module",
                return_value=mod,
            ),
            patch("builtins.__import__", side_effect=selective_import),
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.F.scaled_dot_product_attention",
                side_effect=lambda q, k, v, **kwargs: torch.zeros_like(q),
            ),
        ):
            adapter.inject_fa3_placeholders("", wrapper, should_inject=_should_inject)
        return block

    @classmethod
    def _run_new_forward(cls, block, dtype=torch.float32, rotary=False, seq_txt=2, seq_img=4):
        hidden = torch.zeros(1, seq_img, cls.HIDDEN, dtype=dtype)
        encoder = torch.zeros(1, seq_txt, cls.HIDDEN, dtype=dtype)
        mask = torch.ones(1, seq_txt)
        temb = torch.zeros(1, cls.HIDDEN, dtype=dtype)
        rotary_emb = None
        if rotary:
            freqs = torch.zeros(1, seq_img, cls.HEADS, cls.HEAD_DIM, dtype=dtype)
            txt_freqs = torch.zeros(1, seq_txt, cls.HEADS, cls.HEAD_DIM, dtype=dtype)
            rotary_emb = (freqs, txt_freqs)
        # new_forward 在 mindiesd 不可用时会走 SDPA fallback；CPU 不支持 float16 bmm，需 mock。
        with patch(
            "msmodelslim.model.qwen_image_edit.model_adapter.F.scaled_dot_product_attention",
            side_effect=lambda q, k, v, **kwargs: torch.zeros_like(q),
        ):
            return block(
                hidden,
                encoder,
                mask,
                temb,
                image_rotary_emb=rotary_emb,
            )

    def test_inject_fa3_placeholders_set_submodules_when_should_inject(self):
        adapter = _adapter()
        block, mod = self._build_attention_block()
        root = nn.Module()
        root.add_module("blocks_0", block)
        calls = []

        root.set_submodule = lambda path, placeholder: calls.append(path)
        real_import = builtins.__import__
        selective_import = _make_selective_import(real_import, lambda name, _fromlist: name == "mindiesd")

        with (
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.import_module",
                return_value=mod,
            ),
            patch("builtins.__import__", side_effect=selective_import),
        ):
            adapter.inject_fa3_placeholders("root", root, should_inject=lambda name: "blocks_0" in name)

        assert "blocks_0.fa3_q" in calls
        assert "blocks_0.fa3_k" in calls
        assert "blocks_0.fa3_v" in calls

    def test_inject_fa3_placeholders_raise_import_error_when_no_apply_rotary(self):
        adapter = _adapter()
        block = MagicMock()
        block.__class__.__name__ = QWEN_IMAGE_EDIT_ATTENTION_BLOCK_CLASS
        block.forward = MagicMock(__module__="empty_mod")
        root = MagicMock()
        root.named_modules.return_value = [("blocks.0", block)]
        mod = MagicMock(spec=[])  # no apply_rotary_emb_qwen

        with patch(
            "msmodelslim.model.qwen_image_edit.model_adapter.import_module",
            return_value=mod,
        ):
            with pytest.raises(ImportError, match="apply_rotary_emb_qwen"):
                adapter.inject_fa3_placeholders("root", root, lambda n: True)

    def test_inject_fa3_replace_forward_when_block_injected(self):
        """注入后 block.forward 应被替换为包装实现"""
        adapter = _adapter()
        block, mod = self._build_attention_block()
        original_forward = block.forward
        real_import = builtins.__import__
        selective_import = _make_selective_import(real_import, lambda name, _fromlist: name == "mindiesd")

        wrapper = nn.Module()
        wrapper.add_module("blocks_0", block)

        with (
            patch(
                "msmodelslim.model.qwen_image_edit.model_adapter.import_module",
                return_value=mod,
            ),
            patch("builtins.__import__", side_effect=selective_import),
        ):
            adapter.inject_fa3_placeholders("", wrapper, should_inject=lambda name: name == "blocks_0")

        assert block.forward is not original_forward
        assert hasattr(block, "fa3_q")

    def test_inject_fa3_new_forward_return_tensors_when_adaln_fuse_false(self):
        block, mod = self._build_attention_block(adaln_fuse=False)
        self._inject_block(block, mod)
        out_img, out_txt = self._run_new_forward(block)
        assert out_img.shape == (1, 4, self.HIDDEN)
        assert out_txt.shape == (1, 2, self.HIDDEN)

    def test_inject_fa3_new_forward_return_tensors_when_adaln_fuse_true(self):
        block, mod = self._build_attention_block(adaln_fuse=True)
        self._inject_block(block, mod)
        out_img, out_txt = self._run_new_forward(block)
        assert out_img.shape == (1, 4, self.HIDDEN)
        assert out_txt.shape == (1, 2, self.HIDDEN)

    def test_inject_fa3_new_forward_use_mindiesd_when_available(self):
        block, mod = self._build_attention_block()
        self._inject_block(block, mod, use_mindiesd=True)
        out_img, out_txt = self._run_new_forward(block)
        assert out_img.shape == (1, 4, self.HIDDEN)
        assert out_txt.shape == (1, 2, self.HIDDEN)

    def test_inject_fa3_new_forward_apply_norms_and_rotary_when_enabled(self):
        block, mod = self._build_attention_block(with_norms=True)
        self._inject_block(block, mod)
        out_img, out_txt = self._run_new_forward(block, rotary=True)
        assert out_img.shape == (1, 4, self.HIDDEN)
        assert out_txt.shape == (1, 2, self.HIDDEN)

    def test_inject_fa3_new_forward_use_second_dropout_when_to_out_len_gt_one(self):
        block, mod = self._build_attention_block(extra_to_out=True)
        self._inject_block(block, mod)
        out_img, out_txt = self._run_new_forward(block)
        assert len(block.attn.to_out) > 1
        assert out_img.shape == (1, 4, self.HIDDEN)

    def test_inject_fa3_new_forward_clip_fp16_when_dtype_float16(self):
        block, mod = self._build_attention_block()
        self._inject_block(block, mod)
        out_img, out_txt = self._run_new_forward(block, dtype=torch.float16)
        assert out_img.dtype == torch.float16
        assert out_txt.dtype == torch.float16
        assert out_img.abs().max().item() <= 65504
        assert out_txt.abs().max().item() <= 65504

    def test_inject_fa3_new_forward_use_tensor_scale_when_attn_scale_tensor(self):
        block, mod = self._build_attention_block()
        block.attn.scale = torch.tensor(0.125)
        self._inject_block(block, mod)
        out_img, out_txt = self._run_new_forward(block)
        assert out_img.shape == (1, 4, self.HIDDEN)

    def test_inject_fa3_skip_injection_when_should_inject_false(self):
        block, mod = self._build_attention_block()
        original_func = block.forward.__func__
        self._inject_block(block, mod, inject_block=False)
        assert block.forward.__func__ is original_func
        assert not hasattr(block, "fa3_q")


class TestQwenImageEditParser:
    def test_get_parser_contain_expected_args_when_called(self):
        adapter = _adapter()
        parser = adapter._get_parser()
        dests = {a.dest for a in parser._actions}
        for name in [
            "model_path",
            "torch_dtype",
            "num_inference_steps",
            "quant_desc_path",
            "img_paths",
        ]:
            assert name in dests

    def test_get_default_model_args_parse_empty_argv_when_called(self):
        adapter = _adapter()
        assert adapter.model_args is not None
        assert isinstance(adapter.model_args, argparse.Namespace)
