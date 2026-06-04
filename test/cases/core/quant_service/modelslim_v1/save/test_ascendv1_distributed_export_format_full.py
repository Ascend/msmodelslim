#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Distributed AscendV1 full E2E tests for merge correctness."""

# pylint: disable=redefined-outer-name

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1_distributed import (
    DistributedAscendV1Config,
    DistributedAscendV1Saver,
)
from msmodelslim.ir.qal import QDType, QScope


def _copy_file_stub(src_path, dest_path):
    shutil.copy(src_path, dest_path)


class _Adapter:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)


METHODS = [
    "on_w8a8_static",
    "on_w8a16_static_per_channel",
    "on_w8a16_static_per_group",
    "on_w8a8_dynamic_per_channel",
    "on_w8a8_pd_mix",
    "on_w8a8_dynamic_per_group",
    "on_wfp8afp8_dynamic_per_channel",
    "on_w8a8_mx_dynamic_per_block",
    "on_w4a8_dynamic",
    "on_w4a4_dynamic_per_channel",
    "on_w4a4_dynamic_per_group",
    "on_w4a4_mx_dynamic_per_block",
    "on_w4a8_mx_dynamic_per_block",
    "on_float_linear",
    "on_float_module",
    "on_dynamic_cache",
    "on_w16a16s",
    "on_activation_per_head",
    "on_activation_per_token",
    "on_online_rotation_wrapper",
    "on_rotation_wrapper",
    "on_kronecker_rotation_wrapper",
    "on_quarot_extra_info_wrapper",
    "on_flat_clip_wrapper",
    "on_non_fusion_smooth_quant_wrapper",
]


def _build_case(method_name: str):
    if method_name in {"on_w8a8_static", "on_w8a8_pd_mix"}:
        return (
            "model.layers.0.self_attn.q_proj",
            SimpleNamespace(
                weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
                input_scale=torch.tensor([0.5], dtype=torch.float32),
                input_offset=torch.tensor([0.0], dtype=torch.float32),
                weight_scale=torch.ones(4, dtype=torch.float32) * 0.1,
                bias=torch.zeros(4, dtype=torch.float32),
            ),
            "model.layers.0.self_attn.q_proj.weight",
            "W8A8" if method_name == "on_w8a8_static" else "W8A8_MIX",
            0,
            True,
        )
    if method_name in {"on_w8a16_static_per_channel", "on_w8a16_static_per_group"}:
        mod = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones(4, dtype=torch.float32),
            weight_offset=torch.zeros(4, dtype=torch.float32),
            bias=torch.zeros(4, dtype=torch.float32),
        )
        if method_name.endswith("group"):
            mod.group_size = 64
        return (
            "model.layers.0.self_attn.q_proj",
            mod,
            "model.layers.0.self_attn.q_proj.weight",
            "W8A16",
            64 if method_name.endswith("group") else 0,
            True,
        )
    if method_name in {"on_w8a8_dynamic_per_channel", "on_w8a8_dynamic_per_group"}:
        mod = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones(4, dtype=torch.float32),
            bias=torch.zeros(4, dtype=torch.float32),
        )
        if method_name.endswith("group"):
            mod.group_size = 128
        return (
            "model.layers.0.self_attn.q_proj",
            mod,
            "model.layers.0.self_attn.q_proj.weight",
            "W8A8_DYNAMIC",
            128 if method_name.endswith("group") else 0,
            True,
        )
    if method_name == "on_wfp8afp8_dynamic_per_channel":
        return (
            "model.layers.0.self_attn.q_proj",
            SimpleNamespace(weight=torch.randn(4, 4), weight_scale=torch.ones(4), bias=torch.zeros(4)),
            "model.layers.0.self_attn.q_proj.weight",
            "WFP8AFP8_DYNAMIC",
            0,
            True,
        )
    if method_name in {"on_w8a8_mx_dynamic_per_block", "on_w4a4_mx_dynamic_per_block", "on_w4a8_mx_dynamic_per_block"}:
        val = {
            "on_w8a8_mx_dynamic_per_block": "W8A8_MXFP8",
            "on_w4a4_mx_dynamic_per_block": "W4A4_MXFP4",
            "on_w4a8_mx_dynamic_per_block": "W4A8_MXFP",
        }[method_name]
        return (
            "model.layers.0.self_attn.q_proj",
            SimpleNamespace(
                weight=torch.randn(4, 4),
                weight_scale=torch.zeros((4, 1), dtype=torch.int32),
                w_axes=1,
                bias=torch.zeros(4),
            ),
            "model.layers.0.self_attn.q_proj.weight",
            val,
            32,
            True,
        )
    if method_name == "on_w4a8_dynamic":
        return (
            "model.layers.0.self_attn.q_proj",
            SimpleNamespace(
                weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8), weight_scale=torch.ones(4), bias=torch.zeros(4)
            ),
            "model.layers.0.self_attn.q_proj.weight",
            "W4A8_DYNAMIC",
            0,
            True,
        )
    if method_name in {"on_w4a4_dynamic_per_channel", "on_w4a4_dynamic_per_group"}:
        mod = SimpleNamespace(
            weight=torch.randint(-8, 7, (4, 4), dtype=torch.int8),
            weight_scale=torch.ones(4),
            weight_offset=torch.zeros(4),
            bias=torch.zeros(4),
        )
        if method_name.endswith("group"):
            mod.group_size = 32
        return (
            "model.layers.0.self_attn.q_proj",
            mod,
            "model.layers.0.self_attn.q_proj.weight",
            "W4A4_DYNAMIC",
            32 if method_name.endswith("group") else 0,
            True,
        )
    if method_name == "on_float_linear":
        return (
            "model.layers.0.self_attn.q_proj",
            nn.Linear(4, 3, bias=True),
            "model.layers.0.self_attn.q_proj.weight",
            "FLOAT",
            0,
            True,
        )
    if method_name == "on_float_module":
        mod = nn.Module()
        mod.register_parameter("weight", nn.Parameter(torch.ones(2, 2)))
        return "model.layers.0.self_attn.q_proj", mod, "model.layers.0.self_attn.q_proj.weight", "FLOAT", 0, True
    if method_name == "on_dynamic_cache":
        return (
            "model.layers.0.self_attn.key_states",
            SimpleNamespace(kv_cache_scale=torch.ones(2), kv_cache_offset=torch.zeros(2)),
            "model.layers.0.self_attn.k_proj.kv_cache_scale",
            "C8",
            0,
            True,
        )
    if method_name == "on_w16a16s":
        return (
            "m.w16",
            SimpleNamespace(named_parameters=lambda recurse=False, prefix="": [("m.w16.weight", torch.ones(2, 2))]),
            "m.w16.weight",
            "W16A16S",
            0,
            True,
        )
    if method_name == "on_activation_per_head":
        return (
            "model.layers.0.self_attn.q_proj",
            SimpleNamespace(input_scale=torch.ones(2, dtype=torch.float32)),
            "model.layers.0.self_attn.q_proj.scale",
            "FAQuant",
            0,
            True,
        )
    if method_name == "on_activation_per_token":
        return (
            "model.layers.0.self_attn.fa3_q",
            SimpleNamespace(x_q_scheme=SimpleNamespace(dtype=QDType.FP8_E4M3, scope=QScope.PER_TOKEN)),
            "model.layers.0.self_attn.quant_type",
            "FP8_DYNAMIC",
            0,
            False,
        )
    if method_name == "on_online_rotation_wrapper":
        return (
            "model.layers.0.self_attn.q_proj",
            SimpleNamespace(rotation_info=SimpleNamespace(rotation_matrix=torch.eye(4))),
            "model.layers.0.self_attn.q_proj",
            "FLOAT",
            0,
            True,
        )
    if method_name == "on_rotation_wrapper":
        rot_info = SimpleNamespace(heads_rotation=torch.eye(2))
        rot_info.get_quarot_save_info = lambda: {"heads_rotation": {"layers": []}}
        return (
            "model.layers.0.self_attn",
            SimpleNamespace(rotation_info=rot_info),
            "model.layers.0.self_attn.heads_rotation",
            None,
            0,
            True,
        )
    if method_name == "on_kronecker_rotation_wrapper":
        rot_info = SimpleNamespace(kronecker_rotation_m=torch.eye(2), kronecker_rotation_n=torch.eye(2))
        rot_info.get_quarot_save_info = lambda: {"kronecker_rotation": {"layers": []}}
        return (
            "model.layers.0.self_attn",
            SimpleNamespace(rotation_info=rot_info),
            "model.layers.0.self_attn.kronecker_rotation_m",
            None,
            0,
            True,
        )
    if method_name == "on_quarot_extra_info_wrapper":
        return (
            "model",
            SimpleNamespace(rotation_info=SimpleNamespace(global_rotation=torch.eye(2))),
            "optional",
            None,
            0,
            False,
        )
    if method_name == "on_flat_clip_wrapper":
        return (
            "model.layers.0.self_attn.q_proj",
            SimpleNamespace(
                wrapped_module=SimpleNamespace(), save_trans={"left_trans": torch.ones(2, 2)}, clip_factor=torch.ones(1)
            ),
            "model.layers.0.self_attn.q_proj.left_trans",
            "W8A8_FLATQUANT_DYNAMIC",
            0,
            True,
        )
    if method_name == "on_non_fusion_smooth_quant_wrapper":
        return (
            "model.layers.0.self_attn.q_proj",
            SimpleNamespace(wrapped_module=SimpleNamespace(), scales=torch.ones(4)),
            "model.layers.0.self_attn.q_proj.div.mul_scale",
            "FLOAT",
            0,
            True,
        )
    raise AssertionError(method_name)


def _seed_rank1_artifacts(save_dir: Path) -> str:
    rank1_dir = save_dir / "rank_1"
    rank1_dir.mkdir(parents=True, exist_ok=True)
    src_file = "quant_model_weights-00001-of-00001.safetensors"
    (rank1_dir / src_file).write_bytes(b"r1")
    (rank1_dir / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 20}, "weight_map": {"rank1.extra.weight": src_file}}),
        encoding="utf-8",
    )
    (rank1_dir / "quant_model_description.json").write_text(
        json.dumps({"rank1.extra.weight": "FLOAT"}),
        encoding="utf-8",
    )
    return "rank1.extra.weight"


class TestDistributedAscendV1ExportFormat:
    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1_distributed.dist.get_world_size", return_value=2)
    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1_distributed.dist.get_rank", return_value=0)
    @patch(
        "msmodelslim.core.quant_service.modelslim_v1.save.ascendv1_distributed.dist.is_initialized", return_value=True
    )
    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.get_rank", return_value=0)
    @patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.dist.is_initialized", return_value=True)
    @pytest.mark.parametrize("method_name", METHODS)
    def test_distributed_merge_should_generate_expected_artifacts_when_each_on_method_exported(
        self, _ai_ascend, _ar_ascend, _di_dist, _dr_dist, _dw_dist, tmp_path, method_name
    ):
        model_src = tmp_path / "src_model"
        model_src.mkdir()
        (model_src / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization_config": {"bits": 8}}), encoding="utf-8"
        )
        (model_src / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (model_src / "modeling_dummy.py").write_text("DUMMY = True\n", encoding="utf-8")

        save_dir = tmp_path / f"dist_{method_name}"
        save_dir.mkdir()
        saver = DistributedAscendV1Saver(
            nn.Linear(4, 4),
            DistributedAscendV1Config(save_directory=str(save_dir), part_file_size=1),
            _Adapter(str(model_src)),
        )
        saver._global_torch_dtype_is_bf16 = True
        prefix, module, expected_key, expected_value, expected_group_size, expect_in_index = _build_case(method_name)

        if method_name in {"on_flat_clip_wrapper", "on_non_fusion_smooth_quant_wrapper"}:
            saver._process_module = lambda p, m: saver.write_tensor(f"{p}.weight", "W8A8_DYNAMIC", torch.ones(2, 2))

        getattr(saver, method_name)(prefix, module)
        rank1_extra_key = _seed_rank1_artifacts(save_dir)

        with (
            patch(
                "msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.safe_copy_file",
                side_effect=_copy_file_stub,
            ),
            patch(
                "msmodelslim.core.quant_service.modelslim_v1.save.ascendv1_distributed.dist.all_gather_object"
            ) as m_gather,
            patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1_distributed.dist.barrier"),
        ):

            def _gather_all(out, obj):
                for i in range(len(out)):
                    out[i] = obj

            m_gather.side_effect = _gather_all
            saver.post_run()

        desc_path = save_dir / "quant_model_description.json"
        index_path = save_dir / "quant_model_weights.safetensors.index.json"
        assert desc_path.exists()
        assert index_path.exists()
        desc = json.loads(desc_path.read_text(encoding="utf-8"))
        index_data = json.loads(index_path.read_text(encoding="utf-8"))

        assert "metadata" in index_data and "weight_map" in index_data
        assert (save_dir / "tokenizer_config.json").exists()
        assert (save_dir / "modeling_dummy.py").exists()
        copied_config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
        assert "quantization_config" not in copied_config
        assert desc.get("group_size") == expected_group_size

        if method_name == "on_activation_per_head":
            assert desc.get("fa_quant_type") == "FAKQuant"

        if expected_key == "optional":
            assert "optional" in desc
        elif expected_key.endswith("heads_rotation") or expected_key.endswith("kronecker_rotation_m"):
            assert expected_key in index_data["weight_map"]
        else:
            assert expected_key in desc
            if expected_value is not None:
                assert desc[expected_key] == expected_value

        if expect_in_index:
            assert expected_key in index_data["weight_map"]
        assert rank1_extra_key in index_data["weight_map"]
        assert index_data["metadata"]["total_size"] >= 20

        assert not (save_dir / "rank_0").exists()
        assert not (save_dir / "rank_1").exists()
