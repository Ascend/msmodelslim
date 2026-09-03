#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.core.quant_service.modelslim_convert.config_mapper 模块的单元测试
"""

import pytest

from msmodelslim.core.convert.types import IRKind
from msmodelslim.core.quant_service.modelslim_convert.config_mapper import (
    ModelslimConvertServiceConfig,
    spec_to_convert_config,
)


class TestSpecToConvertConfig:
    """测试 spec_to_convert_config 配置映射"""

    def test_spec_to_convert_config_map_rename_when_preprocess_has_rename(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "preprocess": [
                    {
                        "type": "rename",
                        "patterns": [{"from": "ab*", "to": "cd*"}],
                    },
                ],
                "linears": [],
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o")
        assert len(cfg.preprocess_rules) == 1
        assert cfg.preprocess_rules[0].ops[0].type == "rename"
        assert cfg.preprocess_rules[0].source_patterns == ["ab*"]

    def test_spec_to_convert_config_map_chunk_when_preprocess_has_convert(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "preprocess": [
                    {
                        "type": "convert",
                        "source": ["model.layers.*.mlp.experts.gate_up_proj"],
                        "target": [
                            "model.layers.*.mlp.experts.*.gate_proj.weight",
                            "model.layers.*.mlp.experts.*.up_proj.weight",
                        ],
                        "ops": [{"type": "chunk", "dim": 1}],
                    },
                ],
                "linears": [],
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o")
        assert cfg.preprocess_rules[0].ops[0].type == "split_fused_gate_up"
        assert cfg.preprocess_rules[0].ops[0].params["split_dim"] == 1

    def test_spec_to_convert_config_create_module_and_convert_rules_when_linears_given(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "linears": [
                    {
                        "match": ["layers.*.self_attn.*", "layers.*.mlp.*"],
                        "target": "W8A8_MXFP8",
                        "route": "auto",
                    },
                ],
                "save": [{"type": "ascend_v1", "part_file_size": 4}],
                "parallel": {"workers": 8},
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o", model_family="qwen3_5_moe")
        assert len(cfg.module_rules) == 2
        assert len(cfg.convert_rules) == 2
        assert cfg.convert_rules[0].target_ir == IRKind.W8A8_MXFP8
        assert cfg.dst_format == "ascendv1"
        assert cfg.part_file_size == 4
        assert cfg.parallel.max_workers == 8
        assert cfg.model_family == "qwen3_5_moe"

    def test_spec_to_convert_config_map_part_file_size_when_save_given(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "linears": [],
                "save": [{"type": "huggingface", "part_file_size": 0}],
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o")
        assert cfg.dst_format == "huggingface"
        assert cfg.part_file_size == 0

    def test_spec_to_convert_config_default_part_file_size_when_save_empty(self):
        spec = ModelslimConvertServiceConfig.model_validate({"linears": []})
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o")
        assert cfg.part_file_size == 4

    def test_spec_to_convert_config_auto_route_infer_source_ir_from_catalog_later(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "linears": [
                    {
                        "match": ["layers.*.q_proj"],
                        "target": "FLOAT",
                        "route": "auto",
                    },
                ],
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o")
        assert len(cfg.module_rules) == 1
        rule = cfg.module_rules[0]
        assert rule.source_ir is None
        assert rule.source_format is None
        assert rule.tensor_map["weight"] == "{module}.weight"
        assert rule.tensor_map["weight_scale_inv"] == "{module}.weight_scale_inv"
        assert cfg.convert_rules[0].route == "auto"

    def test_spec_to_convert_config_map_workers_gt_one_to_process_backend(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "linears": [],
                "parallel": {"workers": 8},
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o", device_indices=[0, 1])
        assert cfg.parallel.worker_backend == "process"
        assert cfg.parallel.device_indices == [0, 1]

    def test_spec_to_convert_config_map_workers_one_to_thread_backend(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "linears": [],
                "parallel": {"workers": 1},
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o")
        assert cfg.parallel.worker_backend == "thread"

    def test_spec_to_convert_config_use_fixed_dependency_group_and_shard_cache(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "linears": [],
                "parallel": {"workers": 8},
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o")
        assert cfg.parallel.task_granularity == "dependency_group"
        assert cfg.parallel.shard_cache_size == 1
        assert cfg.parallel.worker_threads == 4

    def test_spec_to_convert_config_no_inflight_memory_limit_by_default(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "linears": [],
                "parallel": {"workers": 8},
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o")
        assert cfg.parallel.max_inflight_bytes is None

    def test_spec_to_convert_config_map_merge_op_when_preprocess_has_merge(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "preprocess": [
                    {
                        "type": "convert",
                        "source": ["a"],
                        "target": ["b"],
                        "ops": [{"type": "merge", "dim": 0}],
                    },
                ],
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o")
        assert cfg.preprocess_rules[0].ops[0].type == "merge_gate_up"

    def test_spec_to_convert_config_raise_error_when_preprocess_type_unknown(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "preprocess": [{"type": "unknown_op"}],
            }
        )
        with pytest.raises(ValueError, match="Unsupported preprocess type"):
            spec_to_convert_config(spec, model_path="/m", save_path="/o")
