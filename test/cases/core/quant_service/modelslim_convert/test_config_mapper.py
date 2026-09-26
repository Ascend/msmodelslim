#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.core.quant_service.modelslim_convert.config_mapper 模块的单元测试
"""

import pytest

from msmodelslim.core.const import DeviceType
from msmodelslim.core.convert.types import IRKind
from msmodelslim.core.quant_service.modelslim_convert.config_mapper import (
    ModelslimConvertServiceConfig,
    spec_to_convert_config,
)
from msmodelslim.utils.exception import SchemaValidateError


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
                "parallel": {"cpu_workers": 8},
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
        # YAML 未写 parallel 时默认 cpu_workers=8 → CPU 路径多进程
        assert cfg.parallel.max_workers == 8
        assert cfg.parallel.worker_backend == "process"

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

    def test_spec_to_convert_config_map_cpu_workers_gt_one_to_process_backend(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "linears": [],
                "parallel": {"cpu_workers": 8},
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o", device_indices=[0, 1])
        assert cfg.parallel.worker_backend == "process"
        assert cfg.parallel.device_indices == [0, 1]

    def test_spec_to_convert_config_pass_device_indices_when_given(self):
        """场景：device=npu 且显式传卡号。
        预期：卡号原样透传。
        """
        spec = ModelslimConvertServiceConfig.model_validate({"linears": []})
        cfg = spec_to_convert_config(
            spec,
            model_path="/m",
            save_path="/o",
            device=DeviceType.NPU,
            device_indices=[0, 1, 2],
        )
        assert cfg.parallel.device_indices == [0, 1, 2]

    def test_spec_to_convert_config_default_card_zero_when_npu_without_indices(self):
        """场景：device=npu（CLI 默认）且未传 --device_id。
        预期：对齐 quant 语义，默认卡 0 走 NPU 路径。
        """
        spec = ModelslimConvertServiceConfig.model_validate({"linears": []})
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o", device=DeviceType.NPU)
        assert cfg.parallel.device_indices == [0]

    def test_spec_to_convert_config_empty_indices_when_device_cpu(self):
        """场景：device=cpu（可传单个卡号）。
        预期：忽略 device_id 并清空卡号，走 CPU 路径。
        """
        spec = ModelslimConvertServiceConfig.model_validate({"linears": []})
        cfg = spec_to_convert_config(
            spec,
            model_path="/m",
            save_path="/o",
            device=DeviceType.CPU,
            device_indices=[0],
        )
        assert cfg.parallel.device_indices == []

    def test_spec_to_convert_config_raise_when_device_cpu_with_multi_indices(self):
        """场景：device=cpu 且传多个卡号。
        预期：对齐 quant 语义直接报错（CPU 不支持多设备）。
        """
        spec = ModelslimConvertServiceConfig.model_validate({"linears": []})
        with pytest.raises(SchemaValidateError, match="CPU does not support multi-device"):
            spec_to_convert_config(
                spec,
                model_path="/m",
                save_path="/o",
                device=DeviceType.CPU,
                device_indices=[0, 1],
            )

    def test_spec_to_convert_config_raise_when_parallel_has_removed_device_fields(self):
        """场景：YAML parallel 显式写已删除的 worker_device。
        预期：extra=forbid 校验失败（设备只由 CLI 决定）。
        """
        with pytest.raises(Exception, match="worker_device"):
            ModelslimConvertServiceConfig.model_validate({"linears": [], "parallel": {"worker_device": "npu"}})

    def test_spec_to_convert_config_map_cpu_workers_one_to_thread_backend(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "linears": [],
                "parallel": {"cpu_workers": 1},
            }
        )
        cfg = spec_to_convert_config(spec, model_path="/m", save_path="/o")
        assert cfg.parallel.worker_backend == "thread"

    def test_spec_to_convert_config_use_fixed_dependency_group_and_shard_cache(self):
        spec = ModelslimConvertServiceConfig.model_validate(
            {
                "linears": [],
                "parallel": {"cpu_workers": 8},
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
                "parallel": {"cpu_workers": 8},
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
