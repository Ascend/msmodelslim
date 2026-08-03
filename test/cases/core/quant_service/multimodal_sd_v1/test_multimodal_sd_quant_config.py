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

import pytest

from msmodelslim.core.quant_service.multimodal_sd_v1.quant_config import (
    DumpConfig,
    MultimodalSDConfig,
    MultimodalSDServiceConfig,
    load_specific_config,
)
from msmodelslim.utils.exception import SchemaValidateError


def test_dump_config_default():
    """测试DumpConfig默认值"""
    config = DumpConfig()
    assert config.enable_dump is True
    assert config.capture_mode == "args"
    assert config.dump_data_dir == ""


def test_dump_config_custom():
    """测试DumpConfig自定义值"""
    config = DumpConfig(capture_mode="args", dump_data_dir="/test/path")
    assert config.capture_mode == "args"
    assert config.dump_data_dir == "/test/path"


def test_dump_config_enable_dump_false():
    """测试DumpConfig enable_dump=False 能正确解析"""
    config = DumpConfig(enable_dump=False)
    assert config.enable_dump is False
    config_from_dict = DumpConfig(**{"enable_dump": False, "capture_mode": "args", "dump_data_dir": ""})
    assert config_from_dict.enable_dump is False


def test_multimodal_sd_config_default():
    """测试MultimodalSDConfig默认配置"""
    config = MultimodalSDConfig(dump_config=DumpConfig())
    assert isinstance(config.dump_config, DumpConfig)
    assert config.extra_params == {}


def test_multimodal_sd_config_with_extra_params():
    """测试MultimodalSDConfig包含额外参数"""
    config = MultimodalSDConfig(dump_config=DumpConfig(), extra_param1="value1", extra_param2=123)
    assert config.extra_params == {"extra_param1": "value1", "extra_param2": 123}


def test_multimodal_sd_service_config_with_dict():
    """测试MultimodalSDServiceConfig使用字典配置"""
    config_dict = {"dump_config": {"capture_mode": "args", "dump_data_dir": "/test"}}
    service_config = MultimodalSDServiceConfig(multimodal_sd_config=config_dict)
    assert isinstance(service_config.multimodal_sd_config, MultimodalSDConfig)
    assert service_config.multimodal_sd_config.dump_config.dump_data_dir == "/test"


def test_multimodal_sd_service_config_with_object():
    """测试MultimodalSDServiceConfig使用对象配置"""
    sd_config = MultimodalSDConfig(dump_config=DumpConfig(dump_data_dir="/test"))
    service_config = MultimodalSDServiceConfig(multimodal_sd_config=sd_config)
    assert isinstance(service_config.multimodal_sd_config, MultimodalSDConfig)
    assert service_config.multimodal_sd_config.dump_config.dump_data_dir == "/test"


def test_load_specific_config_valid():
    """测试load_specific_config加载有效配置"""
    yaml_spec = {"multimodal_sd_config": {"dump_config": {"capture_mode": "args", "dump_data_dir": "/valid/path"}}}
    config = load_specific_config(yaml_spec)
    assert isinstance(config, MultimodalSDServiceConfig)
    assert config.multimodal_sd_config.dump_config.dump_data_dir == "/valid/path"
    assert config.multimodal_sd_config.dump_config.enable_dump is True


def test_load_specific_config_enable_dump_false():
    """测试load_specific_config加载 enable_dump: false"""
    yaml_spec = {
        "multimodal_sd_config": {"dump_config": {"enable_dump": False, "capture_mode": "args", "dump_data_dir": ""}}
    }
    config = load_specific_config(yaml_spec)
    assert isinstance(config, MultimodalSDServiceConfig)
    assert config.multimodal_sd_config.dump_config.enable_dump is False


def test_load_specific_config_invalid_type():
    """测试load_specific_config加载非字典配置"""
    with pytest.raises(SchemaValidateError) as excinfo:
        load_specific_config("not a dict")
    assert "task spec must be dict" in str(excinfo.value)


def test_load_specific_config_invalid_content():
    """测试load_specific_config加载无效内容配置"""
    with pytest.raises(SchemaValidateError):
        # 无效配置（缺少必要字段或类型错误）
        load_specific_config({"multimodal_sd_config": {"dump_config": 123}})


def test_service_config_per_expert_optional():
    """per_expert 可选，默认 None"""
    config = MultimodalSDServiceConfig(
        multimodal_sd_config={"dump_config": {"capture_mode": "args", "dump_data_dir": ""}}
    )
    assert config.per_expert is None


def test_service_config_per_expert_loads_and_resolves():
    """per_expert 解析为 Processor 列表；命中整链替换，未命中回退 process；{} 等同未覆盖"""
    config = load_specific_config(
        {
            "process": [{"type": "load"}],
            "per_expert": {
                "low_noise_model": [{"type": "smooth_quant"}],
                "high_noise_model": [{"type": "smooth_quant"}, {"type": "load"}],
            },
            "multimodal_sd_config": {"dump_config": {"capture_mode": "args", "dump_data_dir": ""}},
        }
    )
    assert len(config.per_expert["low_noise_model"]) == 1
    assert config.per_expert["low_noise_model"][0].type == "smooth_quant"
    assert len(config.per_expert["high_noise_model"]) == 2
    assert config.per_expert["high_noise_model"][1].type == "load"

    low = config.resolve_process_for_expert("low_noise_model")
    high = config.resolve_process_for_expert("high_noise_model")
    assert low[0].type == "smooth_quant"
    assert high[0].type == "smooth_quant" and high[1].type == "load"

    empty = load_specific_config(
        {
            "process": [{"type": "load"}],
            "per_expert": {},
            "multimodal_sd_config": {"dump_config": {"capture_mode": "args", "dump_data_dir": ""}},
        }
    )
    assert empty.resolve_process_for_expert("low_noise_model")[0].type == "load"


def test_resolve_process_for_expert_returns_shallow_copy():
    """返回副本，避免后续 Runner 改动污染默认 process / per_expert 列表"""
    config = load_specific_config(
        {
            "process": [{"type": "load"}],
            "per_expert": {"low_noise_model": [{"type": "smooth_quant"}]},
            "multimodal_sd_config": {"dump_config": {"capture_mode": "args", "dump_data_dir": ""}},
        }
    )
    low = config.resolve_process_for_expert("low_noise_model")
    high = config.resolve_process_for_expert("high_noise_model")
    low.append(object())
    high.clear()
    assert len(config.per_expert["low_noise_model"]) == 1
    assert len(config.process) == 1
