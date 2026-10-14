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

from typing import List

import pytest
from torch import nn

from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig, FusionConfig
from msmodelslim.processor.anti_outlier.iter_smooth import IterSmoothProcessor, IterSmoothProcessorConfig
from msmodelslim.processor.anti_outlier.iter_smooth.interface import IterSmoothInterface
from msmodelslim.utils.exception import SchemaValidateError


class MockModel(nn.Module):
    """模拟模型用于测试"""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)


class MockAdapterWithoutInterface:
    """模拟不实现IterSmoothInterface的适配器"""

    def __init__(self):
        pass


class MockAdapterWithIncompleteConfig(IterSmoothInterface):
    """模拟配置不完整的适配器"""

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        # 缺少subgraph_type
        return [AdapterConfig(subgraph_type=None, mapping=MappingConfig(targets=["layer2"], source="layer1"))]


class MockAdapterWithoutMapping(IterSmoothInterface):
    """模拟缺少mapping的适配器"""

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        # 缺少mapping
        return [AdapterConfig(subgraph_type="norm-linear", mapping=None)]


class MockAdapterWithIncompleteFusion(IterSmoothInterface):
    """模拟FusionConfig不完整的适配器"""

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        # FusionConfig缺少必要参数
        fusion_config = FusionConfig(fusion_type="qkv", num_attention_heads=None, num_key_value_heads=None)
        return [
            AdapterConfig(
                subgraph_type="norm-linear",
                mapping=MappingConfig(targets=["layer2"], source="layer1"),
                fusion=fusion_config,
            )
        ]


class TestIterSmoothProcessor:
    """测试IterSmoothProcessor的各种错误情况"""

    def test_adapter_not_implement_interface(self):
        """测试用例1：用户不实现adapter接口"""
        model = MockModel()
        config = IterSmoothProcessorConfig()
        adapter = MockAdapterWithoutInterface()

        # 现在行为：不再抛出异常，而是记录警告并回退到默认适配器逻辑
        processor = IterSmoothProcessor(model, config, adapter)
        assert isinstance(processor, IterSmoothProcessor)
        assert processor.is_defalut_adapter is True

    def test_adapter_missing_subgraph_type(self):
        """测试用例2：用户adapter不配置subgraph_type"""
        _ = MockModel()
        _ = IterSmoothProcessorConfig()
        adapter = MockAdapterWithIncompleteConfig()

        # 现在 AdapterConfig 在构造阶段就会校验 subgraph_type，并抛出 ValueError
        with pytest.raises(ValueError) as exc_info:
            _ = adapter.get_adapter_config_for_subgraph()

        assert "subgraph_type is required" in str(exc_info.value)

    @pytest.mark.skip
    def test_adapter_missing_mapping(self):
        """测试用例3：用户adapter不配置mapping"""
        model = MockModel()
        config = IterSmoothProcessorConfig()
        adapter = MockAdapterWithoutMapping()

        processor = IterSmoothProcessor(model, config, adapter)

        # 在pre_run时调用adapter.get_adapter_config_for_subgraph()会触发AdapterConfig验证
        with pytest.raises(ValueError) as exc_info:
            processor.pre_run()

        assert "mapping is required" in str(exc_info.value)

    def test_adapter_incomplete_fusion_config(self):
        """测试用例4：用户adapter配置了FusionConfig，但是没有给出num_attention_heads和num_key_value_heads"""
        _ = MockModel()
        _ = IterSmoothProcessorConfig()
        adapter = MockAdapterWithIncompleteFusion()

        # FusionConfig 在构造阶段会校验必要字段，并抛出 ValueError
        with pytest.raises(ValueError) as exc_info:
            _ = adapter.get_adapter_config_for_subgraph()

        assert "QKV融合类型必须提供num_attention_heads和num_key_value_heads" in str(exc_info.value)

    def test_yaml_alpha_validation_negative(self):
        """测试用例5：yaml参数校验alpha为浮点类型且大于0"""
        _ = MockModel()
        _ = MockAdapterWithoutInterface()

        # alpha参数校验在创建config时就会触发
        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessorConfig(alpha=-1)

        error_str = str(exc_info.value)
        assert "value must be in the range (0, 1)" in error_str or "range" in error_str.lower()

    def test_yaml_alpha_validation_out_of_range(self):
        """测试用例5：yaml参数校验alpha超出范围"""
        _ = MockModel()
        _ = MockAdapterWithoutInterface()

        # alpha参数校验在创建config时就会触发
        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessorConfig(alpha=1.5)

        error_str = str(exc_info.value)
        assert "alpha must be in the range (0, 1)" in error_str

    def test_yaml_scale_min_validation_negative(self):
        """测试用例6：yaml参数校验scale_min为浮点类型且大于0"""
        _ = MockModel()
        _ = MockAdapterWithoutInterface()

        # scale_min参数校验在创建config时就会触发
        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessorConfig(scale_min=-1)

        error_str = str(exc_info.value)
        assert "scale_min must be in the range (0, 1)" in error_str

    def test_yaml_scale_min_validation_out_of_range(self):
        """测试用例6：yaml参数校验scale_min超出范围"""
        _ = MockModel()
        _ = MockAdapterWithoutInterface()

        # scale_min参数校验在创建config时就会触发
        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessorConfig(scale_min=2.0)

        error_str = str(exc_info.value)
        assert "scale_min must be in the range (0, 1)" in error_str

    def test_yaml_symmetric_validation_non_boolean(self):
        """测试用例7：yaml参数校验symmetric为布尔类型"""
        _ = MockModel()
        _ = MockAdapterWithoutInterface()

        # symmetric参数校验在创建config时就会触发
        # Pydantic会抛出ValidationError，而不是SchemaValidateError
        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessorConfig(symmetric="haha")

        # 检查错误信息中是否包含我们的验证器消息
        error_str = str(exc_info.value)
        assert "value must be a boolean type" in error_str or "boolean" in error_str.lower()

    def test_yaml_enable_subgraph_type_validation_invalid_element(self):
        """测试用例8：yaml参数校验enable_subgraph_type为字符串列表类型，且元素要在四种子图结构之间"""
        model = MockModel()
        # 需要使用实现了IterSmoothInterface的adapter，否则会在接口检查时提前抛出UnsupportedError
        adapter = MockAdapterWithIncompleteConfig()

        # 创建config，然后创建processor，在__init__时会自动调用validate_parameters验证enable_subgraph_type的内容
        config = IterSmoothProcessorConfig(enable_subgraph_type=["haha"])

        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessor(model, config, adapter)

        assert "Elements in enable_subgraph_type must be in" in str(exc_info.value)

    def test_yaml_enable_subgraph_type_validation_non_list(self):
        """测试用例8：yaml参数校验enable_subgraph_type为非列表类型"""
        _ = MockModel()
        _ = MockAdapterWithoutInterface()

        # enable_subgraph_type参数校验在创建config时就会触发
        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessorConfig(enable_subgraph_type="haha")

        error_str = str(exc_info.value)
        assert "value must be a list type" in error_str or "list" in error_str.lower()

    def test_yaml_include_validation_non_list(self):
        """测试用例9：include参数为非列表类型时在创建config时抛出ValidationError"""
        _ = MockModel()
        _ = MockAdapterWithoutInterface()

        # include参数校验在创建config时就会触发，因为Pydantic会自动验证类型
        # include: Optional[List[str]] = None 期望列表类型，传入整数会失败
        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessorConfig(include=1)

        # 检查错误信息
        error_str = str(exc_info.value)
        assert "include" in error_str.lower() or "list" in error_str.lower()

    def test_yaml_exclude_validation_non_list(self):
        """测试用例9：exclude是字符串列表"""
        _ = MockModel()
        _ = MockAdapterWithoutInterface()

        # exclude参数校验在创建config时就会触发（如果使用了AfterValidator）
        # 由于BaseSmoothProcessorConfig中的exclude没有使用AfterValidator，这里测试类型转换
        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessorConfig(exclude=1)

        error_str = str(exc_info.value)
        assert "exclude" in error_str.lower() or "list" in error_str.lower()

    def test_yaml_include_validation_non_string_elements(self):
        """测试用例9：include列表元素不是字符串时在创建config时抛出ValidationError"""
        _ = MockModel()
        _ = MockAdapterWithoutInterface()

        # include参数校验在创建config时就会触发，因为Pydantic会自动验证类型
        # include: Optional[List[str]] = None 期望字符串列表，传入整数列表会失败
        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessorConfig(include=[1, 2, 3])

        # 检查错误信息
        error_str = str(exc_info.value)
        assert "include" in error_str.lower() or "string" in error_str.lower()

    def test_yaml_exclude_validation_non_string_elements(self):
        """测试用例9：exclude列表元素不是字符串时在创建config时抛出ValidationError"""
        _ = MockModel()
        _ = MockAdapterWithoutInterface()

        # exclude参数校验在创建config时就会触发，因为Pydantic会自动验证类型
        # exclude: Optional[List[str]] = None 期望字符串列表，传入整数列表会失败
        with pytest.raises(SchemaValidateError) as exc_info:
            _ = IterSmoothProcessorConfig(exclude=[1, 2, 3])

        # 检查错误信息
        error_str = str(exc_info.value)
        assert "exclude" in error_str.lower() or "string" in error_str.lower()

    def test_adapter_config_valid_for_subgraph_types(self):
        """AdapterConfig 可正常创建，子图处理顺序由 BaseSmoothProcessor.SUBGRAPH_PRIORITY 决定。"""
        for subgraph_type in ["up-down", "ov", "norm-linear", "linear-linear"]:
            config = AdapterConfig(
                subgraph_type=subgraph_type,
                mapping=MappingConfig(targets=["layer1"], source="layer0"),
            )
            assert config.subgraph_type == subgraph_type
            assert config.mapping.targets == ["layer1"]
            assert config.mapping.source == "layer0"

    def test_mapping_config_source_optional(self):
        """MappingConfig.source 可选，非融合场景下为 None（仅配置 targets）。"""
        mapping = MappingConfig(targets=["layer1", "layer2"], source=None)
        assert mapping.source is None
        assert mapping.targets == ["layer1", "layer2"]


if __name__ == "__main__":
    pytest.main([__file__])
