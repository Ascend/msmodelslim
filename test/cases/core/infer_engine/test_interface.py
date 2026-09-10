# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

msmodelslim/core/infer_engine/interface.py 的单元测试。
"""

import pytest
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.core.infer_engine.interface import (
    FakeQuantInferenceInterface,
    IInferenceEngine,
    InferenceConfig,
    InferenceResult,
)
from msmodelslim.utils.exception import SchemaValidateError


class _ConcreteAdapter(FakeQuantInferenceInterface):
    def build_meta_model(self) -> nn.Module:
        return nn.Module()

    def handle_dataset(self, dataset, device=DeviceType.NPU):
        return list(dataset)


class _ConcreteEngine(IInferenceEngine):
    def run(self, adapter, format_loader, inputs, inference_config, device=DeviceType.NPU, device_indices=None):
        return InferenceResult()


class _SuperEngine(IInferenceEngine):
    """调 super().run 触发抽象基类方法体。"""

    def run(  # pylint: disable=useless-parent-delegation
        self, adapter, format_loader, inputs, inference_config, device=DeviceType.NPU, device_indices=None
    ):
        return super().run(adapter, format_loader, inputs, inference_config, device, device_indices)


class TestInferenceConfig:
    """对应 InferenceConfig。"""

    def test_defaults_max_new_tokens_when_omitted(self):
        cfg = InferenceConfig()

        assert cfg.max_new_tokens == 1

    def test_raise_validation_error_when_max_new_tokens_below_one(self):
        with pytest.raises(SchemaValidateError):
            InferenceConfig(max_new_tokens=0)

    def test_accepts_max_new_tokens_when_positive(self):
        assert InferenceConfig(max_new_tokens=8).max_new_tokens == 8


class TestInferenceResult:
    """对应 InferenceResult。"""

    def test_defaults_to_empty_lists_when_omitted(self):
        result = InferenceResult()

        assert result.generated_token_ids == []
        assert result.generated_texts == []

    def test_keeps_provided_values_when_given(self):
        result = InferenceResult(generated_token_ids=[[1]], generated_texts=["a"])

        assert result.generated_token_ids == [[1]]
        assert result.generated_texts == ["a"]


class TestFakeQuantInferenceInterface:
    """对应 FakeQuantInferenceInterface。"""

    def test_cannot_instantiate_when_abstract(self):
        with pytest.raises(TypeError):
            FakeQuantInferenceInterface()  # pylint: disable=abstract-class-instantiated

    def test_instantiable_when_all_abstract_implemented(self):
        assert isinstance(_ConcreteAdapter(), FakeQuantInferenceInterface)

    def test_handle_dataset_default_device_is_npu_when_omitted(self):
        assert _ConcreteAdapter.handle_dataset.__defaults__[0] is DeviceType.NPU


class TestIInferenceEngine:
    """对应 IInferenceEngine。"""

    def test_run_raises_not_implemented_when_super_called(self):
        engine = _SuperEngine()

        with pytest.raises(NotImplementedError):
            engine.run(None, None, [], InferenceConfig())

    def test_instantiable_when_run_implemented(self):
        assert isinstance(_ConcreteEngine(), IInferenceEngine)
