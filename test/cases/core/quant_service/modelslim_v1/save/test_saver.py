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

from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config
from msmodelslim.core.quant_service.modelslim_v1.save.saver import (
    AutoSaverBaseConfig,
    AutoSaverProcessor,
    _convert_hookir_to_wrapper,
    validate_auto_saver_processor_config_list,
)
import msmodelslim.ir as qir


class TestAutoSaverBaseConfig:
    """Tests for AutoSaverBaseConfig via AscendV1Config."""

    def test_set_save_directory_updates_path_when_called(self):
        """场景：AscendV1Config.set_save_directory。预期：save_directory 更新为给定路径。"""
        cfg = AscendV1Config()
        assert isinstance(cfg, AutoSaverBaseConfig)
        assert cfg.save_directory == "."
        cfg.set_save_directory("/data/quant_out")
        assert cfg.save_directory == "/data/quant_out"


class TestValidateAutoSaverConfigList:
    """Tests for validate_auto_saver_processor_config_list."""

    def test_validates_dict_list_when_last_is_saver_config(self):
        """场景：列表末项为 saver 配置 dict。预期：解析为 AutoSaverBaseConfig。"""
        result = validate_auto_saver_processor_config_list([{"type": "ascendv1_saver", "part_file_size": 4}])
        assert isinstance(result[-1], AutoSaverBaseConfig)

    def test_raises_type_error_when_last_not_saver(self):
        """场景：末项校验结果非 AutoSaverBaseConfig。预期：TypeError。"""
        not_saver = MagicMock()
        with patch.object(AutoSaverBaseConfig, "model_validate", return_value=not_saver):
            with pytest.raises(TypeError, match="not a saver config"):
                validate_auto_saver_processor_config_list([{"type": "x"}])

    def test_raises_value_error_when_item_not_dict_or_config(self):
        """场景：列表项类型非法。预期：ValueError。"""
        with pytest.raises(ValueError, match="Invalid config item type"):
            validate_auto_saver_processor_config_list([123])

    def test_raises_value_error_when_not_list(self):
        """场景：入参非 list。预期：ValueError。"""
        with pytest.raises(ValueError, match="Expected a list"):
            validate_auto_saver_processor_config_list("bad")


class _StubSaver(AutoSaverProcessor):  # pylint: disable=abstract-method
    """Concrete saver for exercising base AutoSaverProcessor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._float_called = (None, None)

    def merge_ranks(self) -> None:
        pass

    def on_float_linear(self, prefix: str, module: nn.Linear) -> None:
        self._float_called = (prefix, module)

    def on_float_module(self, prefix: str, module: nn.Module) -> None:
        self._float_called = (prefix, module)


class TestAutoSaverProcessor:
    """Tests for AutoSaverProcessor routing."""

    @pytest.fixture
    def saver(self):
        model = nn.Sequential(nn.Linear(2, 2))
        return _StubSaver(model, AscendV1Config(), adapter=MagicMock())

    def test_is_data_free_returns_true_when_queried(self, saver):
        """场景：查询 is_data_free。预期：True。"""
        assert saver.is_data_free() is True

    def test_support_distributed_returns_false_when_default_instance(self, saver):
        """场景：查询 support_distributed。预期：False。"""
        assert saver.support_distributed() is False

    def test_process_module_calls_on_float_for_unknown_type(self, saver):
        """场景：未知模块类型。预期：走 on_float_module。"""
        mod = nn.ReLU()
        saver._process_module("relu", mod)
        assert mod in saver.processed_modules
        assert saver._float_called[1] is mod

    def test_on_wrapper_ir_atomic_processes_wrapper_only_when_atomic_true(self, saver):
        """场景：原子 WrapperIR。预期：仅处理 wrapper 自身。"""
        inner = nn.Linear(2, 2)
        wrapper = MagicMock(spec=qir.WrapperIR)
        wrapper.wrapped_module = inner
        wrapper.is_atomic.return_value = True
        calls = []

        def _track(prefix, module):
            calls.append(type(module).__name__)

        saver._process_module = _track
        saver.on_wrapper_ir("w", wrapper)
        assert calls == ["MagicMock"]

    def test_on_wrapper_ir_non_atomic_processes_wrapped_first_when_atomic_false(self, saver):
        """场景：非原子 WrapperIR。预期：先处理 wrapped 再处理 wrapper。"""
        inner = nn.Linear(2, 2)
        wrapper = MagicMock(spec=qir.WrapperIR)
        wrapper.wrapped_module = inner
        wrapper.is_atomic.return_value = False
        calls = []

        def _track(prefix, module):
            calls.append(type(module).__name__)

        saver._process_module = _track
        saver.on_wrapper_ir("w", wrapper)
        assert "Linear" in calls[0]

    def test_postprocess_invokes_process_for_submodules_when_called(self, saver):
        """场景：postprocess BatchProcessRequest。预期：处理子模块。"""
        linear = nn.Linear(2, 2)
        req = BatchProcessRequest(name="l", module=linear, datas=[], outputs=None)
        saver.postprocess(req)
        assert linear in saver.processed_modules

    def test_post_run_scans_unprocessed_modules_when_called(self, saver):
        """场景：post_run。预期：未处理子模块被 on_float_module 处理。"""
        saver.post_run()
        assert len(saver.processed_modules) >= 1

    def test_process_map_routes_relu_to_on_float_module(self, saver):
        """场景：ReLU 模块。预期：on_float_module 被调用。"""
        relu = nn.ReLU()
        saver._process_module("relu", relu)
        assert saver._float_called[1] is relu

    def test_validate_raises_value_error_when_invalid_item_in_list(self):
        """场景：validate 非法列表项。预期：ValueError。"""
        with pytest.raises(ValueError, match="Invalid config item type"):
            validate_auto_saver_processor_config_list([object()])

    def test_on_rotation_wrapper_processes_wrapped_module_when_called(self, saver):
        """场景：QuarotOnlineHeadRotationWrapper。预期：处理 wrapped_module。"""
        inner = nn.Linear(2, 2)
        wrapper = MagicMock(spec=qir.QuarotOnlineHeadRotationWrapper)
        wrapper.wrapped_module = inner
        saver.on_rotation_wrapper("w", wrapper)
        assert inner in saver.processed_modules

    def test_on_flat_clip_wrapper_processes_wrapped_module_when_called(self, saver):
        """场景：FlatQuantOnlineWrapper。预期：处理 wrapped_module。"""
        inner = nn.Linear(2, 2)
        wrapper = MagicMock(spec=qir.FlatQuantOnlineWrapper)
        wrapper.wrapped_module = inner
        saver.on_flat_clip_wrapper("f", wrapper)
        assert inner in saver.processed_modules


class TestConvertHookirToWrapper:
    """Tests for _convert_hookir_to_wrapper."""

    def test_converts_hookir_to_wrapper_when_hook_present(self):
        """场景：子模块含 HookIR pre_hook。预期：set_submodule 被调用。"""
        model = nn.Sequential(nn.Linear(2, 2))
        hook_ir = MagicMock(spec=qir.HookIR)
        wrapper_mod = nn.Linear(2, 2)
        hook_ir.wrapper_module.return_value = wrapper_mod
        model[0].register_forward_pre_hook(hook_ir)

        with patch.object(nn.Sequential, "set_submodule") as mock_set:
            _convert_hookir_to_wrapper(model)
            assert mock_set.called
