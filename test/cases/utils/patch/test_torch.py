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
# pylint: disable=pointless-string-statement

"""
msmodelslim.utils.patch.torch 模块的单元测试
"""
from unittest.mock import Mock  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
import pytest  # noqa: E402

from msmodelslim.utils.patch.torch import (  # noqa: E402
    patch_torch,
    _is_torch_nn_module_has_get_submodule,
    _is_torch_nn_module_has_set_submodule,
    _is_torch_has_get_default_device,
)
from msmodelslim.utils.patch.torch import (  # noqa: E402
    _TORCH_DEFAULT_DEVICE,  # noqa: F401
)  # noqa: F401


class TestTorchPatch:
    @staticmethod
    def test_patch_keeps_get_submodule_when_existing():
        if not hasattr(nn.Module, "get_submodule"):
            pytest.skip("当前 PyTorch 无原生 get_submodule，跳过此用例")

        original_method = nn.Module.get_submodule
        patch_torch()
        assert nn.Module.get_submodule is original_method, "补丁不应覆盖原生 get_submodule"

    @staticmethod
    def test_patch_keeps_set_submodule_when_existing():
        if not hasattr(nn.Module, "set_submodule"):
            pytest.skip("当前 PyTorch 无原生 set_submodule，跳过此用例")

        original_method = nn.Module.set_submodule
        patch_torch()
        assert nn.Module.set_submodule is original_method, "补丁不应覆盖原生 set_submodule"

    @staticmethod
    def test_patch_adds_get_default_device_when_missing():
        if hasattr(torch, "get_default_device"):
            delattr(torch, "get_default_device")
        assert not _is_torch_has_get_default_device(), "初始状态应无 get_default_device"

        patch_torch()
        assert _is_torch_has_get_default_device(), "补丁应补充 get_default_device"

        assert torch.get_default_device() == torch.device("cpu"), "初始默认设备应为 CPU"
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(target_device)
        assert torch.get_default_device() == target_device, "默认设备应更新成功"

    @staticmethod
    def test_patch_keeps_get_default_device_when_existing():
        if not hasattr(torch, "get_default_device"):
            pytest.skip("当前 PyTorch 无原生 get_default_device，跳过此用例")

        original_method = torch.get_default_device
        patch_torch()
        assert torch.get_default_device is original_method, "补丁不应覆盖原生 get_default_device"

    @staticmethod
    def test_patch_adds_get_submodule_when_missing(mock_self):
        if hasattr(nn.Module, "get_submodule"):
            delattr(nn.Module, "get_submodule")
        assert not _is_torch_nn_module_has_get_submodule(), "初始状态应无 get_submodule"

        patch_torch()
        assert _is_torch_nn_module_has_get_submodule(), "补丁应补充 get_submodule"

        assert isinstance(mock_self.test_model.get_submodule("conv"), nn.Conv2d)  # 顶层
        assert isinstance(mock_self.test_model.get_submodule("inner.linear"), nn.Linear)  # 嵌套
        assert mock_self.test_model.get_submodule("inner.non_exist") is None  # 不存在的子模块返回 None

    @staticmethod
    def test_patch_adds_set_submodule_when_missing(mock_self):
        if hasattr(nn.Module, "set_submodule"):
            delattr(nn.Module, "set_submodule")
        assert not _is_torch_nn_module_has_set_submodule(), "初始状态应无 set_submodule"

        patch_torch()
        assert _is_torch_nn_module_has_set_submodule(), "补丁应补充 set_submodule"

        new_linear = nn.Linear(5, 2)
        mock_self.test_model.set_submodule("inner.linear", new_linear)  # 嵌套设置
        assert mock_self.test_model.inner.linear is new_linear, "嵌套子模块应设置成功"

        new_conv = nn.Conv2d(16, 32, 3)
        mock_self.test_model.set_submodule("conv", new_conv)  # 顶层设置
        assert mock_self.test_model.conv is new_conv, "顶层子模块应设置成功"

    @pytest.fixture
    def mock_self(self):
        mock = Mock()
        """测试前准备：保存原始方法+初始化测试模型"""
        # 1. 保存 PyTorch 原生方法（用于测试后恢复）
        mock.original_get_submodule = getattr(nn.Module, "get_submodule", None)
        mock.original_set_submodule = getattr(nn.Module, "set_submodule", None)
        mock.original_get_default_device = getattr(torch, "get_default_device", None)
        mock.original_set_default_device = getattr(torch, "set_default_device", None)

        # 2. 初始化嵌套模型（模拟真实场景的子模块结构）
        class InnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.relu = nn.ReLU()

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerModel()
                self.conv = nn.Conv2d(3, 16, 3)

        mock.test_model = TestModel()
        yield mock

        # === teardown ===
        for name, original in [
            ("get_submodule", mock.original_get_submodule),
            ("set_submodule", mock.original_set_submodule),
        ]:
            if original is not None:
                setattr(nn.Module, name, original)
            elif hasattr(nn.Module, name):
                delattr(nn.Module, name)

        for name, original in [
            ("get_default_device", mock.original_get_default_device),
            ("set_default_device", mock.original_set_default_device),
        ]:
            if original is not None:
                setattr(torch, name, original)
            elif hasattr(torch, name):
                delattr(torch, name)

        global _TORCH_DEFAULT_DEVICE
        _TORCH_DEFAULT_DEVICE = torch.device("cpu")
