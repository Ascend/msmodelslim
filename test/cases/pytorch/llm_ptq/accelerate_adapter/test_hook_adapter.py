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
# pylint: disable=no-name-in-module,redefined-outer-name,use-a-generator

import os
import shutil
from typing import Dict, Union, Optional

import pytest
import torch
from accelerate import dispatch_model
from accelerate.hooks import AlignDevicesHook
from torch import nn
from transformers import PreTrainedModel, LlamaConfig, LlamaForCausalLM

# 检查 accelerate 版本，如果不是 0.28.0 则跳过所有测试
try:
    import importlib.metadata

    accelerate_version = importlib.metadata.version('accelerate')
except ImportError:
    try:
        import pkg_resources

        accelerate_version = pkg_resources.get_distribution('accelerate').version
    except Exception:
        accelerate_version = None

if accelerate_version != '0.28.0':
    pytestmark = pytest.mark.skip(reason=f"accelerate version is {accelerate_version}, required version is 0.28.0")

from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import UpdateWeightsMapHook
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import (
    replace_device_align_hook_if_needed,
    move_update_weight_hook_if_need,
    PrepareWeight,
)
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.utils import HF_HOOK

ALL_CPU_DEVICE_MAP = {
    'model.embed_tokens': 'cpu',
    'model.norm': 'cpu',
    'model.rotary_emb': 'cpu',
    'lm_head': 'cpu',
    'model.layers': 'cpu',
}

CPU_DISK_DEVICE_MAP = {
    'model.embed_tokens': 'cpu',
    'model.norm': 'cpu',
    'model.rotary_emb': 'cpu',
    'lm_head': 'cpu',
    'model.layers': 'disk',
}

OFFLOAD_DIR = './offload'


@pytest.fixture
def clear_offload_dir():
    yield

    if os.path.exists(OFFLOAD_DIR):
        shutil.rmtree(OFFLOAD_DIR)


def get_fake_dispatched_llama_model(
    device_map: Dict[str, Union[str, int, torch.device]] = None, main_device: Optional[torch.device] = None
):
    """
    获取一个随机的、非常小的Llama模型，并将其dispatch至相应的设备
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = LlamaConfig.from_json_file(config_path)
    model = LlamaForCausalLM(config)
    device_map = ALL_CPU_DEVICE_MAP if device_map is None else device_map
    # 1. 保存原始 umask
    original_umask = os.umask(0)  # 临时设为 0 并获取原始值
    try:
        # 2. 设置目标 umask（0o026 对应权限 640）
        os.umask(0o026)
        dispatch_model(
            model,
            device_map=device_map,
            main_device=main_device,
            state_dict=model.state_dict(),
            offload_dir=OFFLOAD_DIR,
        )
    finally:
        # 4. 无论是否出错，都恢复原始 umask
        os.umask(original_umask)
    return model


def check_all_hook(model: PreTrainedModel, target_hook_type, allow_none: bool = True):
    for _, module in model.named_modules():
        hook = getattr(module, HF_HOOK, None)

        if allow_none and hook is None:
            continue

        if not isinstance(hook, target_hook_type):
            return False

    return True


def test_no_error_if_model_has_no_device_align_hook(clear_offload_dir):
    model = get_fake_dispatched_llama_model(ALL_CPU_DEVICE_MAP)
    assert not any([value.device.type == 'meta' for _, value in model.state_dict().items()])
    replace_device_align_hook_if_needed(model)
    assert check_all_hook(model, AlignDevicesHook)


def test_replace_result_if_model_has_device_align_hook(clear_offload_dir):
    model = get_fake_dispatched_llama_model(CPU_DISK_DEVICE_MAP)
    assert any([value.device.type == 'meta' for _, value in model.state_dict().items()])
    replace_device_align_hook_if_needed(model)
    assert check_all_hook(model, UpdateWeightsMapHook)


def test_prepare_weight_can_load_weight_from_disk(clear_offload_dir):
    model = get_fake_dispatched_llama_model(CPU_DISK_DEVICE_MAP)
    replace_device_align_hook_if_needed(model)
    assert any([value.device.type == 'meta' for _, value in model.state_dict().items()])

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with PrepareWeight(module):
                assert not any([value.device.type == 'meta' for _, value in module.state_dict().items()])


@torch.no_grad()
def test_prepare_weight_can_update_weight_with_post_force(clear_offload_dir):
    model = get_fake_dispatched_llama_model(CPU_DISK_DEVICE_MAP)
    replace_device_align_hook_if_needed(model)
    assert any([value.device.type == 'meta' for _, value in model.state_dict().items()])

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with PrepareWeight(module, post_force=True):
                assert not any([value.device.type == 'meta' for _, value in module.state_dict().items()])
                old_weight = module.weight.clone().detach()
                module.weight.div_(torch.tensor([0.5]))
                updated_weight_value = module.weight.clone().detach()

            with PrepareWeight(module):
                new_load_weight = module.weight.clone().detach()

            assert not torch.equal(old_weight, updated_weight_value)
            assert torch.equal(updated_weight_value, new_load_weight)


@torch.no_grad()
def test_prepare_weight_can_manage_new_submodule_with_post_recurse(clear_offload_dir):
    model = get_fake_dispatched_llama_model(CPU_DISK_DEVICE_MAP)
    replace_device_align_hook_if_needed(model)
    assert any([value.device.type == 'meta' for _, value in model.state_dict().items()])

    original_umask = os.umask(0)
    try:
        os.umask(0o026)
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with PrepareWeight(module, post_recurse=True):
                    module.sub_module = nn.LayerNorm(128, device=module.weight.device)
                    module.sub_module.weight.data = torch.ones([128, 128], device=module.weight.device)
                    move_update_weight_hook_if_need(module, module, as_submodule=True)

                # new submodule should have same device type with module
                if module.weight.device.type == 'meta':
                    assert module.sub_module.weight.device.type == 'meta'

                # new submodule can be load correctly
                with PrepareWeight(module):
                    assert module.weight.device.type != 'meta'
                    assert module.sub_module.weight.device.type == module.weight.device.type
                    assert torch.equal(torch.ones([128, 128]), module.sub_module.weight)
    finally:
        os.umask(original_umask)


@torch.no_grad()
def test_prepare_weight_can_manage_new_parameters_automatic(clear_offload_dir):
    model = get_fake_dispatched_llama_model(CPU_DISK_DEVICE_MAP)
    replace_device_align_hook_if_needed(model)
    assert any([value.device.type == 'meta' for _, value in model.state_dict().items()])

    original_umask = os.umask(0)
    try:
        os.umask(0o026)
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with PrepareWeight(module):
                    assert not any([value.device.type == 'meta' for _, value in module.state_dict().items()])
                    module.new_param = nn.Parameter(torch.tensor([1.0]))

                # new param should have same device type with module
                if module.weight.device.type == 'meta':
                    assert module.new_param.device.type == 'meta'

                # new param can be load correctly
                with PrepareWeight(module):
                    assert module.new_param.device.type != 'meta'
                    assert module.new_param.device.type == module.weight.device.type
                    assert torch.equal(torch.tensor([1.0]), module.new_param)
    finally:
        os.umask(original_umask)
