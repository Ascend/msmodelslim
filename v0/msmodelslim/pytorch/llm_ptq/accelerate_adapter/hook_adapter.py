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


import os
from typing import Mapping, Dict, Optional, Union

import torch
from accelerate.hooks import ModelHook, AlignDevicesHook, remove_hook_from_module, add_hook_to_module
from accelerate.utils import PrefixedDataset, OffloadedWeightsLoader
from accelerate.utils import offload_state_dict, offload_weight, save_offload_index, load_offloaded_weight
from safetensors import safe_open

from ascend_utils.common.security import get_valid_read_path, MAX_READ_FILE_SIZE_32G
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.utils import clear_device_cache, judge_module_with_accelerate
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.utils import HF_HOOK


class PrepareWeight:
    """
    调用Accelerate的Hook，将模块的权重准备至执行设备上
    """

    def __init__(self, module: torch.nn.Module, post_force=False, post_recurse=False):
        self.module = module
        self.post_force = post_force
        self.post_recurse = post_recurse

    def __enter__(self):
        if not judge_module_with_accelerate(self.module):
            return

        hook = getattr(self.module, HF_HOOK)

        # enable and save old state
        if isinstance(hook, UpdateWeightsMapHook):
            self.post_force = hook.enable_post_force(self.post_force)
            self.post_recurse = hook.enable_post_recurse(self.post_recurse)

        hook.pre_forward(self.module, *[], **{})

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not judge_module_with_accelerate(self.module):
            return

        hook = getattr(self.module, HF_HOOK)

        hook.post_forward(self.module, *[torch.zeros([1])])
        # restore old state
        if isinstance(hook, UpdateWeightsMapHook):
            hook.enable_post_force(self.post_force)
            hook.enable_post_recurse(self.post_recurse)


def upload_module_weights(module: torch.nn.Module, post_force=False, post_recurse=False):
    if not judge_module_with_accelerate(module):
        return

    hook = getattr(module, HF_HOOK)

    if isinstance(hook, UpdateWeightsMapHook):
        hook.enable_post_force(post_force)
        hook.enable_post_recurse(post_recurse)

    hook.pre_forward(module, *[], **{})


def offload_module_weights(module: torch.nn.Module):
    if not judge_module_with_accelerate(module):
        return

    hook = getattr(module, HF_HOOK)
    hook.post_forward(module, *[torch.zeros([1])], **{})


class WritableOffloadedWeightsLoader(OffloadedWeightsLoader):
    """
    对OffloadedWeightsLoader的拓展，支持参数写入
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor] = None,
                 save_folder: Optional[Union[str, os.PathLike]] = None,
                 index: Mapping = None, device=None):
        super().__init__(state_dict, save_folder, index, device)

    def __setitem__(self, key, value):
        if key in self.state_dict or self.save_folder is None:
            self.state_dict[key] = value
            self.update_all_keys()
            return
        index = offload_weight(value, key, self.save_folder, index=self.index)
        save_offload_index(index, self.save_folder)
        offload_state_dict(self.save_folder, {key: value})
        self.update_all_keys()
        return

    def __getitem__(self, key: str):
        # State dict gets priority
        if key in self.state_dict:
            return self.state_dict[key]
        weight_info = self.index[key]
        if weight_info.get("safetensors_file") is not None:
            device = "cpu" if self.device is None else self.device
            tensor = None

            # torch_npu不支持以int的方式加载，先加载到CPU上
            with safe_open(weight_info["safetensors_file"], framework="pt", device="cpu") as f:
                tensor = f.get_tensor(weight_info.get("weight_name", key))

            if "dtype" in weight_info:
                tensor = tensor.to(getattr(torch, weight_info["dtype"]))

            if tensor.device != torch.device(device):
                tensor = tensor.to(device)
            return tensor

        weight_file = os.path.join(self.save_folder, f"{key}.dat")
        weight_file = get_valid_read_path(weight_file, size_max=MAX_READ_FILE_SIZE_32G)
        return load_offloaded_weight(weight_file, weight_info)

    def update_all_keys(self):
        self.all_keys = list(self.state_dict.keys())
        self.all_keys.extend([key for key in self.index if key not in self.all_keys])
        pass


class UpdateWeightsMapHook(ModelHook):
    """
    包装AlignDevicesHook并拓展其功能，可以支持模块动态添加参数
    """

    logger = msmodelslim_logger

    def __init__(self, old_hook: AlignDevicesHook):
        if not isinstance(old_hook, AlignDevicesHook):
            raise ValueError("old_hook must be AlignDevicesHook")
        self.old_hook = old_hook
        self.init_force = False
        self.init_recurse = False
        self.post_force = False
        self.post_recurse = False
        self.depth = 0
        self.old_hook_offload = old_hook.offload

    def enable_init_force(self, enable: bool) -> bool:
        old_enable = self.init_force
        self.init_force = enable
        return old_enable

    def enable_init_recurse(self, enable: bool) -> bool:
        old_enable = self.init_recurse
        self.init_recurse = enable
        return old_enable

    def enable_post_force(self, enable: bool) -> bool:
        old_enable = self.post_force
        self.post_force = enable
        return old_enable

    def enable_post_recurse(self, enable: bool) -> bool:
        old_enable = self.post_recurse
        self.post_recurse = enable
        return old_enable

    def get_prefix(self):
        return self.old_hook.weights_map.prefix if self.old_hook.weights_map else "Unkown"

    def update_weights_map(self, module, force_update=False, recurse=False):

        """
        更新WritableOffloadedWeightsLoader的内容，分两种情况，一是参数值有更新，二是添加了新的参数。
        """

        if not self.old_hook_offload:
            return

        weights_map = self.old_hook.weights_map

        if weights_map is None:
            return

        if not isinstance(weights_map, PrefixedDataset):
            raise ValueError(f"weights_map {weights_map} is not PrefixedDataset")

        for key, item in module.named_parameters(recurse=recurse):
            if force_update or key not in weights_map:

                if key not in weights_map:
                    self.logger.debug(f"{weights_map.prefix + key} not in weight map, add it")

                dataset = weights_map.dataset

                if not isinstance(dataset, WritableOffloadedWeightsLoader):
                    raise ValueError(f"dataset {dataset} must be WritableOffloadedWeightsLoader")

                dataset[weights_map.prefix + key] = item.clone().detach().cpu()
            module._parameters[key] = torch.nn.Parameter(item.clone())

        clear_device_cache()

        return

    def init_hook(self, module):
        self.depth = 0
        self.old_hook.offload = self.old_hook_offload
        self.update_weights_map(module, force_update=self.init_force, recurse=self.init_recurse)
        return self.old_hook.init_hook(module)

    def detach_hook(self, module):
        self.depth = 0
        self.old_hook.offload = self.old_hook_offload
        return self.old_hook.detach_hook(module)

    def pre_forward(self, module, *args, **kwargs):

        self.depth += 1

        if self.depth > 1:
            # 对于嵌套的情况，不要再从WritableOffloadedWeightsLoader中再次加载
            self.old_hook.offload = False

        args, kwargs = self.old_hook.pre_forward(module, *args, **kwargs)

        return args, kwargs

    def post_forward(self, module, output):

        if self.depth > 0:
            self.depth -= 1

        if self.depth == 0:
            # 对于嵌套的情况，所有嵌套都退出后，才更新权重
            self.old_hook.offload = self.old_hook_offload
            self.update_weights_map(module, force_update=self.post_force, recurse=self.post_recurse)

        output = self.old_hook.post_forward(module, output)

        return output


def replace_offloaded_weights_loader_if_need(hook: AlignDevicesHook):
    if hook.weights_map is None:
        return

    if isinstance(hook.weights_map, PrefixedDataset):
        old_loader: OffloadedWeightsLoader = hook.weights_map.dataset
        new_loader = WritableOffloadedWeightsLoader(old_loader.state_dict, old_loader.save_folder, old_loader.index,
                                                    old_loader.device)
        hook.weights_map.dataset = new_loader

    return


def replace_device_align_hook_if_needed(module: torch.nn.Module, recurse=True, prefix=""):
    msmodelslim_logger.debug(f"replace_device_align_hook_if_needed for {prefix}")

    if judge_module_with_accelerate(module) and isinstance(getattr(module, HF_HOOK), AlignDevicesHook):
        old_hook = getattr(module, HF_HOOK)
        replace_offloaded_weights_loader_if_need(old_hook)
        new_hook = UpdateWeightsMapHook(getattr(module, HF_HOOK))
        setattr(module, HF_HOOK, new_hook)

    if not recurse:
        return

    for name, module in module.named_children():
        replace_device_align_hook_if_needed(module, recurse, prefix=f"{prefix}.{name}")


def move_update_weight_hook_if_need(old_module, new_module, as_submodule=False, force_update=False):
    """
    将old_module的hook移动至new_module，as_submodule设置为True时，old_module将会成为new_module的子模块
    """
    if not judge_module_with_accelerate(old_module):
        return

    if id(old_module) == id(new_module):
        hook: UpdateWeightsMapHook = getattr(old_module, HF_HOOK)
        hook.old_hook.place_submodules = as_submodule
        return

    hook: UpdateWeightsMapHook = getattr(old_module, HF_HOOK)
    remove_hook_from_module(old_module)
    hook.old_hook.place_submodules = as_submodule
    old_init_force = hook.enable_init_force(force_update)
    old_init_recurse = hook.enable_init_recurse(as_submodule)
    add_hook_to_module(new_module, hook)
    hook.enable_init_force(old_init_force)
    hook.enable_init_recurse(old_init_recurse)


def get_offloaded_weights_loader_if_have(module, recurse=True) -> WritableOffloadedWeightsLoader:
    loader = None

    if judge_module_with_accelerate(module):
        hook = getattr(module, HF_HOOK)

        if isinstance(hook, UpdateWeightsMapHook):

            if hook.old_hook.weights_map is not None and isinstance(hook.old_hook.weights_map, PrefixedDataset):
                loader = hook.old_hook.weights_map.dataset

    if loader is not None:
        return loader

    if not recurse:
        return None

    for _, submodule in module.named_children():
        loader = get_offloaded_weights_loader_if_have(submodule, recurse)
        if loader is not None:
            return loader

    return None


def get_state_dict_copy(module: torch.nn.Module, skip_keys=None, device='cpu'):
    state_dict = {}
    weights_loader = get_offloaded_weights_loader_if_have(module)
    for key, value in module.state_dict().items():
        msmodelslim_logger.debug(f"{key} is on {value.device.type}")
        if skip_keys is not None and key in skip_keys:
            state_dict[key] = torch.zeros([1])
        elif value.device.type == 'meta':
            # load from model's offload weights loader
            state_dict[key] = weights_loader[key].clone().detach().to(device)
        else:
            state_dict[key] = value.clone().detach().to(device)
    return state_dict


def clear_unused_module(module: torch.nn.Module):
    # 清空该模块的参数、子模块、Buffer
    module._parameters = {}
    module._named_modules = {}
    module._named_buffers = {}
