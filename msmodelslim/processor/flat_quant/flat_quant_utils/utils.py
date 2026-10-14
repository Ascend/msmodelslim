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
# pylint: disable=inconsistent-return-statements,too-many-nested-blocks,useless-return

import math
import torch
import numpy as np
from scipy.linalg import qr
from torch.nn import Module
from typing import Union, Pattern
from msmodelslim.utils.logging import get_logger  # noqa: F401
from typing import List, Dict, Any  # noqa: F401
from msmodelslim.utils.exception import UnexpectedError

npu_available = False
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass
else:
    npu_available = True


def get_init_scale(w_smax, x_smax, alpha=0.5):
    """计算初始化缩放因子：基于权重量化最大值和激活最大值的加权比"""
    return (w_smax.pow(1 - alpha) / x_smax.pow(alpha)).clamp(min=1e-5)


def get_decompose_dim(n):
    """获取分解维度：将整数 n 分解为两个平方数之差，返回其线性组合维度"""
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


def get_random_orthg(size):
    """生成随机正交矩阵：使用 QR 分解生成随机正交阵，并修正符号"""
    h = np.random.randn(size, size)
    q, r = qr(h)
    q_modified = q @ np.diag(np.sign(np.diag(r)))
    return torch.from_numpy(q_modified)


def get_init_weight(dim):
    """初始化正交权重矩阵：生成一个维度为 dim 的随机正交矩阵"""
    return get_random_orthg(dim)


def get_inverse(matrix):
    """计算矩阵的逆：在 CPU 上以双精度计算逆矩阵，再转换回原数据类型"""
    dtype = matrix.dtype
    if not npu_available:
        return matrix.double().inverse().to(dtype)
    else:
        device = matrix.device
        return matrix.cpu().double().inverse().to(device=device, dtype=dtype)


def get_n_set_parameters_byname(model, required_names):
    """根据模块名称获取可训练参数：匹配指定名称模式的参数，并设置 requires_grad=True"""
    params = []
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return params


def set_require_grad_all(model, requires_grad):
    """设置模型所有参数的 requires_grad 属性"""
    for _, param in model.named_parameters():
        param.requires_grad = requires_grad
    return


def stat_input_hook(m, x, y, name, act_stats):
    """统计输入张量的最大值：用于激活量化校准过程中的动态统计"""
    if isinstance(x, tuple):
        x = x[0]
    stat_tensor(act_stats, name, x)


def stat_tensor(act_stats, name, x):
    """更新输入张量的最大绝对值统计：在多个样本上取最大值，用于量化范围估计"""
    if 'input_max' not in act_stats[name]:
        act_stats[name]['input_max'] = x.view(-1, x.shape[-1]).abs().max(0)[0].clone().detach().cpu()
    else:
        tmp = x.view(-1, x.shape[-1]).abs().max(0)[0].clone().detach().cpu()
        act_stats[name]['input_max'] = torch.maximum(act_stats[name]['input_max'].npu(), tmp.npu())


def get_trainable_parameters(model, base_lr=3e-5):
    """获取所有可训练参数及其对应的优化器配置"""
    params = {}
    params["linear_u"] = get_n_set_parameters_byname(model, ['linear_u'])
    params["linear_v"] = get_n_set_parameters_byname(model, ['linear_v'])
    params["trans_linear"] = get_n_set_parameters_byname(model, ['trans_linear'])
    params["linear_diag"] = get_n_set_parameters_byname(model, ['linear_diag'])
    params["diag_scale"] = get_n_set_parameters_byname(model, ['diag_scale'])
    params["clip_factor"] = get_n_set_parameters_byname(model, ['clip_factor'])
    trainable_params = [{"params": params["linear_u"], "lr": base_lr}]
    trainable_params.append({"params": params["linear_v"], "lr": base_lr})
    trainable_params.append({"params": params["trans_linear"], "lr": base_lr})
    trainable_params.append({"params": params["linear_diag"], "lr": base_lr})
    trainable_params.append({"params": params["diag_scale"], "lr": base_lr})
    trainable_params.append({"params": params["clip_factor"], "lr": base_lr * 10})
    need_train = any(len(value) > 0 for value in params.values())
    return params, trainable_params, need_train


def get_para_names():
    """获取所有可训练参数的名称列表"""
    para_names = []
    para_names.append("linear_u")
    para_names.append("linear_v")
    para_names.append("trans_linear")
    para_names.append("linear_diag")
    para_names.append("diag_scale")
    para_names.append("clip_factor")
    return para_names


def match_pattern(pair_name: str, pattern: Union[str, Pattern]):
    """匹配模块名是否符合指定模式：支持字符串前缀匹配或正则表达式"""
    if isinstance(pattern, str):
        return pair_name.startswith(pattern)
    else:
        return pattern.match(pair_name)


def move_tensors_to_device(data, device):
    """递归遍历嵌套数据结构，将所有Tensor移动到指定设备"""
    if isinstance(data, torch.Tensor):
        try:
            return data.to(device)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                raise UnexpectedError("CUDA out of memory occurred during the data transfer phase.") from e

    elif isinstance(data, dict):
        return {key: move_tensors_to_device(value, device) for key, value in data.items()}

    elif isinstance(data, list):
        return [move_tensors_to_device(item, device) for item in data]

    elif isinstance(data, tuple):
        return tuple(move_tensors_to_device(item, device) for item in data)

    else:
        return data


def empty_cache():
    """清空NPU或CUDA缓存，根据设备类型自动选择"""
    if npu_available:
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def get_module_by_name(model: Module, submodule_key: str, prefix: str = None) -> Module:
    """根据名称路径获取模型中的子模块"""
    if prefix is not None:
        submodule_key = submodule_key[len(prefix) + 1 :]
    module_tokens = submodule_key.split('.')
    cur_mod = model
    for s in module_tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


def set_module_by_name(model: Module, submodule_key: str, module: Module, clone_hooks: bool = True, prefix: str = None):
    """根据名称路径设置模型中的子模块，并可选择是否克隆原模块的钩子"""
    if prefix is not None:
        submodule_key = submodule_key[len(prefix) + 1 :]
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    if clone_hooks:
        old_module = getattr(cur_mod, tokens[-1])
        clone_module_hooks(old_module, module)
    setattr(cur_mod, tokens[-1], module)


def clone_module_hooks(source_module: Module, target_module: Module):
    """将源模块的前向、后向钩子克隆到目标模块"""
    hook_types = [
        ('_forward_pre_hooks', 'register_forward_pre_hook'),
        ('_forward_hooks', 'register_forward_hook'),
        ('_backward_pre_hooks', 'register_backward_pre_hook'),
        ('_backward_hooks', 'register_backward_hook'),
    ]
    for hook_attr, register_method in hook_types:
        if hasattr(source_module, hook_attr):
            hooks_dict = getattr(source_module, hook_attr, {})
            if hooks_dict:
                register_func = getattr(target_module, register_method, None)
                if register_func:
                    for hook_fn in hooks_dict.values():
                        try:
                            register_func(hook_fn)
                        except (TypeError, AttributeError):
                            continue


def remove_after_substring(text, substring):
    """移除字符串中指定子串之后的部分，保留子串及其之前内容"""
    index = text.find(substring)
    if index != -1:
        return text[: index + len(substring)]
    return text


def convert_outputs_to_inputs(outputs):
    """将输出列表转换为输入格式：每个输出作为一个列表元素"""
    converted_inputs = []
    for output in outputs:
        converted_inputs.append([output])
    return converted_inputs
