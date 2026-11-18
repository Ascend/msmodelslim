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


import gc
import functools
from typing import OrderedDict
import inspect

from tqdm import tqdm

import torch
import torch.nn as nn

from ascend_utils import ResListToRelease
from ascend_utils.common.security import check_type
from msmodelslim import logger as msmodelslim_logger

try:
    import torch_npu
except ImportError:
    msmodelslim_logger.warning("Unable to import torch_npu.")

from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import (
    extract_dag,
    PatternProcess,
)
from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils import (
    iter_smooth,
    smooth_ln_fcs,
    os_ln_fcs,
    weight_aware,
)


try:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils import attach_op, Multiplier
except ImportError:
    attach_op, Multiplier = None, None
    msmodelslim_logger.warning(
        "The current CANN version does not support importing the attach_op and Multiplier packages."
    )
from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_outlier import (
    replace_rms_norm,
)
from msmodelslim.pytorch.llm_ptq.accelerate_adapter import PrepareWeight

from .modelslim_adapter import Linear

STAT_KEY_MAX = "max"
STAT_KEY_MIN = "min"
STAT_KEY_SHIFT = "shift"
STAT_KEY_THRESHOLD_CHANNEL = "thres_c"
STAT_KEY_THRESHOLD_TENSOR = "thres_t"
STAT_KEY_SMOOTH_SCALE_MASK = "smooth_scale_mask"
STAT_KEY_SMOOTH_SCALE = "smooth_scale"
STAT_KEY_VARIANCE = "std"
SD3_CONTEXT_EMBEDDER = "context_embedder"
SCALE_MIN_LLM = 1e-5
SCALE_MIN_SD3 = 1e-3

_PREDEFINED_FUSIONS = {
    tuple(): ("context_embedder",)
}

_PREDEFINED_FUSION_KWARGS = {
    "check_group_fusions": False
}


class AntiOutlierAdapter(object):
    """Anti-outlier for LLM activation quantization."""
    def __init__(
            self,
            model: nn.Module,
            calib_data=None,
            cfg: AntiOutlierConfig = None,
            norm_class_name=None,
    ):
        self.logger = msmodelslim_logger
        if cfg.anti_method not in ['m3', 'm5']:
            raise ValueError("Mindspeed Model only support m3 and m5 anti-outlier method!")
        
        if norm_class_name is not None and not isinstance(norm_class_name, str):
            raise TypeError("norm_class_name must be str, please check it.")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module, please check it.")
        self.calib_data = self.check_calib_data(calib_data)
        check_type(cfg, AntiOutlierConfig, param_name="config")

        self.cfg = cfg
        self.device = self.cfg.device
        self.is_context_embedder_model = False
        if SD3_CONTEXT_EMBEDDER + ".weight" in model.state_dict().keys():
            self.is_context_embedder_model = True

        if self.cfg.device == "cpu":
            raise ValueError("Mindspeed Model does't support CPU quantize!")


        self.norm_class_name = norm_class_name

        self.model = model

        self.norm_linear_subgraph = None

        try:
            self.init_dag()
        except Exception as e:
            raise Exception("Please check your config, model and input!", e) from e

    def init_dag(self):
        # 1. Mindspeed model要用npu执行dag构图识别
        dummy_input = self.calib_data[0]

        if self.norm_class_name is not None:
            norm_class = list(OrderedDict.fromkeys([m.__class__ for m in self.model.modules() if
                                                    self.norm_class_name.lower() == m.__class__.__name__.lower()]))
        else:
            norm_class = list(
                OrderedDict.fromkeys(
                    [m.__class__ for m in self.model.modules() if "norm" in m.__class__.__name__.lower()]))
            norm_class = [norm_class[0]]
            self.norm_class_name = norm_class[0].__name__.lower()

        # 2. 加入线性层适配器
        norm_class.append(Linear)

        # 不要保存为成员变量，内部会引用模型以及模型的子模块，导致这些模块的参数正常无法释放
        dag = extract_dag(self.model, dummy_input,
                            hook_nodes=norm_class, anti_method=self.cfg.anti_method)
        
        self.norm_linear_subgraph = dag.get_norm_linear_subgraph()
        if self.cfg.anti_method == 'm4':
            self.linear_linear_subgraph = dag.get_linear_linear_subgraph()
            self.norm_linear_subgraph.update(self.linear_linear_subgraph)
        del dag
        if 'rms' in norm_class[0].__name__.lower():
            replace_rms_norm(self.model, self.norm_class_name)
        gc.collect()
        return

    def trans_to_dict(self, data):
        data_dict = {}
        data_dict['x'] = data[0]
        return data_dict
    
    def stat_tensor(self, act_stats, name, tensor):
        # buffer the latest tensor
        if name not in act_stats:
            act_stats[name] = {}
            if not self.is_context_embedder_model:
                act_stats[name]['tensor'] = tensor

        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).detach()  # [N,C]
        coming_max = torch.max(tensor, dim=0)[0]  # [C]
        coming_min = torch.min(tensor, dim=0)[0]  # [C]

        stat_dict = act_stats[name]

        # collect the min-max value
        if STAT_KEY_MAX in stat_dict:
            stat_dict[STAT_KEY_MAX] = torch.max(stat_dict[STAT_KEY_MAX], coming_max)  # [C]
        else:
            stat_dict[STAT_KEY_MAX] = coming_max

        if STAT_KEY_MIN in stat_dict:
            stat_dict[STAT_KEY_MIN] = torch.min(stat_dict[STAT_KEY_MIN], coming_min)  # [C]
        else:
            stat_dict[STAT_KEY_MIN] = coming_min

        # channel shifting
        if STAT_KEY_SHIFT in stat_dict:
            if self.cfg.ch_align:
                stat_dict[STAT_KEY_SHIFT] = (stat_dict[STAT_KEY_MAX] + stat_dict[STAT_KEY_MIN]) / 2  # [C]
            else:
                stat_dict[STAT_KEY_SHIFT] = torch.zeros(coming_max.size(0)).to(coming_max.device)
        else:
            if self.cfg.ch_align:
                stat_dict[STAT_KEY_SHIFT] = (coming_max + coming_min) / 2
            else:
                stat_dict[STAT_KEY_SHIFT] = torch.zeros(coming_max.size(0)).to(coming_max.device)

        # the tensor-wise max threshold
        tensor_max = torch.max(tensor - stat_dict[STAT_KEY_SHIFT])  # [N, C]
        if STAT_KEY_THRESHOLD_TENSOR in stat_dict:
            stat_dict[STAT_KEY_THRESHOLD_TENSOR] = torch.max(stat_dict[STAT_KEY_THRESHOLD_TENSOR],
                                                             tensor_max)  # [N, C]
        else:
            stat_dict[STAT_KEY_THRESHOLD_TENSOR] = tensor_max

        # the channel-wise max threshold
        channel_max = torch.max(tensor - stat_dict[STAT_KEY_SHIFT], dim=0)[0]  # [C]
        if STAT_KEY_THRESHOLD_CHANNEL in stat_dict:
            stat_dict[STAT_KEY_THRESHOLD_CHANNEL] = torch.max(stat_dict[STAT_KEY_THRESHOLD_CHANNEL],
                                                              channel_max)  # [C]
        else:
            stat_dict[STAT_KEY_THRESHOLD_CHANNEL] = channel_max

        # the tensor-wise std
        tensor_std = torch.std(tensor - stat_dict[STAT_KEY_SHIFT])  # [N, C]
        if STAT_KEY_VARIANCE in stat_dict:
            stat_dict[STAT_KEY_VARIANCE] = torch.max(stat_dict[STAT_KEY_VARIANCE], tensor_std)
        else:
            stat_dict[STAT_KEY_VARIANCE] = tensor_std

        tensor_std_unshifted = torch.std(tensor)
        tensor_mean_unshifted = torch.mean(tensor)
        scale_mask = (((tensor - tensor_mean_unshifted).abs() > 7 * tensor_std_unshifted).sum(dim=0).bool())

        if STAT_KEY_SMOOTH_SCALE_MASK in stat_dict:
            stat_dict[STAT_KEY_SMOOTH_SCALE_MASK] = stat_dict[STAT_KEY_SMOOTH_SCALE_MASK] | scale_mask
        else:
            stat_dict[STAT_KEY_SMOOTH_SCALE_MASK] = scale_mask

        # if anti_method is m4, set tensor_shift to False
        tensor_shift = self.cfg.ch_align and not self.cfg.anti_method == 'm4'
        if tensor_shift:
            channel_max = torch.max((tensor - stat_dict[STAT_KEY_SHIFT]).abs().detach(), dim=0)[0]
        else:
            channel_max = torch.max(tensor.abs().detach(), dim=0)[0]

        if STAT_KEY_SMOOTH_SCALE in stat_dict:
            stat_dict[STAT_KEY_SMOOTH_SCALE] = torch.max(stat_dict[STAT_KEY_SMOOTH_SCALE], channel_max)
        else:
            stat_dict[STAT_KEY_SMOOTH_SCALE] = channel_max

    def os_stats(self):
        """Collect the activations into Dict for outlier suppression."""
        self.model.eval()
        act_stats = {}

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            self.stat_tensor(act_stats, name, x)

        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name))
                )

        for i in tqdm(range(len(self.calib_data))):
            inputs = self.calib_data[i]
            if not self.is_context_embedder_model:
                input_dict = self.trans_to_dict(inputs)
                self.model(**input_dict)
            else:
                self.model(**inputs)

        for name in act_stats:
            stat_dict = act_stats[name]
            for feature in stat_dict:
                stat_dict[feature] = stat_dict[feature].to("cpu")

        for h in hooks:
            h.remove()

        return act_stats

    @torch.no_grad()
    def process(self):
        """The processing of anti-outlier."""
        try:
            self._process()
        except Exception as e:
            raise Exception("Please check your config, model and input!", e) from e

    def get_num_attention_heads(self):
        num_attention_heads = None
        key_attention_heads = ["num_attention_heads", "n_head", "num_heads"]
        for key in key_attention_heads:
            if hasattr(self.model.config, key):
                num_attention_heads = getattr(self.model.config, key)
        if not num_attention_heads:
            raise ValueError(
                f"the config of model must have num_attention_heads, n_head or num_heads, \
                                please check or moddify the config file"
            )
        return num_attention_heads

    def check_calib_data(self, calib_data):
        check_type(calib_data, list, param_name='calib_data')
        if len(calib_data) < 1:
            raise ValueError("calib_data must not be empty.")
        for i, calib_data_item in enumerate(calib_data):
            element_not_tensor = False
            check_type(calib_data_item, (list, dict), param_name=f'calib_data[{i}]')
            if len(calib_data_item) < 1:
                raise ValueError("Each data in calib_data must not be empty.")
            if isinstance(calib_data_item, list):
                for item in calib_data_item:
                    # mindspeed模型可以直接输入字符串作为校准集
                    if not isinstance(item, (str, torch.Tensor)):
                        element_not_tensor = True
                        break
            elif isinstance(calib_data_item, dict):
                for _, value in calib_data_item.items():
                    if not isinstance(value, (str, torch.Tensor)):
                        element_not_tensor = True
                        break
        if element_not_tensor:
            self.logger.warning("Not all elements in calib_data are torch.Tensor, "
                                "please make sure that the model can run with model(*(calib_data[0]))")

        return calib_data

    def check_multimodel(self, model):
        return False

    def _process(self):

        act_stats = self.os_stats()
        if self.cfg.anti_method == 'm4':
            num_attention_heads = self.get_num_attention_heads()
            fusion_kwargs = {}
        scale_min = SCALE_MIN_LLM
        
        if self.is_context_embedder_model:
            fusion_kwargs = _PREDEFINED_FUSION_KWARGS
            scale_min = SCALE_MIN_SD3
        
        for norm_name_group in tqdm(self.norm_linear_subgraph.keys()):
            linear_names = self.norm_linear_subgraph[norm_name_group]
            if isinstance(norm_name_group, str):
                norm_module = PatternProcess.get_module_by_name(self.model, norm_name_group)
            elif len(norm_name_group) > 0:
                norm_module = PatternProcess.get_module_by_name(self.model, norm_name_group[0])
            else:
                norm_module = None

            linear_modules = []
            if len(linear_names) < 1:
                raise ValueError("DAG run failed, the generated linear_names are empty.")
            linear_name = linear_names[0]

            if linear_name not in act_stats.keys():
                raise RuntimeError(f"key {linear_name} not in act_stats")

            stats = act_stats[linear_name]

            is_expert = any("expert" in name.lower() for name in linear_names)
            if (is_expert):
                continue

            for name in linear_names:
                mod = PatternProcess.get_module_by_name(self.model, name)
                linear_modules.append(mod)

            if Multiplier is not None and norm_module is None:
                if len(linear_modules) < 1:
                    raise ValueError("DAG run failed, the generated linear_modules are empty.")
                norm_module = Multiplier(
                    torch.ones_like(stats[STAT_KEY_SMOOTH_SCALE]).to(linear_modules[0].weight.device)
                )

            prepare_list = [PrepareWeight(norm_module, post_force=True, post_recurse=True)]
            prepare_list += [PrepareWeight(mod, post_force=True) for mod in linear_modules]

            with ResListToRelease(*prepare_list):
                if self.cfg.anti_method == 'm1' or self.cfg.anti_method == 'm5':
                    smooth_ln_fcs(self.cfg, norm_module, linear_modules, stats, alpha=self.cfg.alpha)
                elif self.cfg.anti_method == 'm2':
                    os_ln_fcs(self.cfg, norm_module, linear_modules, stats, os_k=self.cfg.os_k)
                elif self.cfg.anti_method == 'm3':
                    weight_aware(self.cfg, norm_module, linear_modules, stats)
                elif self.cfg.anti_method == 'm4':
                    if 'scale_min' in inspect.signature(iter_smooth).parameters:
                        fusion_kwargs.update({"scale_min": scale_min})
                    if 'check_group_fusions' not in inspect.signature(iter_smooth).parameters:
                        fusion_kwargs.pop("check_group_fusions", None)
                    iter_smooth(
                        self.cfg, norm_module, linear_modules, stats, num_attention_heads, **fusion_kwargs
                        )
                    if attach_op is not None and Multiplier is not None and isinstance(norm_module, Multiplier):
                        attach_op(self.model, norm_module, linear_modules, linear_names)
