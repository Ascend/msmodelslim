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

from __future__ import absolute_import, division, print_function

import copy
import os
import gc
import stat
import functools
from typing import OrderedDict, Mapping, Optional, Tuple, List, Union
import inspect
from collections import OrderedDict as OrderedDict_CHECK

from tqdm import tqdm
import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from accelerate.hooks import add_hook_to_module, remove_hook_from_module

from ascend_utils import ResListToRelease
from ascend_utils.common.security import get_valid_write_path, check_type
from msmodelslim.pytorch.llm_ptq.model import ModelAdapterRegistry
from msmodelslim import logger as msmodelslim_logger

from msmodelslim.pytorch.llm_ptq.accelerate_adapter import (PrepareWeight,
                                                            move_update_weight_hook_if_need,
                                                            replace_device_align_hook_if_needed)
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.utils import (judge_model_with_accelerate,
                                                                  judge_module_with_accelerate)

try:
    import torch_npu
except ImportError:
    msmodelslim_logger.warning("Unable to import torch_npu.")

from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import (
    GraphOpt,
    PatternProcess,
    NormBias,
    input_to_cpu,
)
from msmodelslim.pytorch.llm_ptq.anti_outlier.config import AntiOutlierConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils import (
    iter_smooth,
    smooth_ln_fcs,
    os_ln_fcs,
    weight_aware,
)
from .anti_block import (
    QuantV2QwenBlock,
    QuantVisualAttentionBlock,
    LlavaQuantDecoder,
    LlavaClipVision,
    QuantQwen2VLDecoderLayer,
    QuantQwen2VLVisionBlock,
    QuantQwen25VLVisionBlock,
    QuantInternLM2DecoderLayer,
    QuantInternVisionEncoderLayer
)

_FLEX_SMOOTH_IMPORTED = False
try:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.flex_smooth import flex_smooth
except ImportError:
    pass
else:
    _FLEX_SMOOTH_IMPORTED = True

try:
    from msmodelslim.pytorch.llm_ptq.anti_outlier.anti_utils import attach_op, Multiplier
except ImportError:
    attach_op, Multiplier = None, None
    msmodelslim_logger.warning(
        "The current CANN version does not support importing the attach_op and Multiplier packages."
    )

STAT_KEY_MAX = "max"
STAT_KEY_MIN = "min"
STAT_KEY_SHIFT = "shift"
STAT_KEY_THRESHOLD_CHANNEL = "thres_c"
STAT_KEY_THRESHOLD_TENSOR = "thres_t"
STAT_KEY_SMOOTH_SCALE_MASK = "smooth_scale_mask"
STAT_KEY_SMOOTH_SCALE = "smooth_scale"
STAT_KEY_VARIANCE = "std"
TENSOR = 'tensor'
SCALE_MIN_LLM = 1e-5
SCALE_MIN_SD3 = 1e-3

SD3_CONTEXT_EMBEDDER = "context_embedder"
SD3_TRANSFORMER = 'SD3Transformer2DModel'
FLUX_TRANSFORMER = 'FluxTransformer2DModel'
HUNYUANVIDEO_TRANSFORMER = 'HYVideoDiffusionTransformer'

_PREDEFINED_FUSIONS = {
    tuple(): ("context_embedder",)
}

_PREDEFINED_FUSION_KWARGS = {
    "check_group_fusions": False
}

STATE_DICT_COPY_DIR = "msmodelslim_copy"


def replace_rms_norm(model: nn.Module, norm_class_name: str):
    for name, module in model.named_modules():
        if module.__class__.__name__.lower() == 'layernorm':
            pass
        elif norm_class_name != 'layernorm' and module.__class__.__name__.lower() == norm_class_name.lower():
            if judge_module_with_accelerate(module):
                with PrepareWeight(module):
                    new_module = NormBias(module)
                    new_module.to(module.weight.data.device)
                    move_update_weight_hook_if_need(module, new_module, as_submodule=True)
                    GraphOpt.set_module(model, name, new_module)
            else:
                new_module = NormBias(module)
                new_module.to(module.weight.data.device)
                GraphOpt.set_module(model, name, new_module)


def is_model_multimodal(model):
    if not hasattr(model, 'config'):
        return False
    if not hasattr(model.config, 'architectures'):
        return False
    if (model.config.architectures[0] == 'LlavaForConditionalGeneration' or
            (model.config.architectures[0] == 'QWenLMHeadModel' and
             hasattr(model.config, 'visual'))):
        return True
    elif (model.config.architectures[0] == 'Qwen2VLForConditionalGeneration'):
        return True
    elif (model.config.architectures[0] == 'Qwen2_5_VLForConditionalGeneration'):
        return True
    elif (model.config.architectures[0] == 'InternVLChatModel'):
        return True
    return False


class AntiOutlier(object):
    """Anti-outlier for LLM activation quantization."""

    @torch.no_grad()
    def __init__(
            self,
            model: nn.Module,
            calib_data=None,
            cfg: AntiOutlierConfig = None,
            norm_class_name=None,
    ):
        self.logger = msmodelslim_logger

        if norm_class_name is not None and not isinstance(norm_class_name, str):
            raise TypeError("norm_class_name must be str, please check it.")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module, please check it.")
        self.calib_data = self.check_calib_data(calib_data)

        if is_model_multimodal(model):
            if cfg.anti_method != 'm2':
                self.logger.warning("multi modal understanding models use other anti_methods(not 'm2')" + \
                                    "may not be supported or the performance may be bad.")
            else:
                self.cfg = cfg
                self.model = model
                self.calib_data = self.check_calib_data(calib_data)
                self.device = self.cfg.device
                return

        check_type(cfg, AntiOutlierConfig, param_name="config")

        # 校验用户指定的不做异常值抑制的层是否存在
        quant_name_list = []
        conv_name_list = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                quant_name_list.append(name)
            if isinstance(mod, nn.Conv2d):
                conv_name_list.append(name)
        for name in cfg.disable_anti_names:
            if name not in quant_name_list and name not in conv_name_list:
                raise ValueError(
                    f"cfg param `disable_anti_names` has invalid name {name}, "
                    "please check your anti_outlier config."
                )

        self.with_accelerate = judge_model_with_accelerate(model)

        self.cfg = cfg
        self.device = self.cfg.device

        if self.cfg.anti_method == "m6" and not _FLEX_SMOOTH_IMPORTED:
            raise ImportError("CANN Toolkit version not match msmodelslim version, `m6` flex smooth not usable.")

        self.is_context_embedder_model = False
        if SD3_CONTEXT_EMBEDDER + ".weight" in model.state_dict().keys() and \
                model.__class__.__name__ == SD3_TRANSFORMER:
            self.is_context_embedder_model = True

        # 多模态生成模型
        self.is_multimodal_generative_model = False
        if model.__class__.__name__ in (FLUX_TRANSFORMER, HUNYUANVIDEO_TRANSFORMER):
            self.is_multimodal_generative_model = True

        if self.cfg.device == "cpu":
            same_device = self.cfg.device == model.device.type
        else:
            same_device = self.cfg.device == model.device

        if not self.with_accelerate and not same_device:
            self.logger.warning("Model is not on the device indicated in `AntiOutlierConfig`, "
                                "Model is on the device `{}` while `AntiOutlierConfig` "
                                "indicates `{}`".format(model.device, self.cfg.device))
            self.logger.info("Transferring model from `{}` to `{}`...".format(model.device, self.cfg.device))
            model = model.to(self.cfg.device)
            self.logger.info("Transfer done. Suggest to check model and calib_data (if provided) on the "
                             "device that `AntiOutlierConfig` indicates.")

        replace_device_align_hook_if_needed(model)

        self.norm_class_name = norm_class_name

        self.model_adapter = ModelAdapterRegistry.get_adapter(model)

        if not self.with_accelerate:
            self.device_org = next(model.parameters()).device
        else:
            self.device_org = None

        self.model = model

        self.norm_linear_subgraph = {}

        try:
            self.init_dag()
        except Exception as e:
            raise Exception("Please check your config, model and input!", e) from e

        # 非m4,m5,m6场景下，保存anti_outlier处理前的原始权重，作为属性存入model中
        if self.cfg.anti_method in ['m4', 'm5', 'm6']:
            setattr(self.model, 'anti_method', self.cfg.anti_method)

    @staticmethod
    def is_flex_enabled(flex_config):
        return flex_config['alpha'] is None or flex_config['beta'] is None

    def init_dag(self):
        if self.is_context_embedder_model:
            self.norm_linear_subgraph = _PREDEFINED_FUSIONS
        else:
            dummy_input = input_to_cpu(self.calib_data[0][0])
            dummy_input = dummy_input[:1]

            if self.norm_class_name is not None:
                norm_class = list(OrderedDict.fromkeys([m.__class__ for m in self.model.modules() if
                                                        self.norm_class_name.lower() == m.__class__.__name__.lower()]))
            else:
                norm_class = list(
                    OrderedDict.fromkeys(
                        [m.__class__ for m in self.model.modules() if "norm" in m.__class__.__name__.lower()]))
                norm_class = [norm_class[0]]
                self.norm_class_name = norm_class[0].__name__.lower()
            self.norm_linear_subgraph = self.model_adapter.get_norm_linear_subgraph(self.cfg, dummy_input, norm_class)

        if self.cfg.anti_method != "m6":
            replace_rms_norm(self.model, self.norm_class_name)
        gc.collect()
        return

    def trans_to_dict(self, data):
        data_dict = {}
        data_dict['input_ids'] = data[0]
        if len(data) == 2:
            data_dict['attention_mask'] = data[1]
        return data_dict

    def stat_tensor(self, act_stats, name, tensor):

        # buffer the latest tensor
        if name not in act_stats:
            act_stats[name] = {}
            if self.cfg.anti_method != 'm6' or not self.is_flex_enabled(self.cfg.flex_config):
                # store collected tensor in cpu while stat_tensor,
                # otherwise it will occupy too much device memory then cause OOM
                act_stats[name][TENSOR] = tensor.cpu()

        hidden_dim = tensor.shape[-1]
        tensor = tensor.reshape(-1, hidden_dim).detach()  # [N,C]
        coming_max = torch.max(tensor, dim=0)[0]  # [C]
        coming_min = torch.min(tensor, dim=0)[0]  # [C]

        stat_dict = act_stats[name]
    
        if self.cfg.anti_method == 'm6' and self.is_flex_enabled(self.cfg.flex_config):
            if TENSOR not in act_stats[name]:
                act_stats[name][TENSOR] = [tensor.to("cpu").reshape(-1, tensor.shape[-1])]
            else:
                act_stats[name][TENSOR].append(tensor.to("cpu").reshape(-1, tensor.shape[-1]))

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
            stat_dict[STAT_KEY_SHIFT] = (stat_dict[STAT_KEY_MAX] + stat_dict[STAT_KEY_MIN]) / 2  # [C]
        else:
            stat_dict[STAT_KEY_SHIFT] = (coming_max + coming_min) / 2

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
        tensor_shift = self.cfg.ch_align and self.cfg.anti_method not in ['m4', 'm6']
        if tensor_shift:
            channel_max = torch.max((tensor - stat_dict[STAT_KEY_SHIFT]).abs().detach(), dim=0)[0]
        else:
            channel_max = torch.max(tensor.abs().detach(), dim=0)[0]

        if STAT_KEY_SMOOTH_SCALE in stat_dict:
            stat_dict[STAT_KEY_SMOOTH_SCALE] = torch.max(stat_dict[STAT_KEY_SMOOTH_SCALE], channel_max)
        else:
            stat_dict[STAT_KEY_SMOOTH_SCALE] = channel_max

    def os_stats(self, feature_sources):
        """Collect the activations into Dict for outlier suppression."""
        self.model.eval()
        act_stats = {}

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            self.stat_tensor(act_stats, name, x)

        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear) and name in feature_sources:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name))
                )

        for i in tqdm(range(len(self.calib_data))):
            inputs = self.calib_data[i]

            if self.is_context_embedder_model or self.is_multimodal_generative_model:
                # 多模态生成模型特殊处理
                if isinstance(inputs, list):
                    self.model(*inputs)
                elif isinstance(inputs, dict):
                    self.model(**inputs)
                else:
                    raise TypeError(f"Unsupported input type for multimodal generative model: {type(inputs)}")
            else:
                input_dict = self.trans_to_dict(inputs)
                self.model(**input_dict)

        for name in act_stats:
            stat_dict = act_stats[name]
            if isinstance(stat_dict[TENSOR], (list, tuple)):
                stat_dict[TENSOR] = torch.cat(stat_dict[TENSOR])
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

    def save_model(self, output_path):

        output_path = get_valid_write_path(output_path)

        write_mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
        write_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

        if os.path.islink(output_path):
            os.unlink(output_path)

        if os.path.exists(output_path):
            os.remove(output_path)

        with os.fdopen(os.open(output_path, write_flags, write_mode), "w") as f:
            torch.save(self.model, f)

    def get_num_attention_heads(self):
        check_type(self.model.config, (PretrainedConfig, OrderedDict_CHECK), param_name="model.config")

        num_attention_heads = None
        key_attention_heads = ["num_attention_heads", "n_head", "num_heads", "heads_num"]
        for key in key_attention_heads:
            if hasattr(self.model.config, key):
                num_attention_heads = getattr(self.model.config, key)
        if not num_attention_heads:
            raise ValueError(
                f"the config of model must have num_attention_heads, n_head or num_heads, \
                                please check or modify the config file"
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
                    if not isinstance(item, torch.Tensor):
                        element_not_tensor = True
                        break
            elif isinstance(calib_data_item, dict):
                for _, value in calib_data_item.items():
                    if not isinstance(value, torch.Tensor):
                        element_not_tensor = True
                        break
        if element_not_tensor:
            self.logger.warning("Not all elements in calib_data are torch.Tensor, "
                                "please make sure that the model can run with model(*(calib_data[0]))")

        return calib_data

    def anti_for_multimodal(self, model):
        def _set_module(ori_mod, submodule_key, module):
            tokens = submodule_key.split('.')
            sub_tokens = tokens[:-1]
            cur_mod = ori_mod
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            setattr(cur_mod, tokens[-1], module)
            
        block_dict = {
            "QWenBlock": QuantV2QwenBlock,
            "VisualAttentionBlock": QuantVisualAttentionBlock,
            "LlamaDecoderLayer": LlavaQuantDecoder,
            "CLIPEncoderLayer": LlavaClipVision,
            "Qwen2VLDecoderLayer": QuantQwen2VLDecoderLayer,
            "Qwen2VLVisionBlock": QuantQwen2VLVisionBlock,
            "Qwen2_5_VLDecoderLayer": QuantQwen2VLDecoderLayer,
            "Qwen2_5_VLVisionBlock": QuantQwen25VLVisionBlock,
            "InternLM2DecoderLayer": QuantInternLM2DecoderLayer,
            "InternVisionEncoderLayer": QuantInternVisionEncoderLayer
        }
        for name, mod in model.named_modules():
            mod_name = mod.__class__.__name__
            if (mod_name in block_dict):
                quant_mod = block_dict[mod_name](mod, self.model.config, name)
                self.logger.info(
                    "multimodal model replace block `{}` to `{}`".format(mod_name, block_dict[mod_name].__name__)
                )
                if hasattr(mod, '_hf_hook'):
                    add_hook_to_module(quant_mod, mod._hf_hook)
                    remove_hook_from_module(mod)
                _set_module(model, name, quant_mod)
                del mod

    def attach_norm_to_linear_modules(self, norm_module, linear_modules, linear_names):
        if attach_op is not None and norm_module is not None and isinstance(norm_module, Multiplier):
            for i, _ in enumerate(linear_modules):
                attach_op(self.model, copy.deepcopy(norm_module), [linear_modules[i]], [linear_names[i]])

    def check_all_names_not_disable_anti(self, names):
        """
        Check if all names in the list are not in the disable_anti set.

        Args:
            names (list): List of names to check.

        Returns:
            bool: True if all names are not in the disable_anti set, False otherwise.
        """
        disable_anti_set = set(self.cfg.disable_anti_names)
        return all(name not in disable_anti_set for name in names)

    def _process(self):
        if is_model_multimodal(self.model):
            if self.cfg.anti_method != 'm2':
                self.logger.warning("multi modal understanding models use other anti_methods(not 'm2')" + \
                                    "may not be supported or the performance may be bad.")
            else:
                self.anti_for_multimodal(self.model)
                data = self.calib_data[0]
                if isinstance(data, tuple) or isinstance(data, list):
                    self.model(*data)
                elif isinstance(data, dict):
                    self.model(**data)
                return

        smooth_kwargs = {}

        feature_sources = {v[0] for v in self.norm_linear_subgraph.values() if v}
        act_stats = self.os_stats(feature_sources)
        if self.cfg.anti_method in ['m4', 'm6']:
            num_attention_heads = self.get_num_attention_heads()
            smooth_kwargs = {
                "num_attention_heads": num_attention_heads
            }
        scale_min = SCALE_MIN_LLM
        
        if self.is_context_embedder_model:
            smooth_kwargs.update(_PREDEFINED_FUSION_KWARGS)
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

            self.logger.debug(f"smooth {norm_name_group} -> {linear_names}")

            for name in linear_names:
                mod = PatternProcess.get_module_by_name(self.model, name)
                linear_modules.append(mod)

            args = []
            args, smooth_kwargs = self.model_adapter.modify_smooth_args(
                self.cfg,
                norm_name_group,
                linear_names,
                args,
                smooth_kwargs
            )

            self.logger.debug(f"smooth_kwargs is {smooth_kwargs}")

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
                    if self.check_all_names_not_disable_anti(linear_names):
                        weight_aware(self.cfg, norm_module, linear_modules, stats)
                        self.attach_norm_to_linear_modules(norm_module, linear_modules, linear_names)

                elif self.cfg.anti_method == 'm4':
                    if 'scale_min' in inspect.signature(iter_smooth).parameters:
                        smooth_kwargs.update({"scale_min": scale_min})
                    if 'check_group_fusions' not in inspect.signature(iter_smooth).parameters:
                        smooth_kwargs.pop("check_group_fusions", None)
                    if self.check_all_names_not_disable_anti(linear_names):
                        iter_smooth(
                            self.cfg,
                            norm_module,
                            linear_modules,
                            stats,
                            **smooth_kwargs
                            )
                        self.attach_norm_to_linear_modules(norm_module, linear_modules, linear_names)

                elif self.cfg.anti_method == 'm6':
                    smooth_kwargs.update(self.cfg.flex_config)
                    if self.check_all_names_not_disable_anti(linear_names):
                        flex_smooth(
                            self.cfg,
                            norm_module,
                            linear_modules,
                            stats,
                            **smooth_kwargs
                        )
                        self.attach_norm_to_linear_modules(norm_module, linear_modules, linear_names)
