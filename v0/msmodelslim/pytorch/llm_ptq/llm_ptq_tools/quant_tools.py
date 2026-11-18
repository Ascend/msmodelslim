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
import itertools
import re
import os
import gc
import inspect
import functools
from collections import defaultdict
from typing import Mapping, List, Optional

from tqdm import tqdm
import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module, remove_hook_from_module

from ascend_utils.common.security.type import check_mapping_element
from ascend_utils.common.security import check_element_type, check_type, get_write_directory, check_int

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.model.deepseek_v2.deepseek_v2 import is_deepseek_v2
from msmodelslim.pytorch.llm_ptq.model.factory import cutting_method_registry
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_config_manager import LayerConfigManager

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config import QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier.graph_utils import class_detect

# KIA part
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import (
    Quantizer, LinearQuantizer, LinearNf4Quantizer, layer_wise_calib
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    fake_quantize_save, get_features, linear_quantization_params, fully_analyze_activation
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.kv_cache_utils import (
    set_kvcache_vari_func, new_forward
)
from msmodelslim.pytorch.llm_sparsequant.sparsequant_modules import LinearSparseQuantizer
from msmodelslim.pytorch.lowbit.atomic_power_outlier import \
    quant_one_weight_by_outliers as quant_one_weight_by_outliers_low_bit
from msmodelslim.pytorch.lowbit.calibration import (
    replace_RMSNorm, QuantXDecoderLayer, LlamaRMSNormBias
)
from msmodelslim.pytorch.lowbit.quant_modules import LinearQuantizer as LowBitLinearQuantizer
from msmodelslim.pytorch.lowbit.quant_modules import Quantizer as LowBitQuantizer

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save import SaverFactory, ComplexQuantifier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType, WeightQuantMethod
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import (
    SAVE_TYPE_LIST,
    SAVE_TYPE_NUMPY,
    SAVE_TYPE_SAFE_TENSOR,
    SAVE_TYPE_ASCENDV1
)

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant import (
    configure_fa,
    enable_fa_calibration,
    disable_fa_calibration,
    enable_fa_quantizer_record,
    collect_fa_quantizer_record,
    is_attn_module_and_then_check_quantizer,
    install_fa_quantizer
)

from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.torch_dag_adapter import TorchDAGAdapter
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.simulate_tp import ParallelLinearCol
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import (PrepareWeight,
                                                                         replace_device_align_hook_if_needed,
                                                                         move_update_weight_hook_if_need,
                                                                         clear_unused_module)
from msmodelslim.pytorch.llm_ptq.accelerate_adapter.utils import judge_model_with_accelerate

import msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.timestep_utils as timestep_utils
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.quantizer import LinearQuantizerTimestep

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant import flat_quant_train
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant import FlatQuantConfig

HF_HOOK = "_hf_hook"
STATE_DICT_COPY_DIR = "copy"
ALLOWED_MIX_TYPES = {"w8a8", "w8a16", "w8a8_dynamic", "float", "w4a8_dynamic"}


class Calibrator(object):
    """ Calibrator for post-training quantization."""

    @torch.no_grad()
    def __init__(self, model,
                 cfg: QuantConfig,
                 calib_data=None,
                 disable_level='L0',
                 all_tensors=None,
                 mix_cfg: Optional[dict] = None):
        check_type(model, nn.Module, param_name="model")
        check_type(cfg, QuantConfig, param_name="cfg")
        check_type(disable_level, str, param_name='disable_level')
        if all_tensors:
            check_element_type(all_tensors, torch.Tensor, dict, param_name="all_tensors")

        if mix_cfg:  # 当 mix_cfg 不为 None 且非空字典时才进行检查
            if not isinstance(mix_cfg, dict):
                raise ValueError("mix_cfg must be a dict if provided.")
            for key, value in mix_cfg.items():
                # key 不做校验, value必须属于指定范围
                if value not in ALLOWED_MIX_TYPES:
                    raise ValueError(
                        f"mix_cfg中'{key}'指定的量化类型'{value}'不在允许列表{ALLOWED_MIX_TYPES}之内，"
                        f"请检查配置或扩展ALLOWED_MIX_TYPES后再使用。")
        if 'progressive' not in inspect.signature(quant_one_weight_by_outliers_low_bit).parameters \
                and cfg.model_quant_type == QuantType.W4A8_DYNAMIC:
            raise ValueError("Current cann version is not support W4A8_DYNAMIC, \
                             if you want to use this feature, please upgrade it.")

        self.cfg = cfg
        self.logger = msmodelslim_logger
        self.calib_data = self.get_calib_data([]) if calib_data is None else self.get_calib_data(calib_data)
        self.use_kvcache_quant = cfg.use_kvcache_quant
        self.norm_class_name = cfg.norm_class_name

        self.model_torch_dtype = self._init_model_torch_dtype(model)

        # kv_cache量化参数应存于量化对象内，目前暂时特殊处理
        self.kv_cache_quant_params = defaultdict(list)

        self.model_with_accelerate = judge_model_with_accelerate(model)

        replace_device_align_hook_if_needed(model)

        # 初始化dag类
        self.dag = self.extract_dag(model)
        # 初始化kvcache类
        self.attention_class = None
        if self.use_kvcache_quant:
            self.get_kvcache(model)
            self.attention_class = class_detect(model, 'attention')
            self.transformer_class = class_detect(model, 'GLMTransformer')

        if self.cfg.do_smooth:
            replace_RMSNorm(model)

        if not hasattr(self.cfg, 'use_fa_quant'):
            self.cfg.use_fa_quant = False
            self.cfg.fa_tp_size = 1
            self.cfg.fa_amp = 0

        if not re.match(r'^L((?!0)\d+|0)$', disable_level):
            raise ValueError('Please check the `disable_level` configuration.')
        self.disable_level = disable_level

        model = self.init_model_device(model)

        if self.cfg.use_fa_quant:
            install_fa_quantizer(model, model.config, self.logger)
            configure_fa(model, tp_size=self.cfg.fa_tp_size)

        self.last_layer_name = None
        self.rollback_names = None
        self.quant_linear_names = None
        self.act_states = None
        self.rollback_names_process(model)
        self.is_deepseek_v2 = is_deepseek_v2(model)

        # for backward compatibility, when quantifying deepseek v2 model with w8a8,
        # the quant config of all mlp layers is w8a8_dynamic, and the quant config of other layers is w8a8
        if self.is_deepseek_v2 and mix_cfg is None and cfg.model_quant_type == QuantType.W8A8:
            mix_cfg = {"*.mlp.*": "w8a8_dynamic", "*": "w8a8"}

        # this initializes is kept for forward compatibility
        # in the future, this will receive a cfg_store object from outside, but now it is None
        self.cfg_store = None

        if self.is_deepseek_v2 and self.cfg_store is None and cfg.model_quant_type == QuantType.W4A8_DYNAMIC:
            self.cfg_store = {
                'rollback': QuantConfig(w_bit=16, a_bit=16),
                'float': QuantConfig(w_bit=16, a_bit=16),
                'w4a8_dynamic': cfg
            }

            if 'w8a8' not in self.cfg_store:
                try:
                    new_cfg = LayerConfigManager.w4a8_dynamic_convert_to_w8a8(cfg)
                    self.cfg_store['w8a8'] = new_cfg
                    self.cfg_store['default'] = copy.deepcopy(new_cfg)
                except Exception as e:
                    self.logger.warning(f"Failed to build w8a8 config. reason: {e}")
            if 'w8a8_dynamic' not in self.cfg_store:
                try:
                    new_cfg = LayerConfigManager.w4a8_dynamic_convert_to_w8a8_dynamic(cfg)
                    self.cfg_store['w8a8_dynamic'] = new_cfg
                except Exception as e:
                    self.logger.warning(f"Failed to build w8a8_dynamic config. reason: {e}")

        if self.cfg_store is None:
            self.cfg_store = {
                "default": copy.deepcopy(cfg),
                'rollback': QuantConfig(w_bit=16, a_bit=16),
                'float': QuantConfig(w_bit=16, a_bit=16)
            }

        # for int8 quantization, we will convert the cfg to w8a8, w8a16, w8a8_dynamic if not in cfg_store
        if cfg.w_bit == 8 or (cfg.w_bit == 4 and cfg.a_bit == 4):
            if 'w8a8' not in self.cfg_store:
                try:
                    new_cfg = LayerConfigManager.convert_to_w8a8(cfg)
                    self.cfg_store['w8a8'] = new_cfg
                except Exception as e:
                    self.logger.warning(f"Failed to build w8a8 config. reason: {e}")
            if 'w8a16' not in self.cfg_store:
                try:
                    new_cfg = LayerConfigManager.convert_to_w816(cfg)
                    self.cfg_store['w8a16'] = new_cfg
                except Exception as e:
                    self.logger.warning(f"Failed to build w8a16 config. reason: {e}")
            if 'w8a8_dynamic' not in self.cfg_store:
                try:
                    new_cfg = LayerConfigManager.convert_to_w8a8_dynamic(cfg)
                    self.cfg_store['w8a8_dynamic'] = new_cfg
                except Exception as e:
                    self.logger.warning(f"Failed to build w8a8_dynamic config. reason: {e}")

        self.layer_cfg_manager = LayerConfigManager(mix_cfg=mix_cfg, cfg_store=self.cfg_store,
                                                    rollback_names=self.rollback_names)
        self.layer_cfg_manager.build_config_map(model)

        if self.cfg.calib_mode == 1:
            if (self.cfg.a_bit <= 8) or self.cfg.w_hessian:
                self.all_tensors = all_tensors
            else:
                self.all_tensors = None

        try:
            if self.cfg.model_quant_type == QuantType.W4A4_FLATQUANT_DYNAMIC:
                self.model = model
            else:
                self.model = self.quantize(model)
                if self.calib_data:
                    self.enable_quant()
        except Exception as e:
            raise Exception("Please check the model and configuration.", e) from e
        self.logger.info("Quantizer initialized successful!")

    def init_model_device(self, model):
        if self.cfg.device == "cpu":
            same_device = self.cfg.device == model.device.type
        else:
            same_device = self.cfg.device == model.device
        if not judge_model_with_accelerate(model) and not same_device:
            self.logger.warning("Model is not on the device indicated in `QuantConfig`, "
                                "Model is on the device `{}` while `QuantConfig` "
                                "indicates `{}`".format(model.device, self.cfg.device))
            self.logger.info("Transferring model from `{}` to `{}`...".format(model.device, self.cfg.device))
            model = model.to(self.cfg.device)
            self.logger.info("Transfer done. Suggest to check model and calib_data (if provided) on the "
                             "device that `QuantConfig` indicates.")
        return model

    def init_dag(self):
        if self.cfg.do_smooth:
            return True
        if hasattr(self.cfg, 'tp_size'):
            return True
        if self.cfg.use_kvcache_quant:
            return True
        return False

    def extract_dag(self, model):
        if not self.init_dag():
            return None
        if not self.calib_data:
            dummy_input = torch.randint(0, 100, (1, 128)).type(torch.int64)
        else:
            dummy_input = self.calib_data[0]
        norm_class = self.get_norm_class(model, norm_class_name=self.norm_class_name)
        dag = TorchDAGAdapter(model, dummy_input, hook_nodes=norm_class)
        return dag

    def get_kvcache(self, model):
        self.kv_sym = True if not hasattr(self.cfg, 'kv_sym') else self.cfg.kv_sym
        kv_linears, num_kv = self.dag.get_kv_linears()
        self.kv_cache = self.get_kvcache_features(model, kv_linears, num_kv)
        self.get_kvcache_quant_param(num_kv)

    def get_kvcache_features(self, model, kv_linears, num_kv):
        model.eval()
        kv_cache = {}

        def update_key_value_extremums(kv_cache, name, max_min, in_hidden_size, chunk_size):
            comming_max, comming_min = max_min[0], max_min[1]
            # 检查模型类型并调用相应的切割方法
            model_type = model.config.model_type

            if model_type in cutting_method_registry.cutting_methods:
                # 判断模型是否属于特殊qkv排列类型，并执行特殊的切割方法
                cut_method = cutting_method_registry.get_cutting_method(model_type)
                key_max, value_max, key_min, value_min = cut_method(comming_max, comming_min, model)
            elif chunk_size == 2:
                key_max, value_max = torch.chunk(comming_max, chunk_size, dim=0)
                key_min, value_min = torch.chunk(comming_min, chunk_size, dim=0)
            elif chunk_size == 3:
                res_dim = comming_max.shape[-1] - in_hidden_size
                _, kv_max = torch.split(comming_max, [in_hidden_size, res_dim])
                _, kv_min = torch.split(comming_min, [in_hidden_size, res_dim])
                key_max, value_max = torch.chunk(kv_max, 2, dim=0)
                key_min, value_min = torch.chunk(kv_min, 2, dim=0)
            else:
                key_max, value_max, key_min, value_min = \
                    cutting_method_registry.default_cut(comming_max, comming_min, in_hidden_size)

            # 获取k_name和v_name
            k_name = 'k_proj'
            v_name = 'v_proj'

            if name not in kv_cache:
                kv_cache[name] = {}

            # 更新key极值
            update_extremum(kv_cache[name], k_name, 'max', torch.max, key_max)
            update_extremum(kv_cache[name], k_name, 'min', torch.min, key_min)
            # 更新value极值
            update_extremum(kv_cache[name], v_name, 'max', torch.max, value_max)
            update_extremum(kv_cache[name], v_name, 'min', torch.min, value_min)

        def update_extremum(kv_cache, name, key, torch_function, value):
            if name not in kv_cache:
                kv_cache[name] = {}

            if key in kv_cache[name]:
                kv_cache[name][key] = torch_function(kv_cache[name][key], value)
            else:
                kv_cache[name][key] = value

        def kv_tensor(name, in_hidden_size, tensor):
            if name not in kv_cache:
                kv_cache[name] = {}

            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).detach()
            comming_max = torch.max(tensor, dim=0)[0]
            comming_min = torch.min(tensor, dim=0)[0]
            max_min = [comming_max, comming_min]

            if num_kv == 2:
                # 更新最大值
                update_extremum(kv_cache, name, 'max', torch.max, comming_max)
                # 更新最小值
                update_extremum(kv_cache, name, 'min', torch.min, comming_min)
            elif num_kv == 1:
                update_key_value_extremums(kv_cache, name, max_min, in_hidden_size, 2)
            elif num_kv == 0:
                update_key_value_extremums(kv_cache, name, max_min, in_hidden_size, 3)

        def kv_cache_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            if isinstance(y, tuple):
                y = y[0]
            kv_tensor(name, x.shape[-1], y)

        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                if name in kv_linears:
                    hooks.append(
                        m.register_forward_hook(
                            functools.partial(kv_cache_hook, name=name))
                    )

        for data in tqdm(self.calib_data):
            if isinstance(data, tuple) or isinstance(data, list):
                with torch.no_grad():
                    model(*data)
            elif isinstance(data, dict):
                with torch.no_grad():
                    model(**data)
        for h in hooks:
            h.remove()

        return kv_cache

    def get_kvcache_quant_param(self, num_kv):
        if num_kv == 2:
            for key in self.kv_cache.keys():
                self.get_quant_module_param(key, key, self.kv_cache[key])
        else:
            for key in self.kv_cache.keys():
                for kv_key in self.kv_cache[key].keys():
                    new_key = key + '.' + kv_key
                    self.get_quant_module_param(key, new_key, self.kv_cache[key][kv_key])

    def get_quant_module_param(self, key, new_key, kv_cache):
        key_weight = key + '.weight'
        new_key_scale, new_key_offset = new_key + '.kv_cache_scale', new_key + '.kv_cache_offset'

        scale, zero_point = linear_quantization_params(8, kv_cache['min'], kv_cache['max'],
                                                       integral_zero_point=True, q_signed=True, sym=self.kv_sym)

        self.kv_cache_quant_params[key].append((new_key_scale, scale.to('cpu')))
        self.kv_cache_quant_params[key].append((new_key_offset, zero_point.to('cpu')))

    def rollback_names_process(self, model):
        # 自动回退lm_head层
        quant_name_list = []
        conv_name_list = []
        attn_name_list = []

        quant_name_set = set()
        conv_name_set = set()
        attn_name_set = set()

        for name, module in list(model.named_modules()):
            if isinstance(module, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
                quant_name_list.append(name)
                quant_name_set.add(name)
            elif isinstance(module, nn.Conv2d):
                conv_name_list.append(name)
                conv_name_set.add(name)
            elif is_attn_module_and_then_check_quantizer(module, name):
                attn_name_list.append(name)
                attn_name_set.add(name)
        if quant_name_list:
            last_layer_name = quant_name_list[-1]
        else:
            raise ValueError("No nn.Linear found in the model, please check the model.")
        if conv_name_list:
            self.logger.info("conv2d is in the model and will not be quantified")
        # 校验用户指定的回退层是否存在
        for name in self.cfg.disable_names:
            if name not in quant_name_set and name not in conv_name_set and name not in attn_name_set:
                raise ValueError(f"`disable_names` has invalid key `{name}`, please check your model configurations.")
        if self.cfg.disable_last_linear:
            self.logger.info(f"Automatically disabling the last linear layer: {last_layer_name} "
                             f"based on the `disable_last_linear` parameter setting.")
            self.rollback_names = list(set(self.cfg.disable_names + [last_layer_name] + conv_name_list))
        else:
            self.rollback_names = list(set(self.cfg.disable_names + conv_name_list))

        # 如果使用low bit稀疏量化，且disable_level不为L0，则使用amp自动回退
        if self.cfg.is_lowbit and self.disable_level != 'L0':
            self.cfg.use_amp = True
            self.logger.info("Automatically disabling layer set while calibrating "
                             "based on the `use_amp` parameter setting.")
            self.cfg.amp_num = int(self.disable_level[1:])
        # 根据disable level获取自动回退层，Data-Free场景下自动回退设为L0
        elif self.calib_data:
            enable_tensor_dump = False  # 模型规模较大的时候，dump_tensor非常消耗计算、内存和存储空间，默认关闭
            try:
                if getattr(self.cfg, 'use_timestep_quant', False):
                    if 'args' not in self.calib_data[0]:
                        raise ValueError('Timestep calib data error: "arg" not in calib_data[0]')
                    calib_data = [x['args'] for x in self.calib_data]
                else:
                    calib_data = self.calib_data
                self.act_states = get_features(model, calib_data[:5], "features.npy", enable_tensor_dump)
            except Exception as e:
                raise Exception("Please check the model and calibration data, "
                                "ensure that your model can run with `model(*(calib_data[0]))`.", e) from e

            label_threshold_dict = self.get_label_threshold_dict()
            self.logger.info(f"The model contains a total of {len(label_threshold_dict)} nn.Linear layers.")
            auto_disable_names = self.get_auto_disable_names(label_threshold_dict)
            self.rollback_names = list(set(self.rollback_names + auto_disable_names))
            self.quant_linear_names = [name for name in quant_name_list if name not in self.rollback_names]

            if self.disable_level != 'L0':
                self.logger.info('Automatically disabled layer names are:\n\t' +
                                 '\n\t'.join([str(name) for name in sorted(auto_disable_names)]))
        # for Data-Free
        else:
            self.logger.info("Running in Data-Free mode, `disable_level` set to `L0`")
        self.logger.info('The subsequent layers will maintain the use of floating-point weights'
                         ' for forward computations.:\n\t'
                         + '\n\t'.join([str(name) for name in sorted(self.rollback_names)]))

    def get_label_threshold_dict(self):
        label_threshold_dict = {}
        for key in self.act_states.keys():
            abs_max_tensor = max(abs(self.act_states[key]['t_max']), abs(self.act_states[key]['t_min']))
            range_param = abs_max_tensor / self.act_states[key]['std']
            label_threshold_dict[key] = range_param.item()
        return label_threshold_dict

    def get_auto_disable_names(self, label_threshold_dict):
        disable_level_number = int(self.disable_level[1:])
        if self.disable_level == 'L0':
            auto_disable_names = []
        else:
            sorted_label_list = list(dict(sorted(label_threshold_dict.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)).keys())
            auto_disable_names = [label for label in sorted_label_list
                                  if label not in self.rollback_names][:disable_level_number]
        return auto_disable_names

    def quantize(self, model):
        if self.model_with_accelerate:
            return self.quantize_model(model)
        else:
            return self.quantize_model(model).to(self.cfg.device)

    def enable_quant(self):
        enable_quantization(self.model, self.act_states, self.logger, self.cfg.use_fa_quant, self.cfg.disable_names)

    def get_calib_data(self, calib_data):
        check_type(calib_data, list, param_name='calib_data')
        if not calib_data:
            QuantType.check_datafree_quant_type(self.cfg.model_quant_type)
            WeightQuantMethod.check_datafree_wmethod(self.cfg.w_method)
        else:
            self.check_calib_data(calib_data)
        return calib_data

    def check_calib_data(self, calib_data):
        element_not_tensor = False
        for i, calib_data_item in enumerate(calib_data):
            check_type(calib_data_item, (list, dict), param_name=f'calib_data[{i}]')
            for _, item in enumerate(calib_data_item):
                if not isinstance(item, torch.Tensor):
                    element_not_tensor = True
        if element_not_tensor:
            self.logger.warning("Not all elements in calib_data are torch.Tensor, "
                                "please make sure that the model can run with model(*(calib_data[0]))")

    @torch.no_grad()
    def run(self, int_infer=False):
        check_type(int_infer, bool, additional_msg="`int_infer` should be boolean type!")

        try:
            self._run(int_infer=int_infer)
        except Exception as ex:
            raise Exception("Please check the model and configuration.", ex) from ex

    @torch.no_grad()
    def save(self, output_path, safetensors_name=None, json_name=None, save_type=None, part_file_size=None):
        check_type(output_path, str, param_name="output_path")
        output_path = get_write_directory(output_path, write_mode=0o750)

        if safetensors_name is not None and not isinstance(safetensors_name, str):
            safetensors_name = None
        if json_name is not None and not isinstance(json_name, str):
            json_name = None

        if not save_type:
            save_type = [SAVE_TYPE_NUMPY]
        check_element_type(save_type, element_type=str, value_type=list, param_name="save_type")
        if [False for save_type_item in save_type if save_type_item not in SAVE_TYPE_LIST]:
            self.logger.warning(
                f"`save_type` should be one of the choices in {SAVE_TYPE_LIST}, but received {save_type}. "
                f"Defaulting to `{SAVE_TYPE_NUMPY}` type."
            )
            save_type = [SAVE_TYPE_NUMPY]
        if self.cfg.model_quant_type == QuantType.W4A8_DYNAMIC and SAVE_TYPE_NUMPY in save_type:
            if len(save_type) == 1:
                raise ValueError("W4A8_DYNAMIC quantization does not support saving to numpy format, "
                                 "please check it.")
            else:
                self.logger.error("W4A8_DYNAMIC quantization does not support saving to numpy format, "
                                  f"removing `{SAVE_TYPE_NUMPY}` from `save_type`.")
                save_type.remove(SAVE_TYPE_NUMPY)
        if part_file_size is not None:
            check_int(part_file_size, min_value=1)

        self._save(output_path, safetensors_name, json_name, save_type, part_file_size)

    def generate_weight_of_model(self, model, weight_of_module_generator):
        with tqdm(desc='Collect quant param', total=sum(1 for _, _ in self.model.named_modules())) as progress:
            for name, module in model.named_modules():
                # kv_cache量化特殊处理
                if name in self.kv_cache_quant_params:
                    for k, v in self.kv_cache_quant_params[name]:
                        yield k, QuantType.KV8, v

                with PrepareWeight(module):
                    yield from weight_of_module_generator(name, module)
                progress.update(1)

    def run_calib_mode(self):
        amp_done = False
        for data in tqdm(self.calib_data):
            if not amp_done and self.cfg.fa_amp:
                enable_fa_quantizer_record(self.model)

            if isinstance(data, tuple) or isinstance(data, list):
                self.model(*data)
            elif isinstance(data, dict):
                self.model(**data)

            if not amp_done and self.cfg.use_amp:
                self.run_amp()
                amp_done = True
            if not amp_done and self.cfg.fa_amp:
                self.run_fa_amp()
                amp_done = True
        self.run_datafree_after_calib()

    def run_datafree_after_calib(self):
        # 将稀疏模型校准集没有运行的部分切换成data-free模式
        for name, module in self.model.named_modules():
            with PrepareWeight(module):
                if isinstance(module, LinearQuantizer) and module.quant_weight.weight_scale is None:
                    if module.quant_weight.w_hessian:
                        module.quant_weight.w_hessian = False
                        self.logger.info(f"run MIN-MAX quantization on linear layer: {name}")
                    module.quant_weight(module.weight)
                elif isinstance(module, LinearNf4Quantizer):
                    self.logger.info(f"Running in Data-Free mode, quantizing the layer into NF4 type: {name}")
                    module.quant_weight()
                elif isinstance(module, LowBitLinearQuantizer) and hasattr(module, 'weight_quant_flag'):
                    if not module.weight_quant_flag:
                        self.logger.info(f"Running in Data-Free mode, quantizing the layer: {name}")
                        if not module.cfg.is_stage_quant:
                            module.fp_weight = module.weight.cpu().clone()
                        kwargs = {
                            'powerquant': module.cfg.nonuniform,
                            'fraction': module.cfg.fraction,
                            'num_bits': module.cfg.w_bit,
                            'isolate_outlier_amax': False,
                            'per_channel': not module.cfg.mm_tensor,
                            'use_cuda': True if module.cfg.dev_type == 'gpu' else False,
                            'use_sigma': module.cfg.use_sigma,
                            'sigma_factor': module.cfg.sigma_factor,
                            'open_outlier': module.cfg.open_outlier,
                            'group_size': module.cfg.group_size,
                            'w_sym': module.cfg.w_sym,
                            'use_hqq': module.cfg.hqq
                        }
                        if 'progressive' in inspect.signature(quant_one_weight_by_outliers_low_bit).parameters:
                            kwargs['progressive'] = module.cfg.is_stage_quant

                        recovered_weight, scale_w, _, offset_w = quant_one_weight_by_outliers_low_bit(
                            module.weight,
                            **kwargs
                        )
                        is_scale_w_list = isinstance(scale_w, list) and len(scale_w) == 2
                        is_offset_w_list = isinstance(offset_w, list) and len(offset_w) == 2
                        if is_scale_w_list and is_offset_w_list:
                            module.quant_weight.weight_scale = scale_w[0].cpu()
                            module.quant_weight.weight_offset = offset_w[0].cpu()
                            module.quant_weight.weight_scale_second = scale_w[1].cpu()
                            module.quant_weight.weight_offset_second = offset_w[1].cpu()
                        else:
                            module.quant_weight.weight_scale = scale_w.cpu()
                            module.quant_weight.weight_offset = offset_w.cpu()
                        module.has_init_quant_para = True
                        if not module.cfg.is_stage_quant:
                            with torch.no_grad():
                                module.weight[:] = recovered_weight[:]

                        module.weight_quant_flag = True

    def run_amp(self):
        max_input_dict = {}

        for name, module in self.model.named_modules():
            if isinstance(module, LowBitLinearQuantizer):
                max_input_dict[name] = module.max_input_num
        layers_to_disable = list(
            dict(sorted(max_input_dict.items(),
                        key=lambda x: x[1], reverse=True)).keys()
        )[:self.cfg.amp_num]
        for name, module in self.model.named_modules():
            if name in layers_to_disable:
                module.disable_input = True
                module.disable_quant_weight()
        self.rollback_names.extend(layers_to_disable)
        self.logger.info('The following linear layers will continue to '
                         'use floating-point weights for forward computation:\n\t'
                         + '\n\t'.join([str(name) for name in sorted(self.rollback_names)]))

    def run_fa_amp(self):
        qkv_states_record = collect_fa_quantizer_record(self.model)
        states_num_per_layer = 3
        model_attention_layer_num = len(qkv_states_record.keys()) // states_num_per_layer
        if model_attention_layer_num < self.cfg.fa_amp:
            self.logger.warning("`fa_amp` exceeds the total attention layer number. Therefore, "
                                "only up to the total attention layers will skip quantization")
        disabled_module_names = fully_analyze_activation(qkv_states_record, self.cfg.fa_amp)
        self.logger.info('The following attention layers will continue to '
                         'use floating-point weights for forward computation:\n\t'
                         + '\n\t'.join([str(name) for name in sorted(disabled_module_names)]))

        for name, module in self.model.named_modules():
            if is_attn_module_and_then_check_quantizer(module, name) and name in disabled_module_names:
                module.fa_quantizer.reset()
                module.fa_quantizer.disable_calibration()

    def run_datafree_mode(self):
        for name, module in self.model.named_modules():
            with PrepareWeight(module):
                if isinstance(module, LinearQuantizer):
                    self.logger.info(f"Running in Data-Free mode, quantizing the layer: {name}")
                    module.quant_weight(module.weight)
                elif isinstance(module, LinearNf4Quantizer):
                    self.logger.info(f"Running in Data-Free mode, quantizing the layer into NF4 type: {name}")
                    module.quant_weight()
                elif isinstance(module, LowBitLinearQuantizer):
                    with torch.no_grad():
                        if not module.cfg.is_stage_quant:
                            module.fp_weight = module.weight.cpu().clone()
                        self.logger.info(f"Running in Data-Free mode, quantizing the layer: {name}")
                        kwargs = {
                            'powerquant': module.cfg.nonuniform,
                            'fraction': module.cfg.fraction,
                            'num_bits': module.cfg.w_bit,
                            'isolate_outlier_amax': False,
                            'per_channel': not module.cfg.mm_tensor,
                            'use_cuda': True if module.cfg.dev_type == 'gpu' else False,
                            'use_sigma': module.cfg.use_sigma,
                            'sigma_factor': module.cfg.sigma_factor,
                            'open_outlier': module.cfg.open_outlier,
                            'group_size': module.cfg.group_size,
                            'w_sym': module.cfg.w_sym,
                            'use_hqq': module.cfg.hqq
                        }
                        if 'progressive' in inspect.signature(quant_one_weight_by_outliers_low_bit).parameters:
                            kwargs['progressive'] = module.cfg.is_stage_quant

                        recovered_weight, scale_w, _, offset_w = quant_one_weight_by_outliers_low_bit(
                            module.weight,
                            **kwargs
                        )
                        is_scale_w_list = isinstance(scale_w, list) and len(scale_w) == 2
                        is_offset_w_list = isinstance(offset_w, list) and len(offset_w) == 2
                        if is_scale_w_list and is_offset_w_list:
                            module.quant_weight.weight_scale = scale_w[0].cpu()
                            module.quant_weight.weight_offset = offset_w[0].cpu()
                            module.quant_weight.weight_scale_second = scale_w[1].cpu()
                            module.quant_weight.weight_offset_second = offset_w[1].cpu()
                        else:
                            module.quant_weight.weight_scale = scale_w.cpu()
                            module.quant_weight.weight_offset = offset_w.cpu()
                        module.has_init_quant_para = True
                        if not module.cfg.is_stage_quant:
                            with torch.no_grad():
                                module.weight[:] = recovered_weight[:]

                    module.weight_quant_flag = True

    def quantize_model(self, model):
        def _set_module(ori_mod, submodule_key, module):
            tokens = submodule_key.split('.')
            sub_tokens = tokens[:-1]
            cur_mod = ori_mod
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            setattr(cur_mod, tokens[-1], module)

        if self.cfg.do_smooth:
            list_infos = self.dag.get_llm_network_pattern_auto()
            attn_list, mhsa_ln_list, ffn_list, ffn_ln_list = list_infos[0], list_infos[2], list_infos[3], list_infos[4]

            if not (len(attn_list) == len(mhsa_ln_list) == len(ffn_list) == len(ffn_ln_list)):
                raise ValueError("Failed to get network pattern by DAG, please check the model!")
            for _, (attn, mhsa_ln, ffn, ffn_ln) in enumerate(zip(attn_list, mhsa_ln_list, ffn_list, ffn_ln_list)):
                modules = [attn, mhsa_ln, ffn, ffn_ln]
                cur_decoder, cur_decoder_type, modules_info, cur_decoder_name = \
                    self.split_module_name_get_info(modules, model, self.cfg.down_proj_type)
                decoder_info = {
                    'attn': modules_info[0],
                    'mlp': modules_info[2],
                    'ln1': modules_info[1],
                    'ln2': modules_info[3],
                    'qkv': modules_info[4],
                    'ffn': modules_info[5]
                }
                quant_mod = QuantXDecoderLayer(cur_decoder, self.cfg, cur_decoder_name, decoder_info)
                if hasattr(cur_decoder, HF_HOOK):
                    add_hook_to_module(quant_mod, cur_decoder._hf_hook)
                    remove_hook_from_module(cur_decoder)
                _set_module(model, cur_decoder_name, quant_mod)

        for name, mod in model.named_modules():
            with PrepareWeight(mod):
                if name in self.rollback_names:
                    continue
                if isinstance(mod, nn.Linear) or isinstance(mod, nn.modules.linear.NonDynamicallyQuantizableLinear):
                    # every linear layer will have a standalone config
                    layer_cfg = self.layer_cfg_manager.get_layer_config(name)

                    if layer_cfg.model_quant_type == QuantType.FLOAT:
                        continue

                    if layer_cfg.is_lowbit:
                        quant_mod = LowBitLinearQuantizer(cfg=layer_cfg, logger=self.logger, name=name)
                    elif layer_cfg.w_method in QuantType.NF4:
                        quant_mod = LinearNf4Quantizer(cfg=layer_cfg, logger=self.logger)
                    elif getattr(self.cfg, 'use_timestep_quant', False):
                        quant_mod = LinearQuantizerTimestep(cfg=layer_cfg, logger=self.logger)
                    elif layer_cfg.model_quant_type is not QuantType.W8A8S:
                        quant_mod = LinearQuantizer(cfg=layer_cfg, logger=self.logger)
                    else:
                        quant_mod = LinearSparseQuantizer(cfg=self.cfg, logger=self.logger)
                    quant_mod.set_param(mod)
                    move_update_weight_hook_if_need(mod, quant_mod)
                    _set_module(model, name, quant_mod)
                    # 可能会有其他地方引用这个模块，但是可能很难找出来，保险起见清空相关参数
                    clear_unused_module(mod)
                    del mod

        if hasattr(self.cfg, 'tp_size'):
            simulate_linear = self.dag.get_allreduce_linear()
            self.logger.info(
                'The following layers will be quant by simulating tp:\n\t'
                + '\n\t'.join([str(name) for name in sorted(simulate_linear)]))
        else:
            simulate_linear = []

        for name, mod in model.named_modules():
            if name in simulate_linear:
                layer_cfg = self.layer_cfg_manager.get_layer_config(name)
                tp_mod = ParallelLinearCol()
                tp_mod.set_param(mod, name, cfg=self.cfg)
                if name in self.rollback_names:
                    tp_mod.quant_type = QuantType.FLOAT
                else:
                    tp_mod.quant_type = layer_cfg.model_quant_type
                if hasattr(mod, HF_HOOK):
                    add_hook_to_module(tp_mod, mod._hf_hook)
                    remove_hook_from_module(mod)

                _set_module(model, name, tp_mod)
                del mod

        gc.collect()

        return model

    # 获取decoder相关信息
    def split_module_name_get_info(self, module_names, model, down_proj_type):
        item0 = module_names[0][0] if isinstance(module_names[0], list) else module_names[0]
        item2 = module_names[2][0] if isinstance(module_names[2], list) else module_names[2]
        module_names_ = [item0, module_names[1], item2, module_names[3]]

        qkv_item = module_names[0] if isinstance(module_names[0], list) else [module_names[0]]
        ffn_item = module_names[2] if isinstance(module_names[2], list) else [module_names[2]]

        qkv_tokens = [item.split('.') for item in qkv_item]
        qkv_list = [item[-1] for item in qkv_tokens]

        ffn_tokens = [item.split('.') for item in ffn_item]
        ffn_list = [item[-1] for item in ffn_tokens if item[-1] not in down_proj_type]

        tokens = [module_names_[i].split('.') for i in range(len(module_names_))]
        cur_mod = model
        layername = ''
        for i, item in enumerate(tokens[0]):
            layername += item + '.'
            cur_mod = getattr(cur_mod, item)
            if item.isnumeric():
                module_infos = [token[i + 1] for token in tokens]
                module_infos.append(qkv_list)
                module_infos.append(ffn_list)
                break
        mod_type = cur_mod.__class__.__name__

        ret = cur_mod, mod_type, module_infos, layername[:-1]
        return ret

    def get_norm_class(self, model, norm_class_name=None):
        if norm_class_name is not None:
            norm_class = list(set([m.__class__ for m in model.modules() \
                                   if norm_class_name.lower() in m.__class__.__name__.lower()]))
        else:
            norm_class = list(set([m.__class__ for m in model.modules() if "norm" in m.__class__.__name__.lower()]))
            if len(norm_class) != 1:
                raise ValueError("No customized normalization class detected! Please check the model and configuration")
        return norm_class

    def enable_kvcache_fake_quantization(self):
        num_layers = getattr(self.model.config, 'num_layers', None) \
            if hasattr(self.model, 'config') else None
        for name, mod in self.model.named_modules():
            # set the variable and function of kvcache in attention class
            if self.attention_class and isinstance(mod, self.attention_class):
                cache_sub_dict = {
                    key: value
                    for key, value in itertools.chain(*self.kv_cache_quant_params.values())
                    if name in key
                }
                set_kvcache_vari_func(mod, cache_sub_dict, self.cfg, num_layers=num_layers)

                setattr(mod, 'original_forward', mod.forward)
                setattr(mod, 'forward', new_forward.__get__(mod, mod.__class__))

    def _init_model_torch_dtype(self, model):
        if hasattr(model.config, 'torch_dtype'):
            if model.dtype != model.config.torch_dtype:
                self.logger.warning(
                    'The model dtype %r is not consistent with the '
                    'model.config.torch_dtype %r. '
                    'The model will be regarded as %r type in subsequent process.' %
                    (model.dtype, model.config.torch_dtype, model.config.torch_dtype)
                )
            return model.config.torch_dtype
        else:
            return model.dtype

    def _save(self, output_path, safetensors_name, json_name, save_type, part_file_size):
        # quantifier 应基于量化方法予以抽象，当前仅实现了与保存相关的逻辑

        # numpy 和 safetensor type 默认不支持 pack
        if isinstance(save_type, list):
            if SAVE_TYPE_ASCENDV1 in save_type:
                save_type = [SAVE_TYPE_ASCENDV1]
                is_new_version = True
                self.logger.warning(f"{SAVE_TYPE_ASCENDV1} is new version, {SAVE_TYPE_NUMPY} "
                                    f"and {SAVE_TYPE_SAFE_TENSOR} "
                                    f"will be ignored, only {SAVE_TYPE_ASCENDV1} will be saved.")
            else:
                is_new_version = False
        else:
            is_new_version = True if save_type in [SAVE_TYPE_ASCENDV1] else False

        saver = SaverFactory.create(save_type,
                                    output_dir=output_path,
                                    cfg=self.cfg,
                                    safetensors_name=safetensors_name,
                                    json_name=json_name,
                                    part_file_size=part_file_size,
                                    group_size=self.cfg.group_size if hasattr(self.cfg, 'group_size') else 0,
                                    enable_communication_quant=self.cfg.enable_communication_quant
                                    if hasattr(self.cfg, 'enable_communication_quant') else False)

        quantifier = ComplexQuantifier(cfg=self.cfg,
                                       rollback_names=self.rollback_names,
                                       torch_dtype=self.model_torch_dtype,
                                       layer_cfg_manager=self.layer_cfg_manager,
                                       is_new_version=is_new_version)
        self._save_weights_of_model(quantifier, saver)

    def _save_weights_of_model(self, quantifier, saver):
        self.model.eval()
        saver.pre_process()

        weight_collector = self.generate_weight_of_model(self.model, quantifier.generate_weight_of_module)
        for name, meta, tensor in weight_collector:
            saver.save(name, meta, tensor)

        saver.post_process()
        self.logger.info('Save successfully!')

    def _get_module_quant_input(self, module):
        fp_weight = module.weight.cpu()
        weight_scale, weight_offset = module.quant_weight.weight_scale, module.quant_weight.weight_offset
        device = weight_scale.device if weight_scale is not None else None
        scale = weight_scale.cpu() if weight_scale is not None else None
        offset = weight_offset.cpu() if weight_offset is not None else None
        round_opt = False if isinstance(module, LowBitLinearQuantizer) else module.quant_weight.round_opt
        ret = fp_weight, device, scale, offset, round_opt
        return ret

    def _run_training_mode(self):
        args = FlatQuantConfig()
        flat_quant_train(self.model, self.calib_data, self.layer_cfg_manager.layer_cfg, args, self.logger)

    def _run(self, calib_amp=5, int_infer=False):
        if not isinstance(calib_amp, int) or calib_amp < 1:
            raise TypeError("`calib_amp` should be an integer greater than 0 and not exceeding the "
                            "length of the calibration data. Please check the value.")

        self.logger.info("Calibration start!")
        self.model.eval()
        if self.calib_data:
            if self.cfg.model_quant_type == QuantType.W4A4_FLATQUANT_DYNAMIC:
                with torch.enable_grad():
                    self._run_training_mode()
                return
            elif self.cfg.calib_mode == 0:
                with torch.no_grad():
                    if getattr(self.cfg, 'use_timestep_quant', False):
                        timestep_utils.run_calib_timestep(self.model, self.calib_data, self.cfg)
                    else:
                        self.run_calib_mode()
            elif self.cfg.calib_mode == 1:
                try:
                    layer_wise_calib(self.model, self.all_tensors, self.cfg.device)
                except Exception as e:
                    raise Exception("Please check the model and configuration.", e) from e
                del self.all_tensors
            else:
                raise ValueError("Calibration mode not supported!")
        else:
            self.run_datafree_mode()

        self.logger.info("Calibration end!")

        disable_calibration(self.model, self.logger, use_fa_quant=self.cfg.use_fa_quant)
        if self.use_kvcache_quant:
            self.enable_kvcache_fake_quantization()
        if self.cfg.a_bit != 8 and int_infer:
            self.logger.warning("`int_infer` works only in W8A8 case. Defaulting to false")
            int_infer = False
        if int_infer:
            enable_int_infer(self.model, self.logger)


def enable_quantization_by_module(name, module, act_states):
    if '.tp_list' in name:
        states_name = ".".join(name.split(".")[:-3])
    else:
        states_name = ".".join(name.split(".")[:-1])
    range_param = 0
    if states_name in act_states:
        abs_max_tensor = max(abs(act_states[states_name]["t_max"]), abs(act_states[states_name]["t_min"]))
        range_param = abs_max_tensor / act_states[states_name]["std"]

    module.enable_quantization(name, range_param)
    if module.is_input:
        module.init_act_and_observer(module.cfg)


def enable_quantization(model, act_states, logger=None, use_fa_quant=False, skip_modules: List[str] = None):
    _ = logger  # Bypassing not using
    if use_fa_quant:
        enable_fa_calibration(model, skip_modules)
    for name, module in model.named_modules():
        if skip_modules and name in skip_modules:
            continue
        if isinstance(module, Quantizer):
            enable_quantization_by_module(name, module, act_states)
        if isinstance(module, LowBitQuantizer):
            module.enable_quantization()
            if module.is_input:
                module.init_act_and_observer(module.cfg)
        if isinstance(module, QuantXDecoderLayer):
            module.calibration = True


def disable_calibration(model, logger=None, custom_class=None, use_fa_quant=False):
    _ = logger  # Bypassing not using
    if use_fa_quant:
        disable_fa_calibration(model)
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.disable_calib()
        if isinstance(module, LowBitQuantizer):
            module.disable_calib()
        if custom_class and isinstance(module, custom_class):
            module.disable_calib()
        if isinstance(module, ParallelLinearCol):
            module.disable_calib()


def enable_calibration(model, logger=None, custom_class=None):
    _ = logger  # Bypassing not using
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.enable_calib()
    if not custom_class and isinstance(module, custom_class):
        module.enable_calib()


def disable_int_infer(model, logger=None):
    _ = logger  # Bypassing not using
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.disable_int_infer()


def enable_int_infer(model, logger=None):
    _ = logger  # Bypassing not using
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.enable_int_infer()


def set_ratio(model, ratio=0.9, logger=None):
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            if logger:
                logger.debug('Set the ratio: %r', name)
            module.set_ratio(ratio)
