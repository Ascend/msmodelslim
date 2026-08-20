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

import re
from collections import defaultdict
from typing import List, Optional, Dict, Literal, Annotated

import torch
from pydantic import AfterValidator, Field
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantizer.attention import DynamicCacheQuantizer
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.ir import FakeQuantDynamicCache
from msmodelslim.ir.qal.qbase import QScope
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.exception import VersionError, UnsupportedError
from msmodelslim.utils.function_hijacker import hijack_function
from msmodelslim.utils.hook_utils import add_before_hook, add_after_hook, restore_target
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.validation.pydantic import validate_str_length

DYNAMIC_AVAILABLE = True
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DYNAMIC_AVAILABLE = False

HOOK_TARGET = (DynamicCache, 'update') if DYNAMIC_AVAILABLE else (None, None)
CACHE_INPUT_NAME = ("key_states", "value_states") if DYNAMIC_AVAILABLE else (None, None)
LAYER_IDX_NAME = "layer_idx" if DYNAMIC_AVAILABLE else None


class DynamicCacheProcessorConfig(AutoProcessorConfig):
    """KV cache 量化处理器配置。

    位于 `spec.process[]`，由 `type: dynamic_cache` 分派；对注意力层的 KV cache 张量
    应用 `qconfig` 指定的量化方式。
    """

    type: Literal['dynamic_cache'] = Field(default="dynamic_cache", description="处理器类型，固定为 `dynamic_cache`。")
    qconfig: QConfig = Field(description="KV cache 张量的量化配置，见《QConfig 配置说明》。")
    include: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=list, description="包含的模块名称模式"
    )
    exclude: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=list, description="排除的模块名称模式，优先级高于 `include`"
    )


def _warning_unmatched_pattern(name: str, config_set: ConfigSet) -> None:
    unmatched_keys = config_set.unmatched_keys()
    unmatched_keys = list(filter(lambda x: x != "*", unmatched_keys))
    if unmatched_keys:
        get_logger().warning(
            "These %s patterns are not matched any module, please ensure this is as expected: %s", name, unmatched_keys
        )


def _get_module_by_name(model: nn.Module, submodule_key: str) -> nn.Module:
    """根据名称获取模块"""
    module_tokens = submodule_key.split('.')
    cur_mod = model
    for s in module_tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


def _detect_attention_layers(model: torch.nn.Module) -> Dict[int, str]:
    """
    Detect all attention layer prefixes in the model.
    Identifies attention modules by checking if class name contains 'attention'.
    """
    attention_layers = {}

    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if 'attention' in class_name or 'attn' in class_name:
            # Extract layer index from module name
            numbers = re.findall(r'\.(\d+)\.', name)
            if numbers:
                layer_idx = int(numbers[0])
                attention_layers[layer_idx] = name

    return attention_layers


def _get_first_layer(model: torch.nn.Module):
    attention_layers = _detect_attention_layers(model)
    layer_name = '.'.join(attention_layers[0].split('.')[:-1])
    mod = _get_module_by_name(model, layer_name)
    return mod


@QABCRegistry.register(dispatch_key=DynamicCacheProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.processor.kvcache_quant")
class DynamicCacheQuantProcessor(AutoSessionProcessor):
    def __init__(
        self,
        model: nn.Module,
        config: DynamicCacheProcessorConfig,
        adapter: Optional[object] = None,
    ):
        super().__init__(model)
        self.config = config
        if not DYNAMIC_AVAILABLE:
            raise VersionError("DynamicCache is not available", action="please install transformers>=4.36.0")
        if self.config.qconfig.scope != QScope.PER_CHANNEL:
            raise ValueError("DynamicCacheQuantProcessor only supports per_channel quantization!")

        self.include = ConfigSet(config.include) if config.include else ConfigSet(["*"])
        self.exclude = ConfigSet(config.exclude) if config.exclude else ConfigSet([])

        self.input_name_map = dict(enumerate(CACHE_INPUT_NAME))
        self.input_layer_idx_name = LAYER_IDX_NAME

        self.cache_quantizers: Dict[int, Dict[str, DynamicCacheQuantizer]] = defaultdict(
            lambda: {quant_name: None for quant_name in self.input_name_map.values()}
        )
        # Track quantizer ready status
        self.quantizer_ready: Dict[int, Dict[str, bool]] = defaultdict(
            lambda: {quant_name: False for quant_name in self.input_name_map.values()}
        )
        self.fake_kvcache_quantizers: Dict[int, Dict[str, FakeQuantDynamicCache]] = defaultdict(
            lambda: {quant_name: None for quant_name in self.input_name_map.values()}
        )
        # add trigger hook on module who first uses kvcache
        self.first_layer = _get_first_layer(self.model)
        self.trigger_hook_target = (self.first_layer, 'forward')
        self.cache_target = HOOK_TARGET
        self._trigger_hook_installed = False
        self._use_global_hook = False
        self._attention_layers_map = _detect_attention_layers(self.model)
        # Hook registry to avoid duplicate hook installation using cache IDs
        self._installed_cache_ids = set()

    def is_data_free(self) -> bool:
        return False

    def need_kv_cache(self):
        return True

    def support_distributed(self) -> bool:
        return False

    def pre_run(self) -> None:
        attention_layers = _detect_attention_layers(self.model)
        for layer_idx, _ in attention_layers.items():
            self._create_quantizer(layer_idx)

        # add quantize hook
        add_before_hook(self.trigger_hook_target, self._add_quantizer_hook)
        self._trigger_hook_installed = True

    def postprocess(self, _: BatchProcessRequest) -> None:
        _warning_unmatched_pattern("include", self.include)
        _warning_unmatched_pattern("exclude", self.exclude)
        attention_layers = _detect_attention_layers(self.model)
        for layer_idx, attention_prefix in attention_layers.items():
            mod = _get_module_by_name(self.model, attention_prefix)
            self._deploy_quantizer(mod, layer_idx)

    def post_run(self) -> None:
        # remove global hook if used
        if self._trigger_hook_installed:
            if self._use_global_hook:
                # Remove global hook
                restore_target(self.cache_target)
                self._use_global_hook = False
            # Remove trigger hook
            restore_target(self.trigger_hook_target)
            self._trigger_hook_installed = False

        # Clear hook registry
        self._installed_cache_ids.clear()

        # Install execution hook for fake quantization
        add_before_hook(self.trigger_hook_target, self._add_fake_quant_hook)
        self._trigger_hook_installed = True

    def _create_quantizer(self, layer_idx: int):
        for _, target_name in self.input_name_map.items():
            if self.cache_quantizers[layer_idx][target_name] is None:
                self.cache_quantizers[layer_idx][target_name] = DynamicCacheQuantizer(self.config.qconfig)

    def _deploy_quantizer(self, mod: nn.Module, layer_idx: int):
        # 只有当量化器准备好时才部署
        for _, target_name in self.input_name_map.items():
            if self.quantizer_ready[layer_idx][target_name] and self.cache_quantizers[layer_idx][target_name]:
                mod.add_module(f'{target_name}_quantizer', self.cache_quantizers[layer_idx][target_name].deploy())
                self.fake_kvcache_quantizers[layer_idx][target_name] = getattr(mod, f'{target_name}_quantizer')

    def _add_quantizer_hook(self, _, kwargs):
        get_logger().debug(
            "dynamic cache quant processor hijack kvcache update, the cache is always empty when calibrating!"
        )
        for _, value in kwargs.items():
            if isinstance(value, self.cache_target[0]):
                # Check if hook already installed using cache ID
                cache_id = id(value)
                if cache_id in self._installed_cache_ids:
                    return
                target = (value, self.cache_target[1])
                hijack_function(target, self._custom_cache_update)
                self._installed_cache_ids.add(cache_id)
                return
        if not self._use_global_hook:
            get_logger().warning(
                "No %s found in the model forward function arguments"
                "try to hook on %s.%s, "
                "this may influence other model's inference!",
                self.cache_target[0].__name__,
                self.cache_target[0].__name__,
                self.cache_target[1],
            )
            hijack_function(self.cache_target, self._custom_cache_update)
            self._use_global_hook = True

    def _add_fake_quant_hook(self, _, kwargs):
        for _, value in kwargs.items():
            if isinstance(value, self.cache_target[0]):
                # Check if hook already installed using cache ID
                cache_id = id(value)
                if cache_id in self._installed_cache_ids:
                    return
                target = (value, self.cache_target[1])
                add_after_hook(target, self._fake_quant_update)
                self._installed_cache_ids.add(cache_id)
                return
        if not self._use_global_hook:
            get_logger().warning(
                "No %s found in the model forward function arguments"
                "try to hook on %s.%s, "
                "this may influence other model's inference!",
                self.cache_target[0].__name__,
                self.cache_target[0].__name__,
                self.cache_target[1],
            )
            add_after_hook(self.cache_target, self._fake_quant_update)
            self._use_global_hook = True

    def _fake_quant_update(self, _, kwargs, result):
        layer_idx = kwargs.get(self.input_layer_idx_name)
        res = []
        for idx, target_name in self.input_name_map.items():
            if isinstance(result, tuple):
                states = result[idx]
            else:
                states = result

            if self._attention_layers_map[layer_idx] not in self.include:
                return result

            if self._attention_layers_map[layer_idx] in self.exclude:
                return result

            quantizer = self.fake_kvcache_quantizers[layer_idx][target_name]
            if quantizer is not None:
                states = quantizer(states)
            res.append(states)
        return tuple(res) if len(result) > 1 else res[0]

    def _custom_cache_update(self, **kwargs):
        """
        Custom cache update function that replaces the original DynamicCache.update method.
        This function will be used during the calibration phase.
        ###################################################################################################
        NOTICE: this function hijack the cache update function, the cache is always empty when calibrating!
        ###################################################################################################
        """
        # Get layer_idx from kwargs using the configured name
        layer_idx = kwargs.get(self.input_layer_idx_name)
        if layer_idx is None:
            raise UnsupportedError(
                "Can't find layer index from cache update function",
                action="please check the implementation of the model's cache update function",
            )
        for _, target_name in self.input_name_map.items():
            states = kwargs.get(target_name)
            if states is None:
                raise UnsupportedError(
                    f"Can't find {target_name} from cache update function",
                    action="please check the implementation of the model's cache update function",
                )
            if self._attention_layers_map[layer_idx] not in self.include:
                return tuple(kwargs.get(name) for name in CACHE_INPUT_NAME)

            if self._attention_layers_map[layer_idx] in self.exclude:
                return tuple(kwargs.get(name) for name in CACHE_INPUT_NAME)

            # Update key and value quantization observers
            quantizer = self.cache_quantizers[layer_idx][target_name]
            if quantizer is not None:
                quantizer(states)
                self.quantizer_ready[layer_idx][target_name] = True

        return tuple(kwargs.get(name) for name in CACHE_INPUT_NAME)
