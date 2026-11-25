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

import fnmatch
import re
from collections import defaultdict
from typing import List, Optional, Dict, Literal, Annotated

import torch
from pydantic import AfterValidator, Field, BaseModel, ConfigDict, model_validator
from torch import nn
import torch.distributed as dist

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantizer.attention import DynamicCacheQuantizer
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantization_schema import AutoFakeQuantDynamicCache, FakeQuantDynamicCache
from msmodelslim.core.quantization_schema.qal.qbase import QScope, QDType, QParam
from msmodelslim.core.quantization_schema.qal.qregistry import QABCRegistry
from msmodelslim.core.quantization_algorithm.base import AutoSessionProcessor, AutoProcessorConfig
from msmodelslim.utils.config_map import ConfigSet
from msmodelslim.utils.distributed.dist_helper import DistHelper
from msmodelslim.utils.exception import VersionError, UnsupportedError
from msmodelslim.utils.function_hijacker import hijack_function, restore_function
from msmodelslim.utils.hook_utils import add_before_hook, add_after_hook, restore_target
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.validation.pydantic import validate_str_length

KV_BRANCHES = ("k_qconfig", "v_qconfig")
CACHE_BRANCH_TO_NAME = {"k_qconfig": "key_states", "v_qconfig": "value_states"}

STATIC_SCOPES = (QScope.PER_CHANNEL, QScope.PER_HEAD)
DYNAMIC_SCOPES = (QScope.PER_TOKEN, QScope.PER_TILE, QScope.PER_BLOCK)

DYNAMIC_AVAILABLE = True
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DYNAMIC_AVAILABLE = False

HOOK_TARGET = (DynamicCache, 'update') if DYNAMIC_AVAILABLE else (None, None)
CACHE_INPUT_NAME = ("key_states", "value_states") if DYNAMIC_AVAILABLE else (None, None)
LAYER_IDX_NAME = "layer_idx" if DYNAMIC_AVAILABLE else None


class DynamicCachePlaceHolder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DynamicCacheDetails(BaseModel):
    """KV cache 的 K/V 分支独立量化配置。

    用于分别指定 key 与 value 分支的量化方式，与 `qconfig` 互斥。
    """

    k_qconfig: QConfig = Field(default=None, description="Key 分支量化配置")
    v_qconfig: QConfig = Field(default=None, description="Value 分支量化配置")

    model_config = ConfigDict(extra="forbid")


class DynamicCacheProcessorConfig(AutoProcessorConfig):
    """KV cache 量化处理器配置。

    位于 `spec.process[]`，由 `type: dynamic_cache` 分派；对注意力层的 KV cache 张量
    应用量化，支持 `qconfig` 共用或 `details` 按 K/V 分支独立指定。
    """

    type: Literal['dynamic_cache'] = Field(default="dynamic_cache", description="处理器类型，固定为 `dynamic_cache`。")
    qconfig: Optional[QConfig] = Field(default=None, description="K/V 共用的量化配置，与 `details` 互斥。")
    details: Optional[DynamicCacheDetails] = Field(
        default=None, description="K/V 分支独立的量化配置，与 `qconfig` 互斥。"
    )
    include: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=list, description="包含的模块名称模式"
    )
    exclude: List[Annotated[str, AfterValidator(validate_str_length())]] = Field(
        default_factory=list, description="排除的模块名称模式，优先级高于 `include`"
    )
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_exclusivity(self) -> "DynamicCacheProcessorConfig":
        """校验 `qconfig` 与 `details` 至多配置一个；两者均未配置时回落 int8/per_channel 对称 minmax。"""

        if self.qconfig is None and self.details is None:
            self.qconfig = QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax")
            return self

        if self.qconfig is not None and self.details is not None:
            raise ValueError("DynamicCacheQuantProcessor supports only one of the qconfig and details configurations.")
        return self

    def branch_qconfig(self, branch: str) -> Optional[QConfig]:
        if self.details is not None:
            return getattr(self.details, branch)
        return self.qconfig


@QABCRegistry.register(dispatch_key=DynamicCacheProcessorConfig, abc_class=AutoSessionProcessor)
@logger_setter(prefix="msmodelslim.core.quantization_algorithm.quant.kvcache")
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

        supported_scopes = STATIC_SCOPES + DYNAMIC_SCOPES
        for branch in KV_BRANCHES:
            branch_qconfig = config.branch_qconfig(branch)
            if branch_qconfig is not None and branch_qconfig.scope not in supported_scopes:
                supported = ", ".join(scope.value for scope in supported_scopes)
                raise ValueError(
                    f"DynamicCacheQuantProcessor does not support scope {branch_qconfig.scope.value} "
                    f"({branch} branch); supported scopes: {supported}"
                )

        self.adapter = adapter

        self.include = ConfigSet(config.include) if config.include else ConfigSet(["*"])
        self.exclude = ConfigSet(config.exclude) if config.exclude else ConfigSet([])

        self.input_name_map = dict(enumerate(CACHE_INPUT_NAME))
        self.input_layer_idx_name = LAYER_IDX_NAME

        self.cache_quantizers: Dict[str, Dict[str, DynamicCacheQuantizer]] = defaultdict(
            lambda: {quant_name: None for quant_name in self.input_name_map.values()}
        )
        self.quantizer_ready: Dict[str, Dict[str, bool]] = defaultdict(
            lambda: {quant_name: False for quant_name in self.input_name_map.values()}
        )
        self.fake_kvcache_quantizers: Dict[str, Dict[str, FakeQuantDynamicCache]] = defaultdict(
            lambda: {quant_name: None for quant_name in self.input_name_map.values()}
        )
        self.attention_layers: Dict[str, nn.Module] = {}
        self._attention_layers_map: Dict[int, str] = {}
        self.dist_helper: Optional[DistHelper] = None

        self.cache_target = HOOK_TARGET
        self._trigger_hook_installed = False
        self._use_global_hook = False
        self._installed_cache_ids = set()

    def _check_module_name(self, full_name: str) -> bool:
        if self.include:
            if not any(fnmatch.fnmatch(full_name, pattern) for pattern in self.include):
                return False
        if self.exclude:
            if any(fnmatch.fnmatch(full_name, pattern) for pattern in self.exclude):
                return False
        return True

    def _should_inject(self, full_name: str) -> bool:
        return self._check_module_name(full_name)

    def _detect_attention_layers(self, root_name: str, root_module: nn.Module) -> Dict[str, nn.Module]:
        result: Dict[str, nn.Module] = {}
        for name, module in root_module.named_modules():
            class_name = module.__class__.__name__.lower()
            if 'attention' not in class_name and 'attn' not in class_name:
                continue
            full_name = f"{root_name}.{name}" if root_name else name
            if not self._should_inject(full_name):
                continue
            result[full_name] = module
        return result

    @staticmethod
    def _target_name_to_branch(target_name: str) -> str:
        for branch, name in CACHE_BRANCH_TO_NAME.items():
            if name == target_name:
                return branch
        raise UnsupportedError(f"Unknown cache target name: {target_name}")

    def branch_qconfig(self, target_name: str) -> Optional[QConfig]:
        branch = self._target_name_to_branch(target_name)
        return self.config.branch_qconfig(branch)

    def _is_static_branch(self, target_name: str) -> bool:
        qconfig = self.branch_qconfig(target_name)
        if qconfig is None:
            return False
        return qconfig.scope in STATIC_SCOPES

    def _is_dynamic_branch(self, target_name: str) -> bool:
        qconfig = self.branch_qconfig(target_name)
        if qconfig is None:
            return False
        return qconfig.scope in DYNAMIC_SCOPES

    def is_data_free(self) -> bool:
        active_branches = [name for name in self.input_name_map.values() if self.branch_qconfig(name) is not None]
        if not active_branches:
            return True
        return all(self._is_dynamic_branch(name) for name in active_branches)

    def need_kv_cache(self):
        return True

    def support_distributed(self) -> bool:
        return True

    def pre_run(self) -> None:
        def cache_update(**kwargs):
            kwargs.pop("self", None)
            return self._custom_cache_update(**kwargs)

        # Replace update before calibration, including its first invocation.
        hijack_function(HOOK_TARGET, cache_update)
        self._trigger_hook_installed = True

    def preprocess(self, request: BatchProcessRequest) -> None:
        if dist.is_initialized() and self.dist_helper is None:
            self.dist_helper = DistHelper(request.module, prefix=request.name)
        try:
            self.attention_layers = self.adapter.inject_kvcache_placeholders(
                request.name,
                request.module,
                self._should_inject,
                DynamicCachePlaceHolder,
            )
        except (UnsupportedError, AttributeError):
            get_logger().info(
                "Adapter %s does not support inject_kvcache_placeholders, falling back to default attention discovery",
                type(self.adapter).__name__,
            )
            self.attention_layers = self._detect_attention_layers(request.name, request.module)

        get_logger().info("selected kvcache attention layers: %s", list(self.attention_layers.keys()))

        for full_name in self.attention_layers:
            numbers = re.findall(r'\.(\d+)\.', full_name)
            if numbers:
                layer_idx = int(numbers[0])
                if layer_idx not in self._attention_layers_map:
                    self._attention_layers_map[layer_idx] = full_name

        for full_name in self.attention_layers:
            self._create_quantizer(full_name)

        if any(self._is_dynamic_branch(name) for name in self.input_name_map.values()):
            for full_name, attn_mod in self.attention_layers.items():
                for _, target_name in self.input_name_map.items():
                    if not self._is_dynamic_branch(target_name):
                        continue
                    placeholder_name = f'{target_name}_quantizer'
                    self.model.set_submodule(
                        f'{full_name}.{placeholder_name}',
                        self._create_dynamic_ir(target_name),
                    )
                    self.fake_kvcache_quantizers[full_name][target_name] = getattr(attn_mod, placeholder_name)

    def postprocess(self, _: BatchProcessRequest) -> None:
        for full_name, attn_mod in self.attention_layers.items():
            self._deploy_quantizer(attn_mod, full_name)
        if dist.is_initialized():
            self.dist_helper = None

    def post_run(self) -> None:
        if self._trigger_hook_installed:
            if self._use_global_hook:
                restore_target(self.cache_target)
                self._use_global_hook = False
            restore_function(HOOK_TARGET)
            restore_target(HOOK_TARGET)
            self._trigger_hook_installed = False

        self._installed_cache_ids.clear()

        add_before_hook(HOOK_TARGET, self._add_fake_quant_hook)
        self._trigger_hook_installed = True

    def _create_quantizer(self, full_name: str):
        for _, target_name in self.input_name_map.items():
            if not self._is_static_branch(target_name):
                continue
            if self.cache_quantizers[full_name][target_name] is None:
                qconfig = self.branch_qconfig(target_name)
                self.cache_quantizers[full_name][target_name] = DynamicCacheQuantizer(qconfig)
                if dist.is_initialized() and self.dist_helper is not None:
                    if self.dist_helper.is_shared(full_name):
                        self.cache_quantizers[full_name][target_name].input_quantizer.enable_sync()

    def _deploy_quantizer(self, mod: nn.Module, full_name: str):
        for _, target_name in self.input_name_map.items():
            if self._is_dynamic_branch(target_name):
                placeholder_name = f'{target_name}_quantizer'
                if self.fake_kvcache_quantizers[full_name][target_name] is None and not hasattr(mod, placeholder_name):
                    mod.add_module(
                        placeholder_name,
                        self._create_dynamic_ir(target_name),
                    )
                    self.fake_kvcache_quantizers[full_name][target_name] = getattr(mod, placeholder_name)
                continue
            if self.quantizer_ready[full_name][target_name] and self.cache_quantizers[full_name][target_name]:
                mod.add_module(f'{target_name}_quantizer', self.cache_quantizers[full_name][target_name].deploy())
                self.fake_kvcache_quantizers[full_name][target_name] = getattr(mod, f'{target_name}_quantizer')

    def _create_dynamic_ir(self, target_name: str) -> nn.Module:
        qconfig = self.branch_qconfig(target_name)
        scheme = qconfig.to_scheme()
        fallback_ext = {"scale": torch.ones(1, dtype=torch.float32)}
        q_param = QParam(scheme=scheme, ext=fallback_ext)
        return AutoFakeQuantDynamicCache.create(q_param)

    def _add_quantizer_hook(self, _, kwargs):
        get_logger().debug(
            "dynamic cache quant processor hijack kvcache update, the cache is always empty when calibrating!"
        )
        for _, value in kwargs.items():
            if isinstance(value, self.cache_target[0]):
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

    def _get_full_name_by_layer_idx(self, layer_idx: int) -> Optional[str]:
        return self._attention_layers_map.get(layer_idx)

    def _fake_quant_update(self, _, kwargs, result):
        layer_idx = kwargs.get(self.input_layer_idx_name)
        full_name = self._get_full_name_by_layer_idx(layer_idx)
        if full_name is None:
            return result

        res = []
        for idx, target_name in self.input_name_map.items():
            if isinstance(result, tuple):
                states = result[idx]
            else:
                states = result

            quantizer = self.fake_kvcache_quantizers.get(full_name, {}).get(target_name)
            if quantizer is not None:
                states = quantizer(states)
            res.append(states)
        return tuple(res) if len(result) > 1 else res[0]

    def _custom_cache_update(self, **kwargs):
        layer_idx = kwargs.get(self.input_layer_idx_name)
        if layer_idx is None:
            raise UnsupportedError(
                "Can't find layer index from cache update function",
                action="please check the implementation of the model's cache update function",
            )

        full_name = self._get_full_name_by_layer_idx(layer_idx)
        if full_name is None:
            return tuple(kwargs.get(name) for name in CACHE_INPUT_NAME)

        for _, target_name in self.input_name_map.items():
            states = kwargs.get(target_name)
            if states is None:
                raise UnsupportedError(
                    f"Can't find {target_name} from cache update function",
                    action="please check the implementation of the model's cache update function",
                )

            quantizer = self.cache_quantizers.get(full_name, {}).get(target_name)
            if quantizer is not None:
                quantizer(states)
                self.quantizer_ready[full_name][target_name] = True

        return tuple(kwargs.get(name) for name in CACHE_INPUT_NAME)
