# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from msmodelslim.format.interface import IFormatLoader
from msmodelslim.utils.buffer import RuntimeBufferStore
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.memory import align_input_dtype_to_module_hook, align_input_to_module_device_hook

# Decoder layers are detected by class name suffix, matching the heuristic used by
# IFormatLoader implementations (AscendV1Format.decoder_prefixes). Keep in sync.
_DECODER_LAYER_SUFFIX = "DecoderLayer"


def _empty_device_cache() -> None:
    """Best-effort device cache release (NPU / CUDA). No-op when neither is present."""
    if hasattr(torch, "npu"):
        gc.collect()
        torch.npu.empty_cache()
    elif hasattr(torch, "cuda"):
        gc.collect()
        torch.cuda.empty_cache()


class WeightManager:
    """Stage decoder-layer weights on/off the compute device around each forward.

    Installs persistent forward hooks once (via ``install_hooks``); during native
    ``model(**inputs)`` the hooks fire per layer:

      - decoder pre-hook  : hydrate_prefix + restore runtime buffers + ``.to(device)``,
                            then align inputs to the device (one combined hook)
      - decoder post-hook : ``.to(offload_device)`` + empty cache

    Shared modules (embed / norm / lm_head / visual, hydrated once on CPU by Session) only
    get an alignment pre-hook so hidden states cross the NPU<->CPU boundary automatically.

    Boundary: WeightManager decides "when/where"; FormatLoader decides "what/how".
    """

    def __init__(
        self,
        format_loader: IFormatLoader,
        device: str,
        buffer_store: RuntimeBufferStore,
        offload_device: str = "meta",
        shared_module_prefixes: Optional[List[str]] = None,
    ) -> None:
        self.format_loader = format_loader
        self.device = device
        self.buffer_store = buffer_store
        self.offload_device = offload_device
        # Optional allow-list for shared modules; when None, shared = param-bearing modules
        # that are neither decoder layers nor descendants of one.
        self._shared_module_prefixes = set(shared_module_prefixes) if shared_module_prefixes is not None else None
        self._model: Optional[nn.Module] = None
        self._hook_handles: List[Any] = []
        self._decoder_name_by_id: Dict[int, str] = {}

    # ------------------------------------------------------------------ public

    def install_hooks(self, model: nn.Module) -> None:
        """Scan decoder + shared modules and register persistent hooks. Idempotent."""
        if self._hook_handles:
            self.remove_all_hooks()  # avoid double registration

        self._model = model
        decoder_names: List[str] = []
        for name, module in model.named_modules():
            if module.__class__.__name__.endswith(_DECODER_LAYER_SUFFIX):
                decoder_names.append(name)
                self._decoder_name_by_id[id(module)] = name

        for name in decoder_names:
            module = model.get_submodule(name)
            pre = module.register_forward_pre_hook(self._decoder_pre_hook, with_kwargs=True)
            post = module.register_forward_hook(self._decoder_post_hook)
            self._hook_handles.extend([pre, post])

        for name, module in model.named_modules():
            if self._is_shared_module(name, module, decoder_names):
                dev_handle = module.register_forward_pre_hook(align_input_to_module_device_hook, with_kwargs=True)
                dtype_handle = module.register_forward_pre_hook(align_input_dtype_to_module_hook, with_kwargs=True)
                self._hook_handles.extend([dev_handle, dtype_handle])

        get_logger().info(
            "WeightManager installed hooks: %d decoder layer(s), %d shared alignment hook handles (device+dtype)",
            len(decoder_names),
            len(self._hook_handles) - 2 * len(decoder_names),
        )

    def remove_all_hooks(self) -> int:
        """Remove every hook installed by this manager. Returns the count removed."""
        count = len(self._hook_handles)
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception as exc:  # pylint: disable=broad-except
                get_logger().warning("Failed to remove a WeightManager hook: %s", exc)
        self._hook_handles.clear()
        self._decoder_name_by_id.clear()
        if count:
            get_logger().debug("WeightManager removed %d hook(s)", count)
        return count

    def set_offload_target(self, offload_device: str) -> None:
        """Switch the offload target at runtime (e.g. meta -> cpu for a layer window)."""
        self.offload_device = offload_device

    # ------------------------------------------------------------------ private

    def _is_shared_module(
        self,
        name: str,
        module: nn.Module,
        decoder_names: List[str],
    ) -> bool:
        """A shared module is param-bearing, not a decoder layer, and not inside one."""
        if self._shared_module_prefixes is not None:
            return any(name == prefix or name.startswith(prefix + ".") for prefix in self._shared_module_prefixes)
        if module.__class__.__name__.endswith(_DECODER_LAYER_SUFFIX):
            return False  # decoder layer itself
        if any(name.startswith(dec_name + ".") for dec_name in decoder_names):
            return False  # managed by the parent layer hook
        try:  # only hook modules that carry parameters (embed / norm / lm_head leaves)
            next(module.parameters())
        except StopIteration:
            return False
        return True

    def _decoder_pre_hook(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Hydrate + move layer weights to device, then align inputs (fused pre-hook)."""
        name = self._decoder_name_by_id.get(id(module))
        if name is None:
            name = self._lookup_module_name(module)  # module swapped out at runtime
        display_name = name or module.__class__.__name__
        if name is not None and self._model is not None:
            # IR was applied to the full tree in Session.build; here only load this layer's
            # weights (hydrate) and restore its runtime buffers.
            get_logger().info(
                "Running decoder layer %r ",
                display_name,
            )
            self.format_loader.hydrate(self._model, prefix=name)
            self.buffer_store.restore(self._model, prefix=name)
        # Move this layer's weights from host memory (CPU) onto the compute device.
        get_logger().debug(
            "Moving decoder layer %r weights to %s",
            display_name,
            self.device,
        )
        module.to(self.device)
        # Align inputs to the post-move device, so alignment must run after .to().
        return align_input_to_module_device_hook(module, args, kwargs)

    def _decoder_post_hook(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        """Offload the layer's weights and release device cache. Output is returned as-is."""
        name = self._decoder_name_by_id.get(id(module)) or self._lookup_module_name(module) or module.__class__.__name__
        get_logger().debug(
            " Forward %r finished, offloading weights from %s to %s",
            name,
            self.device,
            self.offload_device,
        )
        module.to(self.offload_device)
        _empty_device_cache()
        return output  # the next module's alignment hook handles the device crossing

    def _lookup_module_name(self, module: nn.Module) -> Optional[str]:
        if self._model is None:
            return None
        for name, sub in self._model.named_modules():
            if sub is module:
                return name
        return None
