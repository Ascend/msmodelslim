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

from typing import Dict, Optional, Set

import torch
from torch import nn

from msmodelslim.utils.logging import get_logger


def _is_under_prefix(key: str, prefixes: Set[str]) -> bool:
    for prefix in prefixes:
        if key == prefix or key.startswith(prefix + "."):
            return True
    return False


def _name_in_scope(
    full_name: str,
    prefix: Optional[str] = None,
    skip_prefixes: Optional[Set[str]] = None,
) -> bool:
    if prefix is not None:
        return full_name == prefix or full_name.startswith(prefix + ".")
    if skip_prefixes and _is_under_prefix(full_name, skip_prefixes):
        return False
    return True


def collect_nonpersistent_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone non-persistent buffers that currently hold real (non-meta) data."""
    collected: Dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        for buf_name in getattr(module, "_non_persistent_buffers_set", ()):
            tensor = getattr(module, buf_name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
                continue
            key = f"{module_name}.{buf_name}" if module_name else buf_name
            collected[key] = tensor.detach().cpu().clone()
    get_logger().info("Collected %d non-persistent buffer(s) from inference shell", len(collected))
    return collected


class RuntimeBufferStore:
    """Process-local snapshot of non-persistent buffers (RoPE ``inv_freq``, etc.).

    Owned by ``FakeQuantInferenceEngine._run_single``: the same instance is passed to
    ``Session`` (store after collect, restore shared modules) and ``WeightManager``
    (restore per decoder layer). Not written into Context — DP workers must not pickle
    multi-GB RoPE caches through SharedDictContext.
    """

    def __init__(self) -> None:
        self._buffers: Dict[str, torch.Tensor] = {}

    def store(self, mapping: Dict[str, torch.Tensor]) -> None:
        """Replace the snapshot (no merge). ``mapping`` should already be CPU clones."""
        self._buffers = mapping
        get_logger().debug("Stored %d runtime buffer(s) in process-local snapshot", len(mapping))

    def load(self) -> Dict[str, torch.Tensor]:
        """Return the live snapshot dict (may be empty)."""
        return self._buffers

    def clear(self) -> None:
        """Drop all snapshot tensors so they do not outlive ``_run_single``."""
        self._buffers.clear()

    def restore(
        self,
        model: nn.Module,
        prefix: Optional[str] = None,
        skip_prefixes: Optional[Set[str]] = None,
    ) -> int:
        """Restore snapshot buffers onto ``model``.

        Every snapshot buffer is restored by re-registering it from the snapshot clone:

        - missing / ``meta`` / real / shape-drifted targets all go through
          ``register_buffer(name, snapshot, persistent=False)`` (no in-place ``copy_``).

        Always registering (instead of ``copy_``) keeps the module's buffer registration
        authoritative and self-consistent: runtime caches (e.g. long-context RoPE tables)
        rewritten to the current ``seq_len`` during a forward are re-registered every round
        from the full snapshot, avoiding shape-mismatch crashes in multi-round prefill.
        Requires that no code holds a long-lived reference to the old buffer tensor across
        rounds (transformers modeling code re-``getattr``s buffers each forward, so this
        holds). A same-named attribute that is a plain (unregistered) tensor is left as-is;
        ``register_buffer`` raises ``KeyError`` on such a name, which signals a modeling-side
        buffer assignment that bypasses ``register_buffer`` and must be fixed at the source.
        """
        snapshot = self._buffers
        if not snapshot:
            return 0

        restored = 0
        for full_name, value in snapshot.items():
            if not _name_in_scope(full_name, prefix=prefix, skip_prefixes=skip_prefixes):
                continue
            if "." in full_name:
                parent_name, buf_name = full_name.rsplit(".", 1)
                try:
                    module = model.get_submodule(parent_name)
                except AttributeError:
                    continue
            else:
                module, buf_name = model, full_name

            current = getattr(module, buf_name, None)
            if current is not None and not isinstance(current, torch.Tensor):
                continue
            cloned = value.detach().cpu().clone()
            module.register_buffer(buf_name, cloned, persistent=False)
            restored += 1

        if restored:
            get_logger().debug("Restored %d runtime buffer(s) from process-local snapshot", restored)
        return restored
