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

# Expert Parallelism (EP) runtime patches for Kimi-K3 ``KimiSparseMoeBlock``.
#
# Do NOT edit weight-dir ``modeling_*.py``. This module monkey-patches the imported
# class so that:
#
# 1. ``experts`` is a full-length ``ModuleList`` with ``None`` for non-local slots
#    (DistHelper marks local experts as ``local_only``).
# 2. Forward uses DP token gather + local expert compute + ``all_reduce``, then
#    latent up-proj / shared experts on all ranks (DeepSeek-V4 style).
#
# EP activates only when ``torch.distributed`` is initialized with ``world_size > 1``.
# ``num_experts`` must be divisible by ``world_size``.

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from msmodelslim.model.common.utils import resolve_expert_ep_range
from msmodelslim.utils.logging import get_logger

_EP_PATCHED_MODULES = set()

# New HF release renamed modeling_kimi.py -> modeling_kimi_linear.py; keep both.
_KIMI_TEXT_MODELING_FILES = ("modeling_kimi_linear.py", "modeling_kimi.py")
_KIMI_SPARSE_MOE_CLASS_PATHS = (
    "modeling_kimi_linear.KimiSparseMoeBlock",
    "modeling_kimi.KimiSparseMoeBlock",
)


def moe_infer_local_experts(
    experts: nn.ModuleList,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    experts_start_idx: int,
    experts_end_idx: int,
) -> torch.Tensor:
    """EP-safe routed expert inference (GLM-style local loop).

    Args:
        experts: Full-length ModuleList; non-local entries may be ``None``.
        x: ``[T, H]`` token features (possibly in latent space).
        topk_ids / topk_weight: ``[T, K]`` routing from the gate.
        experts_start_idx / experts_end_idx: local expert half-open range.
    """
    y = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.float32, device=x.device)
    for expert_id in range(experts_start_idx, experts_end_idx):
        expert = experts[expert_id]
        if expert is None:
            continue
        token_idx, top_idx = torch.where(topk_ids == expert_id)
        if token_idx.numel() == 0:
            continue
        expert_out = expert(x[token_idx])
        y[token_idx] += expert_out.to(torch.float32) * topk_weight[token_idx, top_idx].unsqueeze(-1).to(torch.float32)
    return y.type_as(x)


def _is_real_kimi_modeling_module(mod: Any) -> bool:
    """True for real Kimi text modeling modules (never ``torch.ops`` proxies).

    Matches both legacy ``modeling_kimi`` and current ``modeling_kimi_linear``.
    """
    if mod is None:
        return False
    name = getattr(mod, "__name__", "") or ""
    # torch.ops / torch._ops namespaces return truthy objects for any attribute.
    if name.startswith("torch.ops") or name.startswith("torch._ops") or name in ("torch.ops", "torch._ops"):
        return False

    block = getattr(mod, "KimiSparseMoeBlock", None)
    mlp = getattr(mod, "KimiBlockSparseMLP", None)
    gate = getattr(mod, "KimiMoEGate", None)
    if block is None or mlp is None or gate is None:
        return False
    # Must be plain Python classes subclassing nn.Module (not torch custom-class proxies).
    if not isinstance(block, type) or not isinstance(mlp, type) or not isinstance(gate, type):
        return False
    try:
        return issubclass(block, nn.Module) and issubclass(mlp, nn.Module)
    except TypeError:
        return False


def _is_ep_already_patched(block_cls: type) -> bool:
    """Avoid ``getattr`` on torch custom-class proxies (raises RuntimeError)."""
    try:
        return bool(block_cls.__dict__.get("_msmodelslim_ep_patched", False))
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _load_modeling_from_weight_dir(model_path: str, modeling_file: Path) -> Optional[Any]:
    """Import a modeling_*.py from the weight directory without going through HF cache."""
    import importlib.util

    # Unique fake module name per file so old/new can coexist in one process.
    stem = modeling_file.stem
    mod_name = f"msmodelslim_kimi_k3_{stem}_ep"
    spec = importlib.util.spec_from_file_location(mod_name, modeling_file)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        get_logger().warning("Failed to exec %s for EP patch (%s): %s", modeling_file.name, model_path, exc)
        return None
    return mod


def _class_paths_for_model_path(model_path: str) -> Tuple[str, ...]:
    """Prefer the modeling file that actually exists in the weight dir."""
    root = Path(model_path)
    paths = []
    if (root / "modeling_kimi_linear.py").is_file():
        paths.append("modeling_kimi_linear.KimiSparseMoeBlock")
    if (root / "modeling_kimi.py").is_file():
        paths.append("modeling_kimi.KimiSparseMoeBlock")
    return tuple(paths) if paths else _KIMI_SPARSE_MOE_CLASS_PATHS


def _modeling_files_for_model_path(model_path: str) -> Tuple[str, ...]:
    root = Path(model_path)
    files = [name for name in _KIMI_TEXT_MODELING_FILES if (root / name).is_file()]
    return tuple(files) if files else _KIMI_TEXT_MODELING_FILES


def _preload_modeling_from_model_path(model_path: str, seen: set) -> Iterable[Any]:
    """Preload text modeling via HF dynamic import, then direct file fallback."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    class_paths = _class_paths_for_model_path(model_path)
    modeling_files = _modeling_files_for_model_path(model_path)

    last_exc: Optional[BaseException] = None
    for class_path in class_paths:
        try:
            cls = get_class_from_dynamic_module(class_path, model_path)
            mod = inspect.getmodule(cls)
            if mod is None or not _is_real_kimi_modeling_module(mod):
                continue
            # Already discovered via sys.modules — nothing more to yield.
            if id(mod) in seen:
                return
            seen.add(id(mod))
            yield mod
            return
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc

    # Fallback: load modeling_*.py directly from the weight directory.
    for filename in modeling_files:
        modeling_file = Path(model_path) / filename
        if not modeling_file.is_file():
            continue
        mod = _load_modeling_from_weight_dir(model_path, modeling_file)
        if mod is None or not _is_real_kimi_modeling_module(mod):
            continue
        if id(mod) in seen:
            return
        seen.add(id(mod))
        yield mod
        return

    # Avoid noisy warnings when sys.modules already provided a patchable module.
    if seen:
        return

    get_logger().warning(
        "Failed to preload Kimi text modeling for EP patch (%s): tried %s; last_error=%s",
        model_path,
        list(class_paths) + list(modeling_files),
        last_exc,
    )


def _iter_modeling_kimi_modules(model_path: Optional[str] = None) -> Iterable[Any]:
    seen = set()
    for _name, mod in list(sys.modules.items()):
        if mod is None or id(mod) in seen:
            continue
        if not _is_real_kimi_modeling_module(mod):
            continue
        seen.add(id(mod))
        yield mod

    if model_path is None:
        return

    yield from _preload_modeling_from_model_path(model_path, seen)


def _patch_kimi_sparse_moe_block(modeling: Any) -> bool:
    if not _is_real_kimi_modeling_module(modeling):
        return False
    block_cls = modeling.KimiSparseMoeBlock
    if _is_ep_already_patched(block_cls):
        return False

    mlp_cls = modeling.KimiBlockSparseMLP
    gate_cls = modeling.KimiMoEGate
    shared_cls = modeling.KimiMLP
    rms_cls = modeling.KimiRMSNorm

    def ep_init(self, config):
        # Skip replaced __init__; initialize nn.Module only.
        super(block_cls, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize

        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)

        ep_size, ep_rank, start, end = resolve_expert_ep_range(config.num_experts)
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_per_rank = end - start
        self.experts_start_idx = start
        self.experts_end_idx = end

        # Full-length ModuleList; None slots → DistHelper local_only for real experts.
        self.experts = nn.ModuleList(
            [
                mlp_cls(
                    config,
                    hidden_size=self.moe_hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                )
                if start <= i < end
                else None
                for i in range(config.num_experts)
            ]
        )
        self.gate = gate_cls(config)
        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = shared_cls(config=config, intermediate_size=intermediate_size)

        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(config.hidden_size, self.moe_hidden_size, bias=False)
            self.routed_expert_up_proj = nn.Linear(self.moe_hidden_size, config.hidden_size, bias=False)
            if self.latent_moe_use_norm:
                self.routed_expert_norm = rms_cls(self.moe_hidden_size, eps=config.rms_norm_eps)

        if ep_size > 1:
            get_logger().info(
                "KimiSparseMoeBlock EP enabled: rank=%s/%s experts=[%s, %s) of %s",
                ep_rank,
                ep_size,
                start,
                end,
                config.num_experts,
            )

    @torch.no_grad()
    def ep_moe_infer(self, x, topk_ids, topk_weight):
        return moe_infer_local_experts(
            self.experts,
            x,
            topk_ids,
            topk_weight,
            self.experts_start_idx,
            self.experts_end_idx,
        )

    def ep_forward(self, hidden_states):
        if self.training:
            raise NotImplementedError("Training mode is not supported in KimiSparseMoeBlock")

        use_dp_ep = dist.is_initialized() and dist.get_world_size() > 1
        if use_dp_ep:
            from msmodelslim.utils.distributed import DistHelper

            seq_len_this_rank = hidden_states.size(1)
            seq_len_tensor = torch.tensor([seq_len_this_rank], dtype=torch.long, device=hidden_states.device)
            seq_len_list = [torch.zeros_like(seq_len_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(seq_len_list, seq_len_tensor)
            seq_lens = [int(s.item()) for s in seq_len_list]
            rank = dist.get_rank()
            start_pos = sum(seq_lens[:rank])
            end_pos = start_pos + seq_len_this_rank
            hidden_states = torch.cat(DistHelper.gather_variable_shapes(hidden_states), dim=1)
        else:
            start_pos = 0
            end_pos = hidden_states.size(1)

        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        flat = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            flat = self.routed_expert_down_proj(flat)

        y = self.moe_infer(flat, topk_idx, topk_weight)

        # Sum partial expert outputs across EP ranks before shared latent / MLP.
        if use_dp_ep:
            dist.all_reduce(y)

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)

        if use_dp_ep:
            y = y[:, start_pos:end_pos, :]
        return y

    block_cls.__init__ = ep_init
    block_cls.moe_infer = ep_moe_infer
    block_cls.forward = ep_forward
    block_cls._msmodelslim_ep_patched = True  # type: ignore[attr-defined]
    return True


def apply_kimi_k3_ep_patches(model_path: Optional[str] = None) -> int:
    """Patch loaded / preloadable ``KimiSparseMoeBlock`` (old or new modeling file).

    Supports both ``modeling_kimi`` (legacy) and ``modeling_kimi_linear`` (current).

    Returns:
        Number of modeling modules newly patched.
    """
    patched = 0
    for modeling in _iter_modeling_kimi_modules(model_path):
        mod_id = id(modeling)
        if mod_id in _EP_PATCHED_MODULES:
            continue
        if _patch_kimi_sparse_moe_block(modeling):
            patched += 1
            get_logger().info(
                "Patched %s.KimiSparseMoeBlock for Expert Parallelism.",
                getattr(modeling, "__name__", str(modeling)),
            )
        _EP_PATCHED_MODULES.add(mod_id)
    return patched


def count_local_experts(moe: nn.Module) -> int:
    """Helper for tests / diagnostics."""
    experts = getattr(moe, "experts", None)
    if experts is None:
        return 0
    return sum(1 for e in experts if e is not None)
