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

# Runtime monkey-patches for Kimi-K3 quantization on Ascend NPU.
#
# Do NOT edit files under the model weight directory. FLA Triton KDA kernels
# timeout on NPU; replace chunk_kda / fused_recurrent_kda with a pure-PyTorch path.

from __future__ import annotations

import inspect
import sys
from typing import Any, Optional

import torch
import torch.nn.functional as F

from msmodelslim.utils.logging import get_logger

_PATCHED_MODULES = set()

# Match both legacy modeling_kimi and current modeling_kimi_linear module names.
_KIMI_TEXT_MODELING_SUFFIXES = ("modeling_kimi_linear", "modeling_kimi")


def _is_kimi_text_modeling_name(name: str) -> bool:
    """True if ``name`` is (or ends with) a known Kimi text modeling module."""
    if not name:
        return False
    leaf = name.rsplit(".", 1)[-1]
    return leaf in _KIMI_TEXT_MODELING_SUFFIXES


def _reject_torch_ops_proxy(mod: Any) -> bool:
    """``torch.ops`` returns truthy objects for arbitrary attribute names."""
    name = getattr(mod, "__name__", "") or ""
    return (
        name.startswith("torch.ops")
        or name.startswith("torch._ops")
        or name
        in (
            "torch.ops",
            "torch._ops",
        )
    )


def _is_kda_modeling_module(mod: Any) -> bool:
    """True for a real Kimi text modeling module that exposes KDA ops."""
    if mod is None or _reject_torch_ops_proxy(mod):
        return False
    chunk = getattr(mod, "chunk_kda", None)
    fused = getattr(mod, "fused_recurrent_kda", None)
    if not callable(chunk) or not callable(fused):
        return False
    # Prefer modules that look like Kimi text modeling (class or module name).
    name = getattr(mod, "__name__", "") or ""
    if _is_kimi_text_modeling_name(name):
        return True
    delta = getattr(mod, "KimiDeltaAttention", None)
    return isinstance(delta, type)


def _run_naive_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    lower_bound: Optional[float],
):
    from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
    from fla.ops.kda.naive import naive_recurrent_kda

    q_in, k_in = q, k
    if use_qk_l2norm_in_kernel:
        q_in = F.normalize(q, p=2, dim=-1)
        k_in = F.normalize(k, p=2, dim=-1)

    if lower_bound is not None:
        g_decay = naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound=lower_bound, output_dtype=g.dtype)
    else:
        g_decay = naive_kda_gate(g, A_log, dt_bias, output_dtype=g.dtype)

    beta_act = torch.sigmoid(beta) if use_beta_sigmoid_in_kernel else beta
    return naive_recurrent_kda(
        q=q_in,
        k=k_in,
        v=v,
        g=g_decay,
        beta=beta_act,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def _wrap_kda_op(original_op):
    def wrapped(*args, **kwargs):
        q = kwargs["q"] if "q" in kwargs else args[0]
        if getattr(q.device, "type", "") != "npu":
            return original_op(*args, **kwargs)

        if args and "q" not in kwargs:
            for i, name in enumerate(("q", "k", "v", "g", "beta")):
                if i < len(args) and name not in kwargs:
                    kwargs[name] = args[i]

        if not kwargs.get("use_gate_in_kernel", False):
            get_logger().warning("naive KDA NPU fallback expects use_gate_in_kernel=True; calling original op.")
            return original_op(*args, **kwargs)

        lower_bound = kwargs.get("lower_bound", None)
        if kwargs.get("safe_gate", False) and lower_bound is None:
            lower_bound = -5.0

        return _run_naive_kda(
            q=kwargs["q"],
            k=kwargs["k"],
            v=kwargs["v"],
            g=kwargs["g"],
            beta=kwargs["beta"],
            A_log=kwargs.get("A_log"),
            dt_bias=kwargs.get("dt_bias"),
            initial_state=kwargs.get("initial_state"),
            output_final_state=kwargs.get("output_final_state", False),
            use_qk_l2norm_in_kernel=kwargs.get("use_qk_l2norm_in_kernel", False),
            use_beta_sigmoid_in_kernel=kwargs.get("use_beta_sigmoid_in_kernel", False),
            lower_bound=lower_bound,
        )

    wrapped._msmodelslim_kda_patched = True  # type: ignore[attr-defined]
    return wrapped


def _resolve_modeling_module(model: Optional[torch.nn.Module] = None) -> Any:
    if model is not None:
        for module in model.modules():
            mod = inspect.getmodule(type(module))
            if _is_kda_modeling_module(mod):
                return mod
        language_model = getattr(model, "language_model", None)
        if language_model is not None:
            mod = inspect.getmodule(type(language_model))
            if _is_kda_modeling_module(mod):
                return mod

    # Prefer explicit module-name matches (legacy + current).
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if _is_kimi_text_modeling_name(name) and _is_kda_modeling_module(mod):
            return mod

    # Fallback: any imported module that looks like Kimi text modeling + KDA.
    for _name, mod in list(sys.modules.items()):
        if _is_kda_modeling_module(mod):
            return mod
    return None


def apply_kimi_k3_runtime_patches(model: Optional[torch.nn.Module] = None) -> None:
    """Patch imported Kimi text-modeling KDA ops for Ascend NPU.

    Works with both ``modeling_kimi`` (legacy) and ``modeling_kimi_linear`` (current).
    """
    modeling = _resolve_modeling_module(model)
    if modeling is None:
        get_logger().warning("Kimi-K3 runtime patches skipped; modeling_kimi / modeling_kimi_linear not found.")
        return

    mod_id = id(modeling)
    if mod_id in _PATCHED_MODULES:
        return

    for name in ("chunk_kda", "fused_recurrent_kda"):
        op = getattr(modeling, name, None)
        if op is None or getattr(op, "_msmodelslim_kda_patched", False):
            continue
        setattr(modeling, name, _wrap_kda_op(op))
        get_logger().info("Patched %s.%s for naive KDA on NPU.", modeling.__name__, name)

    _PATCHED_MODULES.add(mod_id)
