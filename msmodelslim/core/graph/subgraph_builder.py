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

# 由 ``AdapterConfig`` 与 ``nn.Module`` 构建 anti-outlier / TLQ 共用的 ``Subgraph``。

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from torch import nn

from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import get_logger

from .adapter_types import AdapterConfig

if TYPE_CHECKING:
    from msmodelslim.processor.anti_outlier.common.subgraph_type import Subgraph


def _subgraph_type_module():
    """惰性加载，避免 ``core.graph`` 导入时触发 ``processor`` 循环依赖。"""
    from msmodelslim.processor.anti_outlier.common import subgraph_type as module

    return module


def _try_get_submodule(model: nn.Module, name: Optional[str]) -> Optional[nn.Module]:
    """Resolve ``name`` under ``model``; return ``None`` if missing (no raise)."""
    if not name:
        return None
    try:
        return model.get_submodule(name)
    except AttributeError:
        return None


def _non_fusion_smooth_linear_names(adapter_subgraph_type: str, target_names: Sequence[str]) -> List[str]:
    """非融合子图传给 smooth 算法的 linear 名称。"""
    if adapter_subgraph_type == "norm-linear":
        return list(target_names)
    if target_names:
        return [target_names[0]]
    return []


def _resolve_attention_heads(
    model: nn.Module,
    adapter: Optional[Any] = None,
) -> tuple[int, int]:
    """从 adapter 或 ``model.config`` 解析 attention / KV head 数。"""
    get_attn = getattr(adapter, "get_num_attention_heads", None) if adapter else None
    get_kv = getattr(adapter, "get_num_key_value_heads", None) if adapter else None
    if callable(get_attn) and callable(get_kv):
        return int(get_attn()), int(get_kv())

    cfg = getattr(model, "config", None)
    if cfg is None:
        raise UnsupportedError(
            "model must expose config or adapter must implement get_num_attention_heads/get_num_key_value_heads"
        )
    n_attn = None
    for key in ("num_attention_heads", "n_head", "num_heads", "heads_num"):
        n_attn = getattr(cfg, key, None)
        if n_attn is not None:
            break
    if not n_attn:
        raise UnsupportedError("model.config must have num_attention_heads, n_head or num_heads")
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_attn
    return int(n_attn), int(n_kv)


@dataclass
class BuiltSubgraph:
    """``AdapterConfig`` 构建结果：子图对象 + Smooth 目标 Linear 名。"""

    subgraph: Subgraph
    linear_names: List[str]


def build_subgraph_from_adapter(
    model: nn.Module,
    cfg: AdapterConfig,
    *,
    adapter: Optional[Any] = None,
) -> Optional[BuiltSubgraph]:
    """由 adapter 配置构建子图；无法构建时返回 ``None``（与 ``smooth_base`` 警告语义一致）。"""
    if cfg.mapping.source is None and cfg.mapping.targets:
        return _build_non_fusion(model, cfg)
    handler = _BUILDERS.get(cfg.subgraph_type)
    if handler is None:
        raise UnsupportedError(f"unsupported adapter subgraph type: {cfg.subgraph_type!r}")
    return handler(model, cfg, adapter=adapter)


def _build_non_fusion(model: nn.Module, cfg: AdapterConfig) -> Optional[BuiltSubgraph]:
    st = _subgraph_type_module()
    target_modules: List[nn.Module] = []
    for name in cfg.mapping.targets:
        module = _try_get_submodule(model, name)
        if module is None:
            get_logger().warning(
                "Failed to get modules for non-fusion subgraph targets=%s "
                "(missing or invalid path %r); subgraph_type=%s",
                cfg.mapping.targets,
                name,
                cfg.subgraph_type,
            )
            return None
        target_modules.append(module)
    if not target_modules:
        get_logger().warning(
            "No targets specified for non-fusion subgraph: subgraph_type=%s",
            cfg.subgraph_type,
        )
        return None
    subgraph = st.NonFusionSubgraph(
        linears=target_modules,
        linear_names=list(cfg.mapping.targets),
    )
    linear_names = _non_fusion_smooth_linear_names(cfg.subgraph_type, cfg.mapping.targets)
    return BuiltSubgraph(subgraph=subgraph, linear_names=linear_names)


def _build_up_down(
    model: nn.Module,
    cfg: AdapterConfig,
    *,
    adapter: Optional[Any] = None,
) -> Optional[BuiltSubgraph]:
    _ = adapter
    st = _subgraph_type_module()
    up_name = cfg.mapping.source
    if not up_name:
        return None
    up_module = _try_get_submodule(model, up_name)
    down_name = cfg.mapping.targets[0] if cfg.mapping.targets else ""
    down_module = _try_get_submodule(model, down_name) if down_name else None
    if not up_module or not down_module:
        get_logger().warning(
            "Failed to get modules for up-down subgraph: source=%s targets=%s",
            cfg.mapping.source,
            cfg.mapping.targets,
        )
        return None
    subgraph = st.UpDownSubgraph(
        up_proj=up_module,
        down_proj=down_module,
        gate_proj=None,
        up_proj_name=up_name,
        down_proj_name=down_name,
    )
    return BuiltSubgraph(subgraph=subgraph, linear_names=[down_name])


def _build_ov(
    model: nn.Module,
    cfg: AdapterConfig,
    *,
    adapter: Optional[Any] = None,
) -> Optional[BuiltSubgraph]:
    """Build OV subgraph; on failure log and return ``None`` (skip that subgraph).

    Matches anti-outlier / TLQ ``build_subgraph_from_adapter`` contract: one bad OV
    mapping must not abort the whole session. Unexpected errors are logged with type.
    """
    fusion = cfg.fusion
    fusion_flag = fusion is not None and fusion.fusion_type != "none"
    try:
        if fusion_flag:
            return _build_ov_qkv_fusion(model, cfg)
        return _build_ov_standard(model, cfg, adapter=adapter)
    except UnsupportedError as exc:
        get_logger().warning(
            "Skip OV subgraph source=%r targets=%r: %s",
            cfg.mapping.source,
            cfg.mapping.targets,
            exc,
        )
        return None
    except Exception as exc:
        get_logger().error(
            "Error building OV subgraph (type=%s) source=%r targets=%r: %s",
            type(exc).__name__,
            cfg.mapping.source,
            cfg.mapping.targets,
            exc,
        )
        return None


def _kv_fusion_dims(cfg: AdapterConfig) -> Optional[tuple[Any, Any]]:
    """Read KV fusion dims; missing keys → warning and ``None`` (skip subgraph)."""
    fusion = cfg.fusion
    if fusion is None or not fusion.custom_config:
        get_logger().warning(
            "KV fusion missing custom_config; skip OV source=%r targets=%r",
            cfg.mapping.source,
            cfg.mapping.targets,
        )
        return None
    custom = fusion.custom_config
    required = ("qk_nope_head_dim", "v_head_dim")
    missing = [key for key in required if key not in custom]
    if missing:
        get_logger().warning(
            "KV fusion custom_config missing %s; skip OV source=%r targets=%r",
            missing,
            cfg.mapping.source,
            cfg.mapping.targets,
        )
        return None
    return custom["qk_nope_head_dim"], custom["v_head_dim"]


def _build_ov_qkv_fusion(model: nn.Module, cfg: AdapterConfig) -> Optional[BuiltSubgraph]:
    st = _subgraph_type_module()
    from msmodelslim.processor.anti_outlier.common import (
        VirtualVModuleFromKVFused,
        VirtualVModuleFromQKVFused,
    )

    v_name = cfg.mapping.source
    o_name = cfg.mapping.targets[0] if cfg.mapping.targets else ""
    v_module = _try_get_submodule(model, v_name)
    o_module = _try_get_submodule(model, o_name)
    fusion = cfg.fusion
    extra_config = getattr(cfg, "extra_config", None)

    if v_module is None:
        get_logger().warning(
            "V module path %r not found, skipping QKV fusion (targets=%s)",
            v_name,
            cfg.mapping.targets,
        )
        return None
    if not isinstance(v_module, nn.Linear):
        get_logger().warning("V module %s is not Linear type, skipping QKV fusion", v_name)
        return None
    if o_module is None:
        get_logger().warning("O module %s not found, skipping QKV fusion", o_name)
        return None
    if fusion is None:
        return None

    if fusion.fusion_type == "kv":
        dims = _kv_fusion_dims(cfg)
        if dims is None:
            return None
        qk_nope_head_dim, v_head_dim = dims
        virtual_v_module = VirtualVModuleFromKVFused(
            v_module,
            num_attention_heads=fusion.num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
        )
    elif fusion.fusion_type == "qkv":
        virtual_v_module = VirtualVModuleFromQKVFused(
            v_module,
            num_attention_heads=fusion.num_attention_heads,
            num_key_value_heads=fusion.num_key_value_heads,
        )
    else:
        raise UnsupportedError(f"Unsupported fusion type: {fusion.fusion_type}")

    subgraph = st.OVSubgraph(
        o_proj=o_module,
        v_proj=virtual_v_module,
        num_attention_heads=fusion.num_attention_heads,
        key_value_heads=fusion.num_key_value_heads,
        extra_config=extra_config,
        o_proj_name=o_name,
        v_proj_name=v_name,
    )
    return BuiltSubgraph(subgraph=subgraph, linear_names=[o_name])


def _build_ov_standard(
    model: nn.Module,
    cfg: AdapterConfig,
    *,
    adapter: Optional[Any] = None,
) -> Optional[BuiltSubgraph]:
    st = _subgraph_type_module()
    v_name = cfg.mapping.source
    o_name = cfg.mapping.targets[0] if cfg.mapping.targets else ""
    v_module = _try_get_submodule(model, v_name)
    o_module = _try_get_submodule(model, o_name)
    extra_config = getattr(cfg, "extra_config", None)

    if v_module is None:
        get_logger().warning(
            "V module path %r not found, skipping standard OV (targets=%s)",
            v_name,
            cfg.mapping.targets,
        )
        return None
    if not isinstance(v_module, nn.Linear):
        get_logger().warning("V module %s is not Linear type, skipping standard OV", v_name)
        return None
    if o_module is None:
        get_logger().warning("O module %s not found, skipping standard OV", o_name)
        return None

    num_attention_heads, num_key_value_heads = _resolve_attention_heads(model, adapter)
    subgraph = st.OVSubgraph(
        o_proj=o_module,
        v_proj=v_module,
        num_attention_heads=num_attention_heads,
        key_value_heads=num_key_value_heads,
        extra_config=extra_config,
        o_proj_name=o_name,
        v_proj_name=v_name,
    )
    return BuiltSubgraph(subgraph=subgraph, linear_names=[o_name])


def _build_norm_linear(
    model: nn.Module,
    cfg: AdapterConfig,
    *,
    adapter: Optional[Any] = None,
) -> Optional[BuiltSubgraph]:
    _ = adapter
    st = _subgraph_type_module()
    source_name = cfg.mapping.source
    if not source_name:
        return None
    source_module = _try_get_submodule(model, source_name)
    target_modules: List[nn.Module] = []
    for name in cfg.mapping.targets:
        module = _try_get_submodule(model, name)
        if module is None:
            get_logger().warning(
                "Failed to get modules for norm-linear subgraph: source=%s targets=%s (missing %r)",
                source_name,
                cfg.mapping.targets,
                name,
            )
            return None
        target_modules.append(module)
    target_names = list(cfg.mapping.targets)

    if not source_module or not target_modules:
        get_logger().warning("Failed to get modules for norm-linear subgraph: %s", source_name)
        return None

    subgraph = st.NormLinearSubgraph(source_module, target_modules, linear_names=target_names)
    return BuiltSubgraph(subgraph=subgraph, linear_names=target_names)


def _build_linear_linear(
    model: nn.Module,
    cfg: AdapterConfig,
    *,
    adapter: Optional[Any] = None,
) -> Optional[BuiltSubgraph]:
    _ = adapter
    st = _subgraph_type_module()
    source_name = cfg.mapping.source
    if not source_name:
        return None
    source_module = _try_get_submodule(model, source_name)
    target_name = cfg.mapping.targets[0] if cfg.mapping.targets else ""
    target_module = _try_get_submodule(model, target_name) if target_name else None

    if not source_module or not target_module:
        get_logger().warning(
            "Failed to get modules for linear-linear subgraph: source=%s targets=%s",
            source_name,
            cfg.mapping.targets,
        )
        return None

    subgraph = st.LinearLinearSubgraph(
        source_module,
        target_module,
        linear1_name=source_name,
        linear2_name=target_name,
    )
    return BuiltSubgraph(subgraph=subgraph, linear_names=[target_name])


_BUILDERS = {
    "up-down": _build_up_down,
    "ov": _build_ov,
    "norm-linear": _build_norm_linear,
    "linear-linear": _build_linear_linear,
}
