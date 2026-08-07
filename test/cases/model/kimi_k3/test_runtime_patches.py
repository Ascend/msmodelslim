# -*- coding: UTF-8 -*-
# pylint: disable=no-member
"""Unit tests for msmodelslim.model.kimi_k3.runtime_patches."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from msmodelslim.model.kimi_k3 import runtime_patches as target


@pytest.fixture(autouse=True)
def _clear_patched_modules():
    target._PATCHED_MODULES.clear()
    yield
    target._PATCHED_MODULES.clear()


class _NpuTensor:
    """Minimal stand-in with device.type == 'npu'."""

    def __init__(self):
        self.device = SimpleNamespace(type="npu")


class _CpuTensor:
    def __init__(self):
        self.device = SimpleNamespace(type="cpu")


def _make_fake_modeling(name: str = "modeling_kimi_linear", with_kda_class: bool = True):
    mod = ModuleType(name)
    mod.__name__ = name

    def chunk_kda(*args, **kwargs):
        return ("chunk", args, kwargs)

    def fused_recurrent_kda(*args, **kwargs):
        return ("fused", args, kwargs)

    mod.chunk_kda = chunk_kda
    mod.fused_recurrent_kda = fused_recurrent_kda
    if with_kda_class:

        class KimiDeltaAttention:
            pass

        mod.KimiDeltaAttention = KimiDeltaAttention
    return mod


# ---------------------------------------------------------------------------
# Name / module detectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("", False),
        ("modeling_kimi_linear", True),
        ("modeling_kimi", True),
        ("pkg.modeling_kimi_linear", True),
        ("pkg.modeling_kimi", True),
        ("other", False),
        ("modeling_kimi_extra", False),
    ],
)
def test_is_kimi_text_modeling_name(name, expected):
    assert target._is_kimi_text_modeling_name(name) is expected


@pytest.mark.parametrize(
    "mod_name,expected",
    [
        ("torch.ops", True),
        ("torch.ops.foo", True),
        ("torch._ops", True),
        ("torch._ops.bar", True),
        ("modeling_kimi", False),
        ("", False),
    ],
)
def test_reject_torch_ops_proxy(mod_name, expected):
    mod = SimpleNamespace(__name__=mod_name)
    assert target._reject_torch_ops_proxy(mod) is expected


def test_is_kda_modeling_module_given_none_or_proxy_when_called_then_false():
    assert target._is_kda_modeling_module(None) is False
    assert target._is_kda_modeling_module(SimpleNamespace(__name__="torch.ops.x")) is False


def test_is_kda_modeling_module_given_missing_ops_when_called_then_false():
    mod = SimpleNamespace(__name__="modeling_kimi_linear", chunk_kda=None, fused_recurrent_kda=lambda: None)
    assert target._is_kda_modeling_module(mod) is False


def test_is_kda_modeling_module_given_named_module_when_called_then_true():
    mod = _make_fake_modeling("pkg.modeling_kimi_linear", with_kda_class=False)
    assert target._is_kda_modeling_module(mod) is True


def test_is_kda_modeling_module_given_kda_class_without_name_when_called_then_true():
    mod = _make_fake_modeling("some_random_name", with_kda_class=True)
    assert target._is_kda_modeling_module(mod) is True


def test_is_kda_modeling_module_given_ops_but_no_name_or_class_when_called_then_false():
    mod = ModuleType("anonymous_mod")
    mod.__name__ = "anonymous_mod"
    mod.chunk_kda = lambda *a, **k: None
    mod.fused_recurrent_kda = lambda *a, **k: None
    assert target._is_kda_modeling_module(mod) is False


# ---------------------------------------------------------------------------
# apply_kimi_k3_runtime_patches
# ---------------------------------------------------------------------------


def test_apply_patches_given_no_modeling_when_called_then_skip(monkeypatch):
    warnings = []

    class L:
        def warning(self, msg, *args):
            warnings.append(msg)

        def info(self, msg, *args):
            pass

    monkeypatch.setattr(target, "_resolve_modeling_module", lambda model=None: None)
    monkeypatch.setattr(target, "get_logger", L)
    target.apply_kimi_k3_runtime_patches()
    assert any("skipped" in w for w in warnings)
    assert len(target._PATCHED_MODULES) == 0


def test_apply_patches_given_fake_module_when_called_then_patch_once(monkeypatch):
    mod = _make_fake_modeling("modeling_kimi_linear")
    orig_chunk = mod.chunk_kda
    sys.modules["modeling_kimi_linear"] = mod
    try:
        monkeypatch.setattr(target, "_resolve_modeling_module", lambda model=None: mod)
        target.apply_kimi_k3_runtime_patches()
        assert getattr(mod.chunk_kda, "_msmodelslim_kda_patched", False) is True
        assert getattr(mod.fused_recurrent_kda, "_msmodelslim_kda_patched", False) is True
        assert mod.chunk_kda is not orig_chunk
        assert id(mod) in target._PATCHED_MODULES

        # idempotent
        patched_chunk = mod.chunk_kda
        target.apply_kimi_k3_runtime_patches()
        assert mod.chunk_kda is patched_chunk
    finally:
        sys.modules.pop("modeling_kimi_linear", None)


def test_apply_patches_given_already_patched_ops_when_called_then_skip_setattr(monkeypatch):
    mod = _make_fake_modeling("modeling_kimi")

    def already_patched(*a, **k):
        return "done"

    already_patched._msmodelslim_kda_patched = True
    mod.chunk_kda = already_patched
    mod.fused_recurrent_kda = already_patched
    monkeypatch.setattr(target, "_resolve_modeling_module", lambda model=None: mod)
    target.apply_kimi_k3_runtime_patches()
    assert mod.chunk_kda is already_patched
    assert id(mod) in target._PATCHED_MODULES


def test_resolve_modeling_module_given_sys_modules_when_called_then_find(monkeypatch):
    mod = _make_fake_modeling("pkg.modeling_kimi")
    sys.modules["pkg.modeling_kimi"] = mod
    try:
        found = target._resolve_modeling_module(None)
        assert found is mod
    finally:
        sys.modules.pop("pkg.modeling_kimi", None)


def test_resolve_modeling_module_given_fallback_kda_class_when_called_then_find(monkeypatch):
    # Clear name-based matches; inject only class-based module
    mod = _make_fake_modeling("custom_kimi_mod", with_kda_class=True)
    key = "custom_kimi_mod_for_test"
    sys.modules[key] = mod
    try:
        # Prefer path may not match name; fallback should find via KimiDeltaAttention
        found = target._resolve_modeling_module(None)
        assert found is not None
        assert target._is_kda_modeling_module(found)
    finally:
        sys.modules.pop(key, None)


def test_resolve_modeling_module_given_model_modules_when_called_then_find(monkeypatch):
    fake_mod = _make_fake_modeling("modeling_kimi_linear")

    class FakeLayer(torch.nn.Module):
        pass

    # Bind FakeLayer's module lookup to our fake modeling module
    import inspect

    monkeypatch.setattr(inspect, "getmodule", lambda obj: fake_mod if obj is FakeLayer else None)
    model = torch.nn.Sequential(FakeLayer())
    found = target._resolve_modeling_module(model)
    assert found is fake_mod


def test_resolve_modeling_module_given_language_model_when_modules_miss(monkeypatch):
    fake_mod = _make_fake_modeling("modeling_kimi")

    class Lang(torch.nn.Module):
        pass

    import inspect

    def fake_getmodule(obj):
        if obj is Lang:
            return fake_mod
        return None

    monkeypatch.setattr(inspect, "getmodule", fake_getmodule)

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Lang()

    # Root's modules include Lang; getmodule(Lang)=fake_mod → found via modules loop
    found = target._resolve_modeling_module(Root())
    assert found is fake_mod


# ---------------------------------------------------------------------------
# _wrap_kda_op
# ---------------------------------------------------------------------------


def test_wrap_kda_op_given_cpu_when_called_then_passthrough():
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "orig"

    wrapped = target._wrap_kda_op(original)
    q = _CpuTensor()
    assert wrapped(q=q, k=1) == "orig"
    assert calls and calls[0][1]["q"] is q


def test_wrap_kda_op_given_npu_use_gate_false_when_called_then_original(monkeypatch):
    warnings = []

    class L:
        def warning(self, msg, *args):
            warnings.append(msg)

        def info(self, msg, *args):
            pass

    monkeypatch.setattr(target, "get_logger", L)

    def original(*args, **kwargs):
        return "orig-npu"

    wrapped = target._wrap_kda_op(original)
    q = _NpuTensor()
    assert wrapped(q=q, k=1, v=2, g=3, beta=4, use_gate_in_kernel=False) == "orig-npu"
    assert any("use_gate_in_kernel" in w for w in warnings)


def test_wrap_kda_op_given_npu_safe_gate_when_called_then_naive_lower_bound_neg5(monkeypatch):
    captured = {}

    def fake_naive(**kwargs):
        captured.update(kwargs)
        return "naive"

    monkeypatch.setattr(target, "_run_naive_kda", fake_naive)
    wrapped = target._wrap_kda_op(lambda *a, **k: "orig")
    q = _NpuTensor()
    out = wrapped(
        q=q,
        k="k",
        v="v",
        g="g",
        beta="beta",
        use_gate_in_kernel=True,
        safe_gate=True,
        A_log="A",
    )
    assert out == "naive"
    assert captured["lower_bound"] == -5.0
    assert captured["q"] is q


def test_wrap_kda_op_given_npu_explicit_lower_bound_when_safe_gate_then_keep(monkeypatch):
    captured = {}

    def fake_naive(**kwargs):
        captured.update(kwargs)
        return "naive"

    monkeypatch.setattr(target, "_run_naive_kda", fake_naive)
    wrapped = target._wrap_kda_op(lambda *a, **k: "orig")
    out = wrapped(
        q=_NpuTensor(),
        k="k",
        v="v",
        g="g",
        beta="beta",
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-1.0,
    )
    assert out == "naive"
    assert captured["lower_bound"] == -1.0


def test_wrap_kda_op_given_positional_args_when_npu_then_remap(monkeypatch):
    captured = {}

    def fake_naive(**kwargs):
        captured.update(kwargs)
        return "naive"

    monkeypatch.setattr(target, "_run_naive_kda", fake_naive)
    wrapped = target._wrap_kda_op(lambda *a, **k: "orig")
    q, k, v, g, beta = _NpuTensor(), "k", "v", "g", "beta"
    out = wrapped(q, k, v, g, beta, use_gate_in_kernel=True, safe_gate=False, lower_bound=None)
    assert out == "naive"
    assert captured["q"] is q
    assert captured["k"] == "k"
    assert captured["v"] == "v"
    assert captured["g"] == "g"
    assert captured["beta"] == "beta"
    assert captured["lower_bound"] is None


def test_run_naive_kda_given_flags_when_called_then_delegate(monkeypatch):
    """Cover _run_naive_kda by stubbing fla imports via sys.modules."""
    gate_mod = ModuleType("fla.ops.kda.gate")
    naive_mod = ModuleType("fla.ops.kda.naive")
    calls = {"gate": None, "lower": None, "rec": None}

    def naive_kda_gate(g, A_log, dt_bias, output_dtype=None):
        calls["gate"] = True
        return g

    def naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound=None, output_dtype=None):
        calls["lower"] = lower_bound
        return g

    def naive_recurrent_kda(**kwargs):
        calls["rec"] = kwargs
        return "ok"

    gate_mod.naive_kda_gate = naive_kda_gate
    gate_mod.naive_kda_lowerbound_gate = naive_kda_lowerbound_gate
    naive_mod.naive_recurrent_kda = naive_recurrent_kda

    # Ensure package path exists for import
    for pkg in ("fla", "fla.ops", "fla.ops.kda"):
        if pkg not in sys.modules:
            sys.modules[pkg] = ModuleType(pkg)
    sys.modules["fla.ops.kda.gate"] = gate_mod
    sys.modules["fla.ops.kda.naive"] = naive_mod

    q = torch.randn(1, 2, 4)
    k = torch.randn(1, 2, 4)
    v = torch.randn(1, 2, 4)
    g = torch.randn(1, 2, 4)
    beta = torch.randn(1, 2, 4)
    A_log = torch.randn(4)

    out = target._run_naive_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=None,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
    )
    assert out == "ok"
    assert calls["lower"] == -5.0
    assert calls["rec"] is not None

    out2 = target._run_naive_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=None,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        lower_bound=None,
    )
    assert out2 == "ok"
    assert calls["gate"] is True
