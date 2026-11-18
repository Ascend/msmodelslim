#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from msmodelslim.core.train.optimizer import (
    ParamGroupResult,
    QParamGroupBuilder,
    RelativeAlphaAdamW,
    matches_qparam,
)
from msmodelslim.utils.exception import SchemaValidateError

DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.optimizer.get_logger", lambda: fake_logger)
    return fake_logger


def _qparam_group(param, lr, **extra):
    group = {
        "params": [param],
        "lr": lr,
        "betas": DEFAULT_BETAS,
        "eps": DEFAULT_EPS,
        "weight_decay": 0.0,
        "qat_group": "qparam",
    }
    group.update(extra)
    return group


def _log_space_expected(alpha0, grads, lr, betas=DEFAULT_BETAS, eps=DEFAULT_EPS):
    """按 log-space 公式 alpha <- alpha * exp(-lr * m_hat / (sqrt(v_hat) + eps)) 独立重算期望值。"""
    beta1, beta2 = betas
    alpha = float(alpha0)
    exp_avg = 0.0
    exp_avg_sq = 0.0
    for step, grad in enumerate(grads, start=1):
        exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denom = math.sqrt(exp_avg_sq / bias_correction2) + eps
        direction = (exp_avg / bias_correction1) / denom
        alpha *= math.exp(-lr * direction)
    return alpha


class _QatModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=True)
        self.fc.register_parameter("weight_clip_val", nn.Parameter(torch.full((1,), 0.5)))
        self.weight_clip_val = nn.Parameter(torch.full((1,), 0.25))


class _PlainModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=True)


class _NoBiasQatModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        self.fc.register_parameter("weight_clip_val", nn.Parameter(torch.full((1,), 0.5)))


# ---------- matches_qparam ----------


def test_matches_qparam_returns_true_when_name_exactly_equals():
    assert matches_qparam("weight_clip_val", ["bias", "weight_clip_val"]) is True


def test_matches_qparam_returns_true_when_name_ends_with_dotted_suffix():
    assert matches_qparam("fc.weight_clip_val", ["weight_clip_val"]) is True


def test_matches_qparam_returns_false_when_name_not_in_qparam_names():
    assert matches_qparam("fc.weight", ["weight_clip_val", "alpha"]) is False


def test_matches_qparam_returns_false_when_suffix_lacks_dot_separator():
    assert matches_qparam("dummy_weight_clip_val", ["weight_clip_val"]) is False


def test_matches_qparam_returns_false_when_qparam_names_empty():
    assert matches_qparam("weight_clip_val", []) is False


# ---------- ParamGroupResult ----------


def test_ParamGroupResult_defaults_when_constructed_without_args():
    result = ParamGroupResult()
    assert result.groups == []
    assert result.n_decay == 0
    assert result.n_no_decay == 0
    assert result.n_qparam == 0


def test_ParamGroupResult_stores_fields_when_constructed_with_values():
    groups = [{"params": [], "lr": 0.1}]
    result = ParamGroupResult(groups=groups, n_decay=1, n_no_decay=2, n_qparam=3)
    assert result.groups is groups
    assert (result.n_decay, result.n_no_decay, result.n_qparam) == (1, 2, 3)
    with pytest.raises(FrozenInstanceError):
        result.n_decay = 99


# ---------- QParamGroupBuilder ----------


def test_QParamGroupBuilder_matches_uses_qparam_names_when_configured():
    builder = QParamGroupBuilder(["alpha", "weight_clip_val"], 1.0)
    assert builder.matches("blocks.0.alpha") is True
    assert builder.matches("fc.weight_clip_val") is True
    assert builder.matches("fc.weight") is False


def test_QParamGroupBuilder_build_partitions_params_when_model_has_qparams(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    model = _QatModule()
    params = dict(model.named_parameters())
    builder = QParamGroupBuilder(["weight_clip_val"], 0.5, alpha_min_ratio=0.25, alpha_max_ratio=4.0)
    result = builder.build(model, decay_names=("fc.weight",), base_lr=0.1, weight_decay=0.01)

    assert result.n_decay == 1
    assert result.n_no_decay == 1
    assert result.n_qparam == 2
    assert len(result.groups) == 3
    decay_group, no_decay_group, qparam_group = result.groups

    assert decay_group["params"] == [params["fc.weight"]]
    assert decay_group["lr"] == 0.1
    assert decay_group["weight_decay"] == 0.01
    assert decay_group["qat_group"] == "weight"

    assert no_decay_group["params"] == [params["fc.bias"]]
    assert no_decay_group["lr"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert no_decay_group["qat_group"] == "weight"

    assert set(qparam_group["params"]) == {params["weight_clip_val"], params["fc.weight_clip_val"]}
    assert qparam_group["lr"] == pytest.approx(0.1 * 0.5)
    assert qparam_group["weight_decay"] == 0.0
    assert qparam_group["qat_group"] == "qparam"
    assert qparam_group["alpha_min_ratio"] == 0.25
    assert qparam_group["alpha_max_ratio"] == 4.0
    fake_logger.warning.assert_not_called()


def test_QParamGroupBuilder_build_skips_frozen_params_when_requires_grad_false(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    model = _QatModule()
    params = dict(model.named_parameters())
    params["fc.weight"].requires_grad_(False)
    params["weight_clip_val"].requires_grad_(False)
    builder = QParamGroupBuilder(["weight_clip_val"], 1.0)
    result = builder.build(model, decay_names=("fc.weight",), base_lr=0.1, weight_decay=0.01)

    assert result.n_decay == 0
    assert result.n_no_decay == 1
    assert result.n_qparam == 1
    assert [g["qat_group"] for g in result.groups] == ["weight", "qparam"]
    assert result.groups[0]["params"] == [params["fc.bias"]]
    assert result.groups[1]["params"] == [params["fc.weight_clip_val"]]
    fake_logger.warning.assert_not_called()


def test_QParamGroupBuilder_build_omits_no_decay_group_when_bias_absent(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    model = _NoBiasQatModule()
    params = dict(model.named_parameters())
    builder = QParamGroupBuilder(["weight_clip_val"], 2.0)
    result = builder.build(model, decay_names=("fc.weight",), base_lr=0.01, weight_decay=0.1)

    assert result.n_decay == 1
    assert result.n_no_decay == 0
    assert result.n_qparam == 1
    assert [g["qat_group"] for g in result.groups] == ["weight", "qparam"]
    assert result.groups[0]["params"] == [params["fc.weight"]]
    assert result.groups[1]["params"] == [params["fc.weight_clip_val"]]
    assert result.groups[1]["lr"] == pytest.approx(0.01 * 2.0)
    fake_logger.warning.assert_not_called()


def test_QParamGroupBuilder_build_warns_when_no_qparam_matched(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    model = _PlainModule()
    builder = QParamGroupBuilder(["weight_clip_val"], 1.0)
    result = builder.build(model, decay_names=("fc.weight",), base_lr=0.01, weight_decay=0.0)

    assert result.n_qparam == 0
    assert result.n_decay == 1
    assert result.n_no_decay == 1
    assert [g["qat_group"] for g in result.groups] == ["weight", "weight"]
    fake_logger.warning.assert_called_once()
    assert "no trainable qparam matched" in fake_logger.warning.call_args.args[0]
    assert fake_logger.warning.call_args.args[1] == ("weight_clip_val",)


def test_QParamGroupBuilder_build_raises_when_no_trainable_params(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    model = _PlainModule()
    for param in model.parameters():
        param.requires_grad_(False)
    builder = QParamGroupBuilder(["weight_clip_val"], 1.0)
    with pytest.raises(SchemaValidateError, match="no trainable parameters"):
        builder.build(model, decay_names=("fc.weight",), base_lr=0.01, weight_decay=0.0)
    fake_logger.warning.assert_called_once()


# ---------- RelativeAlphaAdamW ----------


def test_RelativeAlphaAdamW_updates_alpha_in_log_space_when_within_ratio_bounds():
    alpha = nn.Parameter(torch.tensor([1.0]))
    ref = torch.tensor([1.0])
    opt = RelativeAlphaAdamW([_qparam_group(alpha, lr=0.1)], alpha_ref={id(alpha): ref})
    alpha.grad = torch.tensor([1.0])
    opt.step()

    expected = _log_space_expected(1.0, [1.0], lr=0.1)
    assert alpha.item() == pytest.approx(expected, rel=1e-6)
    assert 0.5 * ref.item() <= alpha.item() <= 2.0 * ref.item()


def test_RelativeAlphaAdamW_clamps_alpha_when_update_exceeds_ratio_bounds():
    raw_lower = _log_space_expected(1.0, [1.0], lr=10.0)
    raw_upper = _log_space_expected(1.0, [-1.0], lr=10.0)
    assert raw_lower < 0.5 and raw_upper > 2.0

    alpha = nn.Parameter(torch.tensor([1.0, 1.0]))
    ref = torch.tensor([1.0, 1.0])
    opt = RelativeAlphaAdamW([_qparam_group(alpha, lr=10.0)], alpha_ref={id(alpha): ref})
    alpha.grad = torch.tensor([1.0, -1.0])
    opt.step()

    assert torch.allclose(alpha.detach(), torch.tensor([0.5, 2.0]), atol=1e-7)


def test_RelativeAlphaAdamW_uses_instance_ratios_when_group_has_no_override():
    alpha = nn.Parameter(torch.tensor([1.0]))
    ref = torch.tensor([1.0])
    opt = RelativeAlphaAdamW(
        [_qparam_group(alpha, lr=10.0)],
        alpha_ref={id(alpha): ref},
        alpha_min_ratio=0.25,
        alpha_max_ratio=4.0,
    )
    alpha.grad = torch.tensor([1.0])
    opt.step()
    assert alpha.item() == pytest.approx(0.25)


def test_RelativeAlphaAdamW_uses_group_ratios_when_group_overrides_instance():
    alpha = nn.Parameter(torch.tensor([1.0]))
    ref = torch.tensor([1.0])
    opt = RelativeAlphaAdamW(
        [_qparam_group(alpha, lr=10.0, alpha_min_ratio=0.8, alpha_max_ratio=1.25)],
        alpha_ref={id(alpha): ref},
        alpha_min_ratio=0.5,
        alpha_max_ratio=2.0,
    )
    alpha.grad = torch.tensor([1.0])
    opt.step()
    assert alpha.item() == pytest.approx(0.8)


def test_RelativeAlphaAdamW_initializes_alpha_ref_lazily_when_ref_missing():
    alpha = nn.Parameter(torch.tensor([1.0]))
    opt = RelativeAlphaAdamW([_qparam_group(alpha, lr=0.1)])

    alpha.grad = torch.tensor([1.0])
    opt.step()
    expected_after_one = _log_space_expected(1.0, [1.0], lr=0.1)
    state = opt.state[alpha]
    assert state["alpha_ref"] is not None
    assert state["alpha_ref"].item() == pytest.approx(expected_after_one, rel=1e-6)
    assert state["alpha_master"].item() == pytest.approx(expected_after_one, rel=1e-6)
    assert alpha.item() == pytest.approx(expected_after_one, rel=1e-6)

    alpha.grad = torch.tensor([1.0])
    opt.step()
    expected_after_two = _log_space_expected(1.0, [1.0, 1.0], lr=0.1)
    assert opt.state[alpha]["alpha_ref"].item() == pytest.approx(expected_after_one, rel=1e-6)
    assert opt.state[alpha]["alpha_master"].item() == pytest.approx(expected_after_two, rel=1e-6)
    assert alpha.item() == pytest.approx(expected_after_two, rel=1e-6)


def test_RelativeAlphaAdamW_skips_param_when_grad_is_none():
    alpha = nn.Parameter(torch.tensor([1.0]))
    opt = RelativeAlphaAdamW(
        [_qparam_group(alpha, lr=0.1)],
        alpha_ref={id(alpha): torch.tensor([1.0])},
    )
    loss = opt.step()
    assert loss is None
    assert alpha.item() == 1.0
    assert not opt.state.get(alpha)


def test_RelativeAlphaAdamW_invokes_closure_once_when_closure_given():
    alpha = nn.Parameter(torch.tensor([1.0]))
    opt = RelativeAlphaAdamW([_qparam_group(alpha, lr=0.1)])
    closure = MagicMock(return_value=torch.tensor(1.25))
    loss = opt.step(closure)
    closure.assert_called_once()
    assert torch.allclose(loss, torch.tensor(1.25))


def test_RelativeAlphaAdamW_matches_stock_adamw_when_all_groups_are_weight():
    torch.manual_seed(2024)
    inputs = torch.randn(3, 4)
    expected_model = nn.Linear(4, 3, bias=True)
    actual_model = nn.Linear(4, 3, bias=True)
    actual_model.load_state_dict(expected_model.state_dict())

    opt_expected = torch.optim.AdamW(
        expected_model.parameters(), lr=0.05, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01
    )
    opt_actual = RelativeAlphaAdamW(actual_model.parameters(), lr=0.05, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)

    for _ in range(2):
        expected_model.zero_grad()
        expected_model(inputs).pow(2).sum().backward()
        opt_expected.step()

        actual_model.zero_grad()
        actual_model(inputs).pow(2).sum().backward()
        opt_actual.step()

    for p_expected, p_actual in zip(expected_model.parameters(), actual_model.parameters()):
        assert torch.allclose(p_actual, p_expected, rtol=0.0, atol=1e-12)


def test_RelativeAlphaAdamW_defers_qparam_group_when_mixed_groups():
    weight = nn.Parameter(torch.ones(2, 2))
    alpha = nn.Parameter(torch.tensor([1.0]))
    shared = {"betas": DEFAULT_BETAS, "eps": DEFAULT_EPS, "weight_decay": 0.0}
    opt = RelativeAlphaAdamW(
        [
            {"params": [weight], "lr": 0.01, "qat_group": "weight", **shared},
            {"params": [alpha], "lr": 0.1, "qat_group": "qparam", **shared},
        ],
        alpha_ref={id(alpha): torch.tensor([1.0])},
    )
    (weight.sum() + alpha.sum()).backward()
    opt.step()

    assert alpha.grad is None
    assert weight.grad is not None
    expected_alpha = _log_space_expected(1.0, [1.0], lr=0.1)
    assert alpha.item() == pytest.approx(expected_alpha, rel=1e-6)
    expected_weight = 1.0 - 0.01 / (1.0 + DEFAULT_EPS)
    assert torch.allclose(weight.detach(), torch.full((2, 2), expected_weight), atol=1e-6)


def test_RelativeAlphaAdamW_preserves_param_dtype_when_alpha_is_half_precision():
    alpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    opt = RelativeAlphaAdamW(
        [_qparam_group(alpha, lr=0.1)],
        alpha_ref={id(alpha): torch.tensor([1.0])},
    )
    alpha.grad = torch.tensor([1.0], dtype=torch.float16)
    opt.step()

    expected = _log_space_expected(1.0, [1.0], lr=0.1)
    assert alpha.dtype == torch.float16
    assert alpha.item() == pytest.approx(expected, abs=1e-3)
