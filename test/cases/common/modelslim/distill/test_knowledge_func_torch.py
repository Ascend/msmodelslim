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
# pylint: disable=consider-using-from-import

import pytest
import torch

import torch.nn as nn  # noqa: F401
from ascend_utils.pytorch.knowledge_distill.distill_losses_func_torch import (
    update_logits_by_temperature_pt,
    KDMse,
    KDCrossEntropy,
    HardKDCrossEntropy,
    HiddenMse,
    MMD,
    DISTILL_LOSS_FUNC_TORCH,
)


def test_update_logits_by_temperature_pt_given_tensor_temp_when_called_then_scaled():
    logits_s = torch.randn(2, 10)
    logits_t = torch.randn(2, 10)
    temperature = torch.tensor([2.0, 1.0])
    scaled_s, scaled_t = update_logits_by_temperature_pt(logits_s, logits_t, temperature)
    assert scaled_s.shape == logits_s.shape
    assert scaled_t.shape == logits_t.shape


def test_update_logits_by_temperature_pt_given_scalar_temp_when_called_then_scaled():
    logits_s = torch.randn(2, 10)
    logits_t = torch.randn(2, 10)
    temperature = 2.0
    scaled_s, scaled_t = update_logits_by_temperature_pt(logits_s, logits_t, temperature)
    assert scaled_s.shape == logits_s.shape
    assert scaled_t.shape == logits_t.shape


def test_KDMse_given_valid_logits_and_temp_when_forward_then_loss_computed():
    criterion = KDMse()
    logits_s = torch.randn(2, 10)
    logits_t = torch.randn(2, 10)
    loss = criterion(logits_s, logits_t, temperature=2)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_KDCrossEntropy_given_valid_logits_when_forward_then_loss_computed():
    criterion = KDCrossEntropy()
    logits_s = torch.randn(2, 10)
    logits_t = torch.randn(2, 10)
    loss = criterion(logits_s, logits_t, temperature=1)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_HardKDCrossEntropy_given_valid_logits_when_forward_then_loss_computed():
    criterion = HardKDCrossEntropy()
    logits_s = torch.randn(2, 10)
    logits_t = torch.randn(2, 10)
    loss = criterion(logits_s, logits_t)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_HardKDCrossEntropy_given_invalid_logits_when_forward_then_raise_error():
    criterion = HardKDCrossEntropy()
    logits_s = torch.randn(2, 10)
    logits_t = torch.randn(3, 5)  # mismatched shape
    with pytest.raises(Exception):
        criterion(logits_s, logits_t)


def test_HiddenMse_given_valid_states_when_forward_then_loss_computed():
    criterion = HiddenMse()
    state_s = torch.randn(2, 5, 10)
    state_t = torch.randn(2, 5, 10)
    loss = criterion(state_s, state_t)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_HiddenMse_given_state_and_mask_when_forward_then_masked_loss_computed():
    criterion = HiddenMse()
    state_s = torch.randn(2, 5, 10)
    state_t = torch.randn(2, 5, 10)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
    loss = criterion(state_s, state_t, mask=mask)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_HiddenMse_given_invalid_mask_when_forward_then_raise_error():
    criterion = HiddenMse()
    state_s = torch.randn(2, 5, 10)
    state_t = torch.randn(2, 5, 10)
    mask = torch.randn(2, 6)
    with pytest.raises(Exception):
        criterion(state_s, state_t, mask=mask)


def test_MMD_given_valid_states_when_forward_then_loss_computed():
    criterion = MMD(batch_size=2)
    state_s = torch.randn(4, 10)
    state_t = torch.randn(4, 10)
    loss = criterion(state_s, state_t)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_DISTILL_LOSS_FUNC_TORCH_contains_KDCrossEntropy():
    assert "KDCrossEntropy" in DISTILL_LOSS_FUNC_TORCH
    assert isinstance(DISTILL_LOSS_FUNC_TORCH["KDCrossEntropy"](), KDCrossEntropy)
