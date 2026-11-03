#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of the License at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import pytest
import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.lossless.processor import (
    LosslessCheckProcessor,
    LosslessCheckProcessorConfig,
)
from msmodelslim.processor.quarot.offline_quarot.quarot import (
    QuaRotProcessorConfig,
)
from msmodelslim.processor.quarot.offline_quarot.quarot_interface import (
    QuaRotInterface,
)
from msmodelslim.utils.exception import UnexpectedError


R90 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])


class _PreRunModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.block.weight.copy_(torch.tensor([[1.0, 0.5], [-0.25, 1.5]]))

    def forward(self, hidden_states):
        return self.block(hidden_states)


class _PreRunAdapter(QuaRotInterface):
    def get_ln_fuse_map(self):
        return {}, {}

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size):
        del block_size
        return (
            [
                self.RotatePair(
                    left_rot={"block": R90},
                    right_rot={"block": R90},
                )
            ],
            [],
        )

    def generate_model_forward(self, model, inputs):
        del model, inputs
        yield


def _processor(model):
    return LosslessCheckProcessor(
        model,
        LosslessCheckProcessorConfig(
            configs=[
                QuaRotProcessorConfig(
                    block_size=-1,
                    export_extra_info=False,
                )
            ],
            metrics=[{"name": "numerical_equivalence"}],
        ),
        _PreRunAdapter(),
    )


def _request(model):
    return BatchProcessRequest(
        name="",
        module=model,
        datas=[((torch.tensor([[0.75, -1.25]]),), {})],
        outputs=None,
    )


def test_pre_run_rotation_is_deferred_until_matching_session_and_checked():
    model = _PreRunModel()
    original_weight = model.block.weight.detach().clone()
    processor = _processor(model)

    processor.pre_run()

    assert torch.equal(model.block.weight, original_weight)

    processor.process(_request(model))
    processor.post_run()

    assert processor.result["passed"] is True
    assert "block" in processor.result["checked_targets"]
    assert not torch.equal(model.block.weight, original_weight)


def test_missing_pre_run_execution_is_detected_against_frozen_plan():
    model = _PreRunModel()
    processor = _processor(model)
    processor.processor.prepare_lossless_check()
    processor.processor._pre_run_pairs[0].left_rot.clear()
    processor.pre_run()

    with pytest.raises(UnexpectedError, match="block"):
        processor.process(_request(model))

    assert processor.result["anomalies"][0]["structure"] == "block"


def test_unobserved_deferred_pre_run_rotation_is_applied_during_post_run():
    model = _PreRunModel()
    original_weight = model.block.weight.detach().clone()
    processor = _processor(model)

    processor.pre_run()

    assert torch.equal(model.block.weight, original_weight)

    processor.post_run()

    assert not torch.equal(model.block.weight, original_weight)
    assert processor.result["passed"] is None
    assert processor.result["skipped_targets"] == ["block"]
