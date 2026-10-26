#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=use-implicit-booleaness-not-comparison

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

import pytest

from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig


def _make_inter_config():
    return {
        "t_module": "teacher.blocks.11.output",
        "s_module": "student.blocks.5.output",
        "t_output_idx": 0,
        "s_output_idx": 0,
        "loss_func": [{"func_name": "HiddenMse", "func_weight": 1}],
        "shape": [8],
    }


def test_KnowledgeDistillConfig_default_state_is_neutral():
    config = KnowledgeDistillConfig()

    assert config.hard_label_loss_weight == 0.5
    assert config.output_replace_idx == 0
    assert config.train_teacher is False
    assert config.inter_matches == []
    assert config.output_matches == []
    assert config.custom_loss_func == {}


def test_set_hard_label_updates_weight_and_index_and_returns_self():
    config = KnowledgeDistillConfig()

    result = config.set_hard_label(0.3, 2)

    assert result is config
    assert config.hard_label_loss_weight == 0.3
    assert config.output_replace_idx == 2


@pytest.mark.parametrize("weight", [0.0, 1.0, -0.5])
def test_set_hard_label_rejects_weight_out_of_open_unit_interval(weight):
    config = KnowledgeDistillConfig()

    with pytest.raises(ValueError):
        _ = config.set_hard_label(weight, 0)


def test_add_inter_soft_label_appends_config_and_fills_loss_func_defaults():
    config = KnowledgeDistillConfig()

    result = config.add_inter_soft_label(_make_inter_config())

    assert result is config
    assert len(config.inter_matches) == 1
    loss_func = config.inter_matches[0]["loss_func"][0]
    assert loss_func["temperature"] is None
    assert loss_func["func_param"] == []
    assert config.get_soft_label_shape() == {
        "teacher.blocks.11.output": [8],
        "student.blocks.5.output": [8],
    }


def test_add_inter_soft_label_raises_when_required_key_missing():
    config = KnowledgeDistillConfig()

    with pytest.raises(ValueError):
        _ = config.add_inter_soft_label({"t_module": "teacher.blocks.11.output"})


def test_add_output_soft_label_appends_config_and_returns_self():
    config = KnowledgeDistillConfig()
    output_config = {
        "t_output_idx": 0,
        "s_output_idx": 0,
        "loss_func": [{"func_name": "HiddenMse", "func_weight": 2}],
    }

    result = config.add_output_soft_label(output_config)

    assert result is config
    assert config.output_matches == [output_config]


def test_check_loss_func_name_raises_when_func_name_unknown():
    matches = [{"loss_func": [{"func_name": "Nope", "func_weight": 1}]}]

    with pytest.raises(ValueError):
        KnowledgeDistillConfig.check_loss_func_name(matches, {"HiddenMse": None})
