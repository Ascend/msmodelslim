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
# pylint: disable=consider-using-from-import,useless-object-inheritance

import pytest
import torch  # Not using here, just import before mindspore, or will throw error in later tests...  # noqa: F401
import mindspore.nn as nn
from mindspore.nn.cell import Cell

from ascend_utils.mindspore.knowledge_distill.distill_losses_func_ms import DISTILL_LOSS_FUNC_MS
from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig


class TestPruneKnowledgeDistillConfig(object):
    def test_distill_config_given_valid_when_any_then_pass(self):
        distill_config = KnowledgeDistillConfig()
        distill_config.set_hard_label(0.5, 0).add_inter_soft_label(
            {
                "t_module": "uniter.encoder.encoder.blocks.11.output",
                "s_module": "uniter.encoder.encoder.blocks.5.output",
                "t_output_idx": 0,
                "s_output_idx": 0,
                "loss_func": [{"func_name": "KDCrossEntropy", "func_weight": 1}],
                "shape": [2048],
            }
        ).add_output_soft_label(
            {
                "t_output_idx": 0,
                "s_output_idx": 0,
                "loss_func": [{"func_name": "KDCrossEntropy", "func_weight": 1, "temperature": 1, "func_param": []}],
            }
        ).set_teacher_train().add_custom_loss_func("test_loss_function", nn.Cell())

    def test_distill_config_given_invalid_when_any_then_error(self):
        distill_config = KnowledgeDistillConfig()
        with pytest.raises(TypeError):
            distill_config.set_hard_label("1", 1)
        with pytest.raises(ValueError):
            distill_config.add_inter_soft_label({"t_module": "uniter.encoder.encoder.blocks.11.output"})
        with pytest.raises(ValueError):
            distill_config.add_inter_soft_label(
                {
                    "t_module": "uniter.encoder.encoder.blocks.11.output",
                    "s_module": "uniter.encoder.encoder.blocks.5.output",
                    "t_output_idx": 0,
                    "s_output_idx": 0,
                    "loss_func": [{}],
                    "shape": [2048],
                }
            )
        with pytest.raises(TypeError):
            distill_config.add_inter_soft_label(
                {
                    "t_module": "uniter.encoder.encoder.blocks.11.output",
                    "s_module": "uniter.encoder.encoder.blocks.5.output",
                    "t_output_idx": 0,
                    "s_output_idx": 0,
                    "loss_func": [{"func_name": "KDCrossEntropy", "func_weight": 1, "func_param": 1}],
                    "shape": [2048],
                }
            )
        with pytest.raises(ValueError):
            distill_config.add_output_soft_label(
                {
                    "t_output_idx": 0,
                    "s_output_idx": 0,
                    "loss_func": [{"func_name": "KDCrossEntropy", "temperature": 1, "func_param": []}],
                }
            )
        with pytest.raises(TypeError):
            distill_config.add_custom_loss_func("test_name", "nn.Cell()")
            KnowledgeDistillConfig.check_config(
                distill_config, distill_config.custom_loss_func, DISTILL_LOSS_FUNC_MS, Cell
            )
