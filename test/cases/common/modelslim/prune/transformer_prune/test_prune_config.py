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
# pylint: disable=useless-object-inheritance

import pytest

from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig


class TestPruneConfig:
    def test_prune_config_pass_when_set_and_get_valid_params(self):
        prune_config = PruneConfig()
        prune_config.set_steps(['prune_blocks', 'prune_bert_intra_block'])
        prune_config.add_blocks_params('test_name', {0: 1})
        prune_config.get("prune_blocks_params")

    def test_prune_config_given_invalid_when_any_then_error(self):
        prune_config = PruneConfig()
        with pytest.raises(ValueError):
            prune_config.set_steps(None)
        with pytest.raises(ValueError):
            prune_config.set_steps(["fake_step"])
            PruneConfig.check_steps_list(
                prune_config, ['prune_blocks', 'prune_bert_intra_block', 'prune_vit_intra_block']
            )
        with pytest.raises(TypeError):
            prune_config.add_blocks_params(1, {0: 1})
        with pytest.raises(TypeError):
            prune_config.add_blocks_params('test_name', "{0: 1}")
        with pytest.raises(TypeError):
            prune_config.add_blocks_params('test_name', {1.1, 2})
        with pytest.raises(TypeError):
            prune_config.add_blocks_params('test_name', {"1.1", 2})
        with pytest.raises(ValueError):
            prune_config.get("fake_name")
