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

from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig


def _make_valid_config():
    return PruneConfig().set_steps(["prune_blocks"]).add_blocks_params("blocks.(\\d+)", {0: 1, 1: 3})


def test_prune_config_default_state_is_empty():
    config = PruneConfig()

    assert config.prune_state_dict_steps == []
    assert config.prune_blocks_params == []


def test_set_steps_stores_steps_and_returns_self():
    config = PruneConfig()

    result = config.set_steps(["prune_blocks", "prune_bert_intra_block"])

    assert result is config
    assert config.prune_state_dict_steps == ["prune_blocks", "prune_bert_intra_block"]


def test_set_steps_rejects_empty_list():
    config = PruneConfig()

    with pytest.raises(ValueError):
        _ = config.set_steps([])


@pytest.mark.parametrize("steps", ["prune_blocks", 5])
def test_set_steps_rejects_non_list_type(steps):
    config = PruneConfig()

    with pytest.raises(TypeError):
        _ = config.set_steps(steps)


def test_add_blocks_params_appends_pattern_and_id_map_and_returns_self():
    config = PruneConfig()

    result = config.add_blocks_params("blocks.(\\d+)", {0: 1, 2: 5})

    assert result is config
    assert config.prune_blocks_params == [{"pattern": "blocks.(\\d+)", "layer_id_map": {0: 1, 2: 5}}]


@pytest.mark.parametrize("pattern", [1, None])
def test_add_blocks_params_rejects_non_str_pattern(pattern):
    config = PruneConfig()

    with pytest.raises(TypeError):
        _ = config.add_blocks_params(pattern, {0: 1})


@pytest.mark.parametrize("layer_id_map", [[0, 1], "map", 3])
def test_add_blocks_params_rejects_non_dict_layer_id_map(layer_id_map):
    config = PruneConfig()

    with pytest.raises(TypeError):
        _ = config.add_blocks_params("blocks.(\\d+)", layer_id_map)


def test_get_returns_configured_attribute_or_default():
    config = _make_valid_config()

    assert config.get("prune_state_dict_steps") == ["prune_blocks"]
    assert config.get("not_exist", [7]) == [7]
    with pytest.raises(ValueError):
        _ = config.get("not_exist")


def test_check_prune_config_rejects_step_not_in_target():
    config = _make_valid_config()

    with pytest.raises(ValueError):
        PruneConfig.check_prune_config(config, ["prune_bert_intra_block"])


def test_check_prune_config_accepts_step_in_target():
    config = _make_valid_config()

    assert PruneConfig.check_prune_config(config, ["prune_blocks"]) is None
