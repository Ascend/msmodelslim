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
# pylint: disable=anomalous-backslash-in-string,useless-object-inheritance

from ascend_utils.common.security import check_element_type
from ascend_utils.common.security import check_type
from ascend_utils.common.utils import check_model_backend
from msmodelslim import logger


class PruneConfig(object):
    """
    Configuration for prune.

    Example:
        prune_config = PruneConfig()
        prune_config.set_steps(['prune_blocks', 'prune_bert_intra_block'])\
            .add_blocks_params('uniter\.encoder\.encoder\.blocks\.(\d+)\.', {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11})
    """

    def __init__(self):
        self.prune_state_dict_steps = []
        self.prune_blocks_params = []

    @staticmethod
    def check_prune_config(config, target_steps: list):
        check_type(config, PruneConfig, param_name="config")
        PruneConfig.check_steps_list(config, target_steps)

    @staticmethod
    def check_steps_list(config, target_steps):
        if not hasattr(config, "prune_state_dict_steps"):
            return
        prune_state_dict_steps = getattr(config, "prune_state_dict_steps")
        for step in prune_state_dict_steps:
            if step not in target_steps:
                raise ValueError("Step {} not exist! Step must in {}".format(step, target_steps))

    @staticmethod
    def _check_steps(steps: list):
        if not steps:
            raise ValueError("Steps can not be empty!")
        check_element_type(steps, element_type=str, value_type=list, param_name="steps")

    @staticmethod
    def _check_pattern(pattern: str):
        check_type(pattern, str, param_name="pattern")

    @staticmethod
    def _check_layer_id_map(layer_id_map: dict):
        if not isinstance(layer_id_map, dict):
            raise TypeError("layer_id_map must be dict, not {}.".format(type(layer_id_map)))
        for layer_after, layer_before in layer_id_map.items():
            check_type(layer_after, int, param_name="layer_after")
            check_type(layer_before, int, param_name="layer_before")

    def get(self, config_name: str, default=None):
        if hasattr(self, config_name):
            return getattr(self, config_name)
        elif default:
            return default
        else:
            raise ValueError("config_name not in PruneConfig.")

    def set_steps(self, steps: list):
        """
        Set steps for prune.

        Args:
            steps(str): The steps of prune can be set to "prune_blocks" and "prune_bert_intra_block". Must use
                add_blocks_params() if set "prune_blocks".
        """
        self._check_steps(steps)
        self.prune_state_dict_steps = steps
        return self

    def add_blocks_params(self, pattern: str, layer_id_map: dict):
        """
        Must add_blocks_params() if set "prune_blocks" in prune steps.

        Args:
            pattern(str): The regular expression to match model layer name.
            layer_id_map(dict): The dictionary to map the layer's id. A layer can be specified by the regular expression
                and the layer's id. The keys of the dictionary are ids of the model, and the values of the dictionary
                are ids of the weight.
        """
        self._check_pattern(pattern)
        self._check_layer_id_map(layer_id_map)
        block_params = {"pattern": pattern, "layer_id_map": layer_id_map}
        self.prune_blocks_params.append(block_params)
        return self


def prune_model_weight(model, config: PruneConfig, weight_file_path: str):
    """
    Prune the weight of a transformer model and load the weight into the model. Please note that this function doesn't
    prune the model but prunes the weight. Ensure the model has been pruned (a transformer initiated by fewer
    parameters). The model should be fine-tuned after prune_model_weight().

    Args:
        model: The pruned model can be a PyTorch/MindSpore model.
        config(PruneConfig): The config for prune
        weight_file_path(str): The weight file path of the original model

    Examples:
        >>> prune_config = PruneConfig()
        >>> # set prune_config
        >>> prune_model_weight(model, prune_config, weight_file_path)
    """
    logger.info("================ Start pruning the model's weight ===============")

    backend = check_model_backend(model)
    if backend == "mindspore":
        from msmodelslim.mindspore.prune.transformer_prune.prune_model_ms import prune_model_weight_ms

        prune_model_weight_ms(model, config, weight_file_path)
    elif backend == "pytorch":
        from msmodelslim.pytorch.prune.transformer_prune.prune_model_torch import prune_model_weight_torch

        prune_model_weight_torch(model, config, weight_file_path)

    logger.info("================ Finish pruning the model's weight ===============")
