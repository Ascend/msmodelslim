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
# pylint: disable=anomalous-backslash-in-string,broad-exception-raised,no-else-continue

import re


class PruneUtilsBase:
    @staticmethod
    def flip_dict(data):
        return {str(v): str(k) for k, v in data.items()}

    @staticmethod
    def prune_bert_intra_block(model_state_dict, state_dict, is_parameter=True, parameter=None):
        """
        is_parameter and parameter just for mindspore
        parameter is mindspore.common.parameter
        """
        for name, st_weight in state_dict.items():
            model_weight = model_state_dict.get(name, None)
            if model_weight is None:
                continue
            weight_shape = st_weight.shape
            model_weight_shape = model_weight.shape
            if weight_shape == model_weight_shape:
                continue
            if model_weight.dim() == 1:
                state_dict[name] = st_weight[: model_weight_shape[0]]
            elif model_weight.dim() == 2:
                state_dict[name] = st_weight[: model_weight_shape[0], : model_weight_shape[1]]
            else:
                raise NotImplementedError('Other dimension is not implemented')
            if is_parameter and not isinstance(state_dict[name], parameter):
                state_dict[name] = parameter(state_dict[name], name=name)

    def prune_blocks(self, model, state_dict, model_config):
        params = model_config.get('prune_blocks_params', None)
        if params is None:
            raise Exception('prune_blocks failed. prune_blocks_params cannot be None')

        new_state_dict = self.prune_state_dict_blocks(state_dict, params)
        return new_state_dict

    def prune_state_dict_blocks(self, state_dict, params):
        """
        remove unused state dict blocks and rename reserved blocks weight name
        input
            state_dict: torch model.state_dict()
            params: a list, element is dict.
                e.g. [{'pattern': 'bert\.encoder\.layer\.(\d+)\.', 'layer_id_map': {0: 0, 1: 2, 2: 4}}]
                pattern: It's string. It is encoder / decoder block module name pattern
                layer_id_map: It's a dict, means {new_block_id: old_block_id} or {student_block_id: teacher_block_id}
        return
            pruned state dict
        """
        pruning_passes = []
        for one_match in params:
            pattern_str = one_match['pattern']
            layer_id_map = one_match['layer_id_map']
            layer_id_map = self.flip_dict(layer_id_map)
            pattern = re.compile(pattern_str)
            pruning_passes.append({'layer_id_map': layer_id_map, 'pattern': pattern})

        new_state_dict = {}
        for weight_name in state_dict.keys():
            if len(weight_name) > 512:
                raise ValueError("Length of {} key exceeds limitation {}.".format(weight_name, 512))

            match_flag = False
            for pruning_pass in pruning_passes:
                pattern = pruning_pass['pattern']
                ret = pattern.match(weight_name)
                if ret is None:
                    continue

                match_flag = True
                old_num = ret.group(1)
                layer_id_map = pruning_pass['layer_id_map']
                if old_num not in layer_id_map:
                    continue
                else:
                    new_num = layer_id_map[old_num]
                    start_idx = ret.start(1)
                    end_idx = ret.end(1)
                    new_weight_name = weight_name[:start_idx] + new_num + weight_name[end_idx:]
                    new_state_dict[new_weight_name] = state_dict[weight_name]

            if not match_flag:
                new_state_dict[weight_name] = state_dict[weight_name]

        return new_state_dict
