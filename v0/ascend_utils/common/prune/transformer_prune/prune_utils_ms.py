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

import logging

from mindspore.common.parameter import Parameter
from ascend_utils.common.prune.transformer_prune.prune_utils_base import PruneUtilsBase


class PruneUtilsMs(PruneUtilsBase):
    def prune_bert_intra_block_ms(self, model, state_dict, model_config):
        logging.info('Attention, prune_bert_intra_block is used for "separate" qkv weight')
        model_state_dict = model.parameters_dict()
        self.prune_bert_intra_block(model_state_dict, state_dict, True, parameter=Parameter)
        return state_dict


prune_utils_ms = PruneUtilsMs()
PRUNE_STATE_DICT_FUNCS_MS = {
    'prune_blocks': prune_utils_ms.prune_blocks,
    'prune_bert_intra_block': prune_utils_ms.prune_bert_intra_block_ms,
}
