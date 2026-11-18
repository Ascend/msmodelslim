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
# pylint: disable=broad-exception-raised

import logging

import torch

from ascend_utils.common.prune.transformer_prune.prune_utils_base import PruneUtilsBase
from ascend_utils.common.security.pytorch import safe_torch_load

QKV_NUMS = 3  # transform model has 3 vector named q, k, v


class PruneUtilsTorch(PruneUtilsBase):
    @staticmethod
    def find_linear_diff_dim(weight1, weight2):
        """find nn.linear different dimension, only one different dimension"""
        if weight1.dim() != 2 or weight2.dim() != 2:
            raise Exception('This is for 2D p')

        shape1 = torch.tensor(weight1.shape)
        shape2 = torch.tensor(weight2.shape)
        diff_flag = shape1 != shape2
        diff_axis = torch.nonzero(diff_flag)
        return diff_axis.item()

    @staticmethod
    def generate_combined_qkv_index(old_dim, new_dim):
        """
        generate index to extract weight from old weight

        Args:
            old_dim: 原始维度大小
            new_dim: 新维度大小，必须满足 new_dim <= old_dim

        Returns:
            torch.LongTensor: 提取权重的索引

        Raises:
            ValueError: 当 new_dim > old_dim 时抛出异常
        """
        if new_dim > old_dim:
            raise ValueError(
                f"new_dim ({new_dim}) cannot be greater than old_dim ({old_dim}). "
                f"This function only supports dimension reduction (new_dim <= old_dim)."
            )
        mask = torch.zeros(old_dim)
        old_step = old_dim / QKV_NUMS
        new_step = new_dim / QKV_NUMS
        for i in range(3):
            start_ix = int(i * old_step)
            end_ix = int(start_ix + new_step)
            mask[start_ix:end_ix] = 1
        mask = mask.eq(1)
        index: torch.LongTensor = torch.arange(len(mask))[mask].long()
        return index

    @staticmethod
    def get_state_dict(weight_path):
        # avoid load different npu id model to npu
        ckpt = safe_torch_load(weight_path, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        return ckpt

    def prune_bert_intra_block_torch(self, model, state_dict, model_config):
        logging.info('Attention, prune_bert_intra_block is used for "separate" qkv weight')
        model_state_dict = model.state_dict()
        self.prune_bert_intra_block(model_state_dict, state_dict, False)
        return state_dict

    def prune_vit_intra_block(self, model, state_dict, model_config):
        """
        prune heads and ffn intermediates
        """
        qkv_keys = model_config.get('qkv_keys', None)  # string list
        if qkv_keys is None:
            raise Exception('prune_vit_intra_block failed. "qkv_keys" cannot be None, it is a string list')

        logging.info('Attention, prune_vit_intra_block is used for "combined together" qkv weight')
        model_state_dict = model.state_dict()
        for name, st_weight in state_dict.items():
            model_weight = model_state_dict.get(name, None)
            if model_weight is None:
                continue
            weight_shape = st_weight.shape
            model_weight_shape = model_weight.shape
            if weight_shape == model_weight_shape:
                continue

            find_flag = False
            for k in qkv_keys:
                if k in name:
                    find_flag = True
                    break

            if find_flag:
                self.prune_combined_qkv_weight(name, state_dict, model_weight, st_weight)
            else:
                if model_weight.dim() == 1:
                    state_dict[name] = st_weight[: model_weight_shape[0]]
                elif model_weight.dim() == 2:
                    state_dict[name] = st_weight[: model_weight_shape[0], : model_weight_shape[1]]
                else:
                    raise NotImplementedError('Other dimension is not implemented')

        return state_dict

    def prune_combined_qkv_weight(self, name, state_dict, model_weight, st_weight):
        if model_weight.dim() == 1:
            old_dim = st_weight.shape[0]
            new_dim = model_weight.shape[0]
            index = self.generate_combined_qkv_index(old_dim, new_dim)
            state_dict[name] = st_weight[index]
        elif model_weight.dim() == 2:
            diff_axis = self.find_linear_diff_dim(model_weight, st_weight)
            old_dim = st_weight.shape[diff_axis]
            new_dim = model_weight.shape[diff_axis]
            index = self.generate_combined_qkv_index(old_dim, new_dim)
            if diff_axis == 0:
                state_dict[name] = st_weight[index, :]
            elif diff_axis == 1:
                state_dict[name] = st_weight[:, index]
            else:
                raise Exception(f'nn.Linear weight only has 2 dimension, does not have axis {diff_axis}')
        else:
            raise NotImplementedError('Other dimension is not implemented')


prune_utils_torch = PruneUtilsTorch()
PRUNE_STATE_DICT_FUNCS_TORCH = {
    'prune_blocks': prune_utils_torch.prune_blocks,
    'prune_bert_intra_block': prune_utils_torch.prune_bert_intra_block_torch,
    'prune_vit_intra_block': prune_utils_torch.prune_vit_intra_block,
}
