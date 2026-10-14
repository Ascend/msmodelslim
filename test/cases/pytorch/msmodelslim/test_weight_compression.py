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

import os
import shutil
import numpy as np
import torch

from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor


data0 = torch.tensor([[0, 1, 2], [7, 8, 9]], dtype=torch.int8)
quant_weight_dict = {'model.layers.0.mlp.up_proj': data0}


def load_from_file_fake(self, weight_path):
    self.logger.info("Loading data")
    self.weights = quant_weight_dict


def run_fake(self):
    weight_key = 'model.layers.0.mlp.up_proj'
    weight_data = {weight_key: np.array([-88, 1, 0, 1, 9, 12, 2, 7, 8, 9, -80], dtype=np.int8)}
    index_data = {weight_key: np.array([3, -128, 0, 0, 0, 0, 0, 0], dtype=np.int8)}
    info_data = {weight_key: np.array([8, 8, 3, 2, 1])}
    self.compress_result_weight = weight_data
    self.compress_result_index = index_data
    self.compress_result_info = info_data

    return self.compress_result_weight, self.compress_result_index, self.compress_result_info


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o750)
    return path


setattr(Compressor, 'load_from_file', load_from_file_fake)
setattr(Compressor, 'run', run_fake)


def test_weight_compression_should_get_compressed_weight_when_given_right_param():
    fake_npy = 'quant_weight.npy'
    TEST_SAVE_PATH = 'msmodelslim_weight_compression'
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)
    index_root = make_dir(os.path.join(TEST_SAVE_PATH, 'index'))
    weight_root = make_dir(os.path.join(TEST_SAVE_PATH, 'weight'))
    info_root = make_dir(os.path.join(TEST_SAVE_PATH, 'info'))

    config = CompressConfig(
        do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=TEST_SAVE_PATH, multiprocess_num=2
    )
    compressor = Compressor(config, fake_npy)
    compress_weight, compress_index, compress_info = compressor.run()

    compressor.export(compress_weight, weight_root)
    compressor.export(compress_index, index_root)
    compressor.export(compress_info, info_root)

    if os.path.exists(fake_npy):
        os.remove(fake_npy)

    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)


def test_weight_compression_safetensor_should_get_compressed_weight_when_given_right_param():
    TEST_SAVE_PATH = 'msmodelslim_weight_compression_safetensor'
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    compress_config = CompressConfig(
        do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=TEST_SAVE_PATH, multiprocess_num=8
    )

    quant_model_description = {'model_quant_type': 'W8A8S', 'model.layers.0.mlp.up_proj': 'W8A8S'}

    compressor = Compressor(compress_config, weight=quant_weight_dict, quant_model_description=quant_model_description)
    compress_weight, compress_index, compress_info = compressor.run()

    compressor.export_safetensors(TEST_SAVE_PATH, safetensors_name=None, json_name=None)

    if os.path.exists(TEST_SAVE_PATH):
        shutil.rmtree(TEST_SAVE_PATH)
