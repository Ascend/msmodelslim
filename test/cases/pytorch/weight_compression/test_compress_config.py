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

from unittest.mock import patch

import pytest

from msmodelslim.pytorch.weight_compression import CompressConfig


class TestCompressConfig:
    def test_initialize_compress_config_with_default_args(self):
        config = CompressConfig()
        assert config.do_pseudo_sparse is False
        assert config.sparse_ratio == 1
        assert config.is_debug is False
        assert config.record_detail_root == './'
        assert config.multiprocess_num == 1

    def test_arg_do_pseudo_sparse_with_right_data_type(self):
        config = CompressConfig(do_pseudo_sparse=True)
        assert config.do_pseudo_sparse is True
        config = CompressConfig(do_pseudo_sparse=False)
        assert config.do_pseudo_sparse is False

    def test_arg_do_pseudo_sparse_with_wrong_data_type(self):
        with pytest.raises(TypeError):
            config = CompressConfig(do_pseudo_sparse=1)
        with pytest.raises(TypeError):
            config = CompressConfig(do_pseudo_sparse='True')  # noqa: F841

    def test_arg_sparse_ratio_with_right_data_type(self):
        config = CompressConfig(do_pseudo_sparse=True, sparse_ratio=1)
        assert config.sparse_ratio == 1
        config = CompressConfig(do_pseudo_sparse=True, sparse_ratio=0.58)
        assert config.sparse_ratio == 0.58

    def test_arg_sparse_ratio_with_wrong_format(self):
        with pytest.raises(TypeError):
            CompressConfig(do_pseudo_sparse=True, sparse_ratio=True)
        with pytest.raises(TypeError):
            CompressConfig(do_pseudo_sparse=True, sparse_ratio='0.58')
        with pytest.raises(ValueError):
            CompressConfig(do_pseudo_sparse=True, sparse_ratio=1.2)
        with pytest.raises(ValueError):
            CompressConfig(do_pseudo_sparse=True, sparse_ratio=-0.2)

    def test_arg_is_debug_with_right_data_type(self):
        config = CompressConfig(is_debug=True)
        assert config.is_debug is True
        config = CompressConfig(is_debug=False)
        assert config.is_debug is False

    def test_arg_is_debug_with_wrong_data_type(self):
        with pytest.raises(TypeError):
            CompressConfig(is_debug=1)
        with pytest.raises(TypeError):
            CompressConfig(is_debug='True')

    def test_arg_compress_disable_layers_with_right_data_type(self):
        config = CompressConfig(
            compress_disable_layers=['model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.k_proj']
        )
        assert config.compress_disable_layers == ['model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.k_proj']
        config = CompressConfig(compress_disable_layers=('model.layers.1.mlp.up_proj', 'model.layers.1.mlp.gate_proj'))
        assert config.compress_disable_layers == ('model.layers.1.mlp.up_proj', 'model.layers.1.mlp.gate_proj')

    def test_arg_compress_disable_layers_with_wrong_data_type(self):
        with pytest.raises(TypeError):
            CompressConfig(compress_disable_layers=1)
        with pytest.raises(TypeError):
            CompressConfig(compress_disable_layers='True')

    def test_arg_multiprocess_num_with_right_data_type(self):
        config = CompressConfig(multiprocess_num=1)
        assert config.multiprocess_num == 1
        config = CompressConfig(multiprocess_num=12)
        assert config.multiprocess_num == 12

    def test_arg_multiprocess_num_with_wrong_data_type(self):
        with pytest.raises(ValueError):
            CompressConfig(multiprocess_num=0)
        with pytest.raises(TypeError):
            CompressConfig(is_debug='True')

    def test_arg_multiprocess_num_with_exceeded_cpu_num(self):
        with patch('multiprocessing.cpu_count', return_value=64):
            config = CompressConfig(multiprocess_num=9999)
            assert config.multiprocess_num == 1
