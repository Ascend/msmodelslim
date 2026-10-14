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
# pylint: disable=redefined-outer-name

import os
import pytest
import numpy as np
import mindspore as ms

import atheris

with atheris.instrument_imports():
    from ascend_utils.common.utils import count_parameters
    from msmodelslim.mindspore.low_rank_decompose import low_rank_decompose
    from resources.sample_net_mindspore import LrdSampleNetwork


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    pre_device_target = ms.get_context('device_target')
    ms.set_context(device_target='CPU')  # NPU will be rather slow
    yield
    ms.set_context(device_target=pre_device_target)  # Set back


@pytest.fixture(scope="function")
def source_model():
    yield LrdSampleNetwork()


@pytest.fixture(scope="function")
def source_embedding_cell():
    yield ms.nn.Embedding(32, 64)


@pytest.fixture(scope="function")
def source_decomposer():
    yield low_rank_decompose.Decompose(LrdSampleNetwork())


def test_count_parameters_given_valid_when_any_then_pass(source_model):
    assert count_parameters(source_model) == 1038346


def test_decompose_embedding_given_valid_when_any_then_pass(source_embedding_cell):
    target_cell = low_rank_decompose.decompose_embedding(source_embedding_cell, 16)
    assert count_parameters(target_cell) == 1536


def test_decompose_embedding_given_invalid_when_any_then_error(source_embedding_cell):
    with pytest.raises(ValueError):
        _ = low_rank_decompose.decompose_embedding(source_embedding_cell, 0)
    with pytest.raises(ValueError):
        _ = low_rank_decompose.decompose_embedding(source_embedding_cell, "invalid")


def test_get_decomposed_config_given_int_when_any_then_pass(source_model):
    result = {'embedding.1': 16, 'feature.0': (16, 16), 'feature.2': (16, 16), 'inner': 16, 'classifier.0': 16}
    assert low_rank_decompose.get_decomposed_config(source_model, 13, divisor=16) == result


def test_get_decomposed_config_given_float_when_any_then_pass(source_model):
    result = {'embedding.1': 16, 'feature.0': (64, 64), 'feature.2': (32, 32), 'inner': 192, 'classifier.0': 80}
    assert low_rank_decompose.get_decomposed_config(source_model, 0.5, divisor=16) == result

    result = {'feature.0': (64, 64), 'inner': 192, 'classifier.0': 128}
    assert low_rank_decompose.get_decomposed_config(source_model, 0.5) == result

    result = {'embedding.0': 6, 'embedding.1': 9, 'feature.0': (54, 54), 'feature.2': (18, 18), 'classifier.1': 6}
    excludes = ['inner', 'classifier.0']
    assert low_rank_decompose.get_decomposed_config(source_model, 0.5, excludes=excludes, divisor=3) == result


def test_get_decomposed_config_given_vbmf_when_any_then_pass(source_model):
    res = {'embedding.0': 4, 'embedding.1': 4, 'feature.0': (4, 4), 'feature.2': (4, 4), 'inner': 4, 'classifier.0': 4}
    assert low_rank_decompose.get_decomposed_config(source_model, "vbmf", excludes=['classifier.1'], divisor=4) == res


def test_get_decomposed_config_given_dict_when_any_then_pass(source_model):
    result = {'feature.0': (16, 16), 'feature.2': (32, 16), 'inner': 16}
    hidden_channels = {'feature.0': (12, 12), 'feature.2': (31, 13), 'inner': 11}
    assert low_rank_decompose.get_decomposed_config(source_model, hidden_channels, divisor=16) == result


def test_decompose_network_given_valid_when_any_then_pass_1(source_model):
    decompose_config = {'feature.0': (32, 16), 'feature.2': (16, 32)}
    new_model = low_rank_decompose.decompose_network(source_model, decompose_config, do_decompose_weight=False)
    assert count_parameters(new_model) == 971786


def test_decompose_network_given_valid_when_any_then_pass_2(source_model):
    decompose_info = {'embedding.1': 8, 'feature.0': (16, 16), 'feature.2': (16, 16), 'inner': 16, 'classifier.0': 16}
    new_model = low_rank_decompose.decompose_network(source_model, decompose_info)
    assert count_parameters(new_model) == 59658


def test_decompose_network_given_datasets_when_any_then_pass(source_model):
    decompose_info = {'embedding.1': 8, 'feature.0': (16, 16), 'feature.2': (16, 16), 'inner': 16, 'classifier.0': 16}
    datasets = [[ms.Tensor(np.random.uniform(size=[2, 16, 16]).astype('int32'))] for _ in range(10)]
    new_model = low_rank_decompose.decompose_network(source_model, decompose_info, datasets=datasets)
    assert count_parameters(new_model) == 59658


def test_decompose_given_invalid_when_any_then_error(source_model):
    with pytest.raises(ValueError):
        _ = low_rank_decompose.Decompose(source_model, config_file="invalid")


def test_from_ratio_given_valid_when_any_then_pass(source_decomposer):
    source_decomposer.from_ratio(0.5, excludes=["embedding.1", "classifier.0"])
    assert source_decomposer.decompose_config == {'feature.0': (64, 64), 'inner': 192}


def test_from_ratio_given_invalid_when_any_then_error(source_decomposer):
    with pytest.raises(ValueError):
        source_decomposer.from_ratio(0)


def test_from_fixed_given_valid_when_any_then_pass(source_decomposer):
    source_decomposer.from_fixed(16)
    assert source_decomposer.decompose_config == {'feature.0': (64, 64), 'inner': 64, 'classifier.0': 64}


def test_from_fixed_given_invalid_when_any_then_error(source_decomposer):
    with pytest.raises(ValueError):
        source_decomposer.from_fixed(0)


def test_from_vbmf_given_valid_when_any_then_pass(source_decomposer):
    source_decomposer.from_vbmf(excludes=["embedding.1", "classifier.0"], divisor=8)
    result = {'embedding.0': 8, 'feature.0': (8, 8), 'feature.2': (8, 8), 'inner': 8, 'classifier.1': 8}
    assert source_decomposer.decompose_config == result


def test_from_vbmf_given_invalid_when_any_then_error(source_decomposer):
    with pytest.raises(ValueError):
        source_decomposer.from_vbmf(excludes=0)

    with pytest.raises(ValueError):
        source_decomposer.from_vbmf(divisor=0)


def test_from_dict_given_valid_when_any_then_pass(source_decomposer):
    source_decomposer.from_dict({'feature.0': (32, 16), 'feature.2': (16, 28)}, divisor=16)
    assert source_decomposer.decompose_config == {'feature.0': (32, 16), 'feature.2': (16, 32)}


def test_from_dict_given_invalid_when_any_then_error(source_decomposer):
    with pytest.raises(ValueError):
        source_decomposer.from_dict(0)


def test_decomposer_given_valid_when_any_then_pass(source_decomposer):
    source_decomposer.from_dict({'feature.0': (32, 16), 'feature.2': (16, 28)}, divisor=16)
    new_model = source_decomposer.decompose_network(do_decompose_weight=False)
    assert count_parameters(new_model) == 971786


def test_decompose_given_file_when_any_then_pass(source_model):
    config_file = "test_decompose_given_file_when_any_then_pass.json"
    decomposer = low_rank_decompose.Decompose(source_model, config_file=config_file)
    decomposer.from_ratio(0.5, excludes=["embedding.1", "classifier.0"])
    decomposer.from_file()
    assert decomposer.decompose_config == {'feature.0': [64, 64], 'inner': 192}

    if os.path.exists(config_file):
        os.remove(config_file)
