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
from types import SimpleNamespace
from unittest.mock import Mock

from msmodelslim.utils import seed as seed_module


def test_seed_all_sets_environment_and_npu_seed(monkeypatch):
    random_seed = Mock()
    numpy_seed = Mock()
    torch_seed = Mock()
    deterministic = Mock()
    npu = SimpleNamespace(manual_seed_all=Mock(), manual_seed=Mock())
    monkeypatch.setattr(seed_module.random, "seed", random_seed)
    monkeypatch.setattr(seed_module.np.random, "seed", numpy_seed)
    monkeypatch.setattr(seed_module.torch, "manual_seed", torch_seed)
    monkeypatch.setattr(seed_module.torch, "use_deterministic_algorithms", deterministic)
    monkeypatch.setattr(seed_module.torch.version, "cuda", "10.2", raising=False)
    monkeypatch.setattr(seed_module, "is_gpu", False)
    monkeypatch.setattr(seed_module, "torch_npu", SimpleNamespace(npu=npu), raising=False)

    seed_module.seed_all(seed=7, mode=True)

    assert os.environ["PYTHONHASHSEED"] == "7"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["HCCL_DETERMINISTIC"] == "True"
    random_seed.assert_called_once_with(7)
    numpy_seed.assert_called_once_with(7)
    torch_seed.assert_called_once_with(7)
    deterministic.assert_called_once_with(False)
    npu.manual_seed_all.assert_called_once_with(7)
    npu.manual_seed.assert_called_once_with(7)


def test_seed_all_sets_cuda_and_cudnn_state(monkeypatch):
    random_seed = Mock()
    numpy_seed = Mock()
    torch_seed = Mock()
    deterministic = Mock()
    cuda = SimpleNamespace(manual_seed_all=Mock(), manual_seed=Mock())
    cudnn = SimpleNamespace(deterministic=False, enable=True, benchmark=True)
    monkeypatch.setattr(seed_module.random, "seed", random_seed)
    monkeypatch.setattr(seed_module.np.random, "seed", numpy_seed)
    monkeypatch.setattr(seed_module.torch, "manual_seed", torch_seed)
    monkeypatch.setattr(seed_module.torch, "use_deterministic_algorithms", deterministic)
    monkeypatch.setattr(seed_module.torch.version, "cuda", None, raising=False)
    monkeypatch.setattr(seed_module.torch, "cuda", cuda, raising=False)
    monkeypatch.setattr(seed_module.torch.backends, "cudnn", cudnn, raising=False)
    monkeypatch.setattr(seed_module, "is_gpu", True)

    seed_module.seed_all(seed=11, mode=False)

    assert cuda.manual_seed_all.call_count == 1
    assert cuda.manual_seed.call_count == 1
    assert cudnn.deterministic is True
    assert cudnn.enable is False
    assert cudnn.benchmark is False
    assert deterministic.call_args.args == (False,)
