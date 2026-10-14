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
# pylint: disable=undefined-all-variable

__all__ = [
    'FakeQuantizedLinear',
    'FlatQuantizedLinear',
    'FlatNormWrapper',
    'ForwardMode',
    'FakeQuantizedLinearConfig',
    'WeightQuantizer',
    'ActivationQuantizer',
    'asym_quant',
    'asym_dequant',
    'GeneralMatrixTrans',
    'SVDSingleTransMatrix',
    'InvSingleTransMatrix',
    'DiagonalTransMatrix',
]


from .fake_clip_quantizer import (
    WeightQuantizer,
    ActivationQuantizer,
    asym_quant,
    asym_dequant,
)
from .flat_fake_quant_linear import (
    FlatFakeQuantLinear,  # noqa: F401
    FlatNormWrapper,
    FlatFakeQuantLinearConfig,  # noqa: F401
    ForwardMode,
)
from .flat_quant_manager import FlatQuantLayerManager  # noqa: F401

from .structure_pair import (
    StructurePair,  # noqa: F401
    NormLinearPair,  # noqa: F401
    MLPLinearLinearPair,  # noqa: F401
    MLPNormLinearPair,  # noqa: F401
    AttnLinearLinearPair,  # noqa: F401
    AttnNormLinearPair,  # noqa: F401
)
from .trans_matrix import GeneralMatrixTrans, SVDSingleTransMatrix, InvSingleTransMatrix, DiagonalTransMatrix

from .utils import (
    get_decompose_dim,  # noqa: F401
    get_init_scale,  # noqa: F401
    get_init_weight,  # noqa: F401
    get_n_set_parameters_byname,  # noqa: F401
    set_require_grad_all,  # noqa: F401
    stat_input_hook,  # noqa: F401
    stat_tensor,  # noqa: F401
    get_trainable_parameters,  # noqa: F401
    get_para_names,  # noqa: F401
    match_pattern,  # noqa: F401
    move_tensors_to_device,  # noqa: F401
    empty_cache,  # noqa: F401
    get_module_by_name,  # noqa: F401
    set_module_by_name,  # noqa: F401
    clone_module_hooks,  # noqa: F401
    remove_after_substring,  # noqa: F401
)
