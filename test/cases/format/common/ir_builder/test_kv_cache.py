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

Unit tests for msmodelslim.format.common.ir_builder.kv_cache.
"""

import pytest
import torch
from torch import nn

from msmodelslim.format.common.ir_builder.base import ApplyMode
from msmodelslim.format.common.ir_builder.kv_cache import KvCacheIrBuilder, KvCacheParamSet, KvCacheStructureSpec
from msmodelslim.ir.attention import FakeQuantDynamicCache
from msmodelslim.ir.const import int8_per_channel_sym
from msmodelslim.utils.exception import SchemaValidateError


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Identity()


class TestKvCacheIrBuilder:
    def test_transform_shell_inserts_dynamic_cache(self):
        model = _Holder()
        structure = KvCacheStructureSpec(scale_shape=(2, 1), offset_shape=(2, 1), scheme=int8_per_channel_sym)

        result = KvCacheIrBuilder().transform_shell(model, "attn", nn.Identity(), structure=structure)

        assert result.mode is ApplyMode.INSERT
        assert isinstance(result.module, FakeQuantDynamicCache)
        assert model.attn is result.module
        assert result.module.x_q_scheme == int8_per_channel_sym

    def test_transform_shell_raises_when_structure_type_mismatch(self):
        with pytest.raises(SchemaValidateError):
            KvCacheIrBuilder().transform_shell(_Holder(), "attn", nn.Identity(), structure=object())

    def test_bind_from_param_set_raises_when_ir_module_wrong_type(self):
        param_set = KvCacheParamSet(
            kv_cache_scale=torch.ones(2), kv_cache_offset=torch.zeros(2), scheme=int8_per_channel_sym
        )

        with pytest.raises(SchemaValidateError):
            KvCacheIrBuilder().bind_from_param_set(nn.Identity(), param_set)
