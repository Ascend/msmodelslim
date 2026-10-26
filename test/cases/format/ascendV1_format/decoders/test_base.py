# -*- coding: UTF-8 -*-

# pylint: disable=abstract-class-instantiated

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

msmodelslim/format/ascendV1_format/decoders/base.py 的单元测试。
"""

import pytest

from msmodelslim.format.ascendV1_format.decoders.base import (
    Fa3IrDecoder,
    IrDecoder,
    KvCacheIrDecoder,
    LinearIrDecoder,
)


class TestIrDecoder:
    """对应 IrDecoder。"""

    def test_keeps_store_reference_when_constructed(self, fake_store_cls):
        store = fake_store_cls()

        assert IrDecoder(store).store is store


class TestAbstractDecoderBases:
    """LinearIrDecoder / Fa3IrDecoder / KvCacheIrDecoder 为抽象基类。"""

    def test_linear_decoder_raises_type_error_when_instantiated(self, fake_store_cls):
        with pytest.raises(TypeError):
            LinearIrDecoder(fake_store_cls())

    def test_fa3_decoder_raises_type_error_when_instantiated(self, fake_store_cls):
        with pytest.raises(TypeError):
            Fa3IrDecoder(fake_store_cls())

    def test_kv_cache_decoder_raises_type_error_when_instantiated(self, fake_store_cls):
        with pytest.raises(TypeError):
            KvCacheIrDecoder(fake_store_cls())
