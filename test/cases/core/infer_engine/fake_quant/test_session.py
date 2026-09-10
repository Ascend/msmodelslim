# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

msmodelslim/core/infer_engine/fake_quant/session.py 的单元测试。
"""

from unittest.mock import Mock

from torch import nn

from msmodelslim.core.infer_engine.fake_quant.session import Session
from msmodelslim.utils.buffer import RuntimeBufferStore


class TestSession:
    """对应 Session。"""

    @staticmethod
    def _make_loader():
        loader = Mock()
        loader.decoder_prefixes.return_value = {"model.layers.0", "model.layers.1"}
        return loader

    def test_build_returns_eval_shell_when_hydrated_shared_only(self):
        adapter = Mock()
        model = nn.Module()
        adapter.build_meta_model.return_value = model
        loader = self._make_loader()
        store = RuntimeBufferStore()

        built = Session(adapter, loader, store).build()

        assert built is model
        assert not model.training
        loader.bind_adapter.assert_called_once_with(adapter)
        loader.apply_ir.assert_called_once_with(model)
        loader.decoder_prefixes.assert_called_once_with(model)
        loader.hydrate.assert_called_once_with(model, skip_prefixes={"model.layers.0", "model.layers.1"})
        # 快照先 store 后 restore
        assert store.load() == {}

    def test_build_keeps_shared_weights_when_no_decoder_layers(self):
        adapter = Mock()
        model = nn.Module()
        adapter.build_meta_model.return_value = model
        loader = Mock()
        loader.decoder_prefixes.return_value = set()

        Session(adapter, loader, RuntimeBufferStore()).build()

        loader.hydrate.assert_called_once_with(model, skip_prefixes=set())
