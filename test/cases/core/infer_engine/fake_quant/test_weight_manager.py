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

msmodelslim/core/infer_engine/fake_quant/weight_manager.py 的单元测试。
"""

from unittest.mock import MagicMock, Mock, patch

import torch
from torch import nn

from msmodelslim.core.infer_engine.fake_quant.weight_manager import WeightManager, _empty_device_cache
from msmodelslim.utils.buffer import RuntimeBufferStore


class _DecoderLayer(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.net = nn.Linear(dim, dim)

    def forward(self, x):
        return self.net(x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.layers = nn.ModuleList([_DecoderLayer(), _DecoderLayer()])
        self.head = nn.Linear(4, 4)
        self.act = nn.ReLU()  # 无参数，不视为 shared


def _make_manager(loader=None, **kwargs):
    return WeightManager(
        loader if loader is not None else Mock(),
        device="cpu",
        buffer_store=RuntimeBufferStore(),
        **kwargs,
    )


class TestWeightManager:
    """对应 WeightManager。"""

    def test_install_hooks_registers_decoder_and_shared_hooks_when_model_has_both(self):
        model = _Model()
        wm = _make_manager()
        wm.install_hooks(model)

        assert len(wm._hook_handles) == 12  # 2 decoder * 2 + 4 shared 容器/叶子 * 2
        assert set(wm._decoder_name_by_id.values()) == {"layers.0", "layers.1"}

    def test_install_hooks_is_idempotent_when_called_twice(self):
        model = _Model()
        wm = _make_manager()
        wm.install_hooks(model)
        wm.install_hooks(model)

        assert len(wm._hook_handles) == 12

    def test_remove_all_hooks_returns_count_when_hooks_exist(self):
        model = _Model()
        wm = _make_manager()
        wm.install_hooks(model)

        count = wm.remove_all_hooks()

        assert count == 12
        assert not wm._hook_handles
        assert not wm._decoder_name_by_id

    def test_remove_all_hooks_returns_zero_when_none_installed(self):
        assert _make_manager().remove_all_hooks() == 0

    def test_set_offload_target_updates_device_when_called(self):
        wm = _make_manager()

        wm.set_offload_target("cpu")

        assert wm.offload_device == "cpu"

    def test_decoder_pre_hook_hydrates_and_aligns_when_layer_known(self):
        model = _Model()
        loader = Mock()
        wm = _make_manager(loader)
        wm._model = model
        wm._decoder_name_by_id = {id(model.layers[0]): "layers.0"}
        x = torch.randn(2, 4)

        args, kwargs = wm._decoder_pre_hook(model.layers[0], (x,), {})

        loader.hydrate.assert_called_once_with(model, prefix="layers.0")
        assert len(args) == 1
        assert torch.equal(args[0], x)

    def test_decoder_pre_hook_falls_back_to_lookup_when_unknown_id(self):
        model = _Model()
        loader = Mock()
        wm = _make_manager(loader)
        wm._model = model
        x = torch.randn(2, 4)

        args, _ = wm._decoder_pre_hook(model.layers[1], (x,), {})

        loader.hydrate.assert_called_once_with(model, prefix="layers.1")

    def test_decoder_post_hook_offloads_and_returns_output_when_layer(self):
        model = _Model()
        wm = _make_manager()
        wm._model = model
        wm._decoder_name_by_id = {id(model.layers[0]): "layers.0"}
        output = torch.randn(2, 4)

        with patch("msmodelslim.core.infer_engine.fake_quant.weight_manager._empty_device_cache") as mock_empty:
            returned = wm._decoder_post_hook(model.layers[0], (torch.randn(2, 4),), output)

        assert returned is output
        mock_empty.assert_called_once()
        assert model.layers[0].net.weight.device.type == "meta"

    def test_is_shared_module_defaults_to_param_bearing_non_decoder_when_model(self):
        model = _Model()
        wm = _make_manager()
        decoders = ["layers.0", "layers.1"]

        assert wm._is_shared_module("embed", model.embed, decoders) is True
        assert wm._is_shared_module("head", model.head, decoders) is True
        assert wm._is_shared_module("layers.0", model.layers[0], decoders) is False
        assert wm._is_shared_module("layers.0.net", model.layers[0].net, decoders) is False
        assert wm._is_shared_module("act", model.act, decoders) is False

    def test_is_shared_module_uses_allowlist_when_given(self):
        model = _Model()
        wm = _make_manager(shared_module_prefixes=["embed"])

        assert wm._is_shared_module("embed", model.embed, []) is True
        assert wm._is_shared_module("embed.lin", Mock(), []) is True
        assert wm._is_shared_module("head", model.head, []) is False

    def test_lookup_module_name_returns_name_when_module_present(self):
        model = _Model()
        wm = _make_manager()
        wm._model = model

        assert wm._lookup_module_name(model.layers[1]) == "layers.1"

    def test_lookup_module_name_returns_none_when_no_model(self):
        wm = _make_manager()
        assert wm._lookup_module_name(Mock()) is None

    def test_empty_device_cache_releases_npu_when_available(self):
        npu_mock = MagicMock()
        with patch.object(torch, "npu", npu_mock, create=True):
            _empty_device_cache()

        npu_mock.empty_cache.assert_called_once()

    def test_hooks_removed_stops_hydrating_when_forward_after_remove(self):
        model = _Model()
        loader = Mock()
        wm = _make_manager(loader)
        wm.install_hooks(model)
        wm.remove_all_hooks()
        loader.hydrate.reset_mock()

        model.layers[0](torch.randn(2, 4))

        loader.hydrate.assert_not_called()
