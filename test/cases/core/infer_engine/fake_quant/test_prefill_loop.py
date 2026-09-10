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

msmodelslim/core/infer_engine/fake_quant/prefill_loop.py 的单元测试。
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from msmodelslim.core.infer_engine.fake_quant.prefill_loop import PrefillLoop, _LoggingTextStreamer
from msmodelslim.utils.exception import InvalidModelError


class _Out:
    def __init__(self, sequences):
        self.sequences = sequences


class _GenModel(nn.Module):
    def __init__(self, seq=None, config=None):
        super().__init__()
        self._seq = seq
        self.config = config
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return _Out(self._seq)


class _FakeTokenizer:
    def __init__(self, pad_id=None, eos_id=None, decode=None):
        self.pad_token_id = pad_id
        self.eos_token_id = eos_id
        self._decode = decode

    def decode(self, ids, **kwargs):
        if self._decode is not None:
            return self._decode(ids)
        return "".join(str(i) for i in ids)


def _adapter(tokenizer=None):
    return SimpleNamespace(tokenizer=tokenizer)


def _seq(prompt_len=3, new_len=2, start=10):
    return torch.tensor([[0] * prompt_len + list(range(start, start + new_len))])


class TestPrefillLoop:
    """对应 PrefillLoop。"""

    def test_run_returns_ids_and_str_texts_when_no_tokenizer(self):
        model = _GenModel(seq=_seq())
        loop = PrefillLoop(_adapter(), model)

        result = loop.run([torch.tensor([[0, 0, 0]])], max_new_tokens=2)

        assert result.generated_token_ids == [[10, 11]]
        assert result.generated_texts == ["[10, 11]"]
        assert model.last_kwargs["max_new_tokens"] == 2
        assert model.last_kwargs["use_cache"] is False
        assert model.last_kwargs["num_beams"] == 1

    def test_run_raises_invalid_model_when_max_new_tokens_zero(self):
        loop = PrefillLoop(_adapter(), _GenModel())

        with pytest.raises(InvalidModelError):
            loop.run([torch.tensor([[0, 0, 0]])], max_new_tokens=0)

    def test_run_decodes_texts_when_tokenizer_present(self):
        tok = _FakeTokenizer(decode=lambda ids: "text" + str(ids[0]))
        loop = PrefillLoop(_adapter(tok), _GenModel(seq=_seq()))

        result = loop.run([torch.tensor([[0, 0, 0]])], max_new_tokens=2)

        assert result.generated_texts == ["text10"]

    def test_run_truncates_at_eos_when_disable_eos(self):
        tok = _FakeTokenizer(eos_id=11)
        # new ids 含 eos: [10, 11, 12]
        seq = torch.tensor([[0, 0, 0, 10, 11, 12]])
        loop = PrefillLoop(_adapter(tok), _GenModel(seq=seq))

        result = loop.run([torch.tensor([[0, 0, 0]])], max_new_tokens=3, disable_eos=True)

        assert result.generated_token_ids == [[10]]
        assert loop.model.last_kwargs.get("eos_token_id") is None

    def test_run_accepts_global_sample_indices_when_given(self):
        loop = PrefillLoop(_adapter(), _GenModel(seq=_seq()))

        result = loop.run([torch.tensor([[0, 0, 0]])], max_new_tokens=2, global_sample_indices=[5])

        assert result.generated_token_ids == [[10, 11]]

    def test_generate_one_sets_streamer_when_tokenizer_present(self):
        model = _GenModel(seq=_seq())
        streamer = Mock()
        loop = PrefillLoop(_adapter(_FakeTokenizer(pad_id=0)), model)
        with patch(
            "msmodelslim.core.infer_engine.fake_quant.prefill_loop._LoggingTextStreamer",
            return_value=streamer,
        ):
            ids = loop._generate_one(torch.tensor([[0, 0, 0]]), max_new_tokens=2, sample_idx=0)

        assert ids == [10, 11]
        assert model.last_kwargs["pad_token_id"] == 0
        assert model.last_kwargs["streamer"] is streamer

    def test_generate_one_omits_streamer_when_no_tokenizer(self):
        model = _GenModel(seq=_seq())
        loop = PrefillLoop(_adapter(), model)

        ids = loop._generate_one(torch.tensor([[0, 0, 0]]), max_new_tokens=2, sample_idx=0)

        assert ids == [10, 11]
        assert "streamer" not in model.last_kwargs

    # ---- _to_generate_kwargs ----

    def test_to_generate_kwargs_maps_tensor_when_input(self):
        loop = PrefillLoop(_adapter(), _GenModel())
        x = torch.tensor([[0, 0, 0]])

        assert "input_ids" in loop._to_generate_kwargs(x)

    def test_to_generate_kwargs_maps_list_when_input_pair(self):
        loop = PrefillLoop(_adapter(), _GenModel())
        x = torch.tensor([[0, 0, 0]])
        mask = torch.ones(1, 3, dtype=torch.long)

        kwargs = loop._to_generate_kwargs([x, mask])

        assert kwargs == {"input_ids": x, "attention_mask": mask}

    def test_to_generate_kwargs_keeps_non_none_dict_when_input_dict(self):
        loop = PrefillLoop(_adapter(), _GenModel())
        x = torch.tensor([[0, 0, 0]])

        kwargs = loop._to_generate_kwargs({"input_ids": x, "extra": None, "kv": 1})

        assert kwargs == {"input_ids": x, "kv": 1}

    def test_to_generate_kwargs_raises_when_dict_missing_input_ids(self):
        loop = PrefillLoop(_adapter(), _GenModel())

        with pytest.raises(InvalidModelError):
            loop._to_generate_kwargs({"attention_mask": torch.ones(1, 3)})

    def test_to_generate_kwargs_raises_when_short_list(self):
        loop = PrefillLoop(_adapter(), _GenModel())

        with pytest.raises(InvalidModelError):
            loop._to_generate_kwargs([1])

    def test_to_generate_kwargs_raises_when_unsupported_type(self):
        loop = PrefillLoop(_adapter(), _GenModel())

        with pytest.raises(InvalidModelError):
            loop._to_generate_kwargs(3.14)

    # ---- 内部工具 ----

    def test_build_streamer_returns_none_when_no_tokenizer(self):
        loop = PrefillLoop(_adapter(), _GenModel())

        assert loop._build_streamer(0) is None

    def test_build_streamer_returns_none_when_text_streamer_unavailable(self):
        loop = PrefillLoop(_adapter(_FakeTokenizer()), _GenModel())
        with patch(
            "transformers.generation.streamers.TextStreamer",
            side_effect=ImportError("no transformers"),
        ):
            assert loop._build_streamer(0) is None

    def test_pad_token_id_prefers_tokenizer_when_present(self):
        tok = _FakeTokenizer(pad_id=9)
        loop = PrefillLoop(_adapter(tok), _GenModel(config=SimpleNamespace(pad_token_id=8)))

        assert loop._pad_token_id() == 9

    def test_pad_token_id_falls_back_to_config_when_tokenizer_without_pad(self):
        loop = PrefillLoop(_adapter(_FakeTokenizer()), _GenModel(config=SimpleNamespace(pad_token_id=8)))

        assert loop._pad_token_id() == 8

    def test_pad_token_id_returns_none_when_absent(self):
        assert PrefillLoop(_adapter(), _GenModel())._pad_token_id() is None

    def test_decode_generated_returns_empty_when_no_ids(self):
        loop = PrefillLoop(_adapter(_FakeTokenizer()), _GenModel())

        assert loop._decode_generated([[]]) == [""]

    def test_decode_generated_uses_tokenizer_when_present(self):
        loop = PrefillLoop(_adapter(_FakeTokenizer(decode=lambda ids: "d")), _GenModel())

        assert loop._decode_generated([[1, 2]]) == ["d"]

    def test_resolve_eos_token_ids_merges_tokenizer_and_config(self):
        loop = PrefillLoop(
            _adapter(_FakeTokenizer(eos_id=[1, 2])),
            _GenModel(config=SimpleNamespace(eos_token_id=3)),
        )

        assert loop._resolve_eos_token_ids() == {1, 2, 3}

    def test_resolve_eos_token_ids_returns_empty_when_absent(self):
        assert PrefillLoop(_adapter(), _GenModel())._resolve_eos_token_ids() == set()

    def test_truncate_at_eos_keeps_prefix_when_eos_found(self):
        loop = PrefillLoop(_adapter(_FakeTokenizer(eos_id=7)), _GenModel())

        assert loop._truncate_at_eos([1, 7, 9]) == [1]

    def test_truncate_at_eos_returns_full_when_no_eos(self):
        loop = PrefillLoop(_adapter(_FakeTokenizer(eos_id=7)), _GenModel())

        assert loop._truncate_at_eos([1, 2]) == [1, 2]


class TestLoggingTextStreamer:
    """对应 _LoggingTextStreamer。"""

    def test_init_routes_finalized_text_when_constructed(self):
        fake_streamer = Mock()
        with patch(
            "transformers.generation.streamers.TextStreamer",
            return_value=fake_streamer,
        ):
            streamer = _LoggingTextStreamer(_FakeTokenizer(), sample_idx=2)

        assert streamer._sample_idx == 2
        fake_streamer.on_finalized_text = streamer.on_finalized_text

    def test_on_finalized_text_accumulates_and_skips_when_empty(self):
        fake_streamer = Mock()
        with patch(
            "transformers.generation.streamers.TextStreamer",
            return_value=fake_streamer,
        ):
            streamer = _LoggingTextStreamer(_FakeTokenizer(), sample_idx=0)
        with patch("msmodelslim.core.infer_engine.fake_quant.prefill_loop.get_logger") as mock_log:
            streamer.on_finalized_text("he")
            streamer.on_finalized_text("llo")
            streamer.on_finalized_text("")

        assert streamer._acc == "hello"
        assert mock_log.return_value.info.call_count == 2

    def test_put_put_value_and_end_delegate_when_called(self):
        fake_streamer = Mock()
        with patch(
            "transformers.generation.streamers.TextStreamer",
            return_value=fake_streamer,
        ):
            streamer = _LoggingTextStreamer(_FakeTokenizer(), sample_idx=0)

        streamer.put(torch.tensor([1]))
        streamer.put_value(1)
        streamer.end()

        fake_streamer.put.assert_called_once_with(torch.tensor([1]))
        fake_streamer.put_value.assert_called_once_with(1)
        fake_streamer.end.assert_called_once()
