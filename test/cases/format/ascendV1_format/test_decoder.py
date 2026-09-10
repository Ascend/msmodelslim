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

msmodelslim/format/ascendV1_format/decoder.py 的单元测试。
"""

import pytest

from msmodelslim.format.ascendV1_format.decoder import (
    AscendV1FormatDecoder,
    branch_from_submodule,
    discover_fa3_targets,
    fa3_scheme_from_module,
    fallback_submodule_names,
    _faquant_scale_module_names,
    _parse_fa_quant_type,
)
from msmodelslim.format.ascendV1_format.decoders.fa3_per_block import Fa3PerBlockDecoder
from msmodelslim.format.ascendV1_format.decoders.fa3_per_head import Fa3PerHeadDecoder
from msmodelslim.format.ascendV1_format.decoders.fa3_per_token import Fa3PerTokenDecoder
from msmodelslim.format.ascendV1_format.decoders.kv_cache_c8 import KvCacheC8Decoder
from msmodelslim.format.ascendV1_format.decoders.w8a8_dynamic import W8A8DynamicDecoder
from msmodelslim.format.ascendV1_format.decoders.w8a8_mx import W8A8MxDecoder
from msmodelslim.format.ascendV1_format.decoders.w8a8_static import W8A8StaticDecoder
from msmodelslim.ir.const import (
    fp8_e4m3_per_head_sym,
    fp8_e4m3_per_token_sym,
    int8_per_channel_sym,
    int8_per_head_sym,
    int8_per_token_sym,
    mxfp4_per_block_sym,
    mxfp8_per_block_sym,
)
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


class TestParseFaQuantType:
    """对应 _parse_fa_quant_type 与 FA3 解析。"""

    def test_parse_defaults_branches_when_no_act_prefix(self):
        parsed = _parse_fa_quant_type("INT8")

        assert parsed == {
            "Q": int8_per_head_sym,
            "K": int8_per_head_sym,
            "V": int8_per_head_sym,
        }

    def test_parse_dynamic_default_branches_when_fp8_dynamic(self):
        parsed = _parse_fa_quant_type("FP8_DYNAMIC")

        assert parsed["Q"] == fp8_e4m3_per_token_sym
        assert parsed["V"] == fp8_e4m3_per_token_sym

    def test_parse_single_branch_when_act_prefix_given(self):
        parsed = _parse_fa_quant_type("Q_FP8")

        assert parsed == {"Q": fp8_e4m3_per_head_sym}

    def test_parse_multi_branch_and_multi_segment_when_combined(self):
        parsed = _parse_fa_quant_type("Q_FP8_DYNAMIC_KV_INT8")

        assert parsed["Q"] == fp8_e4m3_per_token_sym
        assert parsed["K"] == int8_per_head_sym
        assert parsed["V"] == int8_per_head_sym

    def test_parse_mxfp_branches_when_dynamic(self):
        parsed = _parse_fa_quant_type("MXFP4_DYNAMIC")

        assert parsed["Q"] == mxfp4_per_block_sym

    def test_parse_raise_when_invalid_dtype(self):
        with pytest.raises(SchemaValidateError):
            _parse_fa_quant_type("Q_WXYZ")

    def test_parse_raise_when_empty_or_non_string(self):
        with pytest.raises(SchemaValidateError):
            _parse_fa_quant_type("")
        with pytest.raises(SchemaValidateError):
            _parse_fa_quant_type(None)


class TestFa3TargetDiscovery:
    """对应 _faquant_scale_module_names / discover_fa3_targets / 反查。"""

    def test_scale_module_names_collects_only_faquant_scale_keys(self):
        description = {
            "lm_attn.fa_q.scale": "FAQuant",
            "lm_attn.fa3_v.scale": "FAQuant",
            "lm_attn.not_branch.scale": "FAQuant",  # branch W 不在回退表
            "lm_attn.weight": "INT8",
        }

        names = _faquant_scale_module_names(description)

        assert names == {"lm_attn": {"Q": "lm_attn.fa_q", "V": "lm_attn.fa3_v"}}

    def test_discover_fa3_targets_combines_quant_type_and_scale_names(self):
        description = {
            "lm_attn.quant_type": "INT8",
            "lm_attn.fa_q.scale": "FAQuant",
            "layer2.quant_type": "Q_FP8_DYNAMIC",
        }

        targets = discover_fa3_targets(description)
        by_attn = {}
        for t in targets:
            by_attn.setdefault(t.attn_prefix, {})[t.branch] = t

        assert set(by_attn["lm_attn"]) == {"Q", "K", "V"}
        assert by_attn["lm_attn"]["Q"].scheme == int8_per_head_sym
        assert by_attn["lm_attn"]["Q"].module_name == "lm_attn.fa_q"
        assert by_attn["layer2"] == {"Q": by_attn["layer2"]["Q"]}
        assert by_attn["layer2"]["Q"].scheme == fp8_e4m3_per_token_sym

    def test_discover_fa3_targets_empty_when_no_quant_type(self):
        assert not discover_fa3_targets({"x.weight": "W8A8"})

    def test_fallback_submodule_names_returns_map_or_empty(self):
        assert fallback_submodule_names("Q") == ("fa_q", "fa3_q")
        assert fallback_submodule_names("X") == ()

    def test_branch_from_submodule_returns_upper_tail(self):
        assert branch_from_submodule("fa_q") == "Q"

    def test_fa3_scheme_from_module_recognises_known_scheme_only(self):
        class _Mod:
            x_q_scheme = int8_per_head_sym

        class _Other:
            x_q_scheme = int8_per_channel_sym

        assert fa3_scheme_from_module(_Mod()) == int8_per_head_sym
        assert fa3_scheme_from_module(_Other()) is None
        assert fa3_scheme_from_module(object()) is None


class TestAscendV1FormatDecoder:
    """对应 AscendV1FormatDecoder。"""

    @staticmethod
    def _decoder(fake_store_cls):
        return AscendV1FormatDecoder(fake_store_cls())

    def test_linear_dispatches_to_decoder_by_label(self, fake_store_cls):
        decoder = self._decoder(fake_store_cls)

        assert isinstance(decoder.linear("W8A8"), W8A8StaticDecoder)
        assert isinstance(decoder.linear("W8A8_DYNAMIC"), W8A8DynamicDecoder)
        assert isinstance(decoder.linear("W8A8_MXFP8"), W8A8MxDecoder)
        assert decoder.has_linear_label("W8A8") is True
        assert decoder.has_linear_label("NOPE") is False

    def test_linear_raises_when_unknown_label(self, fake_store_cls):
        with pytest.raises(UnsupportedError):
            self._decoder(fake_store_cls).linear("NOPE")

    def test_fa3_dispatches_by_scheme(self, fake_store_cls):
        decoder = self._decoder(fake_store_cls)

        assert isinstance(decoder.fa3(int8_per_head_sym), Fa3PerHeadDecoder)
        assert isinstance(decoder.fa3(fp8_e4m3_per_head_sym), Fa3PerHeadDecoder)
        assert isinstance(decoder.fa3(int8_per_token_sym), Fa3PerTokenDecoder)
        assert isinstance(decoder.fa3(fp8_e4m3_per_token_sym), Fa3PerTokenDecoder)
        assert isinstance(decoder.fa3(mxfp8_per_block_sym), Fa3PerBlockDecoder)
        assert isinstance(decoder.fa3(mxfp4_per_block_sym), Fa3PerBlockDecoder)

    def test_fa3_raises_when_unknown_scheme(self, fake_store_cls):
        with pytest.raises(UnsupportedError):
            self._decoder(fake_store_cls).fa3(int8_per_channel_sym)

    def test_kv_cache_dispatches_and_raises_when_unknown(self, fake_store_cls):
        decoder = self._decoder(fake_store_cls)

        assert isinstance(decoder.kv_cache("C8"), KvCacheC8Decoder)
        with pytest.raises(UnsupportedError):
            decoder.kv_cache("NOPE")
