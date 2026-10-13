# -*- coding: UTF-8 -*-
"""Unit tests for modelslim_v0 calib_data helpers."""

from msmodelslim.core.quant_service.modelslim_v0.calib_data import (
    normalize_calib_data_for_calibrator,
    vlm_dict_to_calibrator_args,
)


def test_normalize_passthrough_lists():
    data = [[1, 2, 3]]
    assert normalize_calib_data_for_calibrator(data) == data


def test_normalize_converts_dicts():
    out = normalize_calib_data_for_calibrator([{"input_ids": 7, "attention_mask": 8}])
    assert out[0][0] == 7
    assert out[0][1] == 8
    assert len(out[0]) == 12


def test_vlm_dict_to_calibrator_args_defaults():
    args = vlm_dict_to_calibrator_args({})
    assert args[11] == 0
    assert args[0] is None
