#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from msmodelslim.format.checkpoint_format.constants import (
    FORMAT_NAME,
    STRUCTURE_FILE,
    WEIGHT_FILE,
    canonical_module_name,
)


def test_constants_expose_expected_filenames():
    assert STRUCTURE_FILE == "qat_checkpoint.json"
    assert WEIGHT_FILE == "model.safetensors"
    assert FORMAT_NAME == "msmodelslim_qat_checkpoint"


def test_canonical_module_name_strips_fsdp_wrapper_markers():
    raw = "model.layers.0._fsdp_wrapped_module.mlp.up_proj"
    assert canonical_module_name(raw) == "model.layers.0.mlp.up_proj"


def test_canonical_module_name_strips_multiple_wrapper_markers():
    raw = "a._checkpoint_wrapped_module.b._orig_mod.c"
    assert canonical_module_name(raw) == "a.b.c"


def test_canonical_module_name_returns_same_when_no_wrapper():
    name = "model.embed_tokens"
    assert canonical_module_name(name) == name
