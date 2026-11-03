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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import importlib.util
import os
from pathlib import Path
import random
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
ROTATION_FAULT_TARGET = "blocks.block_00.self_attn.k_rot"


@pytest.mark.parametrize("mode", ["positive", "rotation_fault"])
def test_official_wan_attention_online_quarot_lossless_check(mode):
    runner = os.environ.get("MSMODELSLIM_WAN22_RUNNER")
    source_root = os.environ.get("MSMODELSLIM_WAN22_SOURCE_ROOT")
    if not runner or not source_root:
        pytest.skip(
            "set MSMODELSLIM_WAN22_RUNNER and MSMODELSLIM_WAN22_SOURCE_ROOT "
            "to run the official Wan attention integration"
        )

    module_spec = importlib.util.spec_from_file_location("_wan22_online_quarot_runner", runner)
    assert module_spec is not None and module_spec.loader is not None
    runner_module = importlib.util.module_from_spec(module_spec)
    module_names = (
        "wan",
        "wan.modules",
        "wan.modules.attention",
        "wan.modules.model",
        "diffusers",
        "diffusers.configuration_utils",
        "diffusers.models",
        "diffusers.models.modeling_utils",
    )
    missing = object()
    original_modules = {name: sys.modules.get(name, missing) for name in module_names}
    original_path = list(sys.path)
    original_torch_rng = torch.get_rng_state()
    original_python_rng = random.getstate()
    try:
        module_spec.loader.exec_module(runner_module)
        sys.path.insert(0, str(REPO_ROOT))
        report = runner_module.run(mode, Path(source_root))
    finally:
        sys.path[:] = original_path
        torch.set_rng_state(original_torch_rng)
        random.setstate(original_python_rng)
        for name, module in original_modules.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    case = report["results"][mode]
    assert report["status"] == "passed"
    assert case["status"] == "passed"
    assert case["device"] == "cpu"
    assert case["model_weights"] == "random initialization; no checkpoint loaded"
    assert case["linear_target_count"] == 8
    assert case["rotation_target_count"] == 4
    assert case["module_inventory_before"]["Linear"] == 8
    assert case["result"]["active_capture_count"] == 0

    if mode == "positive":
        assert case["result"]["passed"] is True
        assert len(case["result"]["checked_targets"]) == 12
        assert case["result"]["structure_type_counts"]["Linear"] == 8
        assert case["module_inventory_after"]["OnlineRotationWrapper"] == 4
    else:
        assert case["first_failure"]["structure"] == ROTATION_FAULT_TARGET
        assert case["first_failure"]["reason"] == "metric_threshold_exceeded"
