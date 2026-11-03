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

# SmoothQuant's algorithm-specific implementation of the lossless protocol.

from dataclasses import replace

import torch
from torch import nn

from msmodelslim.processor.lossless.protocol import (
    LosslessCheckContext,
    LosslessCheckPlan,
    LosslessCheckTarget,
)
from msmodelslim.utils.exception import UnsupportedError, UnexpectedError

from .api import compute_smooth_quant_scale


def prepare_smooth_quant_lossless_check(processor):
    if processor.is_defalut_adapter:
        raise UnsupportedError("SmoothQuant lossless check requires declared norm-linear adapter mappings")
    mappings = processor.adapter.get_adapter_config_for_subgraph()
    if any(item.subgraph_type == "norm-linear" and not item.mapping.source for item in mappings):
        raise UnsupportedError("SmoothQuant lossless check requires fused norm-linear subgraphs")
    selected = processor._filter_adapter_configs_by_config(mappings, processor.config, "")
    processor.global_adapter_config = mappings
    processor._lossless_expected_targets = tuple(sorted({name for item in selected for name in item.mapping.targets}))
    processor._lossless_alignments.clear()
    processor._lossless_check_active = True
    return LosslessCheckPlan(expected_targets=processor._lossless_expected_targets)


def create_smooth_quant_lossless_context(processor, request, reference_datas):
    processor._lossless_alignments.clear()
    prefix = f"{request.name}." if request.name else ""
    names = {f"{prefix}{name}" if name else request.name: module for name, module in request.module.named_modules()}
    expected = {
        name
        for name in processor._lossless_expected_targets
        if not request.name or name == request.name or name.startswith(prefix)
    }
    return LosslessCheckContext(
        targets=tuple(
            LosslessCheckTarget(name=name, path=name, decoder=request.name, structure_type=type(module).__name__)
            for name, module in names.items()
            if isinstance(module, nn.Linear)
        ),
        reference_datas=reference_datas,
        covered_targets=tuple(sorted(expected.intersection(names))),
        unresolved_targets=tuple(sorted(expected.difference(names))),
    )


@torch.no_grad()
def freeze_smooth_quant_alignment(processor, subgraph, linear_names, config, context):
    # Freeze independently of the subsequent weight mutation: corrupting the
    # execution must not corrupt the expected inverse transformation as well.
    scales = compute_smooth_quant_scale(subgraph, config, context).detach().clone()
    shift = context.shift.detach().clone() if config.shift else None
    for name in linear_names:
        processor._lossless_alignments[name] = (scales, shift)


def align_smooth_quant_lossless_io(processor, context, target, reference, processed):
    del context
    alignment = processor._lossless_alignments.get(target.name)
    if alignment is None:
        if target.name in processor._lossless_expected_targets:
            raise UnexpectedError(f"No SmoothQuant alignment recorded for executed target {target.name}")
        return reference, processed
    scales, shift = alignment

    def restore(value):
        if isinstance(value, torch.Tensor):
            restored = value * scales.to(value)
            return restored if shift is None else restored + shift.to(value)
        if isinstance(value, tuple):
            return tuple(restore(item) for item in value)
        if isinstance(value, list):
            return [restore(item) for item in value]
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        return value

    return reference, replace(processed, args=restore(processed.args), kwargs=restore(processed.kwargs))
