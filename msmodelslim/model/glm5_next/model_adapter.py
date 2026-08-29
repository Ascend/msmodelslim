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
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import UnsupportedError
from msmodelslim.utils.logging import logger_setter

from ..default.model_adapter import DefaultModelAdapter
from .w8a8_dynamic import DEFAULT_PART_FILE_SIZE, convert_to_w8a8_dynamic, get_quantized_tensor_patterns

# The adapter deliberately does not inherit the anti-outlier / QuaRot / FA3 interfaces: no
# rotation and no smooth-quant fusion is validated for GLM-5.3-Flash, only the plain
# W8A8_DYNAMIC conversion of the FFN projections (see w8a8_dynamic.py).


@logger_setter("msmodelslim.model.glm5_next")
class Glm5NextModelAdapter(DefaultModelAdapter):
    """Model adapter of GLM-5.3-Flash (``model_type=glm5_next``).

    GLM-5.3-Flash is a 45-layer hybrid model: 34 KDA (kimi delta attention) linear-attention
    layers, 11 DeepSeek-style sparse-MLA layers with an indexer, multi-hyper-connection (MHC)
    residual streams, a 288-expert MoE and one MTP draft layer.

    Only the calibration-free BF16 -> W8A8_DYNAMIC conversion is supported for now, because a
    calibration run needs a ``forward`` implementation of the KDA / MHC layers (transformers
    ships ``glm5_next`` from 5.16.0 on, which is newer than the pinned versions of the other
    GLM pedigrees). Please use :meth:`export_w8a8_dynamic`.
    """

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return "glm5_next"

    def get_quantized_tensor_patterns(self) -> tuple:
        """Return the patterns of the tensors that are converted to W8A8_DYNAMIC.

        They match the routed-expert, shared-expert and dense FFN projections. The MTP draft
        layer is filtered out by
        :func:`msmodelslim.model.glm5_next.w8a8_dynamic.should_quantize`.
        """
        return get_quantized_tensor_patterns()

    def get_adapter_config_for_subgraph(self) -> list:
        """No norm-linear / ov fusion subgraph is validated for glm5_next."""
        return []

    def export_w8a8_dynamic(self, save_path: str, part_file_size: int = DEFAULT_PART_FILE_SIZE) -> None:
        """Convert the BF16 checkpoint of this adapter into the ModelSlim W8A8_DYNAMIC format."""
        convert_to_w8a8_dynamic(str(self.model_path), save_path, part_file_size=part_file_size)

    def load_model(self, device: DeviceType = DeviceType.NPU):
        self._raise_calibration_unsupported("load_model")

    def init_model(self, device: DeviceType = DeviceType.NPU):
        self._raise_calibration_unsupported("init_model")

    def _raise_calibration_unsupported(self, entry: str) -> None:
        raise UnsupportedError(
            f"{entry} is not supported by the glm5_next pedigree yet: the calibration pipeline "
            "needs a forward implementation of the KDA linear-attention and MHC layers, which "
            "this adapter does not provide.",
            action="Please use msmodelslim.model.glm5_next.w8a8_dynamic.convert_to_w8a8_dynamic "
            "for the calibration-free W8A8_DYNAMIC conversion",
        )
