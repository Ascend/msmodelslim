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

训练中间产物格式（industry: **checkpoint** / intermediate checkpoint）。

- ``CheckpointSaverFormat``（``IFormatSaver``）：写 ``qat_checkpoint.json`` + ``model.safetensors``
- Deploy 侧不经本包 Loader：由 ``msmodelslim.app.train.deployer.QATDeployer`` 合成
  ``ConvertConfig``，经 convert 管线 ``CheckpointReader`` 读盘并落 AscendV1。
"""

from __future__ import annotations

from msmodelslim.format.checkpoint_format.checkpoint_json_writer_factory_infra import (
    CheckpointJsonWriterFactoryInfra,
)
from msmodelslim.format.checkpoint_format.checkpoint_saver_format import CheckpointSaverFormat
from msmodelslim.format.checkpoint_format.checkpoint_tensors_writer_factory_infra import (
    CheckpointSafetensorsWriterFactoryInfra,
)
from msmodelslim.format.checkpoint_format.constants import (
    _STRUCTURE_FILE,
    _WEIGHT_FILE,
    canonical_module_name,
)

__all__ = [
    "CheckpointSaverFormat",
    "CheckpointJsonWriterFactoryInfra",
    "CheckpointSafetensorsWriterFactoryInfra",
    "canonical_module_name",
    "_STRUCTURE_FILE",
    "_WEIGHT_FILE",
]
