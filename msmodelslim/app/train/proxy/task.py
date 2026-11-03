#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

from pydantic import Field

from msmodelslim.core.train.task import TRAIN_TASK_PLUGIN_GROUP
from msmodelslim.utils.plugin import TypedConfig


@TypedConfig.plugin_entry(entry_point_group=TRAIN_TASK_PLUGIN_GROUP)
class TrainTaskConfig(TypedConfig):
    """Proxy-level train task shell; ``type`` dispatches to backends (e.g. qat)."""

    type: TypedConfig.TypeField = Field(
        default="Unknown",
        description="训练范式：qat 等；YAML 必须显式指定。",
    )


class TrainTask:
    """Proxy recipe: carries typed config for type-based dispatch."""

    def __init__(self, config: TrainTaskConfig):
        self.config = config
