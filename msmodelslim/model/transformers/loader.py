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

from pathlib import Path

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader

from .detect import LLM_ADAPTER_CLASS_PATH, resolve_adapter_class_path


class TransformersAdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = LLM_ADAPTER_CLASS_PATH

    def load(
        self,
        model_type: str,
        model_path: Path,
        trust_remote_code: bool = False,
    ):
        self.ADAPTER_CLASS_PATH = resolve_adapter_class_path(model_path)
        return super().load(model_type, model_path, trust_remote_code)
