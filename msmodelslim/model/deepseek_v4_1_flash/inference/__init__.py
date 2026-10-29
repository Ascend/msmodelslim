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

DeepSeek-V4.1-Flash concrete inference graph, embedded in the model adapter.
The checkpoint source ships its text/VL forward graph under ``inference/``
(``model.py`` plus ``engram.py`` / ``kernel.py`` / ``vision.py`` / ``image_processor.py``).
This package makes that concrete script first-class adapter code: ``model`` is a
direct import (no runtime ``sys.modules`` bridge), ``kernel`` is the CPU
calibration kernel (eager QDQ / sinkhorn / sparse-attention, see ``kernel.py``),
and ``loader`` builds executable on-device blocks / Engram projections from a
source checkpoint exactly as the checkpoint's own ``inference/model.py``
bridge did — its functions and the adapter-side handle (``reference.py``) now
live here, driven off the embedded graph instead of a dynamically loaded script.

The adapter engages this graph for a real forward when the source checkpoint
carries ``inference/config.json``; otherwise it falls back to the structural
(decode-only) model for the data-free W8A8 / W4A8 paths.
"""

from msmodelslim.model.deepseek_v4_1_flash.inference import engram, image_processor, kernel, loader, model, vision

__all__ = ["model", "kernel", "engram", "vision", "image_processor", "loader"]
