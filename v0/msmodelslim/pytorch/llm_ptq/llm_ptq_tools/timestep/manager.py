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
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""


from typing import List, Dict, Any, Optional, Union, Callable
import contextvars

from msmodelslim import logger


class TimestepManager:
    """Manages timestep indices for multi-modal quantization processes."""

    _timestep_var = contextvars.ContextVar("timestep_idx", default=None)

    @classmethod
    def set_timestep_idx(cls, t_idx: int) -> None:
        """Set the current timestep index."""
        if not isinstance(t_idx, int):
            raise ValueError("Timestep index must be an integer.")

        if t_idx < 0:
            raise ValueError("Timestep index must be non-negative.")

        current = cls._timestep_var.get()
        if current is not None and current == t_idx:
            logger.warning("Warning: Setting same timestep value consecutively: %r", t_idx)
        cls._timestep_var.set(t_idx)
        logger.debug("Timestep index set to: %r", t_idx)

    @classmethod
    def get_timestep_idx(cls) -> Optional[int]:
        """Get the current timestep index."""
        t_idx = cls._timestep_var.get()
        if t_idx is None:
            logger.warning("Warning: Timestep index not set. Call set_timestep_idx() before each timestep.")
        return t_idx


