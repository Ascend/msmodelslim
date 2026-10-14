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

from abc import ABC, abstractmethod

from typing import List, Optional  # noqa: F401
from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.tune_strategy import EvaluateResult


class TuningHistoryInfra(ABC):
    """
    Abstract interface for tuning history operations.
    Provides methods for history process management.
    """

    @abstractmethod
    def append_history(self, practice: PracticeConfig, evaluation: EvaluateResult) -> None:
        """
        Append a history record to the database.

        Args:
            practice: PracticeConfig, the practice config
            evaluation: EvaluateResult, the evaluation result
        """
        ...

    @abstractmethod
    def clear_records(self) -> None:
        """
        Clear history records (history.yaml and practice config files), but preserve accuracy cache.
        This is a dangerous operation and should be called explicitly.
        """
        ...


class TuningHistoryManagerInfra(ABC):
    """
    Abstract interface for loading tuning history.
    """

    @abstractmethod
    def load_history(self, database: str) -> TuningHistoryInfra:
        """
        Load tuning history from the specified database path.

        Args:
            database: str, the path to the history database directory

        Returns:
            TuningHistoryInfra: The tuning history instance

        Raises:
            RuntimeError: If failed to create or load history
        """
        ...
