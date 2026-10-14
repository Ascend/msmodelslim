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
from typing import Optional

from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.tune_strategy import EvaluateResult
from .evaluation_service_infra import EvaluateServiceConfig


class TuningAccuracyInfra(ABC):
    """
    Abstract interface for tuning accuracy operations.
    Provides methods for accuracy retrieval and storage.
    """

    @abstractmethod
    def get_accuracy(
        self, practice: PracticeConfig, evaluation_config: EvaluateServiceConfig
    ) -> Optional[EvaluateResult]:
        """
        Get accuracy from history for the given practice and evaluation config.
        practice and evaluation_config form a composite key.

        Args:
            practice: PracticeConfig, the practice config
            evaluation_config: EvaluateServiceConfig, the evaluation configuration

        Returns:
            Optional[EvaluateResult]: The evaluation result if found, None otherwise
        """
        ...

    @abstractmethod
    def append_accuracy(
        self, practice: PracticeConfig, evaluation_config: EvaluateServiceConfig, evaluation: EvaluateResult
    ) -> None:
        """
        Append an accuracy record to the database.
        Accuracy records are append-only (never deleted).
        practice and evaluation_config form a composite key.

        Args:
            practice: PracticeConfig, the practice config
            evaluation_config: EvaluateServiceConfig, the evaluation configuration
            evaluation: EvaluateResult, the evaluation result
        """
        ...

    @abstractmethod
    def get_accuracy_count(self) -> int:
        """
        Get the total number of accuracy records.

        Returns:
            int: The total number of accuracy records
        """
        ...


class TuningAccuracyManagerInfra(ABC):
    """
    Abstract interface for loading tuning accuracy manager.
    """

    @abstractmethod
    def load_accuracy(self, database: str) -> TuningAccuracyInfra:
        """
        Load tuning accuracy from the specified database path.

        Args:
            database: str, the path to the history database directory

        Returns:
            TuningAccuracyInfra: The tuning accuracy instance

        Raises:
            RuntimeError: If failed to create or load accuracy database
        """
        ...
