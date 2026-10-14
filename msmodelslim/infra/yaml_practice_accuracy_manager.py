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

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict

from msmodelslim.app.auto_tuning import TuningAccuracyManagerInfra, TuningAccuracyInfra
from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateServiceConfig
from msmodelslim.core.practice import PracticeConfig
from msmodelslim.core.tune_strategy import EvaluateResult
from msmodelslim.utils.hash import calculate_md5
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import (
    yaml_safe_load,
    yaml_safe_dump,
    get_write_directory,
)
from msmodelslim.utils.yaml_database import YamlDatabase


class YamlDatabaseAccuracy(BaseModel):
    accuracy_database: YamlDatabase  # Accuracy database: manages accuracy.yaml

    model_config = ConfigDict(arbitrary_types_allowed=True)


class YamlTuningAccuracy(TuningAccuracyInfra):
    """
    YAML-based implementation of TuningAccuracyInfra.
    Manages accuracy.yaml (accuracy records).
    Accuracy records are stored with both evaluation_md5 and quant_config_md5 as keys.
    """

    def __init__(self, database: str):
        """
        Initialize YamlTuningAccuracy.

        Args:
            database: str, the path to the history database directory
        """
        self.database = database
        get_write_directory(database)
        self.history_dir = Path(database)

        # Initialize database
        accuracy_database_dir = self.history_dir
        self._database_accuracy = YamlDatabaseAccuracy(
            accuracy_database=YamlDatabase(
                config_dir=accuracy_database_dir,
                read_only=False,
            ),
        )

        # Accuracy file path
        self._accuracy_file_path = self.history_dir / 'accuracy.yaml'

        # Load accuracy cache into memory (for read operations)
        # Structure: {evaluation_md5-quant_config_md5: evaluation_dict}
        self._accuracy_cache: Dict[str, Dict] = self._load_accuracy_database()

    def _load_accuracy_database(self) -> Dict[str, Dict]:
        """
        Load accuracy data from accuracy database interface.
        Expected format: {evaluation_md5-quant_config_md5: evaluation_dict}
        """
        accuracy_database = self._database_accuracy.accuracy_database
        accuracy_key = "accuracy"

        if accuracy_key not in accuracy_database:
            get_logger().debug("Accuracy database key %r does not exist. Starting with empty cache.", accuracy_key)
            return {}

        try:
            content = accuracy_database[accuracy_key]
            if content is None:
                return {}

            if not isinstance(content, dict):
                get_logger().warning("Accuracy database content is not a dict. Starting with empty cache.")
                return {}

            # Format: {evaluation_md5-quant_config_md5: evaluation_dict}
            validated_content = {}
            for composite_key, evaluation_dict in content.items():
                if not isinstance(evaluation_dict, dict):
                    get_logger().warning(
                        "Invalid format in accuracy database for key %r: value is not a dict. Skipping.", composite_key
                    )
                    continue

                if "accuracies" not in evaluation_dict:
                    get_logger().warning(
                        "Invalid format in accuracy database for key %r: "
                        "evaluation_dict is missing 'accuracies' field. Skipping.",
                        composite_key,
                    )
                    continue

                validated_content[composite_key] = evaluation_dict

            return validated_content
        except Exception as e:
            get_logger().debug("Failed to load accuracy cache from database: %s. Starting with empty cache.", e)
            return {}

    def get_accuracy(
        self, practice: PracticeConfig, evaluation_config: EvaluateServiceConfig
    ) -> Optional[EvaluateResult]:
        """
        Get accuracy from history for the given practice and evaluation config.
        practice and evaluation_config form a composite key.
        """
        evaluation_md5 = calculate_md5(evaluation_config)
        quant_config_md5 = calculate_md5(practice.extract_quant_config())
        composite_key = f"{evaluation_md5}-{quant_config_md5}"

        if composite_key not in self._accuracy_cache:
            return None

        evaluation_dict = self._accuracy_cache[composite_key]
        return EvaluateResult.model_validate(evaluation_dict)

    def append_accuracy(
        self, practice: PracticeConfig, evaluation_config: EvaluateServiceConfig, evaluation: EvaluateResult
    ) -> None:
        """
        Append an accuracy record to the database.
        Accuracy records are append-only (never deleted).
        practice and evaluation_config form a composite key.
        If the record already exists, it will be overwritten.
        """
        evaluation_md5 = calculate_md5(evaluation_config)
        quant_config_md5 = calculate_md5(practice.extract_quant_config())
        composite_key = f"{evaluation_md5}-{quant_config_md5}"

        with _get_modifiable_accuracy_cache(self._accuracy_file_path) as accuracy_cache:
            if composite_key in accuracy_cache:
                get_logger().warning(
                    "Accuracy record already exists for evaluation_md5 %r and quant_config_md5 %r. Overwriting with new data.",
                    evaluation_md5,
                    quant_config_md5,
                )

            evaluation_dict = evaluation.model_dump(mode='json')
            accuracy_cache[composite_key] = evaluation_dict
            self._accuracy_cache = accuracy_cache

        get_logger().info(
            "Saved accuracy record for evaluation_md5 %r and quant_config_md5 %r", evaluation_md5, quant_config_md5
        )

    def get_accuracy_count(self) -> int:
        """
        Get the total number of accuracy records.

        Returns:
            int: The total number of accuracy records
        """
        return len(self._accuracy_cache)


class YamlTuningAccuracyManager(TuningAccuracyManagerInfra):
    """Manager for loading YAML-based tuning accuracy."""

    def load_accuracy(self, database: str) -> TuningAccuracyInfra:
        """
        Load the tuning accuracy from the specified database path.
        Returns an accuracy instance even if no records exist (empty accuracy).

        Args:
            database: str, the path to the history database directory

        Returns:
            TuningAccuracyInfra: The tuning accuracy instance

        Raises:
            RuntimeError: If failed to create or load accuracy database
        """
        try:
            return YamlTuningAccuracy(database)
        except Exception as e:
            error_msg = f"Failed to create or load accuracy database from {database}."
            get_logger().error(error_msg)
            raise RuntimeError(error_msg) from e


@contextmanager
def _get_modifiable_accuracy_cache(accuracy_file_path: Path):
    """
    Context manager for modifying accuracy cache.
    Responsible for:
    1. Loading accuracy cache from file
    2. Writing modified accuracy cache back to file
    """
    if accuracy_file_path.exists():
        accuracy_content = yaml_safe_load(str(accuracy_file_path))
        if accuracy_content is None:
            accuracy_cache = {}
        elif isinstance(accuracy_content, dict):
            accuracy_cache = accuracy_content
        else:
            get_logger().warning("Accuracy file content is not a dict. Starting with empty cache.")
            accuracy_cache = {}
    else:
        accuracy_cache = {}

    try:
        yield accuracy_cache
    finally:
        yaml_safe_dump(accuracy_cache, str(accuracy_file_path))
