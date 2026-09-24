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

from pathlib import Path
from typing import List, Any

from msmodelslim.core.quant_service import DatasetLoaderInfra as qsdl
from msmodelslim.core.tune_strategy.dataset_loader_infra import DatasetLoaderInfra as tsdl
from msmodelslim.utils.exception import InvalidDatasetError, SchemaValidateError
from msmodelslim.utils.security import get_valid_read_path, check_read_permission
from msmodelslim.utils.security.model import SafeGenerator
from msmodelslim.utils.security.path import json_safe_load


class FileDatasetLoader(
    qsdl,
    tsdl,
):
    def __init__(self, dataset_dir: Path):
        self.dir = dataset_dir

    @staticmethod
    def _validate_string_list(data: Any, dataset_id: str) -> List[str]:
        """Validate that the loaded data is a list of strings"""
        if not isinstance(data, list):
            raise InvalidDatasetError(
                f'Dataset {dataset_id} must be a list, but got {type(data)}',
                action='Please ensure the dataset file contains a JSON array or JSONL format with string values',
            )

        return data

    def get_dataset_by_name(self, dataset_id: str) -> List[str]:
        if not isinstance(dataset_id, str):
            raise SchemaValidateError('dataset_id must be a str', action='Please make sure the dataset_id is a string')

        # Check if dataset_id is an absolute path or relative path
        dataset_path = Path(dataset_id)
        if dataset_path.is_absolute() or dataset_path.exists():
            # If it's an absolute path or the path exists, use it directly
            dataset_path = dataset_path  # pylint: disable=self-assigning-variable
        else:
            # If it's not a path, combine with self.dir
            dataset_path = self.dir / dataset_id

        # Determine file type and validate path
        check_read_permission(str(dataset_path))
        if dataset_id.endswith('.json'):
            get_valid_read_path(str(dataset_path), "json")
        else:
            get_valid_read_path(str(dataset_path), "jsonl")

        try:
            # Load based on file extension
            if dataset_id.endswith('.json'):
                data = json_safe_load(str(dataset_path))
            else:
                data = SafeGenerator.load_jsonl(dataset_path)

            # Validate that the loaded data is a list of strings
            return self._validate_string_list(data, dataset_id)
        except Exception as e:
            raise InvalidDatasetError(
                f'Failed to load dataset {dataset_id}', action='Please check the dataset path and format'
            ) from e
