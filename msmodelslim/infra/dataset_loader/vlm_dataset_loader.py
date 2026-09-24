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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from msmodelslim.core.quant_service import DatasetLoaderInfra
from msmodelslim.utils.exception import InvalidDatasetError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.security import get_valid_read_path, check_read_permission


@dataclass
class VlmCalibSample:
    """Calibration sample for multimodal VLM."""

    text: str
    image: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None


def _resolve_dataset_path(dataset_name: str, dataset_dir: Optional[Path]) -> Path:
    """
    Resolve dataset_name to an existing path.

    Strategy:
    - If absolute path: use it
    - Else if relative exists: resolve
    - Else if dataset_dir provided: dataset_dir / dataset_name
    """
    dataset_path = Path(dataset_name)

    if dataset_path.is_absolute():
        resolved_path = dataset_path
        get_logger().info("Using absolute path: %s", resolved_path)
        return resolved_path

    if dataset_path.exists():
        resolved_path = dataset_path.resolve()
        get_logger().info("Using existing relative path: %s -> %s", dataset_name, resolved_path)
        return resolved_path

    if dataset_dir is None:
        raise InvalidDatasetError(
            f"Dataset path does not exist: {dataset_name}",
            action="Please provide an absolute path, or a relative path that exists, or set dataset_dir.",
        )

    resolved_path = dataset_dir / dataset_name
    if resolved_path.exists():
        get_logger().info("Resolved short name: %s -> %s", dataset_name, resolved_path)
        return resolved_path

    fallback_resolved = dataset_path.resolve()
    if fallback_resolved.exists():
        get_logger().info("Resolved fallback path: %s -> %s", dataset_name, fallback_resolved)
        return fallback_resolved

    raise InvalidDatasetError(
        f"Dataset path does not exist: {dataset_name}",
        action=(
            "Tried the following candidates but none exists:\n"
            f"- as relative path from cwd: {dataset_path.resolve()}\n"
            f"- as short name under dataset_dir: {dataset_dir / dataset_name}\n"
            "Please check the name/path, or put the dataset under dataset_dir."
        ),
    )


@logger_setter("msmodelslim.infra.vlm_dataset_loader")
class VLMDatasetLoader(DatasetLoaderInfra):
    """
    Refactored VLM dataset loader with chain-of-responsibility routing.

    Entry:
    - get_dataset_by_name(dataset_name)

    Routing chain (in order):
    1) JsonlDatasetLoader
       - only supports dataset_name being a file named index.json or index.jsonl
    2) IndexedDirectoryDatasetLoader
       - only supports dataset_name being a directory containing exactly one of index.json or index.jsonl
         (if both exist -> error)
    3) LegacyDirectoryDatasetLoader
       - legacy directory behavior for TEXT + IMAGE only (no audio/video directory scanning)
    """

    DEFAULT_TEXT = "Describe this image in detail."

    def __init__(self, dataset_dir: Optional[Path] = None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.default_text = self.DEFAULT_TEXT

    def get_dataset_by_name(self, dataset_name: str) -> List[VlmCalibSample]:  # pylint: disable=arguments-renamed
        resolved_path = _resolve_dataset_path(dataset_name, self.dataset_dir)

        if not resolved_path.exists():
            raise InvalidDatasetError(
                f"Dataset path does not exist: {resolved_path}",
                action="Please check the path and permissions.",
            )

        # Validate existence via security layer (file/dir)
        check_read_permission(str(resolved_path))
        resolved_path = Path(
            get_valid_read_path(str(resolved_path), is_dir=resolved_path.is_dir(), check_user_stat=True)
        )

        # Import loaders lazily to avoid circular import with VlmCalibSample definition.
        from .jsonl_dataset_loader import JsonlDatasetLoader
        from .indexed_directory_dataset_loader import IndexedDirectoryDatasetLoader
        from .legacy_directory_dataset_loader import LegacyDirectoryDatasetLoader

        chain = [
            JsonlDatasetLoader(
                default_text=self.default_text,
            ),
            IndexedDirectoryDatasetLoader(
                default_text=self.default_text,
            ),
            LegacyDirectoryDatasetLoader(
                default_text=self.default_text,
            ),
        ]

        for loader in chain:
            if loader.is_support(resolved_path):
                get_logger().info("Using dataset loader: %s", loader.__class__.__name__)
                return loader.load_data(resolved_path)

        raise InvalidDatasetError(
            f"Unsupported dataset type: {resolved_path}",
            action=(
                "Supported inputs: index.json/index.jsonl file; directory with exactly one index.json/index.jsonl; "
                "or legacy text+image directory."
            ),
        )
