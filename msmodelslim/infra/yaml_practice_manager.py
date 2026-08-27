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
from typing import Dict, Generator, List, Optional, Tuple

from msmodelslim.app.auto_tuning import PracticeManagerInfra as atpm
from msmodelslim.app.naive_quantization import PracticeManagerInfra as nqpm
from msmodelslim.core.practice import Metadata, PracticeConfig
from msmodelslim.utils.exception import SecurityError, UnsupportedError, SpecError
from msmodelslim.utils.security import get_valid_read_path, get_write_directory
from msmodelslim.utils.yaml_database import YamlDatabase


class YamlPracticeManager(
    nqpm,
    atpm,
):
    def __init__(
        self,
        official_config_dir: Path,
        custom_config_dir: Optional[Path] = None,
        third_party_config_dirs: Optional[List[Path]] = None,
    ):
        get_valid_read_path(str(official_config_dir), is_dir=True)
        self.official_config_dir = official_config_dir
        if custom_config_dir is not None:
            get_write_directory(str(custom_config_dir))
            get_valid_read_path(str(custom_config_dir), is_dir=True)
        self.custom_config_dir = custom_config_dir

        self.official_databases: Dict[str, YamlDatabase] = {
            model_type_dir.name: YamlDatabase(model_type_dir, read_only=True)
            for model_type_dir in self.official_config_dir.iterdir()
            if model_type_dir.is_dir()
        }

        self.custom_databases: Dict[str, YamlDatabase] = (
            {
                model_type_dir.name: YamlDatabase(model_type_dir, read_only=False)
                for model_type_dir in self.custom_config_dir.iterdir()
                if model_type_dir.is_dir()
            }
            if self.custom_config_dir
            else {}
        )

        # One dict per third-party root: pedigree -> YamlDatabase (read-only)
        self._plugin_database_maps: List[Dict[str, YamlDatabase]] = []
        for plugin_root in third_party_config_dirs or []:
            if not plugin_root.exists() or not plugin_root.is_dir():
                continue
            try:
                get_valid_read_path(str(plugin_root), is_dir=True)
                db_map = {d.name: YamlDatabase(d, read_only=True) for d in plugin_root.iterdir() if d.is_dir()}
                if db_map:
                    self._plugin_database_maps.append(db_map)
            except Exception:  # noqa: S110  # nosec B112
                continue

    def __contains__(self, model_pedigree: str) -> bool:
        model_pedigree = model_pedigree.lower()
        if model_pedigree in self.custom_databases or model_pedigree in self.official_databases:
            return True
        return any(model_pedigree in m for m in self._plugin_database_maps)

    def get_config_by_id(self, model_pedigree: str, config_id: str) -> PracticeConfig:
        model_pedigree = model_pedigree.lower()
        value = None
        if model_pedigree in self.custom_databases and config_id in self.custom_databases[model_pedigree]:
            value = self.custom_databases[model_pedigree][config_id]
        elif model_pedigree in self.official_databases and config_id in self.official_databases[model_pedigree]:
            value = self.official_databases[model_pedigree][config_id]
        else:
            for db_map in self._plugin_database_maps:
                if model_pedigree in db_map and config_id in db_map[model_pedigree]:
                    value = db_map[model_pedigree][config_id]
                    break
            if value is None:
                raise UnsupportedError(
                    f"Practice {config_id} of ModelType {model_pedigree} not found",
                    action='Please check the practice id and model type',
                )

        quant_config = PracticeConfig.model_validate(value)

        if config_id != quant_config.metadata.config_id:
            raise SecurityError(
                f"name {config_id} not match config_id {quant_config.metadata.config_id}",
                action='Please make sure the practice is not tampered',
            )
        return quant_config

    def get_config_url(self, model_pedigree: str, config_id: str) -> Optional[str]:
        """Return the URL/location of the config (same lookup order as get_config_by_id). In file-based use, url is the YAML path."""
        path = self._get_config_path(model_pedigree, config_id)
        return str(path) if path is not None else None

    def _get_config_path(self, model_pedigree: str, config_id: str) -> Optional[Path]:
        """Return the full path to the YAML file for the given config (same lookup order as get_config_by_id)."""
        model_pedigree = model_pedigree.lower()
        if model_pedigree in self.custom_databases and config_id in self.custom_databases[model_pedigree]:
            return self.custom_databases[model_pedigree].config_dir / f"{config_id}.yaml"
        if model_pedigree in self.official_databases and config_id in self.official_databases[model_pedigree]:
            return self.official_databases[model_pedigree].config_dir / f"{config_id}.yaml"
        for db_map in self._plugin_database_maps:
            if model_pedigree in db_map and config_id in db_map[model_pedigree]:
                return db_map[model_pedigree].config_dir / f"{config_id}.yaml"
        return None

    def iter_config(self, model_pedigree) -> Generator[Tuple[Metadata, dict], None, None]:
        tasks = []
        if model_pedigree in self.custom_databases:
            for value in self.custom_databases[model_pedigree].values():
                tasks.append((Metadata.model_validate(value.get('metadata', {})), value))
        for db_map in self._plugin_database_maps:
            if model_pedigree in db_map:
                for value in db_map[model_pedigree].values():
                    tasks.append((Metadata.model_validate(value.get('metadata', {})), value))
        if model_pedigree in self.official_databases:
            for value in self.official_databases[model_pedigree].values():
                tasks.append((Metadata.model_validate(value.get('metadata', {})), value))

        if not tasks:
            raise UnsupportedError(
                f"Model type {model_pedigree} not found in practice repository", action='Please check the model type'
            )

        tasks.sort(key=lambda x: (-x[0].score, x[0].config_id))
        yield from tasks

    def is_saving_supported(self) -> bool:
        return self.custom_config_dir is not None

    def save_practice(self, model_pedigree: str, practice: PracticeConfig) -> None:
        if not self.is_saving_supported():
            raise UnsupportedError(
                "Can NOT save practice without custom practice directory", action="Please set custom practice directory"
            )

        if model_pedigree not in self.custom_databases:
            self.custom_databases[model_pedigree] = YamlDatabase(
                config_dir=self.custom_config_dir / model_pedigree, read_only=False
            )

        if practice.metadata.config_id in self.custom_databases[model_pedigree]:
            raise SpecError(f"Practice {practice.metadata.config_id} already exists")

        self.custom_databases[model_pedigree][practice.metadata.config_id] = practice
