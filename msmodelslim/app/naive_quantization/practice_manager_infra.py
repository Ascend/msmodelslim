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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional, Tuple

from msmodelslim.core.practice import Metadata, PracticeConfig


class PracticeManagerInfra(ABC):
    def get_config_url(self, model_pedigree: str, config_id: str) -> Optional[str]:
        """Return the URL/location of the config; in different scenarios url may be a file path, an HTTP URL, etc."""
        return None

    @abstractmethod
    def __contains__(self, model_pedigree) -> bool:
        """Check if model pedigree is supported"""
        raise NotImplementedError

    @abstractmethod
    def get_config_by_id(self, model_pedigree, config_id: str) -> PracticeConfig:
        """Get configuration by ID"""
        raise NotImplementedError

    @abstractmethod
    def iter_config(self, model_pedigree) -> Generator[Tuple[Metadata, dict], None, None]:
        """Iterate configurations by priority"""
        raise NotImplementedError


class QuantConfigExportInfra(ABC):
    """量化配置导出基础设施抽象基类"""

    @abstractmethod
    def export_quant_config(self, quant_config: PracticeConfig, model_type: str, save_path: Path) -> None:
        """
        导出量化配置到文件

        Args:
            quant_config: 量化配置对象，包含量化任务的配置信息
            model_type: 模型类型，用于生成输出文件名
            save_path: 保存路径，配置文件的输出目录

        Returns:
            None

        Raises:
            NotImplementedError: 子类必须实现此方法
            IOError: 文件写入失败时可能抛出
            ValueError: 参数无效时可能抛出
        """
        raise NotImplementedError
