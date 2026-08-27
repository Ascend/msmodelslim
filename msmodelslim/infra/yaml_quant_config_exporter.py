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

from msmodelslim.app.naive_quantization.practice_manager_infra import QuantConfigExportInfra
from msmodelslim.core.practice.interface import PracticeConfig


class YamlQuantConfigExporter(QuantConfigExportInfra):
    """YAML格式的量化配置导出器"""

    def export_quant_config(self, quant_config: PracticeConfig, model_type: str, save_path: Path) -> None:
        """
        导出量化配置到YAML文件

        Args:
            quant_config: 量化配置对象
            model_type: 模型类型
            save_path: 保存路径
        """
        from msmodelslim.utils.security import yaml_safe_dump

        # 构建文件名
        filename = f"{model_type}_best_practice.yaml"
        file_path = save_path / filename

        # 导出配置
        yaml_safe_dump(quant_config.model_dump(mode='json'), str(file_path), check_user_stat=False)
