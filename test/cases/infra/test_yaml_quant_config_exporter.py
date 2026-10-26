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

msmodelslim.infra.yaml_quant_config_exporter 模块的单元测试
"""

from unittest.mock import MagicMock, patch

from msmodelslim.infra.yaml_quant_config_exporter import YamlQuantConfigExporter


class TestYamlQuantConfigExporter:
    """测试 YamlQuantConfigExporter 的导出文件名与内容"""

    @patch("msmodelslim.utils.security.yaml_safe_dump")
    def test_export_writes_yaml_named_by_model_type(self, mock_yaml_dump, tmp_path):
        """正常路径：以 <model_type>_best_practice.yaml 命名并按 JSON 模式序列化内容"""
        quant_config = MagicMock()
        quant_config.model_dump.return_value = {"apiversion": "modelslim_v1", "spec": {}}

        YamlQuantConfigExporter().export_quant_config(quant_config, "Qwen3", tmp_path)

        expected_path = tmp_path / "Qwen3_best_practice.yaml"
        mock_yaml_dump.assert_called_once_with(
            {"apiversion": "modelslim_v1", "spec": {}}, str(expected_path), check_user_stat=False
        )
        quant_config.model_dump.assert_called_once_with(mode="json")

    @patch("msmodelslim.utils.security.yaml_safe_dump")
    def test_export_name_uses_given_model_type(self, mock_yaml_dump, tmp_path):
        """行为：不同 model_type 产生不同的导出文件名"""
        quant_config = MagicMock()
        quant_config.model_dump.return_value = {"apiversion": "modelslim_v1"}

        YamlQuantConfigExporter().export_quant_config(quant_config, "llama3", tmp_path)

        dumped_path = mock_yaml_dump.call_args.args[1]
        assert dumped_path == str(tmp_path / "llama3_best_practice.yaml")
