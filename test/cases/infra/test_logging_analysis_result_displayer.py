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

msmodelslim/infra/logging_analysis_result_displayer.py 的单元测试。
"""

from msmodelslim.core.analysis_service import AnalysisResult, AnalysisScope
from msmodelslim.infra.logging_analysis_result_displayer import StandardAnalysisResultDisplayer, _save_yaml


class TestSaveYamlSuffix:
    """对应 _save_yaml：写入校验与目录拼接需共用同一份后缀规则（issue #575）。"""

    def test_save_yaml_accepts_yml_suffix(self, tmp_path):
        """正常：save_path 以 .yml 结尾时按原路径写盘，不再报 doesn't endswith ".yaml"。"""
        target = tmp_path / "result.yml"
        output_path = _save_yaml("top 1:\n  - 'block.0'", str(target), 'TinyLlama', 'std')

        assert output_path == str(target)
        assert target.read_text(encoding='utf-8') == "top 1:\n  - 'block.0'"

    def test_save_yaml_accepts_yaml_suffix(self, tmp_path):
        """正常：.yaml 后缀行为保持不变。"""
        target = tmp_path / "result.yaml"
        output_path = _save_yaml("top 0:", str(target), 'TinyLlama', 'std')

        assert output_path == str(target)
        assert target.read_text(encoding='utf-8') == "top 0:"

    def test_save_yaml_appends_default_name_when_path_is_directory(self, tmp_path):
        """正常：save_path 为目录时按 {model_type}-{method}.yaml 拼接文件名。"""
        output_path = _save_yaml("top 0:", str(tmp_path), 'TinyLlama', 'std')

        expected = tmp_path / 'tinyllama-std.yaml'
        assert output_path == str(expected)
        assert expected.read_text(encoding='utf-8') == "top 0:"

    def test_save_yaml_treats_unsupported_suffix_as_directory(self, tmp_path):
        """边界：后缀既非 .yaml 也非 .yml 时按目录处理，文件名由 model_type / method 拼接。"""
        output_path = _save_yaml("top 0:", str(tmp_path / "result.txt"), 'TinyLlama', 'std')

        assert output_path == str(tmp_path / "result.txt" / 'tinyllama-std.yaml')
        assert (tmp_path / "result.txt" / 'tinyllama-std.yaml').read_text(encoding='utf-8') == "top 0:"


class TestStandardAnalysisResultDisplayerSavePath:
    """对应 StandardAnalysisResultDisplayer.display_result 的 save_path 分支。"""

    @staticmethod
    def _result() -> AnalysisResult:
        return AnalysisResult(
            layer_scores=[
                {'name': 'layers.0.self_attn.q_proj', 'score': 2.0},
                {'name': 'layers.1.self_attn.k_proj', 'score': 1.0},
            ],
            method='std',
            patterns=['*'],
        )

    def test_display_result_writes_yml_file(self, tmp_path):
        """端到端：--save_path 传 .yml 时生成文件，且文件含 top-k 层名。"""
        target = tmp_path / "result.yml"
        StandardAnalysisResultDisplayer().display_result(
            self._result(),
            topk=1,
            scope=AnalysisScope.LINEAR,
            save_path=str(target),
            model_type='TinyLlama',
        )

        assert target.read_text(encoding='utf-8') == "top 1:\n  - 'layers.0.self_attn.q_proj'"
