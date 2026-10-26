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

msmodelslim.infra.evaluation.aisbench_server 模块的单元测试
"""

from decimal import Decimal

import pytest

from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateContext
from msmodelslim.infra.evaluation.aisbench_server import (
    AisBenchServer,
    AisbenchConfig,
    AisbenchServerConfig,
    DatasetConfig,
    ModelConfigMeta,
)
from msmodelslim.utils.exception import SpecError


def _make_server(tmp_path, ais_config=None, datasets=None):
    config = AisbenchServerConfig(
        aisbench=ais_config or AisbenchConfig(),
        datasets=datasets or {"gsm8k": DatasetConfig(config_name="gsm8k")},
    )
    context = EvaluateContext(evaluate_id="42", working_dir=tmp_path)
    return AisBenchServer(context, config, ["gsm8k"], tmp_path / "model", tmp_path / "run")


class TestAisBenchServer:
    """测试 AISBench 命令构建与精度日志解析。"""

    def test_build_command_options_merges_global_and_dataset_args(self, tmp_path):
        """正常路径：数据集模式覆盖全局模式，额外参数按全局后数据集顺序合并。"""
        dataset = DatasetConfig(
            config_name="gsm8k",
            mode="accuracy",
            extra_args=["--dataset-arg"],
        )
        server = _make_server(
            tmp_path,
            ais_config=AisbenchConfig(mode="all", extra_args=["--global-arg"]),
            datasets={"gsm8k": dataset},
        )
        server.model_config_name = "model-config"

        result = server._build_command_options(dataset)

        assert result == {
            "options": {
                "--models": "model-config",
                "--datasets": "gsm8k",
                "--mode": "accuracy",
                "--work-dir": tmp_path / "aisbench_output",
            },
            "extra_args": ["--global-arg", "--dataset-arg"],
        }

    def test_parse_accuracy_reads_decimal_from_result_json(self, tmp_path):
        """正常路径：从日志指向目录的 results/{abbr}/*.json 读取 accuracy。"""
        abbr = "custom-abbr"
        server = _make_server(
            tmp_path,
            ais_config=AisbenchConfig(model_meta=ModelConfigMeta(abbr=abbr)),
        )
        exp_folder = tmp_path / "exp"
        results_dir = exp_folder / "results" / abbr
        results_dir.mkdir(parents=True)
        (results_dir / "result.json").write_text('{"accuracy": 0.875}', encoding="utf-8")

        accuracy = server._parse_accuracy(f"Current exp folder: {exp_folder}\n", "gsm8k")

        assert accuracy == Decimal("0.875")

    def test_parse_accuracy_returns_zero_when_exp_folder_missing(self, tmp_path):
        """边界路径：日志中没有实验目录时返回零精度。"""
        server = _make_server(tmp_path)

        assert server._parse_accuracy("AISBench finished without output folder", "gsm8k") == Decimal("0")

    def test_parse_accuracy_raises_when_results_directory_missing(self, tmp_path):
        """异常路径：实验目录存在但结果目录缺失时抛出 SpecError。"""
        server = _make_server(tmp_path)
        exp_folder = tmp_path / "exp"
        exp_folder.mkdir()

        with pytest.raises(SpecError, match="Results directory not found"):
            server._parse_accuracy(f"Current exp folder: {exp_folder}\n", "gsm8k")
