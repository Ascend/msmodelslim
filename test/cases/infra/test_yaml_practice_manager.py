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

msmodelslim.infra.yaml_practice_manager 模块的单元测试
"""

from pathlib import Path

import pytest

from msmodelslim.core.practice import PracticeConfig
from msmodelslim.infra.yaml_practice_manager import YamlPracticeManager
from msmodelslim.utils.exception import SecurityError, SpecError


def _write_practice(path: Path, config_id: str, score: float) -> None:
    path.write_text(
        f"metadata:\n  config_id: {config_id}\n  score: {score}\napiversion: modelslim_v1\nspec: {{}}\n",
        encoding="utf-8",
    )


@pytest.fixture
def official_practice_dir(tmp_path):
    model_dir = tmp_path / "official" / "qwen3"
    model_dir.mkdir(parents=True)
    _write_practice(model_dir / "a.yaml", "a", 0.5)
    _write_practice(model_dir / "b.yaml", "b", 0.9)
    _write_practice(model_dir / "c.yaml", "c", 0.9)
    return tmp_path / "official"


class TestYamlPracticeManager:  # pylint: disable=redefined-outer-name
    """测试实践配置查询、排序、写入与配置 ID 防篡改。"""

    def test_get_config_by_id_is_case_insensitive_and_returns_public_path(self, official_practice_dir):
        """正常路径：模型类型大小写不敏感，配置路径指向官方 YAML。"""
        manager = YamlPracticeManager(official_practice_dir)

        assert "QWEN3" in manager
        config = manager.get_config_by_id("QWEN3", "a")
        assert config.metadata.config_id == "a"
        assert config.apiversion == "modelslim_v1"
        assert Path(manager.get_config_url("qwen3", "a")) == official_practice_dir / "qwen3" / "a.yaml"

    def test_iter_config_sorts_by_score_desc_then_config_id(self, official_practice_dir):
        """正常路径：迭代顺序按分数降序、同分按 config_id 升序。"""
        manager = YamlPracticeManager(official_practice_dir)

        config_ids = [metadata.config_id for metadata, _ in manager.iter_config("qwen3")]
        assert config_ids == ["b", "c", "a"]

    def test_save_practice_writes_custom_yaml_and_rejects_duplicate(self, official_practice_dir, tmp_path):
        """正常/异常路径：自定义目录可写；重复保存同一 config_id 抛出 SpecError。"""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        manager = YamlPracticeManager(official_practice_dir, custom_config_dir=custom_dir)
        practice = PracticeConfig.model_validate(
            {
                "metadata": {"config_id": "new", "score": 1.0},
                "apiversion": "modelslim_v1",
                "spec": {},
            }
        )

        assert manager.is_saving_supported() is True
        manager.save_practice("new-model", practice)

        saved_path = custom_dir / "new-model" / "new.yaml"
        assert saved_path.exists()
        assert manager.get_config_by_id("new-model", "new").metadata.config_id == "new"
        with pytest.raises(SpecError):
            manager.save_practice("new-model", practice)

    def test_get_config_by_id_detects_tampered_config_id(self, official_practice_dir, tmp_path):
        """异常路径：文件名与 YAML 内 config_id 不一致时抛出 SecurityError。"""
        model_dir = official_practice_dir / "qwen3"
        _write_practice(model_dir / "alias.yaml", "different", 1.0)
        manager = YamlPracticeManager(official_practice_dir)

        with pytest.raises(SecurityError):
            manager.get_config_by_id("qwen3", "alias")
