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

msmodelslim.utils.security.model 模块的单元测试
"""
# pylint: disable=no-name-in-module

import json
import os
from unittest.mock import patch, MagicMock, Mock

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from msmodelslim.utils.security.model import SafeGenerator, InvalidModelError


def create_mock_model_dir(tmp_path):
    """创建模拟的预训练模型目录（含config、tokenizer、模型权重文件）"""
    model_dir = tmp_path / "mock_model"
    model_dir.mkdir(exist_ok=True)

    config_data = {"model_type": "gpt2", "vocab_size": 50257}
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    with open(model_dir / "vocab.txt", "w", encoding="utf-8") as f:
        f.write("<|endoftext|>\nhello\nworld")
    tokenizer_config = {"do_lower_case": False}
    with open(model_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f)

    with open(model_dir / "pytorch_model.bin", "wb") as f:
        f.write(b"dummy_weight")

    return str(model_dir)


def create_mock_jsonl(tmp_path, file_name="test_dataset.jsonl", is_humaneval=False):
    """创建模拟的 JSONL 数据集文件"""
    jsonl_path = tmp_path / file_name
    lines = []
    if is_humaneval:
        lines = [
            json.dumps({"prompt": "def add(a, b): return ", "other_key": "val1"}),
            json.dumps({"prompt": "def multiply(x, y): return ", "other_key": "val2"}),
        ]
    else:
        lines = [
            json.dumps({"inputs_pretokenized": "test text 1", "other_key": "val1"}),
            json.dumps({"inputs_pretokenized": "test text 2", "other_key": "val2"}),
        ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(jsonl_path)


class TestSafeGenerator:
    @staticmethod
    def test_load_jsonl_return_prompts_when_humaneval_filename(mock_self):
        """humaneval_x.jsonl 应使用 prompt 字段"""
        lines = [
            json.dumps({"prompt": "def add():", "inputs_pretokenized": "ignored"}),
        ]
        mock_file = MagicMock()
        mock_file.readlines.return_value = [line + "\n" for line in lines]
        mock_cm = MagicMock()
        mock_cm.__enter__ = Mock(return_value=mock_file)
        mock_cm.__exit__ = Mock(return_value=False)
        with patch("os.fdopen", return_value=mock_cm):
            with patch("os.open", return_value=0):
                dataset = mock_self.safe_gen.load_jsonl("humaneval_x.jsonl")
        assert dataset == ["def add():"]

    @staticmethod
    def test_load_jsonl_return_line_when_key_missing(mock_self, tmp_path):
        """缺少默认 key 时应回退为整行 JSON 文本"""
        jsonl_path = tmp_path / "missing_key.jsonl"
        jsonl_path.write_text('{"other": "val"}\n', encoding="utf-8")
        dataset = mock_self.safe_gen.load_jsonl(str(jsonl_path))
        assert len(dataset) == 1

    @staticmethod
    def test_load_jsonl_normal_success(mock_self, tmp_path):
        """测试加载普通 JSONL 数据集（默认 key: inputs_pretokenized）"""
        mock_jsonl_path = create_mock_jsonl(tmp_path, is_humaneval=False)

        dataset = mock_self.safe_gen.load_jsonl(mock_jsonl_path)

        assert len(dataset) == 2
        assert dataset == ["test text 1", "test text 2"]

    @staticmethod
    def test_load_jsonl_fail_invalid_json(mock_self, tmp_path):
        """测试加载格式错误的 JSONL 文件（某行不是合法 JSON）"""
        invalid_jsonl_path = tmp_path / "invalid.jsonl"
        with open(invalid_jsonl_path, "w", encoding="utf-8") as f:
            f.write('{"valid": "line"}\ninvalid json line')

        with pytest.raises(json.JSONDecodeError):
            mock_self.safe_gen.load_jsonl(str(invalid_jsonl_path))

    @staticmethod
    def test_load_jsonl_fail_file_not_exist(mock_self, tmp_path):
        """测试加载不存在的 JSONL 文件"""
        non_exist_path = str(tmp_path / "non_exist.jsonl")

        with pytest.raises(FileNotFoundError):
            mock_self.safe_gen.load_jsonl(non_exist_path)

    @pytest.fixture
    def mock_self(self):
        mock = Mock()
        mock.safe_gen = SafeGenerator()
        return mock

    @patch("msmodelslim.utils.security.model.get_valid_read_path")
    def test_get_config_from_pretrained_return_config_when_valid(self, mock_get_valid, mock_self, tmp_path):
        """正常目录应返回 AutoConfig"""
        mock_model_dir = create_mock_model_dir(tmp_path)
        mock_get_valid.return_value = mock_model_dir
        mock_config = MagicMock()

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config) as mock_from_pretrained:
            result = mock_self.safe_gen.get_config_from_pretrained(mock_model_dir)

        mock_from_pretrained.assert_called_once_with(mock_model_dir, local_files_only=True)
        assert result is mock_config

    @patch("msmodelslim.utils.security.model.get_valid_read_path")
    def test_get_config_from_pretrained_fail_missing_config(self, mock_get_valid, mock_self, tmp_path):
        """测试加载缺失 config.json 的模型目录（异常场景）"""
        invalid_model_dir = tmp_path / "invalid_model"
        invalid_model_dir.mkdir(exist_ok=True)
        mock_get_valid.return_value = str(invalid_model_dir)

        with pytest.raises(InvalidModelError) as excinfo:
            mock_self.safe_gen.get_config_from_pretrained(str(invalid_model_dir))

        assert "Get config from pretrained failed" in str(excinfo.value)
        assert "ensure config files all exist" in str(excinfo.value)

    @patch("msmodelslim.utils.security.model.get_valid_read_path")
    def test_get_model_from_pretrained_success(self, mock_get_valid, mock_self, tmp_path):
        """测试正常加载预训练模型"""
        mock_model_dir = create_mock_model_dir(tmp_path)
        mock_get_valid.return_value = mock_model_dir

        with patch.object(AutoModelForCausalLM, "from_pretrained") as mock_from_pretrained:
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model

            model = mock_self.safe_gen.get_model_from_pretrained(mock_model_dir)

            mock_from_pretrained.assert_called_once_with(mock_model_dir, local_files_only=True)
            assert model is mock_model

    @patch("msmodelslim.utils.security.model.get_valid_read_path")
    def test_get_model_from_pretrained_fail_corrupt_weight(self, mock_get_valid, mock_self, tmp_path):
        """测试加载权重损坏的模型（异常场景）"""
        mock_model_dir = create_mock_model_dir(tmp_path)
        with open(os.path.join(mock_model_dir, "pytorch_model.bin"), "w", encoding="utf-8") as f:
            f.write("corrupt_data")
        mock_get_valid.return_value = mock_model_dir

        with pytest.raises(InvalidModelError) as excinfo:
            mock_self.safe_gen.get_model_from_pretrained(mock_model_dir)

        assert "Get model from pretrained failed" in str(excinfo.value)
        assert "model weights files all exist and are valid" in str(excinfo.value)

    @patch("msmodelslim.utils.security.model.get_valid_read_path")
    def test_get_tokenizer_from_pretrained_success(self, mock_get_valid, mock_self, tmp_path):
        """测试正常加载分词器"""
        mock_model_dir = create_mock_model_dir(tmp_path)
        mock_get_valid.return_value = mock_model_dir

        with patch.object(AutoTokenizer, "from_pretrained") as mock_from_pretrained:
            mock_tokenizer = MagicMock()
            mock_from_pretrained.return_value = mock_tokenizer

            tokenizer = mock_self.safe_gen.get_tokenizer_from_pretrained(mock_model_dir)

            mock_from_pretrained.assert_called_once_with(mock_model_dir, local_files_only=True)
            assert tokenizer is mock_tokenizer

    @patch("msmodelslim.utils.security.model.get_valid_read_path")
    def test_get_tokenizer_from_pretrained_fail_missing_vocab(self, mock_get_valid, mock_self, tmp_path):
        """测试加载缺失 vocab.txt 的分词器（异常场景）"""
        mock_model_dir = tmp_path / "no_vocab_model"
        mock_model_dir.mkdir(exist_ok=True)
        with open(mock_model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"model_type": "gpt2"}, f)
        mock_get_valid.return_value = str(mock_model_dir)

        with pytest.raises(InvalidModelError) as excinfo:
            mock_self.safe_gen.get_tokenizer_from_pretrained(str(mock_model_dir))

        assert "Get tokenizer from pretrained failed" in str(excinfo.value)
        assert "tokenizer files all exist and are valid" in str(excinfo.value)
