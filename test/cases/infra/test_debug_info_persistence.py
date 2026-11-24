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

msmodelslim.infra.debug_info_persistence 模块的单元测试
"""

import json

import pytest
import torch
from safetensors.torch import load_file
from unittest.mock import MagicMock, patch

from msmodelslim.core.context import LocalDictContext
from msmodelslim.infra.debug_info_persistence import DebugInfoPersistence
from msmodelslim.utils.exception import SchemaValidateError, SecurityError


class TestDebugInfoPersistence:
    """测试调试上下文到 JSON 与 SafeTensors 文件的持久化。"""

    def test_init_does_not_create_save_directory(self, tmp_path):
        """初始化不产生文件系统副作用：目录与写入器推迟到保存时创建。"""
        persistence = DebugInfoPersistence(str(tmp_path))

        assert persistence.save_dir == tmp_path / "debug_info"
        assert not (tmp_path / "debug_info").exists()

    def test_save_from_context_writes_debug_info_and_tensor(self, tmp_path):
        """正常路径：非张量写入 JSON，张量写入 SafeTensors 且内容保持一致。"""
        context = LocalDictContext(enable_debug=True)
        context["quant"].debug["step"] = 3
        context["quant"].debug["weights"] = torch.arange(4, dtype=torch.float32)

        DebugInfoPersistence(str(tmp_path)).save_from_context(context)

        save_dir = tmp_path / "debug_info"
        json_payload = json.loads((save_dir / "debug_info.json").read_text(encoding="utf-8"))
        assert json_payload["quant"]["step"] == 3
        assert json_payload["quant"]["weights"]["_type"] == "tensor"
        assert json_payload["quant"]["weights"]["_file"] == "debug_info.safetensors"

        tensor_file = save_dir / "debug_info.safetensors"
        assert tensor_file.stat().st_size > 0
        tensors = load_file(str(tensor_file))
        assert len(tensors) == 1
        assert torch.equal(next(iter(tensors.values())), torch.arange(4, dtype=torch.float32))

    @patch("msmodelslim.infra.debug_info_persistence.get_current_context", return_value=None)
    def test_save_from_context_without_current_context_raises(self, mock_get_context, tmp_path):
        """异常路径：没有当前上下文时抛出 SchemaValidateError。"""
        persistence = DebugInfoPersistence(str(tmp_path))

        with pytest.raises(SchemaValidateError):
            persistence.save_from_context()

        mock_get_context.assert_called_once_with()

    @patch("msmodelslim.infra.debug_info_persistence.SafetensorsWriter")
    def test_writer_failure_leaves_instance_retryable(self, mock_writer_class, tmp_path):
        """写入器构造失败时不留半初始化状态：两个 writer 复位，下次保存重试成功。"""
        mock_writer_class.side_effect = [SecurityError("permission denied"), MagicMock()]
        persistence = DebugInfoPersistence(str(tmp_path))
        context = LocalDictContext(enable_debug=True)
        context["quant"].debug["step"] = 1

        with pytest.raises(SecurityError):
            persistence.save_from_context(context)

        assert persistence.json_writer is None
        assert persistence.safetensors_writer is None

        persistence.save_from_context(context)

        save_dir = tmp_path / "debug_info"
        json_payload = json.loads((save_dir / "debug_info.json").read_text(encoding="utf-8"))
        assert json_payload["quant"]["step"] == 1
