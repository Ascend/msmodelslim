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

msmodelslim.infra.io.safetensors_writer 模块的单元测试
"""

from unittest.mock import MagicMock, patch

import torch

from msmodelslim.infra.io.safetensors_writer import SafetensorsWriter


class TestSafetensorsWriter:
    """测试 SafetensorsWriter 的路径校验、聚合与落盘行为"""

    @patch("msmodelslim.infra.io.safetensors_writer.get_valid_write_path")
    def test_init_bind_validated_path(self, mock_get_valid):
        """正常路径：初始化时对目标文件做安全校验并保存校验后的路径"""
        mock_get_valid.return_value = "/out/w.safetensors"

        writer = SafetensorsWriter(logger=MagicMock(), file_path="w.safetensors")

        assert writer.file_path == "/out/w.safetensors"
        mock_get_valid.assert_called_once_with("w.safetensors", extensions=[".safetensors"])

    @patch("msmodelslim.infra.io.safetensors_writer.get_valid_write_path", return_value="/out/w.safetensors")
    def test_write_aggregate_tensor_keyed_by_name(self, mock_get_valid):
        """正常路径：write 将张量按 key 聚合到内存，值与原张量一致"""
        writer = SafetensorsWriter(logger=MagicMock(), file_path="w.safetensors")
        expected = torch.arange(4).reshape(2, 2).float()

        writer.write("embed.weight", expected)

        assert list(writer.safetensors_weight) == ["embed.weight"]
        assert torch.equal(writer.safetensors_weight["embed.weight"], expected)

    @patch("msmodelslim.infra.io.safetensors_writer.save_file")
    @patch("msmodelslim.infra.io.safetensors_writer.get_valid_write_path", return_value="/out/w.safetensors")
    def test_close_save_file_and_log(self, mock_get_valid, mock_save_file):
        """行为：close 调用 save_file 写盘并按校验路径记录日志"""
        logger = MagicMock()
        writer = SafetensorsWriter(logger=logger, file_path="w.safetensors")
        writer.write("a", torch.ones(2))

        writer.close()

        mock_save_file.assert_called_once()
        saved_tensors, saved_path = mock_save_file.call_args.args
        assert list(saved_tensors) == ["a"]
        assert torch.equal(saved_tensors["a"], torch.ones(2))
        assert saved_path == "/out/w.safetensors"
        logger.info.assert_called_once()
