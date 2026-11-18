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

import torch
from safetensors.torch import save_file

from ascend_utils.common.security import get_valid_write_path, check_type, SafeWriteUmask
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer.base import BaseWriter


class SafetensorsWriter(BaseWriter):
    def __init__(self, logger, file_path):
        super().__init__(logger)

        file_path = get_valid_write_path(file_path, extensions=['safetensors'])
        self.file_path = file_path
        self.safetensors_weight = {}

    def _write(self, key: str, value: torch.Tensor):
        check_type(value, torch.Tensor)
        self.safetensors_weight[key] = value.cpu().contiguous()

    def _close(self):
        with SafeWriteUmask(umask=0o377):
            save_file(self.safetensors_weight, self.file_path)
        self.logger.info(f'Save safetensors to {self.file_path} successfully')