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


import os
import shutil
from typing import Dict

import torch

from safetensors.torch import save_file

from ascend_utils.common.security import json_safe_dump, SafeWriteUmask, check_type, get_valid_write_path, \
    get_write_directory
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save.writer.base import BaseWriter

ONE_GB_FILE_BYTES = 1073741824  # 1G, 1 * 1024 * 1024 * 1024
FILE_TMP_SUFFIX = '-of-00000.safetensors'


def get_index_json(file_map_dict, total_size):
    index_json_dict = {
        'metadata': {'total_size': total_size},
        'weight_map': file_map_dict
    }
    return index_json_dict


class BufferedSafetensorsWriter(BaseWriter):

    def __init__(self, logger, max_gb_size: int = 4, save_directory: str = '.',
                 save_prefix: str = "quant_model_weight"):
        super().__init__(logger)

        self.saved_keys_map: Dict[str, str] = {}
        self.wait_save_keys: Dict[str, torch.Tensor] = {}
        self.max_size: int = max_gb_size * ONE_GB_FILE_BYTES
        self.save_prefix: str = save_prefix
        self.total_size: int = 0
        self._wait_save_size: int = 0
        self._save_count: int = 0
        self.save_directory: str = save_directory

    @property
    def save_directory(self) -> str:
        return self._save_directory

    @save_directory.setter
    def save_directory(self, value: str) -> None:
        self._save_directory = get_write_directory(value, write_mode=0o750)

    def save_index(self):
        # process safetensor index json
        if self._save_count <= 99999:
            suffix = f"-of-{self._save_count:05d}.safetensors"
        else:
            suffix = f"-of-{self._save_count}.safetensors"
        for key in self.saved_keys_map.keys():
            self.saved_keys_map[key] = self.saved_keys_map[key].replace(FILE_TMP_SUFFIX, suffix)

        # save index json
        index_json_dict = get_index_json(self.saved_keys_map, self.total_size)
        index_json_name = os.path.join(self.save_directory, self.save_prefix + '.safetensors.index.json')
        index_json_path = get_valid_write_path(index_json_name, extensions=[".json"])
        json_safe_dump(index_json_dict, index_json_path, indent=2)
        self.logger.debug(f'Save index json to {index_json_path} successfully')

    def save_one_file(self) -> None:
        # no tensors no saving
        if not self.wait_save_keys:
            return

        # one tensor larger than max size
        if self._wait_save_size > self.max_size:
            self.logger.warning(f'Tensor is too large with size {self._wait_save_size / ONE_GB_FILE_BYTES}GB, '
                                f'exceeds file size limit: {self.max_size / ONE_GB_FILE_BYTES}GB')

        self._save_count += 1
        save_file_name = f"{self.save_prefix}-{self._save_count:05d}{FILE_TMP_SUFFIX}"
        full_save_file_name = os.path.join(self.save_directory, save_file_name)
        full_save_file_name = get_valid_write_path(full_save_file_name, extensions=[".safetensors"])

        self.logger.debug(f"Start save {full_save_file_name}")
        with SafeWriteUmask(umask=0o377):
            save_file(self.wait_save_keys, full_save_file_name)
        self.saved_keys_map.update({key: save_file_name for key in self.wait_save_keys.keys()})
        self.wait_save_keys.clear()
        self._wait_save_size = 0
        self.logger.debug(f"End save {full_save_file_name}")

    def _write(self, key: str, value: torch.Tensor) -> None:
        check_type(value, torch.Tensor)
        if value.device.type == 'meta':
            self.logger.warning(f"Skip meta tensor {key}")
            return

        tensor = value.detach().cpu().contiguous()

        tensor_size = tensor.numel() * tensor.element_size()
        if self._wait_save_size + tensor_size >= self.max_size:
            self.save_one_file()

        self.wait_save_keys[key] = tensor
        self.total_size += tensor_size
        self._wait_save_size += tensor_size
        self.logger.debug(
            f"Add new tensor {key}, device: {tensor.device}, size: {tensor_size}, total: {self._wait_save_size}")

    def _close(self) -> None:
        # save last file if needed
        self.save_one_file()

        # rename safetensors
        for i in range(self._save_count):
            src_file = os.path.join(self.save_directory, f"{self.save_prefix}-{i + 1:05d}{FILE_TMP_SUFFIX}")
            # 仿照开源权重命名均为model-0000x-of-0000x.safetensors，超过99999命名为model-x-of-x.safetensors
            if self._save_count <= 99999:
                dst_file_name = f"{self.save_prefix}-{i + 1:05d}-of-{self._save_count:05d}.safetensors"
            else:
                dst_file_name = f"{self.save_prefix}-{i + 1}-of-{self._save_count}.safetensors"
            dst_file = os.path.join(self.save_directory, dst_file_name)
            shutil.move(src_file, dst_file)
            self.logger.debug(f"{src_file} -> {dst_file}")
        self.logger.debug(f'Save .safetensors to {self.save_directory} successfully')

        self.save_index()
        self.logger.info(f'Save safetensors files to {self.save_directory} successfully')
