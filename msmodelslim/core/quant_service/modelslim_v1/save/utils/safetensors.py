#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=duplicate-code

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

from msmodelslim.utils.security import get_valid_write_path, SafeWriteUmask, get_write_directory, json_safe_dump

ONE_GB_FILE_BYTES = 1073741824  # 1G, 1 * 1024 * 1024 * 1024
FILE_TMP_SUFFIX = '-of-00000.safetensors'


def get_index_json(file_map_dict, total_size):
    index_json_dict = {'metadata': {'total_size': total_size}, 'weight_map': file_map_dict}
    return index_json_dict


class SafetensorsWriter:
    def __init__(self, logger, file_path):
        self.logger = logger
        file_path = get_valid_write_path(file_path, extensions=['safetensors'])
        self.file_path = file_path
        self.safetensors_weight = {}

    def write(self, key: str, value: torch.Tensor):
        self.safetensors_weight[key] = value.cpu().contiguous()

    def close(self):
        with SafeWriteUmask(umask=0o377):
            save_file(self.safetensors_weight, self.file_path)
        self.logger.info(f'Save safetensors to {self.file_path} successfully')


class BufferedSafetensorsWriter(SafetensorsWriter):
    def __init__(
        self, logger, max_gb_size: int = 4, save_directory: str = '.', save_prefix: str = "quant_model_weight"
    ):
        super().__init__(logger, save_prefix + '.safetensors')
        self.logger = logger
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
        # process safetensors index json
        if self._save_count <= 99999:
            suffix = f"-of-{self._save_count:05d}.safetensors"
        else:
            suffix = f"-of-{self._save_count}.safetensors"
        for key in self.saved_keys_map:
            self.saved_keys_map[key] = self.saved_keys_map[key].replace(FILE_TMP_SUFFIX, suffix)

        # save index json
        index_json_dict = get_index_json(self.saved_keys_map, self.total_size)
        index_json_name = os.path.join(self.save_directory, self.save_prefix + '.safetensors.index.json')
        index_json_path = get_valid_write_path(index_json_name, extensions=[".json"])
        json_safe_dump(index_json_dict, index_json_path, indent=2)
        self.logger.debug(f'Save index json to {index_json_path} successfully')

    def _dedupe_shared_storage(self, keys_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Resolve shared storage: prefer embed_tokens, write other keys (e.g. lm_head.weight) as clones."""
        seen_ptr: Dict[int, str] = {}
        out: Dict[str, torch.Tensor] = {}
        for k in sorted(keys_dict.keys(), key=lambda x: (0 if "embed_tokens" in x else 1, x)):
            t = keys_dict[k]
            if not t.is_contiguous():
                t = t.contiguous()
            ptr = t.data_ptr()
            if ptr in seen_ptr:
                self.logger.warning(
                    "Saving %s as clone (shares storage with %s) so both keys exist for tie_word_embeddings.",
                    k,
                    seen_ptr[ptr],
                )
                out[k] = t.clone()
                continue
            seen_ptr[ptr] = k
            out[k] = t
        return out

    def save_one_file(self) -> None:
        # no tensors no saving
        if not self.wait_save_keys:
            return

        # one tensor larger than max size
        if self._wait_save_size > self.max_size:
            self.logger.warning(
                f'Tensor is too large with size {self._wait_save_size / ONE_GB_FILE_BYTES}GB, '
                f'exceeds file size limit: {self.max_size / ONE_GB_FILE_BYTES}GB'
            )

        # Dedupe shared storage (e.g. tie_word_embeddings: lm_head.weight vs embed_tokens.weight)
        tensors_to_save = self._dedupe_shared_storage(self.wait_save_keys)

        self._save_count += 1
        save_file_name = f"{self.save_prefix}-{self._save_count:05d}{FILE_TMP_SUFFIX}"
        full_save_file_name = os.path.join(self.save_directory, save_file_name)
        full_save_file_name = get_valid_write_path(full_save_file_name, extensions=[".safetensors"])

        self.logger.debug(f"Start save {full_save_file_name}")
        with SafeWriteUmask(umask=0o377):
            save_file(tensors_to_save, full_save_file_name)
        self.saved_keys_map.update({key: save_file_name for key in self.wait_save_keys})
        self.wait_save_keys.clear()
        self._wait_save_size = 0
        self.logger.debug(f"End save {full_save_file_name}")

    def write(self, key: str, value: torch.Tensor) -> None:
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

    def close(self) -> None:
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
