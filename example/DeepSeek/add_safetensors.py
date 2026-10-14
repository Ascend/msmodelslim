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
"""
# pylint: disable=consider-iterating-dictionary,logging-fstring-interpolation

import json  # noqa: F401
import os
import re  # noqa: F401
import glob
import argparse
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

from convert_fp8_to_bf16 import weight_dequant

from example.common.security.path import (
    json_safe_load,
    json_safe_dump,
    get_valid_read_path,
    get_valid_write_path,
    MAX_READ_FILE_SIZE_512G,
)
from msmodelslim import logger as msmodelslim_logger


def find_file_with_pattern(target_dir, pattern):
    """查找目录下的符合pattern的文件"""
    pattern = os.path.join(target_dir, pattern)
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"can't find {pattern} in {target_dir}")
    if len(files) > 1:
        raise ValueError("find mutiple json files")
    return files[0]


def calculate_tensor_size(tensor):
    # 计算单个张量的总字节数
    return tensor.numel() * tensor.element_size()


def get_weight_map(float_index_path):
    org_data = json_safe_load(float_index_path)
    return org_data.get("weight_map", {})


def get_tensor(tensor_name, safetensor_path, weight_map):
    filename = weight_map[tensor_name]
    file_path = os.path.join(safetensor_path, filename)
    file_path = get_valid_read_path(file_path, size_max=MAX_READ_FILE_SIZE_512G)
    with safe_open(file_path, framework="pt", device="cpu") as f:
        if tensor_name in f.keys():
            tensor = f.get_tensor(tensor_name)
        else:
            raise KeyError(f"tensor {tensor_name} not found in {file_path}")
    return tensor


def get_prefix(name, last_index=-1):
    key_list = name.split(".")[:last_index]
    return ".".join(key_list)


def add_safetensors(org_paths, target_dir, safetensors_prefix, max_file_size_gb=5, prefix=None):
    """将原始模型的tensor添加到量化模型中，支持分文件保存

    Args:
        org_paths (str): 原始模型safetensors文件所在目录路径
        target_dir (str): 目标量化模型目录路径
        safetensors_prefix (str): 新生成的safetensors文件的前缀名
        max_file_size_gb (float): 单个safetensors文件的最大大小(GB)，默认5GB
        prefix (str, optional): 只添加指定前缀的tensor，默认None表示添加所有tensor
    """
    quant_type = "FLOAT"
    # 验证输入输出路径
    org_paths = get_valid_read_path(org_paths, is_dir=True, check_user_stat=True)
    target_dir = get_valid_read_path(target_dir, is_dir=True, check_user_stat=True)
    index_path = find_file_with_pattern(target_dir, "quant_model_weight_*.index.json")
    desc_path = find_file_with_pattern(target_dir, "quant_model_description*.json")

    msmodelslim_logger.info(f"find file in target_dir: \nindex: {index_path}\ndescription: {desc_path}")

    float_index_path = find_file_with_pattern(org_paths, "*.index.json")
    msmodelslim_logger.info(f"find index file in org_path: \n{float_index_path}")

    weight_map = get_weight_map(float_index_path)

    index_data = json_safe_load(index_path)
    desc_data = json_safe_load(desc_path)
    if "metadata" not in index_data:
        index_data["metadata"] = {}
    if "weight_map" not in index_data:
        index_data["weight_map"] = {}
    current_total_size = index_data.get("metadata", {}).get("total_size", 0)
    tensor_names = weight_map.keys()

    if prefix:
        tensor_names = [name for name in tensor_names if name.startswith(prefix)]

    max_file_size = max_file_size_gb * (1024**3)
    current_file_size = 0
    new_data = {}
    file_count = 0

    for tensor_name in tqdm(tensor_names):
        if "weight_scale_inv" not in tensor_name:
            tensor = get_tensor(tensor_name, org_paths, weight_map)
            tensor_size = calculate_tensor_size(tensor)
            current_total_size += tensor_size

            mod_name = get_prefix(tensor_name)
            if mod_name + ".weight_scale_inv" in tensor_names:
                try:
                    weight_scale_inv = get_tensor(mod_name + ".weight_scale_inv", org_paths, weight_map)
                    tensor = weight_dequant(tensor, weight_scale_inv)
                except KeyError:
                    msmodelslim_logger.warning(
                        f"{mod_name + '.weight_scale_inv'} not found in org_paths, \
                                               skip convert {mod_name} from fp8 to bf16"
                    )

            # 如果当前文件大小超过限制，保存当前文件并开始新文件
            if (current_file_size + tensor_size) > max_file_size and new_data:
                file_name = f"{safetensors_prefix}-{file_count + 1}.safetensors"
                ori_mask = os.umask(0o377)
                save_path = os.path.join(target_dir, file_name)
                save_path = get_valid_write_path(save_path, is_dir=False)
                save_file(new_data, save_path)
                os.umask(ori_mask)
                # 更新索引
                for name in new_data.keys():
                    index_data["weight_map"][name] = file_name
                    desc_data[name] = quant_type
                new_data = {}
                current_file_size = 0
                file_count += 1

            new_data[tensor_name] = tensor
            current_file_size += tensor_size

    # 保存最后一个文件
    if new_data:
        file_name = f"{safetensors_prefix}-{file_count + 1}.safetensors"
        ori_mask = os.umask(0o377)
        save_path = os.path.join(target_dir, file_name)
        save_path = get_valid_write_path(save_path, is_dir=False)
        save_file(new_data, save_path)
        os.umask(ori_mask)
        for name in new_data.keys():
            index_data["weight_map"][name] = file_name
            desc_data[name] = "FLOAT"

    index_data["metadata"]["total_size"] = current_total_size
    json_safe_dump(index_data, index_path, indent=4)
    json_safe_dump(desc_data, desc_path, indent=4)
    msmodelslim_logger.info("add success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='添加新的safetensors文件到现有模型')
    parser.add_argument('--quant_dir', help='量化模型文件所在目录')
    parser.add_argument('--float_dir', help='浮点safetensors文件所在目录')

    args = parser.parse_args()

    add_safetensors(args.float_dir, args.quant_dir, "mtp", max_file_size_gb=5, prefix='model.layers.61.')
