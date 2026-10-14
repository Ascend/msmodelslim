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
# pylint: disable=arguments-differ

import os
import shutil
import argparse
from dataclasses import dataclass
from typing import List, Optional, Any
from example.common.security.path import json_safe_load, json_safe_dump
from example.common.security.path import get_valid_read_path, get_valid_write_path
from example.common.utils import SafeGenerator


@dataclass
class ModifyConfigParams:
    """修改配置文件的参数封装"""

    model_dir: str
    dest_dir: str
    torch_dtype: Any
    quantize_type: str
    args: Optional[Any] = None
    quantize_config_parts: Optional[List[str]] = None

    def __post_init__(self):
        """后初始化处理"""
        if self.quantize_config_parts is None:
            self.quantize_config_parts = []


@dataclass
class CopyTokenizerParams:
    """复制tokenizer文件的参数封装"""

    model_dir: str
    dest_dir: str


class VlmSafeGenerator(SafeGenerator):
    # 默认量化配置
    DEFAULT_QUANTIZATION_CONFIG = {
        'group_size': 0,
        'act_method': 2,
        'anti_method': 'm2',
        'is_lowbit': False,
        'mm_tensor': False,
        'w_sym': True,
        'open_outlier': True,
        'is_dynamic': False,
    }

    # 支持的文件扩展名和排除的文件名
    SUPPORTED_EXTENSIONS = {'.json', '.py'}
    EXCLUDED_FILES = {'config.json', 'model.safetensors.index.json'}
    MAX_FILE_NUM = 1024

    @staticmethod
    def modify_config(params: ModifyConfigParams):
        """修改配置文件"""
        # 验证路径
        model_dir = get_valid_read_path(params.model_dir, is_dir=True, check_user_stat=True)
        dest_dir = get_valid_write_path(params.dest_dir, is_dir=True)

        # 加载源配置
        src_config_filepath = os.path.join(model_dir, 'config.json')
        data = json_safe_load(src_config_filepath, check_user_stat=True)

        # 生成量化描述文件路径
        dest_quant_description_filepath = VlmSafeGenerator._get_quantization_filename(
            dest_dir, params.quantize_type, getattr(params.args, 'mindie_format', False)
        )
        dest_quant_description_filepath = get_valid_write_path(dest_quant_description_filepath, is_dir=False)
        quant_description_data = json_safe_load(dest_quant_description_filepath, check_user_stat=True)

        # 更新配置
        data['torch_dtype'] = str(params.torch_dtype).split('.')[1]

        # 处理mindie格式
        if params.args and getattr(params.args, 'mindie_format', False):
            data['quantize'] = params.quantize_type
            for config_part in params.quantize_config_parts:
                data[config_part]['quantize'] = params.quantize_type

        # 构建量化配置
        quantization_config = VlmSafeGenerator._build_quantization_config(params.args)
        if quantization_config:
            quant_description_data.update(quantization_config)
            if params.args and getattr(params.args, 'mindie_format', False):
                data['quantization_config'] = quantization_config

        # 保存配置
        dest_config_filepath = os.path.join(dest_dir, 'config.json')
        json_safe_dump(data, dest_config_filepath, 4)

    @staticmethod
    def copy_tokenizer_files(params: CopyTokenizerParams):
        """复制tokenizer文件"""
        model_dir = get_valid_read_path(params.model_dir, is_dir=True, check_user_stat=True)

        # 确保目标目录存在
        if not os.path.exists(params.dest_dir):
            os.makedirs(params.dest_dir, mode=0o750, exist_ok=True)
        dest_dir = get_valid_write_path(params.dest_dir, is_dir=True)

        # 检查文件数量限制
        filenames = os.listdir(model_dir)
        if len(filenames) > VlmSafeGenerator.MAX_FILE_NUM:
            raise argparse.ArgumentTypeError(
                f"The file num in dir is {len(filenames)}, which exceeds the limit {VlmSafeGenerator.MAX_FILE_NUM}."
            )

        # 复制符合条件的文件
        for filename in filenames:
            # 检查文件扩展名
            _, ext = os.path.splitext(filename)
            if ext not in VlmSafeGenerator.SUPPORTED_EXTENSIONS:
                continue

            # 跳过排除的文件
            if filename in VlmSafeGenerator.EXCLUDED_FILES:
                continue

            # 复制文件
            src_filepath = os.path.join(model_dir, filename)
            dest_filepath = os.path.join(dest_dir, filename)
            shutil.copyfile(src_filepath, dest_filepath)
            os.chmod(dest_filepath, 0o600)

    @staticmethod
    def _get_quantization_filename(dest_dir, quantize_type, mindie_format):
        """生成量化描述文件名"""
        if mindie_format:
            filename = f"quant_model_description_{quantize_type.lower()}.json"
        else:
            filename = "quant_model_description.json"
        return os.path.join(dest_dir, filename)

    @staticmethod
    def _build_quantization_config(args):
        """构建量化配置字典"""
        if args is None:
            return {}

        # 使用getattr获取属性值，提供默认值
        config = {}
        for key, default_value in VlmSafeGenerator.DEFAULT_QUANTIZATION_CONFIG.items():
            config[key] = getattr(args, key, default_value)

        # 特殊处理必需的属性
        required_attrs = ['w_bit', 'a_bit', 'device_type']
        for attr in required_attrs:
            if hasattr(args, attr):
                if attr == 'device_type':
                    config['dev_type'] = getattr(args, attr)
                else:
                    config[attr] = getattr(args, attr)

        # 特殊逻辑：当is_lowbit为True且open_outlier为False时，group_size生效
        if hasattr(args, 'group_size') and hasattr(args, 'is_lowbit') and hasattr(args, 'open_outlier'):
            if args.is_lowbit and not args.open_outlier:
                config['group_size'] = args.group_size

        return config
