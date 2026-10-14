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
# pylint: disable=no-name-in-module

import os
import argparse
import json
import shutil
from typing import Any, Dict, Union

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    from msmodelslim import logger

    logger.warning("torch_npu is not available, if you are using NPU, please install torch_npu")

from example.common.security.path import json_safe_load, json_safe_dump
from example.common.security.path import get_valid_read_path, get_valid_write_path


MAX_KEY_LENGTH = 256
MAX_JSON_LENGTH = 4096


class SafeGenerator:
    def __init__(self):
        pass

    @staticmethod
    def get_config_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=True)
        try:
            config = AutoConfig.from_pretrained(model_path, local_files_only=True, **kwargs)  # nosec B615
        except EnvironmentError as env_err:
            raise EnvironmentError(
                f"Get model from pretrained failed, please check model weights files in the model path. "
                f"If the file exists, make sure the folder's owner has execute permission."
                f"Original error: {env_err}"
            ) from env_err
        except Exception as err:
            raise ValueError(
                f"Get model from pretrained failed, please check model weights files in the model path. "
                f"If the file exists, make sure the folder's owner has execute permission."
                f"Original error: {err}"
            ) from err
        return config

    @staticmethod
    def get_model_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **kwargs)  # nosec B615
        except EnvironmentError as env_err:
            raise EnvironmentError(
                f"Get model from pretrained failed, please check model weights files in the model path. "
                f"If the file exists, make sure the folder's owner has execute permission."
                f"Original error: {env_err}"
            ) from env_err
        except Exception as err:
            raise ValueError(
                f"Get model from pretrained failed, please check model weights files in the model path. "
                f"If the file exists, make sure the folder's owner has execute permission."
                f"Original error: {err}"
            ) from err
        return model

    @staticmethod
    def get_tokenizer_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, **kwargs)  # nosec B615
        except EnvironmentError as env_err:
            raise EnvironmentError(
                f"Get model from pretrained failed, please check model weights files in the model path. "
                f"If the file exists, make sure the folder's owner has execute permission."
                f"Original error: {env_err}"
            ) from env_err
        except Exception as err:
            raise ValueError(
                f"Get model from pretrained failed, please check model weights files in the model path. "
                f"If the file exists, make sure the folder's owner has execute permission."
                f"Original error: {err}"
            ) from err
        return tokenizer

    @staticmethod
    def copy_tokenizer_files(model_dir, dest_dir):
        model_dir = get_valid_read_path(model_dir, is_dir=True, check_user_stat=True)
        if os.path.exists(dest_dir):
            dest_dir = get_valid_write_path(dest_dir, is_dir=True)
        else:
            os.makedirs(dest_dir, mode=0o750, exist_ok=True)
            dest_dir = get_valid_write_path(dest_dir, is_dir=True)
        filenames = os.listdir(model_dir)
        max_file_num = 1024
        if len(filenames) > max_file_num:
            raise argparse.ArgumentTypeError(
                f"The file num in dir is {len(filenames)}, which exceeds the limit {max_file_num}."
            )
        for filename in filenames:
            need_move = False
            file_names = ['tokenizer', 'tokenization', 'special_token_map', 'generation', 'configuration', 'tiktoken']
            for f in file_names:
                if f in filename:
                    need_move = True
                    break
            if need_move:
                src_filepath = os.path.join(model_dir, filename)
                dest_filepath = os.path.join(dest_dir, filename)
                shutil.copyfile(src_filepath, dest_filepath)
                os.chmod(dest_filepath, int("600", 8))

    @staticmethod
    def modify_config(model_dir, dest_dir, torch_dtype, quantize_type, args=None):
        model_dir = get_valid_read_path(model_dir, is_dir=True, check_user_stat=True)
        src_config_filepath = os.path.join(model_dir, 'config.json')
        data = json_safe_load(src_config_filepath, check_user_stat=True)
        dest_dir = get_valid_write_path(dest_dir, is_dir=True)

        if args.mindie_format:
            dest_quant_description_filepath = os.path.join(
                dest_dir, f"quant_model_description_{quantize_type.lower()}.json"
            )
        else:
            dest_quant_description_filepath = os.path.join(dest_dir, "quant_model_description.json")
        dest_quant_description_filepath = get_valid_write_path(dest_quant_description_filepath, is_dir=False)
        quant_description_data = json_safe_load(dest_quant_description_filepath, check_user_stat=True)

        data['torch_dtype'] = str(torch_dtype).split(".")[1]
        if args.mindie_format:
            data['quantize'] = quantize_type
        if args is not None:
            quantization_config = {
                # 当is_lowbit为True，open_outlier为False时，group_size生效
                'group_size': args.group_size if args.is_lowbit and not args.open_outlier else 0,
                'kv_quant_type': "C8" if args.use_kvcache_quant else None,
                "fa_quant_type": "FAQuant" if args.use_fa_quant else None,
                'w_bit': args.w_bit,
                'a_bit': args.a_bit,
                'dev_type': args.device_type,
                'fraction': args.fraction,
                'act_method': args.act_method,
                'co_sparse': args.co_sparse,
                'anti_method': args.anti_method,
                'disable_level': args.disable_level,
                'do_smooth': args.do_smooth,
                'use_sigma': args.use_sigma,
                'sigma_factor': args.sigma_factor,
                'is_lowbit': args.is_lowbit,
                'mm_tensor': False,
                'w_sym': args.w_sym,
                'open_outlier': args.open_outlier,
                'is_dynamic': args.is_dynamic,
            }
            if hasattr(args, 'pdmix') and args.pdmix:
                quantization_config.update({"pdmix": args.pdmix})
            if args.use_reduce_quant:
                quantization_config.update({"reduce_quant_type": "per_channel"})
            quant_description_data.update(quantization_config)

            if args.mindie_format:
                data['quantization_config'] = quantization_config

        dest_config_filepath = os.path.join(dest_dir, 'config.json')
        json_safe_dump(data, dest_config_filepath, 4)

    @staticmethod
    def load_jsonl(dataset_path, key_name='inputs_pretokenized'):
        dataset = []
        if dataset_path == "humaneval_x.jsonl":
            key_name = 'prompt'
        with os.fdopen(os.open(dataset_path, os.O_RDONLY, 0o600), 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                data = json.loads(line)
                text = data.get(key_name, line)
                dataset.append(text)
        return dataset


class ArgumentValidator:
    context = None

    def __init__(self, *args, allow_none: bool = False, **kwargs):
        self.allow_none = allow_none
        self.validation_pipeline = []
        self.create_validation_pipeline()

    def validate(self, value: Any) -> None:
        if value is None and self.allow_none:
            return
        for method in self.validation_pipeline:
            method(value)

    def add_validation_method(self, method, position: int = None, target_method=None):
        if position is not None:
            self.validation_pipeline.insert(position, method)
        elif target_method and target_method in self.validation_pipeline:
            target_index = self.validation_pipeline.index(target_method)
            self.validation_pipeline.insert(target_index + 1, method)
        else:
            self.validation_pipeline.append(method)

    def delete_validation_method(self, method=None, position: int = None):
        if position is not None:
            if 0 <= position < len(self.validation_pipeline):
                self.validation_pipeline.pop(position)
        elif method and method in self.validation_pipeline:
            self.validation_pipeline.remove(method)

    def create_validation_pipeline(self):  # 空方法，子类可以覆盖
        pass

    def _create_validation_pipeline(self, *methods):
        self.validation_pipeline.clear()
        self.validation_pipeline.extend(methods)


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.argument_validators = {}
        super().__init__(*args, **kwargs)

    def add_argument(
        self, *args, validator: Union[ArgumentValidator, Dict[Any, ArgumentValidator]] = None, **kwargs
    ) -> argparse.Action:
        arguments = super().add_argument(*args, **kwargs)
        if validator is not None:
            self.argument_validators.update({arguments.dest: validator})
        return arguments

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        args_all = super().parse_args(args, namespace)
        ArgumentValidator.context = vars(args_all)
        for arg, value in vars(args_all).items():
            if arg in self.argument_validators:
                validator = self.argument_validators[arg]
                type_of_value = type(value)
                try:
                    if isinstance(validator, dict):
                        if type_of_value is list:
                            type_of_value_to_validate = type(value[0])
                        else:
                            type_of_value_to_validate = type_of_value
                        if type_of_value_to_validate in validator:
                            validator[type_of_value_to_validate].validate(value)
                        else:
                            raise argparse.ArgumentTypeError(
                                f"Validation failed for argument '{arg}': \
                                type {type_of_value_to_validate} not supported"
                            )
                    else:
                        validator.validate(value)
                except argparse.ArgumentTypeError as e:
                    raise argparse.ArgumentTypeError(f"Validation failed for argument '{arg}': {e}")
        return args_all

    def update_argument(self, old_name: str, new_name: str = None, **kwargs) -> None:
        old_name = old_name.lstrip('-')
        if new_name:
            kwargs.update({'dest': new_name.lstrip('-')})
        for action in self._actions:
            if action.dest == old_name:
                for key, value in kwargs.items():
                    setattr(action, key, value)


class StringArgumentValidator(ArgumentValidator):
    def __init__(self, min_length: int = 0, max_length: int = float('inf'), allow_none: bool = False):
        super().__init__(allow_none=allow_none)
        self.min_length = min_length
        self.max_length = max_length

    @staticmethod
    def validate_type(value: str) -> None:
        if not isinstance(value, str):
            raise argparse.ArgumentTypeError("Value must be a string")

    def validate_length(self, value: str) -> None:
        if not (self.min_length <= len(value) <= self.max_length):
            raise argparse.ArgumentTypeError(f"String length must be between {self.min_length} and {self.max_length}")

    def create_validation_pipeline(self):
        super()._create_validation_pipeline(self.validate_type, self.validate_length)


def cmd_bool(cmd_arg):
    if cmd_arg == "True":
        return True
    elif cmd_arg == "False":
        return False
    raise ValueError(f"{cmd_arg} should be True or False")


def parse_tokenizer_args(input_str, default=None):
    default = {} if default is None else default
    try:
        args_dict = json.loads(input_str)
        if not isinstance(args_dict, dict):
            raise ValueError("Parsed JSON must be a dictionary")
        return args_dict
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        if not isinstance(default, dict):
            raise ValueError("Default value must be a dictionary") from e
        return default
