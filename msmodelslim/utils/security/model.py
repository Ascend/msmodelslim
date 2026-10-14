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

import json
import os


from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.exception_decorator import exception_handler
from msmodelslim.utils.security import get_valid_read_path

MAX_KEY_LENGTH = 256
MAX_JSON_LENGTH = 4096


class SafeGenerator:
    def __init__(self):
        pass

    @staticmethod
    def get_config_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True)
        with exception_handler(
            f'Get config from pretrained failed in {model_path}.',
            err_cls=Exception,
            ms_err_cls=InvalidModelError,
            action="Please ensure config files all exist and are valid. "
            "Otherwise, the transformers version is not compatible with the model."
            "Before using msModelSlim, please make sure the model load and infer properly.",
        ):
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path, local_files_only=True, **kwargs)  # nosec
            return config

    @staticmethod
    def get_model_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True)
        with exception_handler(
            f'Get model from pretrained failed in {model_path}.',
            err_cls=Exception,
            ms_err_cls=InvalidModelError,
            action="Please ensure the model weights files all exist and are valid. "
            "Otherwise, the transformers version is not compatible with the model."
            "Before using msModelSlim, please make sure the model load and infer properly.",
        ):
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **kwargs)  # nosec
            return model

    @staticmethod
    def get_tokenizer_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True)
        with exception_handler(
            f'Get tokenizer from pretrained failed in {model_path}.',
            err_cls=Exception,
            ms_err_cls=InvalidModelError,
            action="Please ensure the tokenizer files all exist and are valid. "
            "Otherwise, the transformers version is not compatible with the model."
            "Before using msModelSlim, please make sure the model load and infer properly.",
        ):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, **kwargs)  # nosec
            return tokenizer

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
