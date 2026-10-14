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
# pylint: disable=expression-not-assigned

import os
import sys
import argparse
import functools
import random
import json

import numpy as np

import torch
import torch_npu
import transformers

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from example.common.security.path import get_valid_read_path, get_write_directory  # noqa: E402
from example.common.security.type import check_number  # noqa: E402
from example.common.utils import SafeGenerator, cmd_bool  # noqa: E402
from example.common.copy_config_files import copy_config_files, modify_config_json  # noqa: E402
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402
from msmodelslim.utils.logging import set_logger_level  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier  # noqa: E402


def seed_everything(seed=0) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    transformers.set_seed(seed)
    torch_npu.npu.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    (parser.add_argument('--model_path', type=str, help="The path of float model and tokenizer"),)
    (parser.add_argument('--save_directory', type=str, help="The path to save quant model"),)
    parser.add_argument('--layer_count', type=int, default=0, help="Layer count when loading model")
    parser.add_argument('--calib_file', type=str, default="../common/wiki.jsonl", help="The calib data for calibration")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for anti and calibration")
    parser.add_argument(
        '--mindie_format',
        action="store_true",
        help="Compatible with quantization formats \
                        supported by before 2.1.RC1 version of MindIE",
    )
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=False)
    return parser.parse_args()


def custom_hook(model_config):
    model_config["quantize"] = "w4a4_flatquant_dynamic"


def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, device="npu"):
    calib_dataset = []
    calib_list = [calib_list[i : i + batch_size] for i in range(0, len(calib_list), batch_size)]
    for calib_data in calib_list:
        inputs = model_tokenizer(calib_data, return_tensors='pt', padding=True).to(device)
        calib_dataset.append(
            [value.to(device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)]
        )
    return calib_dataset


def pre_check_files(path):
    """
    预先检查模型路径的json和py文件权限配置是否符合要求
    """
    for file in os.listdir(path):
        if not (file.endswith('.json') or file.endswith('.py')):
            continue
        _ = get_valid_read_path(os.path.join(path, file), extensions=['.json', '.py'])


def main():
    args = parse_args()
    set_logger_level("info")
    # 显示整个量化过程各个步骤的进度条
    seed_everything()
    model_path = args.model_path
    batch_size = args.batch_size

    save_directory = get_write_directory(args.save_directory, write_mode=0o750)
    pre_check_files(model_path)
    check_number(batch_size, int, 1, 16, "batch_size")

    safe_generator = SafeGenerator()

    config = safe_generator.get_config_from_pretrained(model_path=model_path, trust_remote_code=args.trust_remote_code)
    num_layer = config.num_hidden_layers
    if args.layer_count < 0 or args.layer_count > num_layer:
        raise ValueError(
            f"Invalid value for parameter layer_count: {args.layer_count}.Must be between 0 and {num_layer}."
        )
    # Set layer count to 0 means use all layers, otherwise it will only use the first layer_count layers
    config.num_hidden_layers = args.layer_count if args.layer_count != 0 else config.num_hidden_layers

    # Disable use cache because we don't need to use cache, otherwise it will use too much device memory then cause OOM
    config.use_cache = False

    tokenizer = safe_generator.get_tokenizer_from_pretrained(
        model_path=model_path,
        config=config,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        add_eos_token=True,
    )

    model = safe_generator.get_model_from_pretrained(
        model_path=model_path,
        config=config,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation='eager',
    )
    if args.calib_file.endswith('.jsonl'):
        calib_dataset_path = get_valid_read_path(args.calib_file, "jsonl", is_dir=False)
        calib_prompt = []
        with open(calib_dataset_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                calib_prompt.append(json.loads(line)['inputs_pretokenized'])
    elif args.calib_file.endswith('.json'):
        calib_dataset_path = get_valid_read_path(args.calib_file, "json", is_dir=False)
        with open(calib_dataset_path, "r", encoding="utf-8") as file:
            calib_prompt = json.load(file)
    else:
        raise ValueError("calib_file must be a jsonl or json file")
    dataset_calib = get_calib_dataset_batch(tokenizer, calib_prompt, batch_size, model.device)
    anti_disable_names = ["model.layers.{}.self_attn.o_proj".format(i) for i in range(config.num_hidden_layers)]
    anti_config = AntiOutlierConfig(
        w_bit=8,
        a_bit=8,
        anti_method='m6',
        dev_type='npu',
        disable_anti_names=anti_disable_names,
        flex_config={'alpha': 0.4, 'beta': 0.325},
        dev_id=model.device.index,
    )
    anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
    anti_outlier.process()
    disable_names = []
    for i in range(config.num_hidden_layers):
        disable_names.append(f'model.layers.{i}.mlp.down_proj')

    quant_config = QuantConfig(
        a_bit=4,
        w_bit=4,
        disable_names=disable_names,
        dev_type='npu',
        dev_id=model.device.index,
        act_method=1,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=True,
    )

    calibrator = Calibrator(
        model,
        quant_config,
        calib_data=dataset_calib,
        disable_level="L0",
        mix_cfg={
            "*.o_proj": "w8a8_dynamic",
            "*.down_proj": "w8a8_dynamic",
        },
    )
    calibrator.run()

    if args.mindie_format:
        quant_model_description_json_name = "quant_model_description_w4a4_flatquant_dynamic.json"
    else:
        quant_model_description_json_name = "quant_model_description.json"
    save_type = "safe_tensor" if args.mindie_format else "ascendV1"
    calibrator.save(
        save_directory,
        json_name=quant_model_description_json_name,
        safetensors_name="quant_model_weight_w4a4_flatquant_dynamic.safetensors",
        save_type=[save_type],
        part_file_size=4,
    )

    custom_hooks = {'config.json': functools.partial(modify_config_json, custom_hook=custom_hook)}
    copy_config_files(
        input_path=model_path,
        output_path=save_directory,
        quant_config=quant_config,
        mindie_format=args.mindie_format,
        custom_hooks=custom_hooks,
    )


if __name__ == "__main__":
    main()
