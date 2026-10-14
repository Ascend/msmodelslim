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
# pylint: disable=expression-not-assigned,unspecified-encoding

import argparse
import functools
import json
import os
import sys
from unittest.mock import patch

import torch
import torch_npu
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from convert_fp8_to_bf16 import auto_convert_model_fp8_to_bf16, OpsType  # noqa: E402
from add_safetensors import add_safetensors  # noqa: E402
from mtp_quant_module import warp_mtp_model, post_process_mtp_quant  # noqa: E402
from example.common.rot_utils.rot_ds import rot_model  # noqa: E402

from example.common.security.path import get_valid_read_path, get_write_directory  # noqa: E402
from example.common.security.path import json_safe_load, json_safe_dump, get_valid_write_path  # noqa: E402
from example.common.security.type import check_number  # noqa: E402
from example.common.utils import SafeGenerator, cmd_bool  # noqa: E402
from example.common.copy_config_files import copy_config_files, modify_config_json, modify_vllm_config_json  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier  # noqa: E402
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402
from msmodelslim.utils.logging import set_logger_level  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    (parser.add_argument('--model_path', type=str, help="The path of float model and tokenizer"),)
    (parser.add_argument('--save_path', type=str, help="The path to save quant model"),)
    parser.add_argument('--layer_count', type=int, default=0, help="Layer count when loading model")
    parser.add_argument(
        '--anti_dataset',
        type=str,
        default="../common/deepseek_anti_prompt.json",
        help="The calib data for anti outlier",
    )
    parser.add_argument(
        '--calib_dataset',
        type=str,
        default="../common/deepseek_calib_prompt.json",
        help="The calib data for calibration",
    )
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for anti and calibration")
    parser.add_argument('--from_fp8', action='store_true', help="Origin model is of fp8")
    parser.add_argument('--from_bf16', action='store_true', help="Origin model is of bf16")
    parser.add_argument('--fa_quant', action="store_true", help="Enable fa quant")
    parser.add_argument(
        '--mindie_format',
        action='store_true',
        help="Compatible with quantization formats \
                        supported by before 2.1.RC1 version of MindIE",
    )
    parser.add_argument('--dynamic', action='store_true', help="Dynamic Quantization")
    parser.add_argument('--disable_anti', action='store_true', help="No AntiOutlier")
    parser.add_argument(
        '--quant_mtp',
        type=str,
        choices=['mix', 'float', 'none'],
        default='none',
        help="Quantization mode: 'mix(w8a8 mix quant)' , or \
                            'float(save float mtp weight)' (default: %(default)s)",
    )
    parser.add_argument('--anti_method', type=str, choices=['m4', 'm6'], default='m4')
    parser.add_argument('--rot', action='store_true', help="rot model")
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=False)
    return parser.parse_args()


def create_custom_hook(quant_mtp, mindie_format, fa_quant):
    def custom_hook(model_config):
        # mindie暂不支持ascendV1格式，临时适配，后续在ascendV1移除
        model_config["mla_quantize"] = "w8a8"
        if quant_mtp == 'mix':
            model_config["mtp_quantize"] = "w8a8_dynamic"
        model_config["quantize"] = "w8a8_dynamic"
        if mindie_format:
            model_config["model_type"] = "deepseekv2"
        if fa_quant:
            if 'quantization_config' not in model_config:
                model_config['quantization_config'] = {}
            model_config['quantization_config']['fa_quant_type'] = 'FAKQuant'

    return custom_hook


def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, device="npu"):
    calib_dataset = []
    calib_list = [calib_list[i : i + batch_size] for i in range(0, len(calib_list), batch_size)]
    for calib_data in calib_list:
        inputs = model_tokenizer(calib_data, return_tensors='pt', padding=True).to(device)
        calib_dataset.append(
            [value.to(device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)]
        )
    return calib_dataset


def set_initialized_submodules(model, state_dict_keys):
    """
    Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state
    dict.
    """
    state_dict_keys = set(state_dict_keys)
    not_initialized_submodules = {}
    for module_name, module in model.named_modules():
        if module_name == "":
            # When checking if the root module is loaded there's no need to prepend module_name.
            module_keys = set(module.state_dict())
        else:
            module_keys = {f"{module_name}.{k}" for k in module.state_dict()}
        if module_keys.issubset(state_dict_keys):
            module._is_hf_initialized = True
        else:
            not_initialized_submodules[module_name] = module
    return not_initialized_submodules


def remove_module_entries(save_path, json_filename="quant_model_description_w8a8_dynamic.json"):
    """
    移除JSON文件中键包含"module"的条目

    参数:
    save_path (str): 输入JSON文件路径，json_filename (str): 输入JSON文件名称

    """
    json_file_path = os.path.join(save_path, json_filename)
    # 读取JSON文件
    description_data = json_safe_load(json_file_path)
    # 过滤掉键中包含"module"的条目
    filtered_data = {
        key: value
        for key, value in description_data.items()
        if "norm.module." not in key
        # 检查键中是否包含"norm.module."字符串
    }
    # 写回更新后的JSON文件（如需保留原始文件，可改为写入新文件）
    json_safe_dump(filtered_data, json_file_path, indent=4)


def update_quant_model_description_content(save_path, json_filename):
    """
    更新量化类型为 W8A8_DYNAMIC

    参数:
    save_path (str): 输出JSON文件路径
    json_filename (str): 输入JSON文件名称

    返回值:
    None
    """
    dest_quant_description_filepath = os.path.join(save_path, json_filename)
    dest_quant_description_filepath = get_valid_write_path(dest_quant_description_filepath, is_dir=False)
    quant_description_data = json_safe_load(dest_quant_description_filepath, check_user_stat=True)
    quant_description_data["model_quant_type"] = "W8A8_DYNAMIC"
    if (
        ("version" in quant_description_data.keys())
        and ("fa_quant_type" in quant_description_data.keys())
        and (quant_description_data["fa_quant_type"] == "FAQuant")
    ):
        quant_description_data["fa_quant_type"] = "FAKQuant"
    json_safe_dump(quant_description_data, dest_quant_description_filepath, indent=4)


def main():
    args = parse_args()
    set_logger_level("info")
    # 显示整个量化过程各个步骤的进度条
    pbar = tqdm(total=5, position=0, desc="Total Process")

    model_path = args.model_path
    batch_size = args.batch_size

    save_path = get_write_directory(args.save_path, write_mode=0o750)
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

    # mtp量化需要加载61层
    if args.quant_mtp == "mix":
        config.num_hidden_layers = config.num_hidden_layers + 1

    # Set model type to deepseekv2 because we only support deepseekv2 now,
    # but v3's architecture is same as v2 without mtp layers so that we can reuse
    config.model_type = "deepseekv2"
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
        device_map={
            "model.embed_tokens": 0,
            "model.layers": "cpu",
            "model.norm": "cpu",
            "lm_head": 0,
        },
        torch_dtype="auto",
        attn_implementation='eager',
    )

    # 内存自动反量化fp8到bf16，如果没有反量化参数则认为是bf16，跳过
    auto_convert_model_fp8_to_bf16(model, model_path, OpsType.get_ops_type(args.from_bf16, args.from_fp8))

    # mtp量化封装原模型为mtp model
    if args.quant_mtp == "mix":
        model = warp_mtp_model(config, model, model_path)

    if args.rot:
        if args.quant_mtp == "float":
            raise NotImplementedError(
                "original float mtp is not supported rotation, "
                "please use mix mode and disable all MTP Linear to get float rotated MTP model!"
            )
        rot_model(model)

    pbar.update(1)

    anti_dataset_path = get_valid_read_path(args.anti_dataset, "json", is_dir=False)
    calib_dataset_path = get_valid_read_path(args.calib_dataset, "json", is_dir=False)

    with open(anti_dataset_path, "r") as file:
        anti_prompt = json.load(file)
    with open(calib_dataset_path, "r") as file:
        calib_prompt = json.load(file)

    anti_dataset = get_calib_dataset_batch(tokenizer, anti_prompt, batch_size, model.device)
    dataset_calib = get_calib_dataset_batch(tokenizer, calib_prompt, batch_size, model.device)

    if not args.disable_anti:
        with torch.no_grad():
            anti_config = AntiOutlierConfig(
                w_bit=8, a_bit=8, anti_method=args.anti_method, dev_type='npu', dev_id=model.device.index
            )
            anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
            anti_outlier.process()
    pbar.update(1)

    disable_names = []
    for ids in range(config.num_hidden_layers):
        disable_names.append("model.layers." + str(ids) + ".self_attn.kv_b_proj")

    if args.fa_quant:
        if 'R1' in args.model_path.upper():
            disable_attn_layers = [0, 1, 2, 46, 47, 50, 54, 55, 56, 57, 58, 59, 60]
        elif 'V3.1' in args.model_path.upper():
            disable_attn_layers = [0, 1, 2, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60]
        elif 'V3' in args.model_path.upper():
            disable_attn_layers = [0, 1, 2, 46, 50, 51, 53, 54, 55, 56, 57, 59, 60]
        else:
            disable_attn_layers = []
        for ids in disable_attn_layers:
            if ids < num_layer:
                disable_names.append("model.layers." + str(ids) + ".self_attn")
    if args.quant_mtp == "mix":
        disable_names.append("lm_head")
        disable_names.append("mtp_decoder.self_attn.kv_b_proj")
        disable_names.append("mtp_layer.shared_head.head")

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        disable_names=disable_names,
        dev_type='npu',
        dev_id=model.device.index,
        act_method=1,
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=args.dynamic,
    )

    if args.fa_quant:
        quant_config = quant_config.fa_quant()
    calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level="L0")
    calibrator.run()
    pbar.update(1)

    if args.mindie_format:
        quant_model_description_json_name = "quant_model_description_w8a8_dynamic.json"
    else:
        quant_model_description_json_name = "quant_model_description.json"

    save_type = "safe_tensor" if args.mindie_format else "ascendV1"
    calibrator.save(
        save_path,
        json_name=quant_model_description_json_name,
        safetensors_name="quant_model_weight_w8a8_dynamic.safetensors",
        save_type=[save_type],
        part_file_size=4,
    )
    # w8a8 混合量化中 MindIE 要求 description 中的 model_quant_type 为 W8A8_DYNAMIC
    update_quant_model_description_content(save_path, quant_model_description_json_name)

    # 适配mindie删除description里的module字段
    if args.mindie_format:
        remove_module_entries(save_path)

    custom_hook_instance = create_custom_hook(args.quant_mtp, args.mindie_format, args.fa_quant)
    custom_hooks = {
        'config.json': functools.partial(modify_config_json, custom_hook=custom_hook_instance)
        if args.mindie_format
        else functools.partial(modify_vllm_config_json, custom_hook=custom_hook_instance)
    }
    copy_config_files(
        input_path=model_path,
        output_path=save_path,
        quant_config=quant_config,
        mindie_format=args.mindie_format,
        custom_hooks=custom_hooks,
    )
    pbar.update(1)
    if args.quant_mtp == "float":
        add_safetensors(
            org_paths=model_path,
            target_dir=save_path,
            safetensors_prefix="mtp_float",
            max_file_size_gb=5,
            prefix="model.layers.61.",
        )
    if args.quant_mtp == "mix":
        post_process_mtp_quant(save_path)
    pbar.update(1)


if __name__ == "__main__":
    # torch_npu will fork a new process to init,
    # it's lazy_init will fail after we load a big model,so we need to init it here
    torch_npu.npu.init()
    # This patch is for speeding up transformer loading big model
    patch("transformers.modeling_utils.set_initialized_submodules", new=set_initialized_submodules).start()
    # Invoke main process
    main()
