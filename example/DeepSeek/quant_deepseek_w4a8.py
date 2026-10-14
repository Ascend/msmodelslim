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
# pylint: disable=expression-not-assigned,no-name-in-module,unspecified-encoding

import argparse
import functools
import json
import os
import sys

import torch
import torch_npu
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from convert_fp8_to_bf16 import auto_convert_model_fp8_to_bf16, OpsType  # noqa: E402
from add_safetensors import add_safetensors  # noqa: E402
from mtp_quant_module import warp_mtp_model, post_process_mtp_quant  # noqa: E402

from example.common.security.path import get_valid_read_path, get_write_directory  # noqa: E402
from example.common.security.path import json_safe_load, json_safe_dump  # noqa: E402
from example.common.security.type import check_number  # noqa: E402
from example.common.utils import cmd_bool  # noqa: E402
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
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for anti and calibration")
    parser.add_argument('--from_fp8', action='store_true', help="Origin model is of fp8")
    parser.add_argument('--from_bf16', action='store_true', help="Origin model is of bf16")
    parser.add_argument(
        '--mindie_format',
        action='store_true',
        help="Compatible with quantization formats \
                        supported by before 2.1.RC1 version of MindIE",
    )
    parser.add_argument(
        '--quant_mtp',
        type=str,
        choices=['mix', 'float', 'none'],
        default='none',
        help="Quantization mode: 'mix(w8a8 mix quant)' , or \
                            'float(save float mtp weight)' (default: %(default)s)",
    )
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=False)
    return parser.parse_args()


def create_custom_hook(quant_mtp, mindie_format):
    def custom_hook(model_config):
        model_config["mla_quantize"] = "w8a8"
        if quant_mtp == 'mix':
            model_config["mtp_quantize"] = "w8a8_dynamic"
        model_config["quantize"] = "w8a8_dynamic"
        model_config["moe_quantize"] = "w4a8_dynamic"
        if mindie_format:
            model_config["model_type"] = "deepseekv2"

    return custom_hook


def get_calib_dataset_batch(model_tokenizer, calib_list, batch_size, max_len=512, device="npu"):
    calib_dataset = []

    def truncate_strings(strings: list[str], max_len=max_len) -> list[str]:
        result = []
        for s in strings:
            current = s
            while True:
                # 截断前 512 字符作为一个块
                chunk = current[:max_len]
                result.append(chunk)
                # 剩余部分继续处理
                current = current[max_len:]
                # 没有剩余内容时结束循环
                if not current:
                    break
        return result

    calib_list = truncate_strings(calib_list)
    calib_list = [calib_list[i : i + batch_size] for i in range(0, len(calib_list), batch_size)]
    for calib_data in calib_list:
        inputs = model_tokenizer(calib_data, return_tensors='pt', padding=True).to(device)
        calib_dataset.append(
            [value.to(device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)]
        )
    return calib_dataset


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


def update_quant_type(save_path, json_filename):
    """
    更新量化类型为 W8A8_DYNAMIC

    参数:
    save_path (str): 输出JSON文件路径
    json_filename (str): 输入JSON文件名称

    返回值:
    None
    """
    json_file_path = os.path.join(save_path, json_filename)
    description_data = json_safe_load(json_file_path)
    description_data["model_quant_type"] = "W8A8_DYNAMIC"
    json_safe_dump(description_data, json_file_path, indent=4)


def main():
    args = parse_args()
    set_logger_level("info")
    # 显示整个量化过程各个步骤的进度条
    pbar = tqdm(total=5, position=0, desc="Total Process")
    model_path = args.model_path
    save_path = args.save_path
    anti_path = args.anti_dataset
    calib_path = args.calib_dataset
    batch_size = args.batch_size

    model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=True)
    save_path = get_write_directory(save_path, write_mode=0o750)
    anti_path = get_valid_read_path(anti_path, is_dir=False, check_user_stat=True)
    calib_path = get_valid_read_path(calib_path, is_dir=False, check_user_stat=True)
    check_number(batch_size, int, 1, 16, "batch_size")

    config = AutoConfig.from_pretrained(  # nosec B615
        pretrained_model_name_or_path=model_path, trust_remote_code=args.trust_remote_code
    )
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

    if config.num_hidden_layers < 0:
        raise ValueError("model num_hidden_layers is invalid, please check it.")

    # Set model type to deepseekv2 because we only support deepseekv2 now,
    # but v3's architecture is same as v2 without mtp layers so that we can reuse
    config.model_type = "deepseekv2"
    # Disable use cache because we don't need to use cache, otherwise it will use too much device memory then cause OOM
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        pretrained_model_name_or_path=model_path,
        config=config,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        add_eos_token=True,
    )

    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        pretrained_model_name_or_path=model_path,
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

    pbar.update(1)

    with open(anti_path, "r") as file:
        anti_prompt = json.load(file)
    with open(calib_path, "r") as file:
        calib_prompt = json.load(file)

    anti_dataset = get_calib_dataset_batch(tokenizer, anti_prompt, batch_size, device=model.device)
    dataset_calib = get_calib_dataset_batch(tokenizer, calib_prompt, batch_size, device=model.device)

    with torch.no_grad():
        anti_config = AntiOutlierConfig(w_bit=8, a_bit=8, anti_method='m4', dev_type='npu', dev_id=model.device.index)
        anti_outlier = AntiOutlier(model, calib_data=anti_dataset, cfg=anti_config)
        anti_outlier.process()
    pbar.update(1)

    disable_names = []
    for ids in range(config.num_hidden_layers):
        disable_names.append("model.layers." + str(ids) + ".self_attn.kv_b_proj")
    if args.quant_mtp == "mix":
        disable_names.append("lm_head")
        disable_names.append("mtp_decoder.self_attn.kv_b_proj")
        disable_names.append("mtp_layer.shared_head.head")

    w4a8_pertoken_config = QuantConfig(
        a_bit=8,
        w_bit=4,
        w_sym=True,
        dev_id=model.device.index,
        dev_type='npu',
        is_lowbit=True,
        mm_tensor=False,
        group_size=256,
        is_dynamic=True,
        do_smooth=False,
        use_sigma=True,
        open_outlier=False,
        disable_names=disable_names,
    )

    # w4a8_dynamic 和 w8a8 & w8a8_dynamic 混合量化配置
    mix_cfg = {
        "model.layers.[012].mlp.gate_proj": "w8a8_dynamic",
        "model.layers.[012].mlp.down_proj": "w8a8_dynamic",
        "model.layers.[012].mlp.up_proj": "w8a8_dynamic",
        "model.layers.[0-9]*.self_attn.*": "w8a8",
        "model.layers.[0-9]*.mlp.shared_experts.*": "w8a8_dynamic",
        "model.layers.[0-9]*.mlp.experts.*": "w4a8_dynamic",
        "mtp_decoder.mlp.shared_experts.*": "w8a8_dynamic",
        "mtp_decoder.mlp.experts.*": "w8a8_dynamic",
    }

    calibrator = Calibrator(model, w4a8_pertoken_config, calib_data=dataset_calib, disable_level="L0", mix_cfg=mix_cfg)
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
    # w4a8 混合量化中 MindIE 要求 description 中的 model_quant_type 为 W8A8_DYNAMIC
    update_quant_type(save_path, quant_model_description_json_name)
    # 适配mindie删除description里的module字段
    if args.mindie_format:
        remove_module_entries(save_path)

    custom_hook_instance = create_custom_hook(args.quant_mtp, args.mindie_format)
    custom_hooks = {
        'config.json': functools.partial(modify_config_json, custom_hook=custom_hook_instance)
        if args.mindie_format
        else functools.partial(modify_vllm_config_json, custom_hook=custom_hook_instance)
    }
    copy_config_files(
        input_path=model_path,
        output_path=save_path,
        quant_config=w4a8_pertoken_config,
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
    main()
