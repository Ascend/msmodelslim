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
# pylint: disable=no-member,no-name-in-module

import argparse
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(parent_directory)

from example.common.utils import cmd_bool  # noqa: E402
from example.common.security.path import get_valid_read_path, get_write_directory  # noqa: E402
from example.common.vlm_utils import VlmSafeGenerator, ModifyConfigParams, CopyTokenizerParams  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier  # noqa: E402
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402


CPU = "cpu"
NPU = "npu"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--calib_images', type=str, default='../calibImages')
    parser.add_argument('--save_directory', type=str, default='')
    parser.add_argument('--part_file_size', type=int, default=None)
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--device_type', type=str, choices=[CPU, NPU], default=NPU)
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=False)
    parser.add_argument(
        '--mindie_format',
        action="store_true",
        help="Compatible with quantization formats \
                    supported by MindIE",
    )
    args = parser.parse_args()

    # check args
    args.model_path = get_valid_read_path(args.model_path, is_dir=True, check_user_stat=True)
    args.calib_images = get_valid_read_path(args.calib_images, is_dir=True, check_user_stat=True)
    args.save_directory = get_write_directory(args.save_directory, write_mode=0o750)

    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        args.model_path, trust_remote_code=args.trust_remote_code, local_files_only=True
    )
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id

    ##1.加载模型
    device_map = CPU if args.device_type == CPU else "auto"
    config = AutoConfig.from_pretrained(  # nosec B615
        args.model_path, trust_remote_code=args.trust_remote_code, local_files_only=True
    )
    dtype = config.torch_dtype if args.device_type == NPU else torch.float32
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.model_path,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        local_files_only=True,
    ).eval()

    ##2.设置回退层
    num_layers = config.num_hidden_layers
    disable_names = [f"transformer.h.{layer}.mlp.c_proj" for layer in range(num_layers)]
    disable_names.append("transformer.visual.attn_pool.attn.out_proj")
    disable_names.append("lm_head")

    ##3.校准集
    images_list = os.listdir(args.calib_images)
    prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    calib_data = []
    for i in images_list:
        imagePath = os.path.join(args.calib_images, i)
        imagePath = get_valid_read_path(imagePath)
        query = tokenizer.from_list_format(
            [
                {'image': imagePath},  # Either a local path or an url
                {'text': prompt},
            ]
        )
        inputs = tokenizer(query, return_tensors='pt')
        inputs = inputs.to(model.device)
        calib_data.append([inputs['input_ids'], None, inputs['attention_mask']])

    ##4.异常值抑制
    anti_config = AntiOutlierConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        anti_method="m2",
        dev_type=args.device_type,
        dev_id=model.device.index,
    )
    anti_outlier = AntiOutlier(model, calib_data=calib_data, cfg=anti_config)
    anti_outlier.process()
    ##5.模型量化
    quant_config = QuantConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        disable_names=disable_names,
        dev_type=args.device_type,
        dev_id=model.device.index,
        act_method=2,
        mm_tensor=False,
    )
    calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level='L0')
    calibrator.run()
    ##6.保存权重
    save_type = "safe_tensor" if args.mindie_format else "ascendV1"
    calibrator.save(args.save_directory, save_type=[save_type], part_file_size=args.part_file_size)

    quant_type = quant_config.model_quant_type.lower()
    checker = VlmSafeGenerator()
    auto_config = checker.get_config_from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    # 使用dataclass参数
    modify_params = ModifyConfigParams(
        model_dir=args.model_path,
        dest_dir=args.save_directory,
        torch_dtype=auto_config.torch_dtype,
        quantize_type=quant_type,
        args=args,
        quantize_config_parts=['visual'],
    )
    checker.modify_config(modify_params)

    copy_params = CopyTokenizerParams(model_dir=args.model_path, dest_dir=args.save_directory)
    checker.copy_tokenizer_files(copy_params)
