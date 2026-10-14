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

import os
import sys
import argparse
from transformers import AutoModel, AutoTokenizer, AutoConfig

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(parent_directory)

from internvl2_utils import get_tokenized_data, get_textvqa_calibration, cmd_bool  # noqa: E402
from example.common.security.path import get_valid_read_path, get_write_directory  # noqa: E402
from example.common.vlm_utils import VlmSafeGenerator, ModifyConfigParams, CopyTokenizerParams  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier  # noqa: E402
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402


CPU = "cpu"
NPU = "npu"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--calib_images', type=str, default='./textvqa_val')
    parser.add_argument('--calib_num', type=int, default=30, help='random sample calib num')
    parser.add_argument('--save_directory', type=str, default='')
    parser.add_argument('--part_file_size', type=int, default=None)
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--act_method', type=int, default=1)
    parser.add_argument('--device_type', type=str, choices=[CPU, NPU], default=NPU)
    parser.add_argument('--is_8B_model', action="store_true", help='whether to use 8B model')
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

    # 1.加载模型
    device_map = CPU if args.device_type == CPU else "auto"
    config = AutoConfig.from_pretrained(  # nosec B615
        args.model_path, local_files_only=True, trust_remote_code=args.trust_remote_code
    )
    dtype = config.torch_dtype
    model = AutoModel.from_pretrained(  # nosec B615
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map=device_map,
        use_safetensors=True,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        args.model_path, local_files_only=True, trust_remote_code=args.trust_remote_code, use_fast=False
    )

    # 2.调用chat接口
    model.forward = model.chat

    # 3.设置回退层
    disable_names = []
    vision_name = []
    if args.is_8B_model:
        llm_name = ["language_model.output", "mlp1.1", "mlp1.3"]
        for i in range(config.vision_config.num_hidden_layers):
            vision_name.extend([f"vision_model.encoder.layers.{i}.mlp.fc2"])
        for i in range(config.llm_config.num_hidden_layers):
            llm_name.extend([f"language_model.model.layers.{i}.feed_forward.w2"])
    else:
        llm_name = ["language_model.lm_head", "mlp1.1", "mlp1.3"]
        for i in range(config.vision_config.num_hidden_layers):
            vision_name.extend(
                [
                    f"vision_model.encoder.layers.{i}.mlp.fc1",
                    f"vision_model.encoder.layers.{i}.mlp.fc2",
                    f"vision_model.encoder.layers.{i}.attn.proj",
                    f"vision_model.encoder.layers.{i}.attn.qkv",
                ]
            )
        for i in range(config.llm_config.num_hidden_layers):
            llm_name.extend([f"language_model.model.layers.{i}.mlp.down_proj"])
    disable_names.extend(vision_name)
    disable_names.extend(llm_name)

    # 4.配置校准集
    if isinstance(args.calib_num, int) and args.calib_num > 0:
        calib_num = args.calib_num
    else:
        raise ValueError("calib_num should be a int number > 0")
    calibration_dataset = get_textvqa_calibration(args.calib_images, calib_num)
    calib_data = get_tokenized_data(tokenizer, calibration_dataset, dtype=dtype)

    # 5.异常值抑制
    anti_config = AntiOutlierConfig(a_bit=8, w_bit=8, anti_method='m2', dev_type='npu', dev_id=model.device.index)
    anti_outlier = AntiOutlier(model, calib_data=calib_data, cfg=anti_config)
    anti_outlier.process()

    # 6.模型量化
    quant_config = QuantConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        disable_names=disable_names,
        dev_type=args.device_type,
        dev_id=model.device.index,
        act_method=args.act_method,
        mm_tensor=False,
    )
    calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level='L0')
    calibrator.run()

    # 7.保存权重
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
        quantize_config_parts=['llm_config', 'vision_config'],
    )
    checker.modify_config(modify_params)

    copy_params = CopyTokenizerParams(model_dir=args.model_path, dest_dir=args.save_directory)
    checker.copy_tokenizer_files(copy_params)
