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
# pylint: disable=function-redefined,possibly-used-before-assignment,redefined-outer-name,ungrouped-imports

import os
import sys
import functools
import torch

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from example.common.security.path import get_valid_write_path  # noqa: E402
from example.common.utils import (  # noqa: E402
    SafeGenerator,
    ArgumentParser,
    StringArgumentValidator,
    MAX_KEY_LENGTH,
    MAX_JSON_LENGTH,
    cmd_bool,
    parse_tokenizer_args,
)
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig  # noqa: E402
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig  # noqa: E402
from example.common.copy_config_files import copy_config_files, modify_config_json  # noqa: E402


CPU = "cpu"
NPU = "npu"


def cmd_bool(cmd_arg):  # noqa: F811
    if cmd_arg == "True":
        return True
    elif cmd_arg == "False":
        return False
    raise ValueError(f"{cmd_arg} should be True or False")


def get_disable_names(num_layers: int) -> list:
    return [f"model.layers.{i}.mlp.gate.wg" for i in range(num_layers)]


def custom_hook(model_config):
    model_config["quantize"] = "w8a8"
    model_config["moe_quantize"] = "w8a8_dynamic"


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path")
    parser.add_argument('--save_directory', type=str)
    parser.add_argument('--part_file_size', type=int, default=5)
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--disable_names', type=str, nargs='+', default=None)
    parser.add_argument('--device_type', type=str, choices=[CPU, NPU], default=NPU)
    parser.add_argument('--fraction', type=float, default=0.01)
    parser.add_argument(
        "--act_method", type=int, choices=[1, 2, 3], default=1, help=" 1: MinMax, 2: Histogram, 3: Auto"
    )
    parser.add_argument('--co_sparse', type=cmd_bool, default=False)
    parser.add_argument('--anti_method', type=str, default='')
    parser.add_argument('--disable_level', type=str, default='L0')
    parser.add_argument('--do_smooth', type=cmd_bool, default=False)
    parser.add_argument('--use_sigma', type=cmd_bool, default=False)
    parser.add_argument('--use_reduce_quant', type=cmd_bool, default=False)
    parser.add_argument('--sigma_factor', type=float, default=3.0)
    parser.add_argument('--is_lowbit', type=cmd_bool, default=False)
    parser.add_argument('--mm_tensor', type=cmd_bool, default=True)
    parser.add_argument('--w_sym', type=cmd_bool, default=True)
    parser.add_argument('--use_kvcache_quant', type=cmd_bool, default=False)
    parser.add_argument('--use_fa_quant', type=cmd_bool, default=False)
    parser.add_argument('--fa_amp', type=int, default=0)
    parser.add_argument('--open_outlier', type=cmd_bool, default=True)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--is_dynamic', type=cmd_bool, default=False)
    parser.add_argument(
        '--input_ids_name',
        type=str,
        default='input_ids',
        validator=StringArgumentValidator(min_length=1, max_length=MAX_KEY_LENGTH),
    )
    parser.add_argument(
        '--attention_mask_name',
        type=str,
        default='attention_mask',
        validator=StringArgumentValidator(min_length=1, max_length=MAX_KEY_LENGTH),
    )
    parser.add_argument(
        '--tokenizer_args',
        type=str,
        default='{}',
        validator=StringArgumentValidator(min_length=2, max_length=MAX_JSON_LENGTH),
    )
    parser.add_argument('--disable_last_linear', type=cmd_bool, default=True)
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        validator=StringArgumentValidator(min_length=1, max_length=MAX_KEY_LENGTH, allow_none=True),
    )
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=False)
    parser.add_argument(
        '--mindie_format',
        action="store_true",
        help="Compatible with quantization formats \
                        supported by before 2.1.RC1 version of MindIE",
    )
    return parser.parse_args()


class Quantifier:
    def __init__(
        self,
        model_path_or_name,
        quant_config=None,
        anti_outlier_config=None,
        device_type='cpu',
        trust_remote_code=False,
        **kwargs,
    ):
        safe_generator = SafeGenerator()
        self.device_type = device_type
        device_map = CPU if self.device_type == CPU else "auto"  # noqa: F841
        self.trust_remote_code = trust_remote_code

        self.quant_config = quant_config
        self.anti_outlier_config = anti_outlier_config
        self.model_path_or_name = model_path_or_name
        self.config = safe_generator.get_config_from_pretrained(
            self.model_path_or_name, trust_remote_code=self.trust_remote_code
        )
        self.dtype = self.config.torch_dtype if self.device_type == NPU else torch.float32
        self.model = safe_generator.get_model_from_pretrained(
            self.model_path_or_name,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            device_map={
                "model.embed_tokens": 0,
                "model.layers": "cpu",
                "model.norm": "cpu",
                "lm_head": 0,
            },
        )

        tokenizer_args = kwargs.get("tokenizer_args", {})
        self.tokenizer = safe_generator.get_tokenizer_from_pretrained(
            self.model_path_or_name,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
            add_eos_token=True,
            **tokenizer_args,
        )
        self.model_name = kwargs.get("model_name", None)

    def get_tokenized_data(self, input_texts, input_ids_name='input_ids', attention_mask_name='attention_mask'):
        tokenized_data = []
        for input_text in input_texts:
            inputs = self.tokenizer(input_text, return_tensors='pt', padding=True).to(self.device_type)
            tokenized_data.append([inputs.data[input_ids_name], inputs.data[attention_mask_name]])
        return tokenized_data

    def convert(self, tokenized_data, save_path, disable_level, part_file_size=None):
        if self.device_type == NPU:
            # 避免在线编译算子，使用二进制编译的算子
            torch.npu.set_compile_mode(jit_compile=False)

        if self.anti_outlier_config is not None:
            anti_outlier = AntiOutlier(self.model, calib_data=tokenized_data, cfg=self.anti_outlier_config)
            anti_outlier.process()

        mix_cfg = {"*.experts.*": "w8a8_dynamic", "*": "w8a8"}
        calibrator = Calibrator(
            self.model, self.quant_config, calib_data=tokenized_data, disable_level=disable_level, mix_cfg=mix_cfg
        )
        calibrator.run()
        save_type = "safe_tensor" if args.mindie_format else "ascendV1"
        calibrator.save(save_path, save_type=[save_type], part_file_size=part_file_size)


if __name__ == '__main__':
    args = parse_arguments()
    checker = SafeGenerator()
    rank: int = int(os.getenv("RANK", "0"))

    model_path = args.model_path
    save_directory = args.save_directory
    num_layers = checker.get_config_from_pretrained(
        model_path, trust_remote_code=args.trust_remote_code
    ).num_hidden_layers

    disable_names = args.disable_names
    if not disable_names:
        disable_names = get_disable_names(num_layers)

    quant_conf = QuantConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        disable_names=disable_names,
        dev_type=args.device_type,
        dev_id=rank,
        act_method=args.act_method,
        w_sym=args.w_sym,
        mm_tensor=False,
        co_sparse=args.co_sparse,
        fraction=args.fraction,
        sigma_factor=args.sigma_factor,
        use_sigma=args.use_sigma,
        is_lowbit=args.is_lowbit,
        do_smooth=args.do_smooth,
        open_outlier=args.open_outlier,
        group_size=args.group_size,
        use_kvcache_quant=args.use_kvcache_quant,
        is_dynamic=args.is_dynamic,
        disable_last_linear=args.disable_last_linear,
    )

    if args.use_fa_quant:
        quant_conf = quant_conf.fa_quant(fa_amp=args.fa_amp)

    anti_outlier_config_val = None
    if args.anti_method == 'm3':
        anti_outlier_config_val = AntiOutlierConfig(
            a_bit=args.a_bit,
            w_bit=args.w_bit,
            anti_method=args.anti_method,
            w_sym=args.w_sym,
            dev_type=args.device_type,
            dev_id=rank,
        )
    elif args.anti_method:
        anti_outlier_config_val = AntiOutlierConfig(
            anti_method=args.anti_method, dev_type=args.device_type, dev_id=rank
        )
    tokenizer_args = parse_tokenizer_args(args.tokenizer_args, default={})
    quantifier = Quantifier(
        model_path,
        quant_conf,
        anti_outlier_config_val,
        device_type=args.device_type,
        tokenizer_args=tokenizer_args,
        model_name=args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    tokenized_calib_data = None

    calib_texts = [
        "Where is the capital of China?",
        "Please make a poem:",
        "I want to learn python, how should I learn it?",
        "Please help me write a job report on large model inference optimization:",
        "What are the most worth visiting scenic spots in China?",
    ]

    if calib_texts is not None:
        tokenized_calib_data = quantifier.get_tokenized_data(
            calib_texts, input_ids_name=args.input_ids_name, attention_mask_name=args.attention_mask_name
        )

    if not os.path.exists(save_directory):
        os.makedirs(save_directory, mode=0o750, exist_ok=True)

    # check dst dir
    save_directory = get_valid_write_path(save_directory, is_dir=True)
    quantifier.convert(tokenized_calib_data, save_directory, args.disable_level, part_file_size=args.part_file_size)

    custom_hooks = {'config.json': functools.partial(modify_config_json, custom_hook=custom_hook)}
    copy_config_files(
        input_path=model_path,
        output_path=save_directory,
        quant_config=quant_conf,
        mindie_format=args.mindie_format,
        custom_hooks=custom_hooks,
    )
    checker.copy_tokenizer_files(model_path, save_directory)
