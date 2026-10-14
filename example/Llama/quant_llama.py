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
# pylint: disable=logging-fstring-interpolation,no-member,possibly-used-before-assignment,redefined-outer-name,reimported,unnecessary-list-index-lookup,unspecified-encoding

import os
import json
import sys
import torch
import torch.nn.functional as F

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', ".."))
sys.path.append(parent_directory)

from example.common.security.path import get_valid_read_path, get_write_directory  # noqa: E402
from example.common.security.path import get_valid_write_path, get_valid_read_path, json_safe_load, get_write_directory  # noqa: E402, F401, F811
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
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.layer_select import LayerSelector  # noqa: E402
from msmodelslim import logger  # noqa: E402

CPU = "cpu"
NPU = "npu"


def get_down_proj_disable_names(num_layers: int) -> list:
    disable_names = []
    # 遍历层数并添加对应的 disable_names
    for i in range(num_layers):
        disable_names.append(f"model.layers.{i}.mlp.down_proj")
    return disable_names


def get_llama3_1_disable_names(num_layers: int) -> list:
    disable_names = []
    for i in range(num_layers):
        disable_names.append(f"model.layers.{i}.mlp.down_proj")
    # llama 3.1 70b 推荐回退所有down层和前5层下述各层
    for i in range(5):
        disable_names.append(f"model.layers.{i}.self_attn.q_proj")
        disable_names.append(f"model.layers.{i}.self_attn.k_proj")
        disable_names.append(f"model.layers.{i}.self_attn.v_proj")
        disable_names.append(f"model.layers.{i}.self_attn.o_proj")
        disable_names.append(f"model.layers.{i}.mlp.gate_proj")
        disable_names.append(f"model.layers.{i}.mlp.up_proj")
    disable_names.append("lm_head")
    return disable_names


def get_llama3_disable_names(num_layers: int) -> list:
    disable_names = []
    # llama 3 70b 推荐回退前5层下述各层
    for i in range(5):
        disable_names.append(f"model.layers.{i}.mlp.down_proj")
        disable_names.append(f"model.layers.{i}.self_attn.q_proj")
        disable_names.append(f"model.layers.{i}.self_attn.k_proj")
        disable_names.append(f"model.layers.{i}.self_attn.v_proj")
        disable_names.append(f"model.layers.{i}.self_attn.o_proj")
        disable_names.append(f"model.layers.{i}.mlp.gate_proj")
        disable_names.append(f"model.layers.{i}.mlp.up_proj")
    disable_names.append("lm_head")
    return disable_names


def get_padding_data(tokenizer, calib_list, device_type):
    calib_dataset = []
    max_len = 0
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt', add_special_tokens=False)
        calib_dataset.append(inputs.data['input_ids'].to(device_type))
        max_len = max(max_len, inputs.data['input_ids'].size(1))
    new_calib_dataset = []
    for inputs in calib_dataset:
        new_inputs = F.pad(inputs, (0, max_len - inputs.size(1)), value=0)
        new_calib_dataset.append(new_inputs)
    return [torch.cat(new_calib_dataset)]


def get_batch_tokenized_data(tokenizer, input_texts, device_type, batch_size=4):
    batch_ant_calib_texts = [input_texts[i : i + batch_size] for i in range(0, len(input_texts), batch_size)]
    tokenized_ant_calib_data = []
    for prompt in batch_ant_calib_texts:
        tmp = get_padding_data(tokenizer, prompt, device_type)
        tokenized_ant_calib_data.append(tmp)
    return tokenized_ant_calib_data


def auto_layer_select(model, disable_names, disable_threshold, select_layer_data):
    layer_selector = LayerSelector(model=model, layer_names=disable_names)
    layer_selector.run(select_layer_data)
    return layer_selector.select_layers_by_threshold(disable_threshold)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path")
    parser.add_argument('--save_directory', type=str)
    parser.add_argument('--part_file_size', type=int, default=None)
    parser.add_argument('--calib_texts', type=str, nargs='+', default=None)
    parser.add_argument(
        '--calib_file',
        type=str,
        help='A jsonl file contains calibration data.',
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'common', 'teacher_qualification.jsonl'),
    )
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--disable_names', type=str, nargs='+', default=None)
    parser.add_argument('--device_type', type=str, choices=[CPU, NPU], default=CPU)
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
    parser.add_argument('--tp_size', type=int, default=1)
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
        default='{"padding_side":"left","pad_token":"<unk>"}',
        validator=StringArgumentValidator(min_length=2, max_length=MAX_JSON_LENGTH),
    )
    parser.add_argument('--disable_last_linear', type=cmd_bool, default=True)
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        validator=StringArgumentValidator(min_length=1, max_length=MAX_KEY_LENGTH, allow_none=True),
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='llama2',
        choices=['llama', 'llama2', 'llama3', 'llama3.1_bf', 'llama3.1_fp', 'llama3.1_instruct'],
        help='Specify the type of llama model \
                            (choices: llama, llama2, llama3, llama3.1_bf, llama3.1_fp, llama3.1_instruct)',
    )
    parser.add_argument(
        '--anti_calib_file', type=str, default=None, help='Path to anti-calibration data file (.json or .jsonl)'
    )
    parser.add_argument(
        '--disable_threshold', type=float, default=0, help='Disable threshold when auto select disable names'
    )
    parser.add_argument('--pdmix', type=cmd_bool, default=False, help='use pdmix quantization type')
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=False)
    parser.add_argument(
        '--mindie_format',
        action="store_true",
        help="Compatible with quantization formats \
                        supported by before 2.1.RC1 version of MindIE",
    )
    return parser.parse_args()


class Quantifier:
    def __init__(self, model_path_or_name, args, anti_outlier_config=None, device_type='cpu', rank=0, **kwargs):
        self.args = args
        self.rank = rank
        safe_generator = SafeGenerator()
        self.device_type = device_type
        device_map = CPU if self.device_type == CPU else "auto"
        self.trust_remote_code = args.trust_remote_code

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
            device_map=device_map,
            trust_remote_code=self.trust_remote_code,
        )
        tokenizer_args = kwargs.get("tokenizer_args", {})
        self.tokenizer = safe_generator.get_tokenizer_from_pretrained(
            self.model_path_or_name,
            use_fast=False,
            trust_remote_code=self.trust_remote_code,
            legacy=False,
            **tokenizer_args,
        )
        self.model_name = kwargs.get("model_name", None)
        self.quant_config = None

    def create_quant_config(self, num_layers, select_layer_data=None):
        args = self.args
        # Check if disable_names is provided, if not and a_bit is 8, generate disable_names
        disable_names = args.disable_names
        if not disable_names and args.a_bit == 8:
            if args.disable_threshold > 0:
                disable_names = get_down_proj_disable_names(num_layers)
            elif args.model_type == 'llama3':
                disable_names = get_llama3_disable_names(num_layers)
            elif args.model_type == 'llama3.1_fp':
                disable_names = get_llama3_1_disable_names(num_layers)
            elif args.model_type == 'llama3.1_instruct':
                disable_names = get_llama3_1_disable_names(num_layers)
            else:
                disable_names = get_down_proj_disable_names(num_layers)
        if args.disable_threshold > 0:
            disable_names = auto_layer_select(self.model, disable_names, args.disable_threshold, select_layer_data)

        quant_config = QuantConfig(
            w_bit=args.w_bit,
            a_bit=args.a_bit,
            disable_names=disable_names,
            dev_type=args.device_type,
            dev_id=self.rank,
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
            pdmix=args.pdmix,
        )

        if args.use_fa_quant:
            quant_config = quant_config.fa_quant(fa_amp=args.fa_amp)
        self.quant_config = quant_config

    def get_batch_tokenized_data(self, input_texts, batch_size=4):
        return get_batch_tokenized_data(self.tokenizer, input_texts, self.device_type, batch_size)

    def get_tokenized_data(self, input_texts, input_ids_name='input_ids', attention_mask_name='attention_mask'):
        tokenized_data = []
        for input_text in input_texts:
            inputs = self.tokenizer(input_text, return_tensors='pt', padding=True).to(self.device_type)
            tokenized_data.append([inputs.data[input_ids_name], inputs.data[attention_mask_name]])
        return tokenized_data

    def convert(self, tokenized_data, save_path, disable_level, part_file_size=None, tokenized_ant_calib_data=None):
        if self.device_type == NPU:
            # 避免在线编译算子，使用二进制编译的算子
            torch.npu.set_compile_mode(jit_compile=False)
        if tokenized_ant_calib_data is None:
            tokenized_ant_calib_data = tokenized_data

        if self.anti_outlier_config is not None:
            if self.model_name == "baichuan":
                anti_outlier = AntiOutlier(
                    self.model,
                    calib_data=tokenized_ant_calib_data,
                    cfg=self.anti_outlier_config,
                    norm_class_name="RMSNorm",
                )
            else:
                anti_outlier = AntiOutlier(
                    self.model, calib_data=tokenized_ant_calib_data, cfg=self.anti_outlier_config
                )
            anti_outlier.process()

        if not os.path.exists(save_path):
            os.mkdir(save_path, mode=0o750)

        calibrator = Calibrator(self.model, self.quant_config, calib_data=tokenized_data, disable_level=disable_level)
        calibrator.run()
        save_type = "safe_tensor" if args.mindie_format else "ascendV1"
        calibrator.save(save_path, save_type=[save_type], part_file_size=part_file_size)


def get_anti_dataset(tokenizer, calib_list, device_type):
    calib_dataset = []
    max_len = 0
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append(inputs.data['input_ids'].to(device_type))
        max_len = max(max_len, inputs.data['input_ids'].size(1))
    for i, cur_calib in enumerate(calib_dataset):
        calib_dataset[i] = F.pad(cur_calib, (0, max_len - cur_calib.size(1)), value=0)
    return torch.cat(calib_dataset)


if __name__ == '__main__':
    args = parse_arguments()
    checker = SafeGenerator()

    try:
        rank: int = int(os.getenv("RANK", "0"))
    except ValueError as e:
        logger.warning(f"Error converting 'RANK' environment variable to integer: {e}")
        logger.info("Defaulting to 0.")
        rank: int = 0

    model_path = get_valid_read_path(args.model_path, is_dir=True, check_user_stat=True)
    save_directory = get_write_directory(args.save_directory, write_mode=0o750)

    num_layers = checker.get_config_from_pretrained(
        model_path, trust_remote_code=args.trust_remote_code
    ).num_hidden_layers

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
    elif args.anti_method == 'm6':
        keys = ['.o_proj']
        anti_disable_names = ["model.layers.{}.self_attn.o_proj".format(i) for i in range(num_layers)]
        anti_outlier_config_val = AntiOutlierConfig(
            anti_method=args.anti_method,
            dev_type=args.device_type,
            disable_anti_names=anti_disable_names,
            flex_config={},
        )
    elif args.anti_method:
        anti_outlier_config_val = AntiOutlierConfig(anti_method=args.anti_method, dev_type=args.device_type)
    tokenizer_args = parse_tokenizer_args(args.tokenizer_args, default={"padding_side": "left", "pad_token": "<unk>"})  # nosec B105
    quantifier = Quantifier(
        model_path,
        args,
        anti_outlier_config_val,
        device_type=args.device_type,
        rank=rank,
        tokenizer_args=tokenizer_args,
        model_name=args.model_name,
    )
    tokenized_calib_data = []
    calib_file = args.calib_file
    if calib_file:
        calib_file = get_valid_read_path(calib_file)
        calib_texts = checker.load_jsonl(calib_file)
    else:
        calib_texts = args.calib_texts

    if calib_texts is not None:
        tokenized_calib_data = quantifier.get_tokenized_data(
            calib_texts, input_ids_name=args.input_ids_name, attention_mask_name=args.attention_mask_name
        )

    tokenized_ant_calib_data = tokenized_calib_data
    if args.anti_calib_file:
        if args.model_type == "llama3.1_instruct":
            anti_calib_file_path = get_valid_read_path(args.anti_calib_file, "json", is_dir=False)
            with open(anti_calib_file_path, "r") as f:
                anti_prompt = json.load(f)
            anti_data = []
            for i, _ in enumerate(anti_prompt):
                tmp = get_anti_dataset(quantifier.tokenizer, anti_prompt[i], args.device_type)
                anti_data.append(tmp)

            tokenized_ant_calib_data = []
            for data in anti_data:
                tokenized_ant_calib_data.append([data])
        else:
            args.anti_calib_file = get_valid_read_path(args.anti_calib_file)
            ant_calib_texts = checker.load_jsonl(args.anti_calib_file)
            if ant_calib_texts is not None:
                tokenized_ant_calib_data = quantifier.get_batch_tokenized_data(ant_calib_texts)

    if isinstance(args.disable_threshold, float) and args.disable_threshold > 0:
        quantifier.create_quant_config(num_layers, tokenized_ant_calib_data)
    elif args.disable_threshold == 0:
        quantifier.create_quant_config(num_layers)
    else:
        raise ValueError("disable_threshold should be a float number >= 0")

    # 为适配工具稀疏量化传入w_bit=4,a_bit=8暂时修改quant_type
    quantifier.convert(
        tokenized_calib_data,
        save_directory,
        args.disable_level,
        part_file_size=args.part_file_size,
        tokenized_ant_calib_data=tokenized_ant_calib_data,
    )

    # 通过 model_quant_type 获得 quant_type
    quant_type = quantifier.quant_config.model_quant_type.lower()

    auto_config = checker.get_config_from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    checker.modify_config(model_path, save_directory, auto_config.torch_dtype, quant_type, args)
    checker.copy_tokenizer_files(model_path, save_directory)
