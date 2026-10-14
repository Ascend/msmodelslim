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
from tqdm import tqdm
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModel


from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator as SparseQuantCalibrator
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig as SparseQuantConfig


def parse_args():
    parser = argparse.ArgumentParser(description="generate Model weights for quant.")
    parser.add_argument(
        "--fp16_path",
        default=f"{os.environ['PROJECT_PATH']}/resource/sparse/chatglm2",
        help="Location of Model weights, which contains pytorch_model-00001-of-00007.bin and others",
    )
    parser.add_argument(
        "--data_path",
        default=f"{os.environ['PROJECT_PATH']}/resource/CEval",
        help="Location to read and write the quant weights",
    )
    parser.add_argument(
        "--save_path",
        default=f"{os.environ['PROJECT_PATH']}/output/sparse",
        help="Location to read and write the quant weights",
    )
    parsed = parser.parse_args()
    return parsed


args = parse_args()
data_path = args.data_path
fp16_path = args.fp16_path
save_path = args.save_path
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=fp16_path, trust_remote_code=True, local_files_only=True
)

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path=fp16_path, torch_dtype=torch.float32, trust_remote_code=True, local_files_only=True
)

choices = ["A", "B", "C", "D"]
choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]


def build_prompt(text):
    return "[Round {}]\n\n问：{}\n\n答：".format(1, text)


extraction_prompt = '综上所述，ABCD中正确的选项是：'


def get_dataset(bs=1):
    with torch.no_grad():
        dataset_all = []
        entry = data_path + "/val/Other/civil_servant.jsonl"
        dataset_cur = []
        dataset = []
        with open(entry, encoding='utf-8') as file:
            for line in file:
                dataset.append(json.loads(line))
        dataset = dataset[:1]
        correct = 0  # noqa: F841
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs)
        for batch in tqdm(dataloader):
            texts = batch["inputs_pretokenized"]
            queries = [build_prompt(query) for query in texts]
            inputs = tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cpu')

            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
            intermediate_outputs = []
            for idx in range(len(outputs)):
                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]) :]
                response = tokenizer.decode(output)
                print('response: ', response)
                intermediate_outputs.append(response)
            answer_texts = [
                text + intermediate + "\n" + extraction_prompt
                for text, intermediate in zip(texts, intermediate_outputs)
            ]
            input_tokens = [build_prompt(answer_text) for answer_text in answer_texts]
            inputs = tokenizer(input_tokens, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(
                'cpu'
            )
            dataset_tmp = [
                inputs.data['input_ids'],
                inputs.data['position_ids'],
                inputs.data['attention_mask'],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                True,
            ]
            dataset_cur.append(dataset_tmp)
        dataset_all.extend(dataset_cur)
    return dataset_all


dataset_calib = get_dataset()
print('len of calib_dataset: ', len(dataset_calib))

w_bit = 4
fraction = 0.011
powerquant = False
mm_tensor = False

quant_config = SparseQuantConfig(
    w_bit=w_bit,
    disable_names=[
        'transformer.encoder.layers.0.self_attention.query_key_value',
        'transformer.encoder.layers.0.self_attention.dense',
        'transformer.encoder.layers.0.mlp.dense_h_to_4h',
        'transformer.encoder.layers.0.mlp.dense_4h_to_h',
        'transformer.output_layer',
    ],
    dev_type='cpu',
    act_method=3,
    pr=2.0,
    fraction=fraction,
    nonuniform=powerquant,
    mm_tensor=mm_tensor,
    co_sparse=True,
)

calibrator = SparseQuantCalibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')  # 内部回退两层
calibrator.run(int_infer=False)
calibrator.save(
    save_path
)  # 存储量化参数用于部署，在存储量化参数过程中，存在反序列化风险，已通过将保存的量化结果文件夹权限设置为750，将量化结果文件权限设置为400来消减该风险
