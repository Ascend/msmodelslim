#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import os
import json
import re
import random
import argparse

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoTokenizer, PreTrainedTokenizerBase
from ascend_utils.common.security import json_safe_load, get_valid_path, get_valid_read_path, get_valid_write_path, \
    SafeWriteUmask, check_type, json_safe_dump, check_dict_character

from msmodelslim.pytorch.llm_ptq.mix_calibration.dataset_processor_base import DatasetProcessorBase
from msmodelslim import logger as msmodelslim_logger

SUPPORTED_DATASET_NAME = ["boolq", "gsm8k", "ceval_5_shot", "mmlu"]
SHOT = 5


class BoolqProcessor(DatasetProcessorBase):
    def __init__(self, dataset_path, tokenizer=None, model=None):
        super().__init__(dataset_path, tokenizer, model)
        self.choice_tokens = []
        self.dataset_path = get_valid_read_path(self.dataset_path)
        with open(self.dataset_path, encoding='utf-8') as f:
            for line in f:
                line_json = json.loads(line)
                self.ori_prompts.append(line_json)

        if self.model is not None:
            sample_yes = "How can we learning machine learning: yes"
            sample_no = "How can we learning machine learning: no"
            self.choice_tokens = [
                self.tokenizer([sample_yes], return_tensors="pt", max_length=2048,
                               add_special_tokens=False).input_ids[0, -1].item(),
                self.tokenizer([sample_no], return_tensors="pt", max_length=2048,
                               add_special_tokens=False).input_ids[0, -1].item()
            ]

    def process_data(self, indexs):
        def build_prompt(title, quest, passage):
            return f"{title} -- {passage}\nQuestion: {quest}?\nAnswer:"

        prmpts_anses = []
        for idx in indexs:
            try:
                sample_data = self.ori_prompts[idx]
                title = sample_data["title"]
                quest = sample_data["question"]
                passage = sample_data["passage"]
                ans = sample_data["answer"]

                prompt = build_prompt(title, quest, passage)
                prmpts_anses.append({"prompt": prompt, "ans": ans})
            except KeyError as e:
                msmodelslim_logger.error(
                    f"sample_data has no key: {self.dataset_name}, please check your dataset."
                )
        return prmpts_anses

    def verify_positive_prompt(self, prompts, labels):
        prpt_ans = []

        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", truncation=True).to(self.model.device)
        answers = self.model(**inputs)

        logits = answers.logits[:, -1, :]
        logits_softmax = F.log_softmax(logits.float(), dim=-1)
        logits_softmax = logits_softmax[:, self.choice_tokens]

        for idx, label in enumerate(labels):
            choice = logits_softmax[idx, 0] > logits_softmax[idx, 1]
            if choice == label:
                prpt_ans.append({"prompt": prompts[idx], "ans": label})

        return prpt_ans


class CEval5ShotProcessor(DatasetProcessorBase):
    def __init__(self, dataset_path, tokenizer=None, model=None):
        super().__init__(dataset_path, tokenizer, model)
        self.choices = ("A", "B", "C", "D")
        self.shot = SHOT
        subject_mapping = self.get_subject_mapping()
        self.ori_prompts = []
        self.ori_answers = []
        required_column = len(self.choices) + 1
        for task_name in tqdm(subject_mapping):
            dev_df, val_df = self.load_csv_by_task_name(task_name, self.dataset_path)
            task_len = val_df.shape[0]
            subject_prompt = []
            if val_df.shape[1] <= required_column:
                # 处理列数不足的情况
                msmodelslim_logger.warning(f"val_df has only {val_df.shape[1]} columns, "
                                            f"but requires at least {required_column+1} columns. "
                                            f"Skipping task {task_name}.")
                continue

            for i in range(task_len):
                subject_prompt.append(self.format_example(val_df, i, include_answer=False))
                self.ori_answers.append(val_df.iloc[i, len(self.choices) + 1])
            train_prompts = [self.gen_prompt(dev_df, task_name, self.shot)] * len(subject_prompt)
            self.ori_prompts.extend([t + p for t, p in zip(train_prompts, subject_prompt)])
        self.ori_prompts = [prompt.encode().decode(encoding="utf8") for prompt in self.ori_prompts]

    def get_subject_mapping(self):
        subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
        subject_mapping_path = get_valid_read_path(subject_mapping_path)
        with open(subject_mapping_path) as f:
            subject_mapping = json.load(f)
        return subject_mapping

    def load_csv_by_task_name(self, task_name, dataset_path):
        dev_file_path = get_valid_path(os.path.join(dataset_path, "dev", task_name + "_dev.csv"))
        dev_df = pd.read_csv(dev_file_path, header=None)[:self.shot + 1]
        val_file_path = get_valid_path(os.path.join(dataset_path, "val", task_name + "_val.csv"))
        val_df = pd.read_csv(val_file_path, header=None)

        dev_df = dev_df.iloc[1:, 1:]
        val_df = val_df.iloc[1:, 1:]
        return dev_df, val_df

    def format_subject(self, subject):
        str_units = subject.split("_")
        res = ""
        for entry in str_units:
            res += " " + entry
        return res

    def format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = len(self.choices)
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = "The following are multiple choice questions about {}.\n\n".format(
            self.format_subject(subject))
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(train_df, i)
        return prompt

    def process_data(self, indexs):
        prmpts_anses = []
        for idx in indexs:
            prmpts_anses.append({"prompt": self.ori_prompts[idx], "ans": self.ori_answers[idx]})
        return prmpts_anses

    def verify_positive_prompt(self, prompts, labels):
        prpt_ans = []
        with torch.no_grad():
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", truncation=True,
                                    return_token_type_ids=False).to(self.model.device)
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=20)

            answers = []
            for idx in range(len(outputs)):
                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                response = self.tokenizer.decode(output)
                answers.append(response)
            answers = [answer.lstrip()[0] if answer.lstrip() else "-1" for answer in answers]

            for ans, label, prmpt in zip(answers, labels, prompts):
                if ans == label:
                    prpt_ans.append({"prompt": prmpt, "ans": ans})

        return prpt_ans


class Gsm8kProcessor(DatasetProcessorBase):
    def __init__(self, dataset_path, tokenizer=None, model=None):
        super().__init__(dataset_path, tokenizer, model)
        self.cot_prompt = ''
        self.dataset_path = get_valid_read_path(self.dataset_path)
        with open(self.dataset_path, encoding='utf-8') as f:
            for line in f:
                line_json = json.loads(line)
                self.ori_prompts.append(line_json)

    def build_prompt(self, question, answer):
        if self.cot_prompt == '':
            question = 'Question: ' + question + '\n'
        else:
            question = self.cot_prompt + '\nQuestion: ' + question + '\n'
        answer = answer.split('####')[1].strip()

        return {"prompt": question, "ans": answer}

    def process_data(self, indexs):
        prmpts_anses = []
        for idx in indexs:
            sample_data = self.ori_prompts[idx]
            question = sample_data["question"]
            answer = sample_data["answer"]
            prmpts_anses.append(self.build_prompt(question, answer))
        return prmpts_anses

    def verify_positive_prompt(self, prompts, labels):
        def process_answers(questions, answers, split=False):
            outputs = []
            for i, answer in enumerate(answers):
                output = answer.strip()[len(questions[i]):]
                if split:
                    output = output.split('\n\n')[0]
                outputs.append(output)
            return outputs

        prpt_ans = []

        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256, eos_token_id=self.tokenizer.eos_token_id,
                                      do_sample=True)

        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers = process_answers(prompts, outputs, split=True)

        for answer_result, label, prompt in zip(answers, labels, prompts):
            response_number = re.findall(r'\d+', answer_result)
            if response_number is not None and len(response_number) > 0:
                last_number = response_number[-1]
            else:
                last_number = ''
            if label == last_number:
                prpt_ans.append({"prompt": prompt, "ans": label})

        return prpt_ans


class MmluProcessor(DatasetProcessorBase):
    def __init__(self, dataset_path, tokenizer=None, model=None):
        super().__init__(dataset_path, tokenizer, model)
        self.shot = SHOT
        self.choices = ("A", "B", "C", "D")

        subject_mapping = self.get_subject_mapping()
        for task_name in tqdm(subject_mapping):
            dev_df, val_df = self.load_csv_by_task_name(task_name, self.dataset_path)
            task_len = val_df.shape[0]
            subject_prompt = []

            for i in range(task_len):
                self.ori_prompts.append(self.format_example(val_df, i, include_answer=False))
                self.ori_answers.append(val_df.iloc[i, len(self.choices) + 1])
            train_prompts = [self.gen_prompt(dev_df, task_name, self.shot)] * len(subject_prompt)
            self.ori_prompts.extend([t + p for t, p in zip(train_prompts, subject_prompt)])
        self.ori_prompts = [prpt.encode().decode(encoding="utf8") for prpt in self.ori_prompts]

    def get_subject_mapping(self):
        subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
        subject_mapping = json_safe_load(subject_mapping_path)
        return subject_mapping["mmlu_all_sets"]

    def load_csv_by_task_name(self, task_name, dataset_path):
        row_begin_idx = 0
        col_begin_idx = 0
        dev_file_path = get_valid_path(os.path.join(dataset_path, "dev", task_name + "_dev.csv"))
        ori_dev_df = pd.read_csv(dev_file_path, header=None)
        val_file_path = get_valid_path(os.path.join(dataset_path, "test",
                                                    f"{task_name}_test.csv"))
        ori_val_df = pd.read_csv(val_file_path, header=None)

        dev_df = ori_dev_df.iloc[row_begin_idx:row_begin_idx + self.shot, col_begin_idx:]
        val_df = ori_val_df.iloc[row_begin_idx:, col_begin_idx:]
        return dev_df, val_df

    def format_subject(self, subject):
        sub_set = subject.split("_")
        res = ""
        for entry in sub_set:
            res += " " + entry
        return res

    def format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = len(self.choices)
        if k + 1 >= len(df.columns):
            raise IndexError(f"Can not get column {k + 1}, please check your dataset.")
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = "The following are multiple choice questions " \
                 "(with answers) about {}.\n\n".format(self.format_subject(subject))
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(train_df, i)
        return prompt

    def process_data(self, indexs):
        prmpts_anses = []
        for idx in indexs:
            prmpts_anses.append({"prompt": self.ori_prompts[idx], "ans": self.ori_answers[idx]})
        return prmpts_anses

    def verify_positive_prompt(self, prompts, labels):
        prpt_ans = []

        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", truncation=True,
                                return_token_type_ids=False).to(self.model.device)
        tokenizer_out_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        outputs = self.model.generate(inputs=tokenizer_out_ids, attention_mask=attention_mask, do_sample=False,
                                      max_new_tokens=20, pad_token_id=self.tokenizer.eos_token_id)

        answers = []
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = self.tokenizer.decode(output)
            answers.append(response)
        answers = [answer.lstrip()[0] if answer.lstrip() else "-1" for answer in answers]

        for ans, label, prompt in zip(answers, labels, prompts):
            if ans == label:
                prpt_ans.append({"prompt": prompt, "ans": label})

        return prpt_ans


class CalibHandler:
    """提供混合校准集通用方法"""

    def __init__(self, dataset_name, dataset_processor, shuffle_seed, batch_size):
        self.dataset_processor = dataset_processor
        self.dataset_name = dataset_name
        self.sample_size = 0
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.positive_prompt_verified = set()
        self.output_dataset = []  # [{"prpt":"", "ans":""}]

    def set_sample_size(self, sample_size):
        self.sample_size = sample_size
        self.positive_prompt_verified = set()
        self.output_dataset = []

    def evaluate(self, need_positive_prompt=False):
        dataset_size = self.dataset_processor.get_dataset_size()
        random.seed(self.shuffle_seed)
        random_idxs = list(range(dataset_size))
        random.shuffle(random_idxs)
        idxs_batches = [random_idxs[i:i + self.batch_size] for i in range(0, len(random_idxs), self.batch_size)]
        sampling_pbar = tqdm(total=self.sample_size, leave=False, desc=f"{self.dataset_name} sampling progress")

        for idx_batch in idxs_batches:
            if len(self.output_dataset) >= self.sample_size:
                self.output_dataset = self.output_dataset[:self.sample_size]
                return

            prmpts_anses = self.dataset_processor.process_data(idx_batch)
            if not need_positive_prompt:
                self.output_dataset.extend(prmpts_anses)
                sampling_pbar.update(len(prmpts_anses))
            else:
                prompts = [d["prompt"] for d in prmpts_anses if "prompt" in d]
                labels = [d['ans'] for d in prmpts_anses if "ans" in d]
                prmpts_gts = self.dataset_processor.verify_positive_prompt(prompts, labels)
                self.output_dataset.extend(prmpts_gts)
                sampling_pbar.update(len(prmpts_gts))

        msmodelslim_logger.warning(
            f"Dataset {self.dataset_name} : available samples is less than the number of samples.")
        return

    def run(self, need_positive_prompt=False):
        self.evaluate(need_positive_prompt)
        return self.output_dataset


class DatasetFactory:
    """Dataset工厂类，返回对应数据集的processor"""

    def __init__(self):
        pass

    @staticmethod
    def create_dataset_processor(dataset_name, dataset_path, customized_dataset_processor, tokenizer=None, model=None):
        if dataset_name == "boolq":
            processor = BoolqProcessor(dataset_path, tokenizer, model)
        elif dataset_name == "ceval_5_shot":
            processor = CEval5ShotProcessor(dataset_path, tokenizer, model)
        elif dataset_name == "gsm8k":
            processor = Gsm8kProcessor(dataset_path, tokenizer, model)
        elif dataset_name == "mmlu":
            processor = MmluProcessor(dataset_path, tokenizer, model)
        elif dataset_name in customized_dataset_processor.keys():
            processor = customized_dataset_processor[dataset_name]
        else:
            msmodelslim_logger.warning(f"Dataset {dataset_name} hasn't provide processor")
            processor = None

        return processor


class CalibrationData(object):
    """Calibration data for get Calibration Set for Anti-outlier and Calibrator"""

    def __init__(
            self,
            config_path="",
            save_path="",
            tokenizer=None,
            model: nn.Module = None,
    ):
        self.logger = msmodelslim_logger
        self.shuffle_seed = 0

        self.model = model
        if self.model is not None and not self.verify_model():
            raise TypeError("Invalid model")
        self.tokenizer = tokenizer
        if tokenizer is not None:
            check_type(tokenizer, PreTrainedTokenizerBase, param_name="tokenizer")
        if self.model is not None and not self.verify_tokenizer():
            raise TypeError("Tokenizer must be matched with model.")

        self.batch_size = 4
        self.handlers = {}
        self.custormized_dataset_processor = {}

        self.read_config(config_path)
        self.mixed_dataset = []
        self.save_path = None
        check_type(save_path, str, "save_path")
        if len(save_path) > 0:
            self.save_path = get_valid_write_path(path=save_path, extensions="json", is_dir=False)

    def verify_model(self):
        if not isinstance(self.model, PreTrainedModel):
            self.logger.error(f"Model is not a valid model, type is {type(self.model)}")
            return False
        return True

    def verify_tokenizer(self):
        text = "hello world"
        inputs = (self.tokenizer(text, return_tensors="pt",
                                 return_token_type_ids=False).to(self.model.device))
        try:
            with torch.no_grad():
                self.model(**inputs)
        except Exception as e:
            self.logger.error(f"{e}")
            self.logger.error("Model and tokenizer are not compatible. Cannot simply use model API.")
            self.logger.error(f"Model type is {type(self.model)}")
            self.logger.error(f"Tokenizer type is {type(self.tokenizer)}")
            return False
        try:
            with torch.no_grad():
                self.model.generate(**inputs, do_sample=False, max_new_tokens=1)
        except Exception as e:
            self.logger.error(f"{e}")
            self.logger.error("Model and tokenizer are not compatible. Cannot simply use generate API.")
            self.logger.error(f"Model type is {type(self.model)}")
            self.logger.error(f"Tokenizer type is {type(self.tokenizer)}")
            return False
        self.logger.info("Model and Tokenizer are compatible.")
        return True

    def read_config(self, config_path):
        config_path = get_valid_read_path(config_path, extensions=[".json", ".jsonl"])
        with open(config_path, "r") as f:
            config = json.load(f)
        if isinstance(config, dict):
            check_dict_character(config, 512)
        if "configurations" not in config.keys():
            raise ValueError(f"config should have attribute configurations")
        dataset_cfg = config["configurations"]

        for cfg in dataset_cfg:
            if "dataset_name" not in cfg.keys():
                raise ValueError(f"config should have attribute dataset_name")
            if "dataset_path" not in cfg.keys():
                raise ValueError(f"config should have attribute dataset_path")

            dataset_name = cfg["dataset_name"]
            path = cfg["dataset_path"]
            if dataset_name not in SUPPORTED_DATASET_NAME:
                self.logger.warning(f"Dataset is customized. Please provide the processor.")
            processor = DatasetFactory.create_dataset_processor(dataset_name, path, self.custormized_dataset_processor,
                                                                self.tokenizer, self.model)
            if processor is not None:
                self.handlers[dataset_name] = CalibHandler(dataset_name, processor, self.shuffle_seed, self.batch_size)

    def add_custormized_dataset_processor(self, dataset_name, processor):
        """添加自定义dataset_processor"""
        if not isinstance(dataset_name, str):
            raise argparse.ArgumentTypeError("%r is not an str." % dataset_name)
        if not isinstance(processor, DatasetProcessorBase):
            raise argparse.ArgumentTypeError("%r is not an DatasetProcessorBase." % processor)
        self.custormized_dataset_processor[dataset_name] = processor
        self.handlers[dataset_name] = CalibHandler(dataset_name, processor, self.shuffle_seed, self.batch_size)

    def process(self):
        """执行混合校准集"""
        self.mixed_dataset = []
        for _, handler in self.handlers.items():
            self.mixed_dataset.extend(handler.run(self.model is not None))

        random.seed(self.shuffle_seed)
        random.shuffle(self.mixed_dataset)

        if self.save_path:
            with SafeWriteUmask(umask=0o377):
                with open(self.save_path, 'w') as file:
                    json.dump(self.mixed_dataset, file, indent=2)
                    msmodelslim_logger.info(f"Save mixed dataset success, save path:{self.save_path}")
        return self.mixed_dataset

    def set_sample_size(self, sample_size: dict):
        """设置每个数据集的采样数量"""
        if not isinstance(sample_size, dict):
            raise TypeError("sample_size should be dict")
        for dataset_name, size in sample_size.items():
            if not isinstance(dataset_name, str):
                raise TypeError("sample_size keys should be string")
            if not isinstance(size, int):
                raise TypeError("sample_size values should be int")
            if dataset_name not in self.handlers.keys():
                self.logger.error(f"Dataset {dataset_name} has no handler")
                return
            self.handlers[dataset_name].set_sample_size(size)

    def set_batch_size(self, batch_size: int):
        """设置batch_size"""
        if not isinstance(batch_size, int):
            raise argparse.ArgumentTypeError("%r is not an int." % batch_size)
        if batch_size == 0:
            raise ValueError("batch_size cant be zero")
        self.batch_size = batch_size

    def set_shuffle_seed(self, shuffle_seed):
        """设置随机种子"""
        if not isinstance(shuffle_seed, int):
            raise argparse.ArgumentTypeError("%r is not an int." % shuffle_seed)
        self.shuffle_seed = shuffle_seed