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
# pylint: disable=anomalous-backslash-in-string,logging-format-interpolation,logging-fstring-interpolation,logging-not-lazy,no-name-in-module,undefined-variable,unnecessary-comprehension,unnecessary-negation,unspecified-encoding

import glob
import json
import os
import stat
from pathlib import Path
import math
import re
import collections

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import pandas as pd
import numpy as np
from transformers import PreTrainedModel, AutoTokenizer  # noqa: F401
from tqdm import tqdm

from security import json_safe_load, json_safe_dump, get_valid_path, get_valid_write_path, get_valid_read_path  # noqa: F401
from precision_tool import logger
from precision_tool import truthfulqa_eval
from ascend_utils.common.security.type import check_type


supported_dataset = ["boolq", "ceval_0_shot", "ceval_5_shot", "humaneval", "truthfulqa", "mmlu"]

supported_hardware = ["npu"]

TENSOR_TYPE_PYTORCH = "pt"
DATASET_HUMAN_EVAL = "humaneval"

CalcParam = collections.namedtuple(
    'CalcParam', ['tag', 'frame', 'idx', 'scores_true', 'scores_false', 'ref_true', 'ref_best']
)


def is_running_on_npu(device_name):
    return device_name.lower() == "npu"


class PrecisionTest:
    def __init__(self, model, tokenizer, dataset, batch_size, hardware_type, tokenizer_return_type_id=False, shot=5):
        """
        @param model:
            llm to run the test, should be an instance of transformers.PreTrainedModel
        @param dataset:
            dataset to test precision
        @param batch_size:
            batch_size to run inference
        @param hardware_type:
            currently only npu is supported
        @param tokenizer_return_type_id:
            tokenizer return token type id
        @param shot
            shot to test precision
        """
        self.logger = logger
        self.logger.info("Ready to init precision tool.")
        self.dataset = dataset
        if not self.__verify_dataset():
            raise TypeError("Dataset setting failed.")
        self.model = model
        if not self.__verify_model():
            raise TypeError("Invalid model.")
        self.device = model.device
        self.tokenizer_return_type_id = tokenizer_return_type_id
        check_type(self.tokenizer_return_type_id, bool, param_name="tokenizer_return_type_id")
        self.tokenizer = tokenizer
        if not self.__verify_tokenizer():
            raise TypeError("Tokenizer must be matched with model.")
        self.batch_size = batch_size
        if not self.__verify_batch_size():
            raise TypeError("Batch size must be a positive integer.")
        self.shot = shot
        if not self.__verify_shot():
            raise TypeError("shot must be a nonnegative integer.")
        self.hardware_type = hardware_type
        if self.hardware_type not in supported_hardware:
            raise TypeError(f"Hardware type {self.hardware_type} is not supported.")
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(self.script_path, "dataset", self.dataset)
        if not os.path.exists(self.dataset_path):
            raise EnvironmentError(f"Dataset was not found, valid path should be '{self.dataset_path}")
        self.dataset_path = get_valid_path(self.dataset_path)
        self.result_file = ""
        self.logger.info("Precision test was inited.")

    @staticmethod
    def __postprocess(text: str, options: str, cushion=True) -> str:
        patterns = [
            f'答案是?\s?([{options}])',
            f'答案是?\s?：([{options}])',
            f'答案是?\s?:([{options}])',
            f'答案应该?是\s?([{options}])',
            f'答案应该?选\s?([{options}])',
            f'答案为\s?([{options}])',
            f'答案选\s?([{options}])',
            f'选择?\s?([{options}])',
            f'故选?\s?([{options}])只有选?项?\s?([{options}])\s?是?对',
            f'只有选?项?\s?([{options}])\s?是?错',
            f'只有选?项?\s?([{options}])\s?不?正确',
            f'只有选?项?\s?([{options}])\s?错误',
            f'说法不?对选?项?的?是\s?([{options}])',
            f'说法不?正确选?项?的?是\s?([{options}])',
            f'说法错误选?项?的?是\s?([{options}])',
            f'([{options}])\s?是正确的',
            f'([{options}])\s?是正确答案',
            f'选项\s?([{options}])\s?正确',
            f'所以答\s?([{options}])',
            f'所以\s?([{options}][.。$]?$)',
            f'所有\s?([{options}][.。$]?$)',
            f'[\s，：:,]([{options}])[。，,\.]?$',
            f'[\s，,：:][故即]([{options}])[。\.]?$',
            f'[\s，,：:]因此([{options}])[。\.]?$',
            f'[是为。]\s?([{options}])[。\.]?$',
            f'因此\s?([{options}])[。\.]?$',
            f'显然\s?([{options}])[。\.]?$',
            '答案是\s?(\S+)(?:。|$)',
            '答案应该是\s?(\S+)(?:。|$)',
            '答案为\s?(\S+)(?:。|$)',
            f'[Tt]he answer is \(?([{options}])\)?',
            f'[Tt]he answer is option \(?([{options}])\)?',
            f'[Tt]he correct answer is \(?([{options}])\)?',
            f'[Tt]he correct answer is option \(?([{options}])\)?',
            f'[Tt]he answer to the question is \(?([{options}])\)?',
            f'^选项\s?([{options}])',
            f'^([{options}])\s?选?项',
            f'(\s|^)[{options}][\s。，,：:\.$]',
            f'(\s|^)[{options}](\s|$)',
            '1.\s?(.*?)$',
            f'1.\s?([{options}])[.。$]?$',
        ]
        cushion_patterns = [
            f'([{options}]):',
            f'[{options}]',
        ]

        if cushion:
            patterns.extend(cushion_patterns)
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
        return ''

    def test(self):
        self.logger.info("Begin to run precision test.")
        torch.manual_seed(1)
        self.__run_test()
        self.logger.info("Precision test has finished.")

    def __verify_batch_size(self):
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            self.logger.error("Batch size must be an integer")
            return False
        if not self.batch_size > 0:
            self.logger.error("Batch size must be positive")
            return False
        return True

    def __verify_dataset(self):
        if not isinstance(self.dataset, str):
            self.logger.error("Dataset should be a string")
            return False
        if self.dataset not in supported_dataset:
            self.logger.error("Dataset %r is not supported. Use %s instead." % (self.dataset, supported_dataset))
            return False
        if self.dataset == DATASET_HUMAN_EVAL:
            try:
                from human_eval.evaluation import evaluate_functional_correctness  # noqa: F401
            except Exception as e:
                self.logger.error(f"Import human_eval test package failed. Error info is as below. \n {e}")
                return False
        return True

    def __verify_model(self):
        if not isinstance(self.model, PreTrainedModel):
            self.logger.error(f"Model is not a valid model, type is {type(self.model)}")
            return False
        return True

    def __verify_tokenizer(self):
        text = "hello world"
        inputs = self.tokenizer(
            text, return_tensors=TENSOR_TYPE_PYTORCH, return_token_type_ids=self.tokenizer_return_type_id
        ).to(self.model.device)
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

    def __verify_shot(self):
        if not isinstance(self.shot, int) or isinstance(self.shot, bool):
            self.logger.error("shot must be an integer, can not be {}.".format(type(self.shot).__name__))
            return False
        if not self.shot >= 0:
            self.logger.error("shot must be nonnegative integer, can not be {}.".format(self.shot))
            return False
        return True

    def __run_test(self):
        if self.dataset == 'ceval_0_shot':
            self.__run_full_dataset_ceval_0_shot()
        elif self.dataset == 'ceval_5_shot':
            self.__run_full_dataset_ceval_5_shot()
        elif self.dataset == 'boolq':
            self.__run_full_dataset_boolq()
        elif self.dataset == 'humaneval':
            self.__run_full_dataset_humaneval()
        elif self.dataset == 'truthfulqa':
            self.__run_full_dataset_truthfulqa()
        elif self.dataset == 'mmlu':
            self.__run_full_dataset_mmlu()
        else:
            raise TypeError("Unsupported dataset")

    def __run_full_dataset_ceval_0_shot(self):
        choices = ["A", "B", "C", "D"]
        choice_tokens = [self.tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]

        extraction_prompt = '综上所述，ABCD中正确的选项是：'

        def build_prompt(text):
            return "[Round {}]\n\n问：{}\n\n答：".format(1, text)

        def run_test():
            correct_total, sum_total = 0, 0
            for entry in glob.glob((Path(self.dataset_path) / "val/**/*.jsonl").as_posix(), recursive=True):
                correct, dataset = 0, []
                entry = get_valid_read_path(entry)
                with open(entry, encoding='utf-8') as file:
                    for line in file:
                        single_json = json.loads(line)
                        dataset.append(single_json)

                dataset_num = len(dataset)

                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for batch in tqdm(dataloader):
                    texts = batch["inputs_pretokenized"]
                    queries = [build_prompt(query) for query in texts]
                    inputs = self.tokenizer(
                        queries,
                        padding=True,
                        return_tensors=TENSOR_TYPE_PYTORCH,
                        truncation=True,
                        return_token_type_ids=self.tokenizer_return_type_id,
                    ).to(self.device)
                    outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=512)
                    intermediate_outputs = []
                    for idx in range(len(outputs)):
                        output = outputs.tolist()[idx][len(inputs["input_ids"][idx]) :]
                        response = self.tokenizer.decode(output)
                        intermediate_outputs.append(response)
                    answer_texts = []
                    for text, intermediate in zip(texts, intermediate_outputs):
                        answer_texts.append(text + intermediate + "\n" + extraction_prompt)
                    input_tokens = []
                    for answer_text in answer_texts:
                        input_tokens.append(build_prompt(answer_text))
                    inputs = self.tokenizer(
                        input_tokens,
                        padding=True,
                        return_tensors=TENSOR_TYPE_PYTORCH,
                        truncation=True,
                        return_token_type_ids=self.tokenizer_return_type_id,
                    ).to(self.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :]
                    logits = logits[:, choice_tokens]
                    preds = logits.argmax(dim=-1)
                    correct += (preds.cpu() == batch["label"]).sum().item()
                correct_total += correct
                sum_total += dataset_num
            return correct_total, sum_total

        with torch.no_grad():
            correct_total, sum_total = run_test()

        if sum_total == 0:
            self.logger.error("Did not ran any test, maybe wrong cevel_0_shot dataset folder.")
        else:
            self.logger.info(
                f"correct rate:{correct_total / sum_total}, correct total:{correct_total}, sum_total:{sum_total}"
            )

    def __run_full_dataset_ceval_5_shot(self):
        choices = ["A", "B", "C", "D"]
        shot = 5

        def get_subject_mapping():
            subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
            subject_mapping_path = get_valid_read_path(subject_mapping_path)
            with open(subject_mapping_path) as f:
                subject_mapping = json.load(f)
            return subject_mapping

        def load_csv_by_task_name(task_name, dataset_path):
            dev_file_path = get_valid_path(os.path.join(dataset_path, "dev", task_name + "_dev.csv"))
            dev_df = pd.read_csv(dev_file_path, header=None)[: shot + 1]
            val_file_path = get_valid_path(os.path.join(dataset_path, "val", task_name + "_val.csv"))
            val_df = pd.read_csv(val_file_path, header=None)

            dev_df = dev_df.iloc[1:, 1:]
            val_df = val_df.iloc[1:, 1:]
            return dev_df, val_df

        def format_subject(subject):
            str_units = subject.split("_")
            res = ""
            for entry in str_units:
                res += " " + entry
            return res

        def format_example(df, idx, include_answer=True):
            prompt = df.iloc[idx, 0]
            k = len(choices)
            for j in range(k):
                prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
            prompt += "\nAnswer:"
            if include_answer:
                prompt += " {}\n\n".format(df.iloc[idx, k + 1])
            return prompt

        def gen_prompt(train_df, subject, k=-1):
            prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
                format_subject(subject)
            )
            if k == -1:
                k = train_df.shape[0]
            for i in range(k):
                prompt += format_example(train_df, i)
            return prompt

        correct_total = 0
        sum_total = 0

        subject_mapping = get_subject_mapping()
        index = 1
        for task_name in tqdm(subject_mapping):
            dev_df, val_df = load_csv_by_task_name(task_name, self.dataset_path)
            correct = 0
            task_len = val_df.shape[0]
            for i in range(math.ceil(task_len / self.batch_size)):
                q_num = self.batch_size if (i + 1) * self.batch_size <= task_len else task_len - i * self.batch_size
                prompt_ends = []
                for j in range(q_num):
                    prompt_ends.append(format_example(val_df, i * self.batch_size + j, include_answer=False))
                train_prompts = [gen_prompt(dev_df, task_name, shot)] * q_num
                prompt = [t + p for t, p in zip(train_prompts, prompt_ends)]
                labels = [val_df.iloc[i * self.batch_size + j, val_df.shape[1] - 1] for j in range(q_num)]
                prompts = [prpt.encode().decode(encoding="utf8") for prpt in prompt]
                inputs = self.tokenizer(
                    prompts,
                    padding=True,
                    return_tensors=TENSOR_TYPE_PYTORCH,
                    truncation=True,
                    return_token_type_ids=self.tokenizer_return_type_id,
                ).to(self.device)
                outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=20)
                answers = []
                for idx in range(len(outputs)):
                    output = outputs.tolist()[idx][len(inputs["input_ids"][idx]) :]
                    response = self.tokenizer.decode(output)
                    answers.append(response)

                answer_results = [answer.lstrip()[0] if answer.lstrip() else "-1" for answer in answers]
                is_correct = []
                for answer_result, label in zip(answer_results, labels):
                    is_correct.append("Correct" if answer_result == label else "Wrong")
                correct += is_correct.count("Correct")
            correct_total += correct
            sum_total += task_len
            index += 1
        if sum_total == 0:
            self.logger.error(
                f"Did not ran any test, maybe wrong cevel_5_shot dataset folder. Current folder is {self.dataset_path}"
            )
        else:
            self.logger.info(
                f"correct rate:{correct_total / sum_total}, correct total:{correct_total}, sum_total:{sum_total}"
            )

    def __run_full_dataset_humaneval(self):
        def cleanup_code(code: str) -> str:
            code_splits = code.split("\n")
            is_empty_line = False
            ind_empty_line = None
            for i, line in enumerate(code_splits):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    is_empty_line = True
                    ind_empty_line = i
                    break
            if is_empty_line:
                code = "\n".join(code_splits[:ind_empty_line])
            else:
                end_words = ["\ndef", "\nclass", "\n#", "\nassert", '\n"""', "\nprint", "\nif", "\n\n\n"]
                for w in end_words:
                    if w in code:
                        code = code[: code.rfind(w)]
            return code

        def run_test():
            for entry in tqdm(
                glob.glob((Path(self.dataset_path) / "*.jsonl").as_posix(), recursive=True), desc='global'
            ):
                dataset = []
                entry = get_valid_read_path(entry)
                with open(entry, encoding='utf-8') as f:
                    for line in f:
                        line_json = json.loads(line)
                        dataset.append(line_json)

                samples = []
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for _, batch in enumerate(tqdm(dataloader)):
                    task_ids = [task_id.split('/')[1] for task_id in batch["task_id"]]
                    queries = [prompt.strip() for prompt in batch["prompt"]]
                    inputs = self.tokenizer(
                        queries, padding=True, return_tensors=TENSOR_TYPE_PYTORCH, truncation=True
                    ).to(self.device)
                    outputs = self.model.generate(
                        inputs=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        do_sample=False,
                        max_new_tokens=512,
                    )
                    for idx, output in enumerate(outputs.tolist()):
                        output = output[len(inputs["input_ids"][idx]) :]
                        response = self.tokenizer.decode(output)
                        response_cleaned_up = cleanup_code(response)
                        result = dict(
                            task_id="HumanEval/" + task_ids[idx],
                            completion=response_cleaned_up,
                        )
                        samples += [result]
                self.__save_humaneval_res(samples)
                self.logger.info(f"Single question check result has been written to {self.result_file}.")
                result = evaluate_functional_correctness(self.result_file, [1], 4, 3.0, entry)
                self.logger.info(result)

        try:
            from human_eval.evaluation import evaluate_functional_correctness
        except ImportException as e:  # noqa: F821
            self.logger.error(f"Import human_eval test package failed. Error info is as below. \n {e}")
            return

        with torch.no_grad():
            try:
                run_test()
            except Exception as e:
                self.logger.error(f"Unable to run humaneval test. {e}")

    def __run_full_dataset_boolq(self):
        sample_yes = "How can we learning machine learning: yes"
        sample_no = "How can we learning machine learning: no"
        choice_tokens = [
            self.tokenizer([sample_yes], return_tensors=TENSOR_TYPE_PYTORCH, max_length=2048, add_special_tokens=False)
            .input_ids[0, -1]
            .item(),
            self.tokenizer([sample_no], return_tensors=TENSOR_TYPE_PYTORCH, max_length=2048, add_special_tokens=False)
            .input_ids[0, -1]
            .item(),
        ]

        def build_prompt(title, text, passage):
            prompt = f"{title} -- {passage}\nQuestion: {text}?\nAnswer:"
            return prompt

        def run_test(choice_tokens, correct_total, sum_total):
            for entry in tqdm(
                glob.glob((Path(self.dataset_path) / "*.jsonl").as_posix(), recursive=True), desc='global'
            ):
                dataset = []
                entry = get_valid_read_path(entry)
                with open(entry, encoding='utf-8') as f:
                    for line in f:
                        line_json = json.loads(line)
                        dataset.append(line_json)

                correct = 0
                dataset_num = len(dataset)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for _, batch in enumerate(tqdm(dataloader)):
                    titles = batch["title"]
                    texts = batch["question"]
                    passages = batch["passage"]
                    queries = []
                    for title, query, passage in zip(titles, texts, passages):
                        queries.append(build_prompt(title, query, passage))
                    inputs = self.tokenizer(
                        queries,
                        padding=True,
                        return_tensors=TENSOR_TYPE_PYTORCH,
                        truncation=True,
                        return_token_type_ids=self.tokenizer_return_type_id,
                    ).to(self.model.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :]
                    logits_softmax = F.log_softmax(logits.float(), dim=-1)

                    logits_softmax = logits_softmax[:, choice_tokens]
                    for idx, ans in enumerate(batch['answer']):
                        choice = (logits_softmax[idx, 0] > logits_softmax[idx, 1]).cpu()
                        correct += 1 if choice == ans else 0
                correct_total += correct
                sum_total += dataset_num
            return correct_total, sum_total

        correct_total = 0
        sum_total = 0
        with torch.no_grad():
            correct_total, sum_total = run_test(choice_tokens, correct_total, sum_total)
        if sum_total == 0:
            self.logger.error("Did not ran any test, maybe wrong humaneval dataset folder.")
        else:
            self.logger.info(
                f"correct rate:{correct_total / sum_total}, correct total:{correct_total}, sum_total:{sum_total}"
            )

    def __save_humaneval_res(self, results):
        self.result_file = os.path.dirname(os.path.abspath(__file__)) + os.sep + "result.jsonl"
        self.result_file = get_valid_write_path(self.result_file)
        mode = stat.S_IWUSR | stat.S_IRUSR
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(self.result_file, flags=flags, mode=mode), "w") as fp:
            for result in results:
                fp.write((json.dumps(result) + "\n"))

    def __run_full_dataset_truthfulqa(self):
        best_col = 'Best Answer'
        answer_col = 'Correct Answers'
        incorrect_col = 'Incorrect Answers'

        def run_answers():
            frame = pd.read_csv((Path(self.dataset_path) / "TruthfulQA.csv").as_posix())
            frame.dropna(axis=1, how='all', inplace=True)

            if tag not in frame.columns:
                frame[tag] = ''

            frame[tag].fillna('', inplace=True)
            frame[tag] = frame[tag].astype(str)

            num_rows = frame.shape[0]
            num_batches = math.ceil(num_rows / self.batch_size)

            seq_start = np.array(tokenizer('A:')['input_ids'])
            seq_end = np.array(tokenizer('Q:')['input_ids'])
            try:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # nosec B105
            except AttributeError:
                self.logger.info("this model can't set attribute 'pad_token'")
            with torch.no_grad():
                for batch in tqdm(range(num_batches)):
                    q_num = (
                        self.batch_size
                        if (batch + 1) * self.batch_size <= num_rows
                        else num_rows - self.batch_size * batch
                    )
                    idx_list = [i for i in range(batch * self.batch_size, batch * self.batch_size + q_num)]
                    prompt = [truthfulqa_eval.format_prompt(frame.loc[idx]) for idx in idx_list]
                    input_ids = tokenizer(
                        prompt, padding=True, return_tensors=TENSOR_TYPE_PYTORCH, truncation=True
                    ).input_ids
                    max_len = input_ids.shape[-1] + 50
                    input_ids = input_ids.to(self.device)
                    outputs = self.model.generate(
                        input_ids, do_sample=False, max_length=max_len, pad_token_id=self.tokenizer.eos_token_id
                    )
                    output_token_ids_list = [
                        output[len(input_ids[idx]) :] for idx, output in enumerate(outputs.tolist())
                    ]
                    gen_arrs = np.array(output_token_ids_list)

                    idx_start = [
                        truthfulqa_eval.find_subsequence(gen_arr, seq_start, start=True) for gen_arr in gen_arrs
                    ]
                    idx_end = [truthfulqa_eval.find_subsequence(gen_arr, seq_end, start=False) for gen_arr in gen_arrs]

                    output_token_ids_list = [
                        output_token_ids[idx_start[output_token_ids_idx] : idx_end[output_token_ids_idx]]
                        for output_token_ids_idx, output_token_ids in enumerate(output_token_ids_list)
                    ]
                    output_strs = [
                        tokenizer.decode(output_token_ids, skip_special_tokens=True)
                        for output_token_ids in output_token_ids_list
                    ]
                    output_str = [output_str.strip() for output_str in output_strs]

                    for idx in idx_list:
                        frame.loc[idx, tag] = output_str[idx % self.batch_size]
            return frame

        def run_probs(frame):
            truthfulqa_eval.set_columns(tag, frame)
            with torch.no_grad():
                for idx in tqdm(frame.index):
                    if pd.isnull(frame.loc[idx, incorrect_col]):
                        self.logger.warning("References missing for {0}!".format(idx))
                        continue
                    if len(frame.loc[idx, incorrect_col]) == 0:
                        self.logger.warning("References missing for {0}!".format(idx))
                        continue

                    ref_best = truthfulqa_eval.format_best(frame.loc[idx, best_col])
                    ref_true = truthfulqa_eval.split_multi_answer(frame.loc[idx, answer_col])
                    ref_false = truthfulqa_eval.split_multi_answer(frame.loc[idx, incorrect_col])

                    input_prompt = truthfulqa_eval.format_prompt(frame.loc[idx])

                    scores_true = get_scores(input_prompt, frame, idx, ref_true)
                    scores_false = get_scores(input_prompt, frame, idx, ref_false)
                    calc_param = CalcParam(tag, frame, idx, scores_true, scores_false, ref_true, ref_best)
                    frame = truthfulqa_eval.mc_calcs(calc_param)
            return frame

        def get_scores(input_prompt, frame, idx, ref_answer):
            scores_answer = []
            for temp_ans in ref_answer:
                prompt = [truthfulqa_eval.format_prompt_with_answer_strings(frame.loc[idx, 'Question'], temp_ans)]
                input_ids = tokenizer(input_prompt, return_tensors=TENSOR_TYPE_PYTORCH).input_ids
                prompt_ids = tokenizer(prompt, return_tensors=TENSOR_TYPE_PYTORCH).input_ids
                input_ids = input_ids.to(self.device)
                prompt_ids = prompt_ids.to(self.device)
                logits = self.model(prompt_ids)[0].squeeze(0)
                logits = logits[input_ids.shape[-1] - 1 : -1, :]

                logits_softmax = F.log_softmax(logits.float(), dim=-1)
                prompt_ids = prompt_ids[0, input_ids.shape[-1] :]
                log_probs = logits_softmax[range(logits_softmax.shape[0]), prompt_ids.squeeze(0)]
                log_probs = log_probs[3:]
                scores_answer.append(log_probs.sum().item())
            return scores_answer

        tokenizer = self.tokenizer
        tag = "test_model"

        frame = run_answers()
        frame = run_probs(frame)
        frame = truthfulqa_eval.run_bleu_and_rouge(tag, frame)

        results = truthfulqa_eval.format_frame(frame)

        results = results.mean(axis=0)
        results = results.reset_index().rename(columns={'level_0': 'Model', 'level_1': 'Metric', 0: 'Value'})

        results = results[results['Metric'].isin(['MC1', 'MC2', 'bleu diff', 'rouge1 diff', 'BLEURT diff'])]

        results = pd.pivot_table(results, 'Value', 'Model', 'Metric')

        results = results.rename(columns={'bleu diff': 'BLEU', 'rouge1 diff': 'ROUGE', 'BLEURT diff': 'BLEURT'})
        self.logger.info(f"results:{results}")

    def __run_full_dataset_mmlu(self):
        if self.shot == 0:
            self.__run_full_dataset_mmlu_0_shot()
        else:
            self.__run_full_dataset_mmlu_few_shots()

    def __run_full_dataset_mmlu_0_shot(self):
        def get_subject_mapping():
            subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
            subject_mapping = json_safe_load(subject_mapping_path)
            return subject_mapping

        def load_csv_by_task_name(task_name, dataset_path):
            csv_file_path = get_valid_path(os.path.join(dataset_path, "test", task_name + "_test.csv"))
            val_df = pd.read_csv(csv_file_path, header=None)
            return val_df

        def format_example(name, df, idx):
            question = df.iloc[idx, 0]
            a = df.iloc[idx, 1]
            b = df.iloc[idx, 2]
            c = df.iloc[idx, 3]
            d = df.iloc[idx, 4]
            prompt = (
                "\nThere is a single choice question about {}. Answer the question by replying A, B, C or D.\n"
                "Q: {}\nA. {}\nB. {}\nC. {}\nD. {}\n"
                "Let's think step by step. A: ".format(name.replace("_", " "), question, a, b, c, d)
            )
            return prompt

        correct_total = 0
        sum_total = 0
        result_total = []
        try:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # nosec B105
        except AttributeError:
            self.logger.info("this model can't set attribute 'pad_token'")

        subject_mapping = get_subject_mapping()
        subject_mapping = subject_mapping["mmlu_all_sets"]
        index = 1
        for task_name in tqdm(subject_mapping):
            self.logger.info(f"dataset {index} start, task name: {task_name}")
            val_df = load_csv_by_task_name(task_name, self.dataset_path)
            correct = 0
            task_len = val_df.shape[0]
            if task_len == 0:
                raise ValueError(f"dateset {task_name} is empty, please check.")
            for i in range(math.ceil(task_len / self.batch_size)):
                q_num = self.batch_size if (i + 1) * self.batch_size <= task_len else task_len - i * self.batch_size
                name = task_name
                prompt = [format_example(name, val_df, i * self.batch_size + j) for j in range(q_num)]
                labels = [val_df.iloc[i * self.batch_size + j, val_df.shape[1] - 1] for j in range(q_num)]
                prompts = [prpt.encode().decode(encoding="utf8") for prpt in prompt]
                inputs = self.tokenizer(prompts, padding=True, return_tensors=TENSOR_TYPE_PYTORCH, truncation=True)
                inputs = inputs.to(self.device)
                tokenizer_out_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                outputs = self.model.generate(
                    inputs=tokenizer_out_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=1024,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                answers = []
                for idx, output in enumerate(outputs.tolist()):
                    output = output[len(inputs["input_ids"][idx]) :]
                    answers.append(self.tokenizer.decode(output))

                answer_results = [self.__postprocess(answer, "ABCD") for answer in answers]
                is_correct = []
                for answer_result, label in zip(answer_results, labels):
                    is_correct.append("Correct" if answer_result == label else "Wrong")
                correct += is_correct.count("Correct")

            result = [task_name, correct / task_len, correct, task_len]
            self.logger.info(f"dataset {index} finish, result:{result}")
            result_total.append(result)
            correct_total += correct
            sum_total += task_len
            index += 1
        if sum_total == 0:
            self.logger.error(
                f"Did not run any test, maybe wrong mmlu dataset folder. Current folder is {self.dataset_path}"
            )
            return
        total = ["total", correct_total / sum_total, correct_total, sum_total]
        self.logger.info(f"total result:{total}")

    def __run_full_dataset_mmlu_few_shots(self):
        choices = ["A", "B", "C", "D"]
        test_set = {"mmlu": "test"}

        def get_subject_mapping():
            subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
            subject_mapping = json_safe_load(subject_mapping_path)
            return subject_mapping["mmlu_all_sets"]

        def load_csv_by_task_name(task_name, dataset_path):
            row_begin_idx = 0
            col_begin_idx = 0
            dev_file_path = get_valid_path(os.path.join(dataset_path, "dev", task_name + "_dev.csv"))
            ori_dev_df = pd.read_csv(dev_file_path, header=None)
            val_file_path = get_valid_path(
                os.path.join(dataset_path, test_set.get(self.dataset), f"{task_name}_{test_set.get(self.dataset)}.csv")
            )
            ori_val_df = pd.read_csv(val_file_path, header=None)

            dev_df = ori_dev_df.iloc[row_begin_idx : row_begin_idx + self.shot, col_begin_idx:]
            val_df = ori_val_df.iloc[row_begin_idx:, col_begin_idx:]
            return dev_df, val_df

        def format_subject(subject):
            sub_set = subject.split("_")
            res = ""
            for entry in sub_set:
                res += " " + entry
            return res

        def format_example(df, idx, include_answer=True):
            prompt = df.iloc[idx, 0]
            k = len(choices)
            for j in range(k):
                prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
            prompt += "\nAnswer:"
            if include_answer:
                prompt += " {}\n\n".format(df.iloc[idx, k + 1])
            return prompt

        def gen_prompt(train_df, subject, k=-1):
            prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
                format_subject(subject)
            )
            if k == -1:
                k = train_df.shape[0]
            for i in range(k):
                prompt += format_example(train_df, i)
            return prompt

        correct_total = 0
        sum_total = 0
        result_total = []
        is_result = False  # noqa: F841
        try:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # nosec B105
        except AttributeError:
            self.logger.info("this model can't set attribute 'pad_token'")

        subject_mapping = get_subject_mapping()
        index = 1
        for task_name in tqdm(subject_mapping):
            self.logger.info(f"dataset {index} start, task name: {task_name}")
            dev_df, val_df = load_csv_by_task_name(task_name, self.dataset_path)
            correct = 0
            task_len = val_df.shape[0]
            if task_len == 0:
                raise ValueError(f"dateset {task_name} is empty, please check.")
            for i in range(math.ceil(task_len / self.batch_size)):
                q_num = self.batch_size if (i + 1) * self.batch_size <= task_len else task_len - i * self.batch_size
                prompt_ends = []
                for j in range(q_num):
                    prompt_ends.append(format_example(val_df, i * self.batch_size + j, include_answer=False))
                train_prompts = [gen_prompt(dev_df, task_name, self.shot)] * q_num
                prompt = [t + p for t, p in zip(train_prompts, prompt_ends)]
                labels = [val_df.iloc[i * self.batch_size + j, val_df.shape[1] - 1] for j in range(q_num)]
                prompts = [prpt.encode().decode(encoding="utf8") for prpt in prompt]
                inputs = self.tokenizer(prompts, padding=True, return_tensors=TENSOR_TYPE_PYTORCH, truncation=True)
                inputs = inputs.to(self.device)
                tokenizer_out_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                outputs = self.model.generate(
                    inputs=tokenizer_out_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=20,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                answers = []
                for idx in range(len(outputs)):
                    output = outputs.tolist()[idx][len(inputs["input_ids"][idx]) :]
                    response = self.tokenizer.decode(output)
                    answers.append(response)

                answer_results = [answer.lstrip()[0] if answer.lstrip() else "-1" for answer in answers]
                is_correct = []
                for answer_result, label in zip(answer_results, labels):
                    is_correct.append("Correct" if answer_result == label else "Wrong")
                correct += is_correct.count("Correct")

            result = [task_name, correct / task_len, correct, task_len]
            self.logger.info(f"dataset {index} finish, result:{result}")
            result_total.append(result)
            correct_total += correct
            sum_total += task_len
            index += 1

        if sum_total == 0:
            self.logger.error(
                f"Did not run any test, maybe wrong mmlu dataset folder. Current folder is {self.dataset_path}"
            )
            return
        total = ["total", correct_total / sum_total, correct_total, sum_total]
        self.logger.info(f"total result:{total}")
