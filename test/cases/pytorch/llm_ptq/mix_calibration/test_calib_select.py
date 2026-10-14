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
# pylint: disable=no-name-in-module,pointless-string-statement,redefined-outer-name,unspecified-encoding

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, call
import pandas as pd
import pytest
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch
from msmodelslim.pytorch.llm_ptq.mix_calibration.calib_select import (
    DatasetFactory,
    CalibHandler,
    CalibrationData,
    SUPPORTED_DATASET_NAME,
    CEval5ShotProcessor,
)


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.eos_token_id = "[EOS]"
    tokenizer.pad_token_id = "[PAD]"
    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock(spec=PreTrainedModel)
    model.device = "cpu"
    return model


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as tmp:
        """创建临时测试配置文件"""
        config = {
            "configurations": [
                {"dataset_name": "boolq", "dataset_path": str(os.path.join(tmp, "boolq", "boolq.jsonl"))},
                {"dataset_name": "ceval_5_shot", "dataset_path": str(os.path.join(tmp, "ceval_5_shot"))},
                {"dataset_name": "gsm8k", "dataset_path": str(os.path.join(tmp, "gsm8k", "GSM8K.jsonl"))},
                {"dataset_name": "mmlu", "dataset_path": str(os.path.join(tmp, "mmlu"))},
            ]
        }

        # 1. 保存原始 umask
        original_umask = os.umask(0)  # 临时设为 0 并获取原始值
        try:
            # 2. 设置目标 umask（0o027 对应权限 640）
            os.umask(0o027)
            with open(f"{tmp}/test_config.json", "w") as f:
                json.dump(config, f)

            # 创建临时数据目录
            os.mkdir(os.path.join(tmp, "boolq"))
            os.mkdir(os.path.join(tmp, "ceval_5_shot"))
            os.mkdir(os.path.join(tmp, "ceval_5_shot", "val"))
            os.mkdir(os.path.join(tmp, "ceval_5_shot", "dev"))
            os.open(os.path.join(tmp, "ceval_5_shot", "dev", "test_subject_dev.csv"), os.O_CREAT, 0o644)
            os.open(os.path.join(tmp, "ceval_5_shot", "val", "test_subject_val.csv"), os.O_CREAT, 0o644)
            os.mkdir(os.path.join(tmp, "gsm8k"))
            os.mkdir(os.path.join(tmp, "mmlu"))
            os.mkdir(os.path.join(tmp, "mmlu", "val"))
            os.mkdir(os.path.join(tmp, "mmlu", "dev"))

            # 创建示例数据文件
            with open(os.path.join(tmp, "boolq", "boolq.jsonl"), "w") as f:
                f.write(
                    json.dumps(
                        {
                            "title": "Test Title",
                            "question": "Is this a test?",
                            "passage": "This is a test passage",
                            "answer": True,
                        }
                    )
                    + "\n"
                )

            # 创建ceval测试数据
            subject_map = {}
            with open(os.path.join(tmp, "ceval_5_shot", "subject_mapping.json"), "w") as f:
                json.dump(subject_map, f)

            dev_data = [["Q1", "A", "B", "C", "D", "A"], ["Q2", "A", "B", "C", "D", "A"]]
            val_data = [["Q1", "A", "B", "C", "D", "A"], ["Q2", "A", "B", "C", "D", "A"]]

            pd.DataFrame(dev_data).to_csv(
                os.path.join(tmp, "ceval_5_shot", "dev", "test_subject_dev.csv"), header=False, index=False
            )
            pd.DataFrame(val_data).to_csv(
                os.path.join(tmp, "ceval_5_shot", "val", "test_subject_val.csv"), header=False, index=False
            )

            # 创建mmlu测试数据
            subject_map = {"mmlu_all_sets": []}
            with open(os.path.join(tmp, "mmlu", "subject_mapping.json"), "w") as f:
                json.dump(subject_map, f)

            # 创建gsm8k测试数据
            with open(os.path.join(tmp, "gsm8k", "GSM8K.jsonl"), "w") as f:
                f.write(json.dumps({"question": "What is 2 + 2?", "answer": "The answer is <<2+2=4>>4#### 4"}) + "\n")
        finally:
            # 4. 无论是否出错，都恢复原始 umask
            os.umask(original_umask)

        yield tmp


class TestDatasetFactory:
    @pytest.mark.parametrize("dataset_name", SUPPORTED_DATASET_NAME)
    def test_create_supported_processors(self, dataset_name, tmp_path):
        config = {
            "boolq": str(os.path.join(tmp_path, "boolq", "boolq.jsonl")),
            "ceval_5_shot": str(os.path.join(tmp_path, "ceval_5_shot")),
            "gsm8k": str(os.path.join(tmp_path, "gsm8k", "GSM8K.jsonl")),
            "mmlu": str(os.path.join(tmp_path, "mmlu")),
        }

        processor = DatasetFactory.create_dataset_processor(
            dataset_name=dataset_name, dataset_path=config[dataset_name], customized_dataset_processor={}
        )
        assert processor is not None
        assert (
            processor.__class__.__name__ == f"{dataset_name.capitalize()}Processor"
            if dataset_name != "ceval_5_shot"
            else processor.__class__.__name__ == "CEval5ShotProcessor"
        )

    def test_create_invalid_processor(self):
        processor = DatasetFactory.create_dataset_processor(
            dataset_name="invalid_dataset", dataset_path="", customized_dataset_processor={}
        )
        assert processor is None


class TestCalibHandler:
    def test_basic_sampling(self, mock_tokenizer, tmp_path):
        # 初始化boolq processor
        processor = DatasetFactory.create_dataset_processor(
            "boolq", str(os.path.join(tmp_path, "boolq", "boolq.jsonl")), {}, mock_tokenizer
        )

        handler = CalibHandler(dataset_name="boolq", dataset_processor=processor, shuffle_seed=42, batch_size=2)
        handler.set_sample_size(1)

        samples = handler.run()
        assert len(samples) == 1
        assert "prompt" in samples[0]
        assert "ans" in samples[0]

    def test_positive_prompt_verification(self, mock_tokenizer, mock_model, tmp_path):
        # 配置mock model返回值
        mock_model.generate.return_value = ["aaaaa"]
        mock_tokenizer.decode.return_value = "A"
        mock_model.device = "cpu"

        processor = DatasetFactory.create_dataset_processor(
            "gsm8k", str(os.path.join(tmp_path, "gsm8k", "GSM8K.jsonl")), {}, mock_tokenizer, mock_model
        )

        handler = CalibHandler(dataset_name="gsm8k", dataset_processor=processor, shuffle_seed=42, batch_size=2)
        handler.set_sample_size(1)

        samples = handler.run(need_positive_prompt=True)
        assert len(samples) <= 1  # 验证可能过滤部分样本


class TestCalibrationData:
    def test_config_loading(self, tmp_path, mock_tokenizer):
        calib = CalibrationData(config_path=str(os.path.join(tmp_path, "test_config.json")), tokenizer=mock_tokenizer)

        assert len(calib.handlers) == 4
        assert "boolq" in calib.handlers
        assert "ceval_5_shot" in calib.handlers

    def test_mixed_dataset_generation(self, tmp_path, mock_tokenizer):
        save_path = str(os.path.join(tmp_path, "output.json"))
        calib = CalibrationData(
            config_path=str(os.path.join(tmp_path, "test_config.json")), tokenizer=mock_tokenizer, save_path=save_path
        )
        calib.set_sample_size({"boolq": 1, "ceval_5_shot": 1})

        mixed_data = calib.process()
        assert len(mixed_data) == 1
        assert os.path.exists(save_path)

    def test_invalid_model_handling(self, tmp_path):
        with pytest.raises(TypeError):
            CalibrationData(
                config_path=str(os.path.join(tmp_path, "test_config.json")),
                tokenizer=MagicMock(),
                model="invalid_model_type",
            )


# 验证各processor的数据解析逻辑
@pytest.mark.parametrize("dataset_name", SUPPORTED_DATASET_NAME)
def test_dataset_processors(tmp_path, dataset_name):
    config = {
        "boolq": str(os.path.join(tmp_path, "boolq", "boolq.jsonl")),
        "ceval_5_shot": str(os.path.join(tmp_path, "ceval_5_shot")),
        "gsm8k": str(os.path.join(tmp_path, "gsm8k", "GSM8K.jsonl")),
        "mmlu": str(os.path.join(tmp_path, "mmlu")),
    }

    # 初始化processor
    processor = DatasetFactory.create_dataset_processor(
        dataset_name=dataset_name, dataset_path=config[dataset_name], customized_dataset_processor={}
    )

    # 验证基础属性
    assert processor.dataset_path == str(config[dataset_name])


# ------------------------------ 测试类（包含内部 fixture，无命名冲突） ------------------------------
class TestCEval5ShotProcessor:
    """Test class for CEval5ShotProcessor（内聚 fixture，避免重名冲突）"""

    # ============================== 测试方法（使用内部 fixture） ==============================
    @staticmethod
    def test___init___given_valid_dataset_when_columns_sufficient_then_ori_populated(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        assert len(processor.ori_prompts) == 2
        assert len(processor.ori_answers) == 2
        assert processor.choices == ("A", "B", "C", "D")
        assert processor.shot == 5

    @staticmethod
    def test___init___given_valid_dataset_when_val_df_columns_insufficient_then_skip_task(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model, mocker
    ):
        mock_logger = mocker.patch("msmodelslim.pytorch.llm_ptq.mix_calibration.calib_select.msmodelslim_logger")
        CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        mock_logger.warning.assert_has_calls(
            [call("val_df has only 5 columns, but requires at least 6 columns. Skipping task math.")], any_order=False
        )

    @staticmethod
    def test___init___given_missing_subject_mapping_when_init_then_raise_error(
        test_ceval_missing_subject_mapping_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        with pytest.raises(ValueError) as excinfo:
            CEval5ShotProcessor(
                dataset_path=test_ceval_missing_subject_mapping_dataset,
                tokenizer=test_ceval_mock_tokenizer,
                model=test_ceval_mock_model,
            )
        assert "subject_mapping.json" in str(excinfo.value)

    @staticmethod
    def test___init___given_missing_dev_csv_when_init_then_raise_error(
        test_ceval_missing_dev_csv_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        with pytest.raises(FileNotFoundError) as excinfo:
            CEval5ShotProcessor(
                dataset_path=test_ceval_missing_dev_csv_dataset,
                tokenizer=test_ceval_mock_tokenizer,
                model=test_ceval_mock_model,
            )
        assert "physics_dev.csv" in str(excinfo.value)

    # ------------------------------ Test get_subject_mapping ------------------------------
    @staticmethod
    def test_get_subject_mapping_given_valid_dataset_when_call_then_return_correct_mapping(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        subject_mapping = processor.get_subject_mapping()
        assert subject_mapping == {"physics": "Physics", "math": "Mathematics", "chemistry": "Chemistry"}

    @staticmethod
    def test_get_subject_mapping_given_non_existent_path_when_call_then_raise_error(
        tmp_path, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        with pytest.raises(FileNotFoundError):
            non_exist_path = os.path.join(tmp_path, "non_exist")
            processor = CEval5ShotProcessor(
                dataset_path=non_exist_path, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
            )
            processor.get_subject_mapping()

    # ------------------------------ Test load_csv_by_task_name ------------------------------
    @staticmethod
    def test_load_csv_by_task_name_given_valid_task_when_call_then_return_correct_dfs(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        dev_df, val_df = processor.load_csv_by_task_name(task_name="physics", dataset_path=test_ceval_tmp_dataset)
        assert dev_df.shape == (5, 6)
        assert val_df.shape == (2, 6)
        assert dev_df.iloc[0, 5] == "A"
        assert val_df.iloc[1, 5] == "A"

    @staticmethod
    def test_load_csv_by_task_name_given_missing_val_csv_when_call_then_raise_error(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        with pytest.raises(FileNotFoundError) as excinfo:
            processor.load_csv_by_task_name(task_name="missing_val", dataset_path=test_ceval_tmp_dataset)
        assert "missing_val_dev.csv" in str(excinfo.value)

    # ------------------------------ Test format_subject ------------------------------
    @staticmethod
    def test_format_subject_given_underscored_subject_when_call_then_return_space_separated(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        test_cases = [
            ("math_algebra", " math algebra"),
            ("physics_quantum_mechanics", " physics quantum mechanics"),
            ("computer_science_ai", " computer science ai"),
        ]
        for input_subj, expected in test_cases:
            assert processor.format_subject(subject=input_subj) == expected

    @staticmethod
    def test_format_subject_given_no_underscore_subject_when_call_then_return_single_word(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        test_cases = [("chemistry", " chemistry"), ("biology", " biology"), ("english", " english")]
        for input_subj, expected in test_cases:
            assert processor.format_subject(subject=input_subj) == expected

    @staticmethod
    def test_format_subject_given_empty_string_when_call_then_return_space(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        assert processor.format_subject(subject="") == " "

    # ------------------------------ Test format_example ------------------------------
    @staticmethod
    def test_format_example_given_valid_df_when_include_answer_true_then_prompt_has_answer(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        dev_df, val_df = processor.load_csv_by_task_name(task_name="physics", dataset_path=test_ceval_tmp_dataset)
        idx = 0
        prompt = processor.format_example(df=val_df, idx=idx, include_answer=True)
        assert val_df.iloc[idx, 0] in prompt
        assert f"A. {val_df.iloc[idx, 1]}" in prompt
        assert f"Answer: {val_df.iloc[idx, 5]}\n\n" in prompt

    @staticmethod
    def test_format_example_given_insufficient_columns_when_call_then_raise_index_error(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        _, val_df = processor.load_csv_by_task_name(task_name="math", dataset_path=test_ceval_tmp_dataset)
        with pytest.raises(IndexError):
            processor.format_example(df=val_df, idx=0, include_answer=True)

    @staticmethod
    def test_format_example_given_out_of_range_idx_when_call_then_raise_index_error(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        _, val_df = processor.load_csv_by_task_name(task_name="physics", dataset_path=test_ceval_tmp_dataset)
        with pytest.raises(IndexError):
            processor.format_example(df=val_df, idx=10, include_answer=False)

    # ------------------------------ Test gen_prompt ------------------------------
    @staticmethod
    def test_gen_prompt_given_valid_train_df_when_k_negative_then_use_all_rows(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        dev_df, _ = processor.load_csv_by_task_name(task_name="physics", dataset_path=test_ceval_tmp_dataset)
        prompt = processor.gen_prompt(train_df=dev_df, subject="physics", k=-1)
        assert prompt.count("Answer:") == 5
        assert prompt.startswith("The following are multiple choice questions about  physics.\n\n")

    @staticmethod
    def test_gen_prompt_given_valid_train_df_when_k_positive_then_use_k_rows(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        dev_df, _ = processor.load_csv_by_task_name(task_name="physics", dataset_path=test_ceval_tmp_dataset)
        prompt = processor.gen_prompt(train_df=dev_df, subject="physics", k=2)
        assert prompt.count("Answer:") == 2

    # ------------------------------ Test process_data ------------------------------
    @staticmethod
    def test_process_data_given_out_of_range_index_when_call_then_raise_index_error(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        with pytest.raises(IndexError):
            processor.process_data(indexs=[3])

    @staticmethod
    def test_process_data_given_empty_indexs_when_call_then_return_empty(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer, test_ceval_mock_model
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=test_ceval_mock_model
        )
        assert len(processor.process_data(indexs=[])) == 0

    # ------------------------------ Test verify_positive_prompt ------------------------------
    @staticmethod
    def test_verify_positive_prompt_given_model_not_initialized_when_call_then_raise_attribute_error(
        test_ceval_tmp_dataset, test_ceval_mock_tokenizer
    ):
        processor = CEval5ShotProcessor(
            dataset_path=test_ceval_tmp_dataset, tokenizer=test_ceval_mock_tokenizer, model=None
        )
        with pytest.raises(AttributeError):
            processor.verify_positive_prompt(prompts=["P1"], labels=["A"])

    # ============================== 类内 fixture（仅当前类可用，无冲突） ==============================
    @pytest.fixture(scope="function")
    def test_ceval_tmp_dataset(self, tmp_path):
        """内部 fixture：创建 fake CEval 5-shot 数据集（加前缀避免冲突）"""
        tmp_path = Path(tmp_path)
        dataset_root = tmp_path / "ceval_dataset"
        dataset_root.mkdir(exist_ok=True)

        # 1. Subject mapping file
        subject_mapping = {"physics": "Physics", "math": "Mathematics", "chemistry": "Chemistry"}
        with open(dataset_root / "subject_mapping.json", "w", encoding="utf-8") as f:
            json.dump(subject_mapping, f)
        os.chmod(dataset_root / "subject_mapping.json", 0o640)

        # 2. Dev/Val directories
        dev_dir = dataset_root / "dev"
        val_dir = dataset_root / "val"
        dev_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

        # ---------------- Valid Task: Physics ----------------
        dev_physics_data = [
            [0, "Question", "Option A", "Option B", "Option C", "Option D", "Answer"],
            [1, "F=ma means?", "Force=mass×acceleration", "Energy=mass×c²", "Power=V×I", "Work=F×d", "A"],
            [2, "E=mc² means?", "Force=mass×acceleration", "Energy=mass×c²", "Power=V×I", "Work=F×d", "B"],
            [3, "P=VI means?", "Force=mass×acceleration", "Energy=mass×c²", "Power=V×I", "Work=F×d", "C"],
            [4, "W=Fd means?", "Force=mass×acceleration", "Energy=mass×c²", "Power=V×I", "Work=F×d", "D"],
            [5, "V=IR means?", "Ohm's Law", "Newton's Law", "Kepler's Law", "Boyle's Law", "A"],
        ]
        pd.DataFrame(dev_physics_data).to_csv(dev_dir / "physics_dev.csv", index=False, header=False)

        val_physics_data = [
            [0, "Question", "Option A", "Option B", "Option C", "Option D", "Answer"],
            [1, "Test Q1: Gravity law?", "F=GMm/r²", "F=ma", "E=mc²", "P=VI", "A"],
            [2, "Test Q2: Ohm's law?", "V=IR", "F=ma", "E=mc²", "P=VI", "A"],
        ]
        pd.DataFrame(val_physics_data).to_csv(val_dir / "physics_val.csv", index=False, header=False)

        # ---------------- Invalid Task: Math ----------------
        dev_math_data = [
            [0, "Question", "Option A", "Option B", "Option C", "Option D"],
            [1, "2+2=?", "3", "4", "5", "6"],
            [2, "3×3=?", "6", "7", "8", "9"],
        ]
        pd.DataFrame(dev_math_data).to_csv(dev_dir / "math_dev.csv", index=False, header=False)

        val_math_data = [
            [0, "Question", "Option A", "Option B", "Option C", "Option D"],
            [1, "2+2=?", "3", "4", "5", "6"],
            [2, "3×3=?", "6", "7", "8", "9"],
        ]
        pd.DataFrame(val_math_data).to_csv(val_dir / "math_val.csv", index=False, header=False)

        # ---------------- Valid Task: Chemistry ----------------
        dev_chem_data = [
            ["Question", "A", "B", "C", "D", "Answer"],
            ["H2O is?", "Water", "Oxygen", "Hydrogen", "Carbon", "A"],
        ]
        pd.DataFrame(dev_chem_data).to_csv(dev_dir / "chemistry_dev.csv", index=False, header=False)

        val_chem_data = [
            ["Question", "A", "B", "C", "D", "Answer"],
            ["CO2 is?", "Carbon Dioxide", "Oxygen", "Carbon", "Water", "A"],
        ]
        pd.DataFrame(val_chem_data).to_csv(val_dir / "chemistry_val.csv", index=False, header=False)

        return dataset_root

    @pytest.fixture(scope="function")
    def test_ceval_mock_tokenizer(self):
        """内部 fixture：Mock Tokenizer（加前缀避免冲突）"""

        def tokenize_side_effect(text, **kwargs):
            return {
                "input_ids": torch.tensor([[101] + [100 + i for i in range(len(text.split()))] + [102]]),
                "attention_mask": torch.tensor([[1] * (len(text.split()) + 2)]),
            }

        def decode_side_effect(token_ids, **kwargs):
            if token_ids[-1] == 103:
                return "A"
            elif token_ids[-1] == 104:
                return "B"
            else:
                return ""

        def pad_side_effect(batch, **kwargs):
            return {
                "input_ids": torch.cat([x["input_ids"] for x in batch]),
                "attention_mask": torch.cat([x["attention_mask"] for x in batch]),
            }

        tokenizer = Mock(spec=PreTrainedTokenizerBase)
        tokenizer.side_effect = tokenize_side_effect
        tokenizer.decode.side_effect = decode_side_effect
        tokenizer.pad.side_effect = pad_side_effect

        return tokenizer

    @pytest.fixture(scope="function")
    def test_ceval_mock_model(self):
        """内部 fixture：Mock Model（加前缀避免冲突）"""

        def generate_side_effect(**kwargs):
            return torch.cat([kwargs["input_ids"], torch.tensor([[103]] * kwargs["input_ids"].shape[0])], dim=1)

        def model_side_effect(**kwargs):
            return Mock(logits=torch.randn(kwargs["input_ids"].shape[0], kwargs["input_ids"].shape[1], 1000))

        model = Mock(spec=PreTrainedModel)
        model.device = torch.device("cpu")
        model.generate.side_effect = generate_side_effect
        model.side_effect = model_side_effect

        return model

    @pytest.fixture(scope="function")
    def test_ceval_missing_subject_mapping_dataset(self, tmp_path):
        """内部 fixture：缺失 subject_mapping.json 的数据集"""
        tmp_path = Path(tmp_path)
        dataset_root = tmp_path / "missing_subject_mapping"
        dataset_root.mkdir(exist_ok=True)
        (dataset_root / "dev").mkdir(exist_ok=True)
        (dataset_root / "val").mkdir(exist_ok=True)
        return dataset_root

    @pytest.fixture(scope="function")
    def test_ceval_missing_dev_csv_dataset(self, tmp_path):
        """内部 fixture：缺失 dev CSV 的数据集"""
        tmp_path = Path(tmp_path)
        dataset_root = tmp_path / "missing_dev_csv"
        # 保存原 umask → 设置新 umask（0o027 对应文件权限 640）→ 最后恢复
        old_umask = os.umask(0o027)  # 0o027: 组用户减写权限，其他用户减所有权限
        try:
            dataset_root.mkdir(exist_ok=True)

            with open(dataset_root / "subject_mapping.json", "w") as f:
                json.dump({"physics": "Physics"}, f)

            (dataset_root / "dev").mkdir(exist_ok=True)
            val_dir = dataset_root / "val"
            val_dir.mkdir(exist_ok=True)
            pd.DataFrame([["Q", "A", "B", "C", "D", "A"]]).to_csv(
                val_dir / "physics_val.csv", index=False, header=False
            )

        finally:
            os.umask(old_umask)  # 恢复原 umask，避免影响其他测试
        return dataset_root

    @pytest.fixture(scope="function")
    def test_ceval_invalid_json_dataset(self, tmp_path):
        """内部 fixture：包含无效 JSON 的数据集"""
        tmp_path = Path(tmp_path)
        dataset_root = tmp_path / "invalid_json"
        dataset_root.mkdir(exist_ok=True)
        with open(dataset_root / "subject_mapping.json", "w") as f:
            f.write("{invalid json syntax}")
        os.chmod(dataset_root / "subject_mapping.json", 0o640)
        (dataset_root / "dev").mkdir(exist_ok=True)
        (dataset_root / "val").mkdir(exist_ok=True)
        return dataset_root
