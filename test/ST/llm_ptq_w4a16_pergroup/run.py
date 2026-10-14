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
# pylint: disable=consider-using-from-import,logging-fstring-interpolation,no-member,no-name-in-module,redefined-outer-name,ungrouped-imports

import os
import json

import torch
import numpy as np  # noqa: F401
import torch.nn as nn  # noqa: F401
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.pytorch.lowbit.quant_modules import LinearQuantizer as LowBitLinearQuantizer


# Constants for model validation and quantization
EXPECTED_OUTPUT_FILES = 2
W4A8_MIN_VALUE = -8  # Minimum value for 4-bit weight quantization
W4A8_MAX_VALUE = 7  # Maximum value for 4-bit weight quantization
W4A8_MIN_PACK_VALUE = -128  # Minimum packed value for AscendV1 format
W4A8_MAX_PACK_VALUE = 127  # Maximum packed value for AscendV1 format


def log_validation_step(step_name: str, success: bool = True):
    """
    Helper function to log validation steps with consistent formatting

    Args:
        step_name (str): Name of the validation step being performed
        success (bool): Whether the validation step passed or failed
    """
    status = "✓" if success else "✗"
    msmodelslim_logger.info(f"[Validation] {step_name}: {status}")


def verify_save_files_name_and_num_safetensors(quant_config):
    """
    Verify the number and names of output files in SafeTensors format

    This function checks:
    - The existence of the output directory
    - The correct number of output files
    - The presence of both description and weight files
    - The correct naming convention for the files

    Args:
        quant_config: Configuration object containing quantization parameters

    Raises:
        AssertionError: If any verification check fails
    """
    msmodelslim_logger.info("Starting File Verification")

    # Check output directory and file count
    output_dir = os.path.join(os.environ['PROJECT_PATH'], "output/llm_ptq_w4a16_pergroup")
    output_files = os.listdir(output_dir)
    assert len(output_files) == EXPECTED_OUTPUT_FILES, (
        f"Expected {EXPECTED_OUTPUT_FILES} output files, found {len(output_files)}"
    )

    # Verify description file existence
    description_file = os.path.join(output_dir, f"quant_model_description_{quant_config.model_quant_type.lower()}.json")
    is_description_file_exist = os.path.exists(description_file)
    log_validation_step("Description file check", is_description_file_exist)
    assert is_description_file_exist, "Description file not found"

    # Verify weight file existence
    weight_file = os.path.join(output_dir, f"quant_model_weight_{quant_config.model_quant_type.lower()}.safetensors")
    is_weight_file_exist = os.path.exists(weight_file)
    log_validation_step("Weight file check", is_weight_file_exist)
    assert is_weight_file_exist, "Weight file not found"

    msmodelslim_logger.info("File verification completed successfully")


def verify_save_files_name_and_num_ascendV1(quant_config):
    """
    Verify the number and names of output files in AscendV1 format

    This function checks:
    - The existence of the output directory
    - The correct number of output files
    - The presence of both description and weight files
    - The correct naming convention for AscendV1 format

    Args:
        quant_config: Configuration object containing quantization parameters

    Raises:
        AssertionError: If any verification check fails
    """
    msmodelslim_logger.info("Starting File Verification")

    # Check output directory and file count
    output_dir = os.path.join(os.environ['PROJECT_PATH'], "output/llm_ptq_w4a16_pergroup")
    output_files = os.listdir(output_dir)
    assert len(output_files) == EXPECTED_OUTPUT_FILES, (
        f"Expected {EXPECTED_OUTPUT_FILES} output files, found {len(output_files)}"
    )

    # Verify description file existence
    description_file = os.path.join(output_dir, "quant_model_description.json")
    is_description_file_exist = os.path.exists(description_file)
    log_validation_step("Description file check", is_description_file_exist)
    assert is_description_file_exist, "Description file not found"

    # Verify weight file existence
    weight_file = os.path.join(output_dir, f"quant_model_weight_{quant_config.model_quant_type.lower()}.safetensors")
    is_weight_file_exist = os.path.exists(weight_file)
    log_validation_step("Weight file check", is_weight_file_exist)
    assert is_weight_file_exist, "Weight file not found"

    msmodelslim_logger.info("File verification completed successfully")


def verify_save_description_safetensors(model, quant_config, disable_names):
    """
    Verify the content of the description file in SafeTensors format

    This function validates:
    - Quantization types for each layer
    - Disabled layers are marked as float
    - Proper quantization marking for LowBitLinearQuantizer layers
    - Layer statistics and counts

    Args:
        model: The model being quantized
        quant_config: Configuration object containing quantization parameters
        disable_names: List of layer names to exclude from quantization

    Raises:
        AssertionError: If any verification check fails
    """
    msmodelslim_logger.info("Starting Description Verification")

    description_file = os.path.join(
        os.environ['PROJECT_PATH'],
        "output/llm_ptq_w4a16_pergroup",
        f"quant_model_description_{quant_config.model_quant_type.lower()}.json",
    )

    with open(description_file, 'r', encoding='utf-8') as f:
        description_data = json.load(f)

    validation_stats = {"total_layers": 0, "disabled_layers": 0, "quantized_layers": 0}

    for name, mod in model.named_modules():
        key = f"{name}.weight"

        # Skip normalization layers
        if "norm" in name:
            continue

        # Check disabled layers
        if name in disable_names:
            validation_stats["disabled_layers"] += 1
            assert description_data[key] == QuantType.FLOAT, f"Expected float for {key}, got {description_data[key]}"

            validation_stats["total_layers"] += 1
            continue

        # Check quantized layers
        if isinstance(mod, LowBitLinearQuantizer):
            validation_stats["quantized_layers"] += 1
            assert description_data[key] == QuantType.W4A16, f"Expected w4a16 for {key}, got {description_data[key]}"

            validation_stats["total_layers"] += 1
            continue

    msmodelslim_logger.info("Description verification stats:")
    msmodelslim_logger.info(f"- Total layers examined: {validation_stats['total_layers']}")
    msmodelslim_logger.info(f"- Disabled layers: {validation_stats['disabled_layers']}")
    msmodelslim_logger.info(f"- Quantized layers: {validation_stats['quantized_layers']}")
    msmodelslim_logger.info("Description verification completed successfully")


def verify_save_description_ascendV1(model, quant_config, disable_names):
    """
    Verify the content of the description file in AscendV1 format

    This function validates:
    - Model version information
    - Quantization type configuration
    - Group size settings
    - Layer-wise quantization settings
    - Disabled layer configurations

    Args:
        model: The model being quantized
        quant_config: Configuration object containing quantization parameters
        disable_names: List of layer names to exclude from quantization

    Raises:
        AssertionError: If any verification check fails
    """
    msmodelslim_logger.info("Starting Description Verification")

    description_file = os.path.join(
        os.environ['PROJECT_PATH'], "output/llm_ptq_w4a16_pergroup", "quant_model_description.json"
    )

    with open(description_file, 'r', encoding='utf-8') as f:
        description_data = json.load(f)

    assert description_data.get("version") is not None, "Version not found"
    assert description_data["model_quant_type"] == quant_config.model_quant_type, (
        f"Expected {quant_config.model_quant_type} for model_quant_type, got {description_data['model_quant_type']}"
    )
    assert description_data["group_size"] == quant_config.group_size, (
        f"Expected {quant_config.group_size} for group_size, got {description_data['group_size']}"
    )

    validation_stats = {"total_layers": 0, "disabled_layers": 0, "quantized_layers": 0}

    for name, mod in model.named_modules():
        key = f"{name}.weight"

        # Skip normalization layers
        if "norm" in name:
            continue

        # Check disabled layers
        if name in disable_names:
            validation_stats["disabled_layers"] += 1
            assert description_data[key] == QuantType.FLOAT, f"Expected float for {key}, got {description_data[key]}"

            validation_stats["total_layers"] += 1
            continue

        # Check quantized layers
        if isinstance(mod, LowBitLinearQuantizer):
            validation_stats["quantized_layers"] += 1
            assert description_data[key] == QuantType.W4A16, f"Expected w4a16 for {key}, got {description_data[key]}"

            validation_stats["total_layers"] += 1
            continue

    msmodelslim_logger.info("Description verification stats:")
    msmodelslim_logger.info(f"- Total layers examined: {validation_stats['total_layers']}")
    msmodelslim_logger.info(f"- Disabled layers: {validation_stats['disabled_layers']}")
    msmodelslim_logger.info(f"- Quantized layers: {validation_stats['quantized_layers']}")
    msmodelslim_logger.info("Description verification completed successfully")


def verify_save_weights_safetensors(quant_config):
    """
    Verify the quantized weights saved in SafeTensors format

    This function checks:
    - Data types of weights and scales
    - Value ranges for quantized weights
    - Shape relationships between weights and scales
    - Group size configurations
    - First and second stage quantization parameters

    Args:
        quant_config: Configuration object containing quantization parameters

    Raises:
        AssertionError: If any verification check fails
    """
    msmodelslim_logger.info("Starting Weight Verification")

    # Load weight file
    weight_file = os.path.join(
        os.environ['PROJECT_PATH'],
        "output/llm_ptq_w4a16_pergroup",
        f"quant_model_weight_{quant_config.model_quant_type.lower()}.safetensors",
    )

    tensors = load_file(weight_file)
    verification_stats = {"total_weights": 0, "quantized_weights": 0, "float_weights": 0}

    for name, mod in model.named_modules():
        key = f"{name}.weight"

        if "norm" in name:
            continue

        if name in disable_names:
            verification_stats["float_weights"] += 1
            assert tensors[key].dtype == model.config.torch_dtype, (
                f"Expected {model.config.torch_dtype} for {key}, got {tensors[key].dtype}"
            )
            verification_stats["total_weights"] += 1
            continue

        if isinstance(mod, LowBitLinearQuantizer):
            verification_stats["quantized_weights"] += 1
            verification_stats["total_weights"] += 1

            # Verify weight properties
            weight_tensor = tensors[key]
            assert weight_tensor.dtype == torch.int8, f"Expected int8 for {key}, got {weight_tensor.dtype}"
            assert W4A8_MIN_VALUE <= weight_tensor.min() <= weight_tensor.max() <= W4A8_MAX_VALUE, (
                f"Values out of range for {key}: [{weight_tensor.min()}, {weight_tensor.max()}]"
            )
            assert weight_tensor.max() >= W4A8_MAX_VALUE, "Weight max value should be greater than W4A8_MAX_VALUE"

            # Verify first stage quantization parameters
            for param_name in ['weight_scale', 'weight_offset']:
                param = tensors[f"{name}.{param_name}"]
                assert param.dtype == torch.float, f"Expected float for {name}.{param_name}, got {param.dtype}"
                # History weight_scale.shape[0] = weight_tensor.shape[0] * weight_tensor.shape[1] / quant_config.group_size
                assert param.shape[0] == weight_tensor.shape[0] * weight_tensor.shape[1] // quant_config.group_size, (
                    f"Shape mismatch in {param_name}: {param.shape[0]} != {weight_tensor.shape[0] * weight_tensor.shape[1] // quant_config.group_size}"
                )
                assert param.shape[1] == 1, f"Shape mismatch in {param_name}: {param.shape[1]} != {1}"

            # Verify second stage quantization parameters should not exist
            assert f"{name}.weight_scale_second" not in tensors, (
                f"Unexpected second stage scale parameter found for {name}"
            )
            assert f"{name}.weight_offset_second" not in tensors, (
                f"Unexpected second stage offset parameter found for {name}"
            )

    msmodelslim_logger.info("\nWeight verification stats:")
    msmodelslim_logger.info(f"- Total weights examined: {verification_stats['total_weights']}")
    msmodelslim_logger.info(f"- Quantized weights: {verification_stats['quantized_weights']}")
    msmodelslim_logger.info(f"- Float weights: {verification_stats['float_weights']}")
    msmodelslim_logger.info("Weight verification completed successfully")


def verify_save_weights_ascendV1(quant_config):
    """
    Verify the quantized weights saved in AscendV1 format

    This function checks:
    - Data types of weights and scales
    - Value ranges for packed quantized weights
    - Shape relationships specific to AscendV1 format
    - Scale and bias parameters
    - First and second stage quantization parameters

    Args:
        quant_config: Configuration object containing quantization parameters

    Raises:
        AssertionError: If any verification check fails
    """
    msmodelslim_logger.info("Starting Weight Verification")

    # Load weight file
    weight_file = os.path.join(
        os.environ['PROJECT_PATH'],
        "output/llm_ptq_w4a16_pergroup",
        f"quant_model_weight_{quant_config.model_quant_type.lower()}.safetensors",
    )

    tensors = load_file(weight_file)
    verification_stats = {"total_weights": 0, "quantized_weights": 0, "float_weights": 0}

    for name, mod in model.named_modules():
        key = f"{name}.weight"

        if "norm" in name:
            continue

        if name in disable_names:
            verification_stats["float_weights"] += 1
            assert tensors[key].dtype == model.config.torch_dtype, (
                f"Expected {model.config.torch_dtype} for {key}, got {tensors[key].dtype}"
            )
            verification_stats["total_weights"] += 1
            continue

        if isinstance(mod, LowBitLinearQuantizer):
            verification_stats["quantized_weights"] += 1
            verification_stats["total_weights"] += 1

            # Verify weight properties
            weight_tensor = tensors[key]
            assert weight_tensor.dtype == torch.int8, f"Expected int8 for {key}, got {weight_tensor.dtype}"
            assert W4A8_MIN_PACK_VALUE <= weight_tensor.min() <= weight_tensor.max() <= W4A8_MAX_PACK_VALUE, (
                f"Values out of range for {key}: [{weight_tensor.min()}, {weight_tensor.max()}]"
            )
            assert weight_tensor.max() >= W4A8_MAX_VALUE, "Weight max value should be greater than W4A8_MAX_VALUE"

            # Verify first stage quantization parameters
            for param_name in ['weight_scale', 'weight_offset']:
                param = tensors[f"{name}.{param_name}"]
                assert param.dtype == torch.float, f"Expected float for {name}.{param_name}, got {param.dtype}"
                assert param.shape[0] == mod.weight.shape[0] * mod.weight.shape[1] // quant_config.group_size, (
                    f"Shape mismatch in {param_name}: {param.shape[0]} != "
                    f"{mod.weight.shape[0] * mod.weight.shape[1] // quant_config.group_size}"
                )
                assert param.shape[1] == 1, f"Shape mismatch in {param_name}: {param.shape[1]} != {1}"

            # Verify second stage quantization parameters should not exist
            assert f"{name}.weight_scale_second" not in tensors, (
                f"Unexpected second stage scale parameter found for {name}"
            )
            assert f"{name}.weight_offset_second" not in tensors, (
                f"Unexpected second stage offset parameter found for {name}"
            )

    msmodelslim_logger.info("\nWeight verification stats:")
    msmodelslim_logger.info(f"- Total weights examined: {verification_stats['total_weights']}")
    msmodelslim_logger.info(f"- Quantized weights: {verification_stats['quantized_weights']}")
    msmodelslim_logger.info(f"- Float weights: {verification_stats['float_weights']}")
    msmodelslim_logger.info("Weight verification completed successfully")


def get_calib_dataset(tokenizer, calib_list, device="cpu"):
    """
    Create calibration dataset from text inputs for model quantization

    This function:
    1. Takes a list of text samples
    2. Tokenizes each sample using the provided tokenizer
    3. Creates input tensors with attention masks
    4. Moves the tensors to the specified device

    Args:
        tokenizer: Tokenizer for processing text inputs
        calib_list: List of text samples for calibration
        device: Target device for the tensors (default: "cpu")

    Returns:
        List of lists, where each inner list contains:
            - input_ids tensor
            - attention_mask tensor
        Both tensors are moved to the specified device
    """
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer(calib_data, return_tensors='pt')
        calib_dataset.append([inputs.data['input_ids'].to(device), inputs.data['attention_mask'].to(device)])
    return calib_dataset


# Set up model path and configuration
model_path = f"{os.environ['PROJECT_PATH']}/resource/llm_ptq/Qwen2.5-7B-Instruct/"

# Initialize tokenizer and model configuration
msmodelslim_logger.info("Initializing Model and Tokenizer")
config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model_path, trust_remote_code=True, local_files_only=True
)
config.num_hidden_layers = 28  # Set number of transformer layers (default is 28)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    config=config,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype="auto",
    device_map="auto",
).eval()

# Prepare calibration dataset with diverse text samples
msmodelslim_logger.info("Preparing Calibration Dataset")
calib_set = [
    "Where is the capital of China?",
    "Please make a poem:",
    "I want to learn python, how should I learn it?",
    "Please help me write a job report on large model inference optimization:",
    "What are the most worth visiting scenic spots in China?",
]
dataset_calib = get_calib_dataset(tokenizer, calib_set, model.device)

# Configure anti-outlier processing for better quantization stability
msmodelslim_logger.info("Configuring Anti-outlier Processing")
anti_config = AntiOutlierConfig(
    w_bit=4,  # Weight quantization bits
    a_bit=16,  # Activation quantization bits
    anti_method="m3",  # Anti-outlier method
    dev_type="npu",  # Target device type
    dev_id=model.device.index,
)

anti_outlier = AntiOutlier(model, calib_data=dataset_calib, cfg=anti_config)
anti_outlier.process()

# Configure layers to exclude from quantization
# Typically we keep certain critical layers in full precision
disable_names = []
for i in range(config.num_hidden_layers):
    disable_names.append(f"model.layers.{i}.mlp.down_proj")
disable_names.append("lm_head")

# Configure quantization parameters for W4A8 dynamic quantization
msmodelslim_logger.info("Setting up Quantization Configuration")
quant_config = QuantConfig(
    a_bit=16,  # Activation bits
    w_bit=4,  # Weight bits
    w_sym=True,  # Symmetric weight quantization
    dev_type='npu',  # Target device type
    dev_id=model.device.index,
    is_lowbit=True,  # Enable low-bit quantization
    mm_tensor=False,  # Disable tensor memory management
    group_size=128,  # Group size for weight quantization
    open_outlier=False,  # Disable outlier handling
    disable_names=disable_names,
)

# Verify quantization type matches W4A8_DYNAMIC
assert quant_config.model_quant_type == QuantType.W4A16, "Incorrect quantization type configured"

# Perform model quantization using calibration data
msmodelslim_logger.info("Executing Model Quantization")
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()

# Test quantized model with a sample prompt
msmodelslim_logger.info("Testing Quantized Model")
SEQ_LEN_OUT = 32  # Maximum number of tokens to generate
test_prompt = "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:"
test_input = tokenizer(test_prompt, return_tensors="pt")
msmodelslim_logger.info("Model inference in progress...")
generate_ids = model.generate(
    test_input.input_ids.to(model.device),
    attention_mask=test_input.attention_mask.to(model.device),
    max_new_tokens=SEQ_LEN_OUT,
)

# Display generation results
msmodelslim_logger.info("Generation Results")
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
msmodelslim_logger.info("Single Round Dialogue Result")
for _, item in enumerate(res):
    msmodelslim_logger.info(item)

# Save quantized model in SafeTensors format
msmodelslim_logger.info("Saving Quantized Model")
save_path = f"{os.environ['PROJECT_PATH']}/output/llm_ptq_w4a16_pergroup"
calibrator.save(save_path, save_type=["safe_tensor"], part_file_size=None)

msmodelslim_logger.info('Quantized model saved in SafeTensors format successfully!')

# Run comprehensive verification for SafeTensors format
verify_save_files_name_and_num_safetensors(quant_config)
verify_save_description_safetensors(model, quant_config, disable_names)
verify_save_weights_safetensors(quant_config)

# Save and verify model in AscendV1 format
# Clean up existing files
os.system(f"rm -rf {save_path}")

# Save model in AscendV1 format
msmodelslim_logger.info("Saving Model in AscendV1 Format")
calibrator.save(save_path, save_type=["ascendV1"], part_file_size=None)
msmodelslim_logger.info('Quantized model saved in AscendV1 format successfully!')

# Run comprehensive verification for AscendV1 format
verify_save_files_name_and_num_ascendV1(quant_config)
verify_save_description_ascendV1(model, quant_config, disable_names)
verify_save_weights_ascendV1(quant_config)
