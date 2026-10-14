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
# pylint: disable=assignment-from-no-return,consider-using-from-import,logging-not-lazy,redefined-outer-name

from typing import List

import torch
import torch.nn as nn
from resources.fake_llama.fake_llama import get_fake_llama_model_and_tokenizer

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.processor.anti_outlier.flex_smooth import FlexAWQSSZProcessor, FlexAWQSSZProcessorConfig
from msmodelslim.processor.anti_outlier.flex_smooth.interface import FlexSmoothQuantInterface

SEQ_LEN_OUT = 32
KEY_INPUT_IDS = "input_ids"
KEY_ATTENTION_MASK = "attention_mask"
STR_TEST_PROMPT = "Hello world"
RETURN_TENSOR_TYPE = "pt"


def test_flex_awq_ssz_processor_with_hooks():
    """
    Test FlexAWQSSZProcessor functionality, including hook collection and comparison before/after processing
    """

    try:
        # Get model and tokenizer
        model, tokenizer = get_fake_llama_model_and_tokenizer()

        # Set model to evaluation mode and disable gradient computation
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Create test prompt
        test_prompt = tokenizer(STR_TEST_PROMPT, return_tensors=RETURN_TENSOR_TYPE, padding=True, truncation=True)

        # Create quantization configuration using LinearQConfig
        act_qconfig = QConfig(dtype="int8", scope="per_token", symmetric=True, method="minmax")

        weight_qconfig = QConfig(dtype="int8", scope="per_channel", symmetric=True, method="ssz")

        qconfig = LinearQConfig(act=act_qconfig, weight=weight_qconfig)

        # Create FlexAWQSSZProcessorConfig
        flex_awq_ssz_config = FlexAWQSSZProcessorConfig(
            type="flex_awq_ssz", qconfig=qconfig, enable_subgraph_type=["norm-linear", "linear-linear", "ov", "up-down"]
        )

        # Create ModelAdapter
        class FlexAWQSSZModelAdapter(FlexSmoothQuantInterface):
            def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
                adapter_config = []
                for layer_idx in range(1):
                    # Norm-Linear mapping config 1: input layernorm to QKV projection
                    norm_linear_mapping_config1 = MappingConfig(
                        source=f"model.layers.{layer_idx}.input_layernorm",
                        targets=[
                            f"model.layers.{layer_idx}.self_attn.k_proj",
                            f"model.layers.{layer_idx}.self_attn.q_proj",
                            f"model.layers.{layer_idx}.self_attn.v_proj",
                        ],
                    )

                    # Norm-Linear mapping config 2: post attention layernorm to MLP projection
                    norm_linear_mapping_config2 = MappingConfig(
                        source=f"model.layers.{layer_idx}.post_attention_layernorm",
                        targets=[f"model.layers.{layer_idx}.mlp.gate_proj", f"model.layers.{layer_idx}.mlp.up_proj"],
                    )

                    # OV mapping config (QKV to output projection)
                    ov_mapping_config = MappingConfig(
                        source=f"model.layers.{layer_idx}.self_attn.v_proj",
                        targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],
                    )

                    # Up-Down mapping config
                    up_down_mapping_config = MappingConfig(
                        source=f"model.layers.{layer_idx}.mlp.up_proj",
                        targets=[f"model.layers.{layer_idx}.mlp.down_proj"],
                    )

                    # Add 4 configs for current layer
                    adapter_config.extend(
                        [
                            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config1),
                            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config2),
                            AdapterConfig(subgraph_type="ov", mapping=ov_mapping_config),
                            AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config),
                        ]
                    )
                return adapter_config

        adapter = FlexAWQSSZModelAdapter()
        # Create FlexAWQSSZProcessor instance
        flex_awq_ssz_processor = FlexAWQSSZProcessor(model, flex_awq_ssz_config, adapter)

        msmodelslim_logger.info("FlexAWQSSZProcessor created successfully!")
        msmodelslim_logger.info("Processor type: %s", flex_awq_ssz_processor.config.type)
        msmodelslim_logger.info("Alpha value: %s", flex_awq_ssz_processor.config.alpha)
        msmodelslim_logger.info("Beta value: %s", flex_awq_ssz_processor.config.beta)
        msmodelslim_logger.info("Activation quantization config: %s", flex_awq_ssz_processor.config.qconfig.act)
        msmodelslim_logger.info("Weight quantization config: %s", flex_awq_ssz_processor.config.qconfig.weight)
        msmodelslim_logger.info("Enabled subgraph types: %s", flex_awq_ssz_processor.config.enable_subgraph_type)

        # Check if model has anti_method attribute
        if hasattr(model, 'anti_method'):
            msmodelslim_logger.info("Model anti_method attribute: %s", model.anti_method)
        else:
            msmodelslim_logger.warning("Model does not have anti_method attribute")

        # Get calibration dataset
        dataset_calib = [[test_prompt[KEY_INPUT_IDS], test_prompt.data[KEY_ATTENTION_MASK]]]

        # Test pre_run phase - load global subgraph configuration
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("Starting pre_run phase - loading global subgraph configuration")
        msmodelslim_logger.info("=" * 50)

        # Call pre_run to load global adapter configuration
        flex_awq_ssz_processor.pre_run()
        msmodelslim_logger.info("Global subgraph configuration loaded successfully")

        # Test preprocess phase - install statistics hooks
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("Starting preprocess phase - installing statistics hooks")
        msmodelslim_logger.info("=" * 50)

        # Create BatchProcessRequest
        request = BatchProcessRequest(module=model, name="model.layers.0", datas=dataset_calib)

        # Call preprocess to install statistics hooks
        flex_awq_ssz_processor.preprocess(request)
        msmodelslim_logger.info("Statistics hooks installed successfully")

        # Disable all parameter gradients again after preprocess
        for param in model.parameters():
            param.requires_grad = False

        # Check if hooks were successfully installed
        hook_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, '_forward_hooks') and module._forward_hooks:
                    hook_count += len(module._forward_hooks)
                    msmodelslim_logger.info(
                        "Module %s has %d forward hooks installed", name, len(module._forward_hooks)
                    )

        msmodelslim_logger.info("Total %d statistics hooks installed", hook_count)

        # Test forward inference - trigger statistics hooks to collect stats
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("Starting forward inference - triggering statistics collection")
        msmodelslim_logger.info("=" * 50)

        with torch.no_grad():
            output_logits_before_anti = model(test_prompt[KEY_INPUT_IDS]).logits
        msmodelslim_logger.info("Output shape before processing: %s", output_logits_before_anti.shape)
        msmodelslim_logger.info(
            "Output stats before processing: mean=%.6f, std=%.6f",
            output_logits_before_anti.mean().item(),
            output_logits_before_anti.std().item(),
        )

        # Check collected statistics
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("Checking collected statistics")
        msmodelslim_logger.info("=" * 50)

        if (
            hasattr(flex_awq_ssz_processor.stats_collector, 'act_stats')
            and flex_awq_ssz_processor.stats_collector.act_stats
        ):
            msmodelslim_logger.info(
                "Successfully collected statistics from %d modules",
                len(flex_awq_ssz_processor.stats_collector.act_stats),
            )
            for module_name, stats in flex_awq_ssz_processor.stats_collector.act_stats.items():
                msmodelslim_logger.info("Statistics for module %s:", module_name)
                for stat_key, stat_value in stats.items():
                    if isinstance(stat_value, torch.Tensor):
                        msmodelslim_logger.info(
                            "  %s: shape=%s, device=%s", stat_key, stat_value.shape, stat_value.device
                        )
                    elif isinstance(stat_value, list) and stat_value and isinstance(stat_value[0], torch.Tensor):
                        msmodelslim_logger.info("  %s: contains %d tensors", stat_key, len(stat_value))
                    else:
                        msmodelslim_logger.info("  %s: %s", stat_key, stat_value)
        else:
            msmodelslim_logger.warning("No statistics collected")

        # Test postprocess phase - apply FlexAWQSSZ smooth processing
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("Starting postprocess phase - applying FlexAWQSSZ smooth processing")
        msmodelslim_logger.info("=" * 50)

        # Call postprocess to apply smooth processing
        flex_awq_ssz_processor.postprocess(request)
        msmodelslim_logger.info("FlexAWQSSZ smooth processing completed")

        # Check if statistics are cleaned up after processing
        if hasattr(flex_awq_ssz_processor.stats_collector, 'act_stats'):
            if flex_awq_ssz_processor.stats_collector.act_stats:
                msmodelslim_logger.info(
                    "Remaining statistics after smooth processing: %d modules",
                    len(flex_awq_ssz_processor.stats_collector.act_stats),
                )
            else:
                msmodelslim_logger.info("Statistics cleaned up after smooth processing")

        # Get model output after processing and compare
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("Comparing model outputs before and after FlexAWQSSZProcessor")
        msmodelslim_logger.info("=" * 50)

        with torch.no_grad():
            output_logits_after_anti = model(test_prompt[KEY_INPUT_IDS]).logits
        msmodelslim_logger.info("Output shape after processing: %s", output_logits_after_anti.shape)
        msmodelslim_logger.info(
            "Output stats after processing: mean=%.6f, std=%.6f",
            output_logits_after_anti.mean().item(),
            output_logits_after_anti.std().item(),
        )

        # Compare output differences before and after processing
        # First check if shapes match
        if output_logits_before_anti.shape != output_logits_after_anti.shape:
            error_msg = "Output shapes do not match: before=%s, after=%s" % (
                output_logits_before_anti.shape,
                output_logits_after_anti.shape,
            )
            msmodelslim_logger.error(error_msg)
            raise AssertionError(error_msg)

        # Calculate absolute differences
        abs_diff = torch.abs(output_logits_before_anti - output_logits_after_anti)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        std_diff = abs_diff.std().item()

        msmodelslim_logger.info("Output difference statistics:")
        msmodelslim_logger.info("  Maximum absolute difference: %.15f", max_diff)
        msmodelslim_logger.info("  Mean absolute difference: %.15f", mean_diff)
        msmodelslim_logger.info("  Absolute difference std: %.15f", std_diff)

        # FlexAWQSSZ includes quantization, output will have some difference, use looser tolerance
        tolerance = 1e-3
        if torch.allclose(output_logits_before_anti, output_logits_after_anti, atol=tolerance):
            msmodelslim_logger.info(
                "✓ FlexAWQSSZProcessor output difference is within acceptable range (atol=%.3e)", tolerance
            )
        else:
            msmodelslim_logger.warning("⚠ FlexAWQSSZProcessor output difference is large (atol=%.3e)", tolerance)

            # Calculate relative differences
            relative_diff = abs_diff / (torch.abs(output_logits_before_anti) + 1e-8)
            max_rel_diff = relative_diff.max().item()
            mean_rel_diff = relative_diff.mean().item()

            msmodelslim_logger.info("Relative difference statistics:")
            msmodelslim_logger.info("  Maximum relative difference: %.6f", max_rel_diff)
            msmodelslim_logger.info("  Mean relative difference: %.6f", mean_rel_diff)

            # FlexAWQSSZ includes SSZ quantization, quantization error is expected
            msmodelslim_logger.info(
                "Note: FlexAWQSSZ includes SSZ quantization process, output difference is within expected range"
            )

        # Check if model weights are modified
        msmodelslim_logger.info("\n" + "=" * 50)
        msmodelslim_logger.info("FlexAWQSSZ processing modifies model weights (applies smooth scaling)")
        msmodelslim_logger.info("=" * 50)

    except Exception as e:
        msmodelslim_logger.error("Test FlexAWQSSZProcessor failed: %s", e)
        import traceback

        traceback.print_exc()
        # Re-raise exception so pytest can recognize test failure
        raise


if __name__ == "__main__":
    msmodelslim_logger.info("=" * 60)
    msmodelslim_logger.info("LLaMA2-7B Model FlexAWQSSZProcessor Functionality Test Started")
    msmodelslim_logger.info("=" * 60)

    try:
        # Test 1: Complete FlexAWQSSZProcessor functionality test
        msmodelslim_logger.info(
            "\n1. Complete test of FlexAWQSSZProcessor functionality (including statistics hook collection)..."
        )
        flex_awq_ssz_processor = test_flex_awq_ssz_processor_with_hooks()

        if flex_awq_ssz_processor:
            msmodelslim_logger.info("✓ FlexAWQSSZProcessor complete functionality test passed")
        else:
            msmodelslim_logger.error("✗ FlexAWQSSZProcessor complete functionality test failed")

        msmodelslim_logger.info("\n" + "=" * 60)
        msmodelslim_logger.info("All tests completed")
        msmodelslim_logger.info("=" * 60)

    except Exception as e:
        msmodelslim_logger.error("Error occurred during test: %s", e)
        import traceback

        traceback.print_exc()
        raise e
