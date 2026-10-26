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
# pylint: disable=redefined-outer-name,logging-not-lazy,assignment-from-no-return

from typing import List

import torch
from torch import nn
from resources.fake_llama.fake_llama import get_fake_llama_model_and_tokenizer

from msmodelslim import logger as msmodelslim_logger
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.processor.anti_outlier.flex_smooth import FlexAWQSSZProcessor, FlexAWQSSZProcessorConfig
from msmodelslim.processor.anti_outlier.flex_smooth.interface import FlexSmoothQuantInterface

KEY_INPUT_IDS = "input_ids"
KEY_ATTENTION_MASK = "attention_mask"
STR_TEST_PROMPT = "Hello world"
RETURN_TENSOR_TYPE = "pt"


class _FlexAWQSSZModelAdapter(FlexSmoothQuantInterface):
    """为 FlexAWQSSZProcessor 提供 4 类子图映射的模型 adapter。"""

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config = []
        for layer_idx in range(1):
            norm_linear_mapping_config1 = MappingConfig(
                source=f"model.layers.{layer_idx}.input_layernorm",
                targets=[
                    f"model.layers.{layer_idx}.self_attn.k_proj",
                    f"model.layers.{layer_idx}.self_attn.q_proj",
                    f"model.layers.{layer_idx}.self_attn.v_proj",
                ],
            )

            norm_linear_mapping_config2 = MappingConfig(
                source=f"model.layers.{layer_idx}.post_attention_layernorm",
                targets=[f"model.layers.{layer_idx}.mlp.gate_proj", f"model.layers.{layer_idx}.mlp.up_proj"],
            )

            ov_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.self_attn.v_proj",
                targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],
            )

            up_down_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.mlp.up_proj",
                targets=[f"model.layers.{layer_idx}.mlp.down_proj"],
            )

            adapter_config.extend(
                [
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config1),
                    AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config2),
                    AdapterConfig(subgraph_type="ov", mapping=ov_mapping_config),
                    AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config),
                ]
            )
        return adapter_config


def _build_flex_awq_ssz_processor_with_hooks():
    """构建一个已完成 pre_run/preprocess、已安装统计 hook 的 FlexAWQSSZProcessor。"""
    model, tokenizer = get_fake_llama_model_and_tokenizer()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    test_prompt = tokenizer(STR_TEST_PROMPT, return_tensors=RETURN_TENSOR_TYPE, padding=True, truncation=True)

    act_qconfig = QConfig(dtype="int8", scope="per_token", symmetric=True, method="minmax")
    weight_qconfig = QConfig(dtype="int8", scope="per_channel", symmetric=True, method="ssz")
    qconfig = LinearQConfig(act=act_qconfig, weight=weight_qconfig)

    flex_awq_ssz_config = FlexAWQSSZProcessorConfig(
        type="flex_awq_ssz", qconfig=qconfig, enable_subgraph_type=["norm-linear", "linear-linear", "ov", "up-down"]
    )

    adapter = _FlexAWQSSZModelAdapter()
    flex_awq_ssz_processor = FlexAWQSSZProcessor(model, flex_awq_ssz_config, adapter)

    dataset_calib = [[test_prompt[KEY_INPUT_IDS], test_prompt.data[KEY_ATTENTION_MASK]]]

    flex_awq_ssz_processor.pre_run()
    request = BatchProcessRequest(module=model, name="model.layers.0", datas=dataset_calib)
    flex_awq_ssz_processor.preprocess(request)

    for param in model.parameters():
        param.requires_grad = False

    return model, test_prompt, flex_awq_ssz_processor, request


def test_flex_awq_ssz_preprocess_installs_stats_hooks_when_enabled():
    """给定启用 norm-linear/ov/up-down 子图，期望 preprocess 为线性层安装 4 个统计 hook。"""
    model, _, processor, _ = _build_flex_awq_ssz_processor_with_hooks()

    hook_count = 0
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook_count += len(module._forward_hooks)

    assert hook_count == 4
    assert len(processor.adapter_config) == 4


def test_flex_awq_ssz_forward_collects_stats_when_hooks_installed():
    """给定已安装统计 hook，期望前向推理后收集到 4 个模块的激活统计。"""
    model, test_prompt, processor, _ = _build_flex_awq_ssz_processor_with_hooks()

    with torch.no_grad():
        model(test_prompt[KEY_INPUT_IDS])

    assert len(processor.stats_collector.act_stats) == 4


def test_flex_awq_ssz_postprocess_clears_stats_and_keeps_logits_when_applied():
    """给定已收集统计信息，期望 postprocess 清空统计且输出与处理前近似一致。"""
    model, test_prompt, processor, request = _build_flex_awq_ssz_processor_with_hooks()

    with torch.no_grad():
        logits_before = model(test_prompt[KEY_INPUT_IDS]).logits

    assert len(processor.stats_collector.act_stats) > 0

    processor.postprocess(request)

    assert len(processor.stats_collector.act_stats) == 0

    with torch.no_grad():
        logits_after = model(test_prompt[KEY_INPUT_IDS]).logits

    assert logits_before.shape == logits_after.shape
    assert torch.allclose(logits_before, logits_after, atol=1e-3)


if __name__ == "__main__":
    msmodelslim_logger.info("=" * 60)
    msmodelslim_logger.info("LLaMA2-7B Model FlexAWQSSZProcessor Functionality Test Started")
    msmodelslim_logger.info("=" * 60)

    test_flex_awq_ssz_preprocess_installs_stats_hooks_when_enabled()
    test_flex_awq_ssz_forward_collects_stats_when_hooks_installed()
    test_flex_awq_ssz_postprocess_clears_stats_and_keeps_logits_when_applied()

    msmodelslim_logger.info("\n" + "=" * 60)
    msmodelslim_logger.info("All tests completed")
    msmodelslim_logger.info("=" * 60)
