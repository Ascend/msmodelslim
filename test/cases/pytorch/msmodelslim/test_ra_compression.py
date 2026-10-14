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
import stat

import torch
import pytest
from transformers import AutoConfig, AutoModelForCausalLM

from ascend_utils.common.security import file_safe_write, safe_delete_path_if_exists
from msmodelslim.pytorch import ra_compression  # noqa: F401
from msmodelslim.pytorch.ra_compression import RACompressConfig

torch.manual_seed(2024)


@pytest.fixture(scope="module", autouse=True)
def model_config():
    model_config_path = "./config.json"
    config = """{
      "architectures": [
        "LlamaForCausalLM"
      ],
      "bos_token_id": 1,
      "eos_token_id": 2,
      "hidden_act": "silu",
      "hidden_size": 1024,
      "initializer_range": 0.02,
      "intermediate_size": 4096,
      "max_position_embeddings": 4096,
      "model_type": "llama",
      "num_attention_heads": 4,
      "num_hidden_layers": 4,
      "num_key_value_heads": 4,
      "pad_token_id": 0,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "tie_word_embeddings": false,
      "torch_dtype": "float16",
      "transformers_version": "4.31.0.dev0",
      "use_cache": true,
      "vocab_size": 32000
    }"""
    file_safe_write(config, model_config_path)
    os.chmod(model_config_path, stat.S_IRUSR | stat.S_IWUSR)
    yield
    safe_delete_path_if_exists(model_config_path)


# @pytest.mark.skipif("RACompressor" not in ra_compression.__all__, reason="requires KIA so")
@pytest.mark.skip()
def test_ra_compression_given_model_then_pass():
    from msmodelslim.pytorch.ra_compression import RACompressor

    config = AutoConfig.from_pretrained("./", local_files_only=True)
    model = AutoModelForCausalLM.from_config(config)
    raconfig = RACompressConfig(theta=0.00001, alpha=100)
    output_model_path = "./win.pt"
    ra = RACompressor(model, raconfig)
    ra.get_alibi_windows(output_model_path)
    loaded_tensor = torch.load(output_model_path)
    target_tensor = torch.tensor([354, 1418, 5675, 22698], dtype=torch.int64)
    assert torch.equal(loaded_tensor[:2], target_tensor[:2])
    if os.path.exists(output_model_path):
        os.remove(output_model_path)
