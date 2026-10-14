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

import pytest

import torch

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType, QuantModelJsonDescription


class TestQuantType:
    def test_get_quant_type(self):
        params = {
            'w_bit': 8,
            'a_bit': 8,
            'w_method': None,  # 根据实际状况填写
            'is_sparse': False,
            'is_dynamic': False,
            'is_lowbit': False,
        }
        quant_type = QuantType.get_quant_type(params)
        assert quant_type is QuantType.W8A8

        params = {
            'w_bit': 8,
            'a_bit': 8,
            'w_method': None,  # 根据实际状况填写
            'is_sparse': False,
            'is_dynamic': True,
            'is_lowbit': False,
        }
        quant_type = QuantType.get_quant_type(params)
        assert quant_type is QuantType.W8A8_DYNAMIC

        params = {
            'w_bit': 8,
            'a_bit': 8,
            'w_method': None,  # 根据实际状况填写
            'is_sparse': True,
            'is_dynamic': False,
            'is_lowbit': False,
        }
        quant_type = QuantType.get_quant_type(params)
        assert quant_type is QuantType.W8A8S

        params = {
            'w_bit': 4,
            'a_bit': 8,
            'w_method': None,  # 根据实际状况填写
            'is_sparse': False,
            'is_dynamic': False,
            'is_lowbit': True,
        }
        quant_type = QuantType.get_quant_type(params)
        assert quant_type is QuantType.W8A8S

        params = {
            'w_bit': 8,
            'a_bit': 16,
            'w_method': None,  # 根据实际状况填写
            'is_sparse': False,
            'is_dynamic': False,
            'is_lowbit': False,
        }
        quant_type = QuantType.get_quant_type(params)
        assert quant_type is QuantType.W8A16

        params = {
            'w_bit': 4,
            'a_bit': 16,
            'w_method': "NF4",  # 根据实际状况填写
            'is_sparse': False,
            'is_dynamic': False,
            'is_lowbit': False,
        }
        quant_type = QuantType.get_quant_type(params)
        assert quant_type is QuantType.NF4

        params = {
            'w_bit': 4,
            'a_bit': 16,
            'w_method': "None",  # 根据实际状况填写
            'is_sparse': False,
            'is_dynamic': False,
            'is_lowbit': False,
        }
        quant_type = QuantType.get_quant_type(params)
        assert quant_type is QuantType.W4A16

        params = {
            'w_bit': 16,
            'a_bit': 16,
            'w_method': None,  # 根据实际状况填写
            'is_sparse': False,
            'is_dynamic': False,
            'is_lowbit': False,
        }
        quant_type = QuantType.get_quant_type(params)
        assert quant_type is QuantType.FLOAT

        params = {
            'w_bit': 1,
            'a_bit': 8,
            'w_method': None,  # 根据实际状况填写
            'is_sparse': False,
            'is_dynamic': False,
            'is_lowbit': False,
        }
        quant_type = QuantType.get_quant_type(params)
        assert quant_type is QuantType.UNKNOWN

    def test_is_value_in_enum_should_return_true_when_quant_type_value_is_right(self):
        result = QuantType.is_value_in_enum("W8A16")
        assert result is True

    def test_is_value_in_enum_should_return_false_when_quant_type_value_is_not_right(self):
        result = QuantType.is_value_in_enum("W0A0")
        assert result is False

    def test_check_instance_of_enum_raise_type_error_when_invalid_quant_type(self):
        with pytest.raises(ValueError):
            QuantType.check_datafree_quant_type("string_type")
        with pytest.raises(ValueError):
            QuantType.check_datafree_quant_type(1)

    def test_check_datafree_quant_type_raise_value_error_when_invalid_datafree_type(self):
        with pytest.raises(ValueError):
            QuantType.check_datafree_quant_type("string_type")
        with pytest.raises(ValueError):
            QuantType.check_datafree_quant_type(1)


class TestQuantModelJsonDescription:
    dummy_input_shape = (1, 1)

    def test_check_description_with_right_format_json_description(self):
        right_format_json_description = {"model_quant_type": "W8A8", "model.embed_tokens.weight": "FLOAT"}
        result = QuantModelJsonDescription.check_description(right_format_json_description)
        assert right_format_json_description == result

    def test_check_description_with_no_args_input(self):
        with pytest.raises(TypeError):
            QuantModelJsonDescription.check_description()

    def test_check_description_with_wrong_format_json_description(self):
        wrong_data_type_json_description = [("model_quant_type", "W8A16"), ("model.layers.0.self_attn.q_proj", "FLOAT")]
        with pytest.raises(TypeError):
            QuantModelJsonDescription.check_description(wrong_data_type_json_description)

        wrong_weight_name_type_json_description = {1: "W4A16", 2: "W4A16"}
        with pytest.raises(TypeError):
            QuantModelJsonDescription.check_description(wrong_weight_name_type_json_description)

        wrong_weight_type_json_description = {"model_quant_type": "W0A0", "model.layers.0.self_attn.k_proj": "COMPLEX"}
        with pytest.raises(TypeError):
            QuantModelJsonDescription.check_description(wrong_weight_type_json_description)

        json_description_without_model_quant_type_name = {"model.layers.0.self_attn.v_proj": "W8A8S"}
        with pytest.raises(ValueError):
            QuantModelJsonDescription.check_description(json_description_without_model_quant_type_name)

    def test_check_safetensor_with_right_format_safetensor(self):
        right_format_json_safetensor = {"model.embed_tokens.weight": torch.rand(self.dummy_input_shape)}
        result = QuantModelJsonDescription.check_safetensor(right_format_json_safetensor)
        assert right_format_json_safetensor == result

    def test_check_safetensor_with_no_args_input(self):
        with pytest.raises(TypeError):
            QuantModelJsonDescription.check_safetensor()

    def test_check_safetensor_with_wrong_format_safetensor(self):
        wrong_data_type_safetensor = [("model.layers.0.self_attn.q_proj", torch.rand(self.dummy_input_shape))]
        with pytest.raises(TypeError):
            QuantModelJsonDescription.check_safetensor(wrong_data_type_safetensor)

        wrong_weight_name_type_safetensor = {1: torch.rand(self.dummy_input_shape)}
        with pytest.raises(TypeError):
            QuantModelJsonDescription.check_safetensor(wrong_weight_name_type_safetensor)

        wrong_weight_type_json_safetensor = {"model.layers.0.self_attn.k_proj": "COMPLEX"}
        with pytest.raises(TypeError):
            QuantModelJsonDescription.check_safetensor(wrong_weight_type_json_safetensor)

    def test_check_description_match_with_right_format(self):
        right_format_json_description = {"model_quant_type": "W8A8SC", "model.layers.0.self_attn.v_proj": "W8A8SC"}
        right_format_json_safetensor = {"model.layers.0.self_attn.v_proj": torch.rand(self.dummy_input_shape)}
        QuantModelJsonDescription.check_description_match(
            quant_model_json_description=right_format_json_description,
            quant_model_safetensor=right_format_json_safetensor,
        )

    def test_check_description_match_with_different_keys(self):
        right_format_json_description = {"model_quant_type": "W8A8", "model.layers.0.self_attn.o_proj": "W8A8"}
        right_format_json_safetensor = {"model.layers.0.mlp.gate_proj": torch.rand(self.dummy_input_shape)}
        with pytest.raises(ValueError):
            QuantModelJsonDescription.check_description_match(
                quant_model_json_description=right_format_json_description,
                quant_model_safetensor=right_format_json_safetensor,
            )
