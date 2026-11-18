# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from __future__ import absolute_import, division, print_function
from dataclasses import dataclass
from typing import List

from ascend_utils.common.security import check_element_type


@dataclass
class Config:
    def __init__(
            self,
            disable_names: List[str] = None,
            input_names: List[str] = None,
            output_names: List[str] = None,
            save_onnx_name: str = None
    ):
        self.disable_names = disable_names
        self.input_names = input_names
        self.output_names = output_names
        self.save_onnx_name = save_onnx_name

        if self.save_onnx_name is not None and not isinstance(self.save_onnx_name, str):
            raise TypeError("save_onnx_name should be str, please check it.")

        if self.disable_names is not None:
            check_element_type(self.disable_names, str, list, param_name="disable_names")

        if self.input_names is not None:
            check_element_type(self.input_names, str, list, param_name="input_names")

        if self.output_names is not None:
            check_element_type(self.output_names, str, list, param_name="output_names")
