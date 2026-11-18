# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from __future__ import absolute_import, division, print_function
from dataclasses import dataclass
from typing import List

from ascend_utils.common.security import check_number, get_valid_read_path
from msmodelslim.pytorch.quant.qat_tools.common.config import Config

# onnx version
ONNX_VERSION_LIST = [11, 13]


@dataclass
class QatConfig(Config):

    def __init__(
            self,
            w_bit: int = 8,
            a_bit: int = 8,
            a_sym: bool = False,
            amp_num: int = 0,
            steps: int = 1,
            ema: float = 0.99,
            is_forward: bool = False,
            ignore_head_tail_node: bool = False,
            disable_names: List[str] = None,
            has_init_quant: bool = False,
            quant_mode: bool = True,
            grad_scale: float = 0.0,
            compressed_model_checkpoint: str = None,
            opset_version: int = 11,
            save_params: bool = False,
            input_names: List[str] = None,
            output_names: List[str] = None,
            save_onnx_name: str = None
    ):
        super().__init__(disable_names, input_names, output_names, save_onnx_name)
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.is_forward = is_forward
        self.grad_scale = grad_scale
        self.a_sym = a_sym
        self.amp_num = amp_num
        self.steps = steps
        self.ema = ema
        self.ignore_head_tail_node = ignore_head_tail_node
        self.compressed_model_checkpoint = compressed_model_checkpoint
        self.opset_version = opset_version
        self.has_init_quant = has_init_quant
        self.quant_mode = quant_mode
        self.save_params = save_params

        self.method = 'aqt'
        self.w_signed = True
        self.a_signed = False
        self.w_sym = True
        self.act_quant = True
        self.fold_bn = True
        self.fix_bn = True
        self.fuse_add = True

        if not isinstance(self.w_bit, int) or self.w_bit != 8:
            raise TypeError("w_bit must be 8, please check it.")
        if not isinstance(self.a_bit, int) or self.a_bit != 8:
            raise TypeError("a_bit must be 8, please check it.")
        check_number(self.amp_num, int, 0, 10, param_name="amp_num")
        if not isinstance(self.steps, int) or self.steps < 1:
            raise TypeError("steps should be int and more than 0 and less than total steps of one epoch, "
                            "please check it.")
        check_number(self.ema, float, 0.1, 1.0, param_name="ema")
        if self.opset_version not in ONNX_VERSION_LIST:
            raise ValueError("opset_version should be int and corresponding "
                             "with onnx version 11 or 13, please check it.")
        check_number(self.grad_scale, float, 0.0, 0.01, param_name="grad_scale")
        if self.compressed_model_checkpoint is not None and not isinstance(self.compressed_model_checkpoint, str):
            raise TypeError("compressed_model_checkpoint should be str, please check it.")
        self.compressed_model_checkpoint = get_valid_read_path(self.compressed_model_checkpoint)
        if not isinstance(self.a_sym, bool):
            raise TypeError("a_sym should be bool, please check it.")
        if not isinstance(self.is_forward, bool):
            raise TypeError("is_forward should be bool, please check it.")
        if not isinstance(self.ignore_head_tail_node, bool):
            raise TypeError("ignore_head_tail_node should be bool, please check it.")
        if not isinstance(self.has_init_quant, bool):
            raise TypeError("has_init_quant should be bool, please check it.")
        if not isinstance(self.quant_mode, bool):
            raise TypeError("quant_mode should be bool, please check it.")
        if not isinstance(self.save_params, bool):
            raise TypeError("save_params should be bool, please check it.")