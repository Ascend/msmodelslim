# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import os
import re
import sys
import copy

from collections import defaultdict
import numpy as np
from tqdm import tqdm

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Parameter
from mindformers.modules.layers import Linear
from mindformers.models.modeling_utils import PreTrainedModel

from ascend_utils.common.security import (get_valid_write_path, get_valid_read_path, check_element_type,
    check_type, check_dict_element, get_write_directory, SafeWriteUmask)

from msmodelslim.mindspore.llm_ptq.quant_funcs_ms import quant_one_weight_by_outliers
from msmodelslim.mindspore.llm_ptq.quant_modules import LinearSparseQuantizer
from msmodelslim import logger as msmodelslim_logger
from msmodelslim.mindspore.llm_ptq.quant_config import QuantConfig


MAX_READ_FILE_SIZE_20G = 20 * 1024 * 1024 * 1024
MAX_RECURSIVE_DEPTH = 100  # 最大递归深度限制


class Calibrator(object):
    def __init__(self,
                 cfg: QuantConfig,
                 model,
                 model_ckpt,
                 calib_data=None) -> None:
        # validation check
        check_type(cfg, QuantConfig, param_name="cfg")
        check_type(model, PreTrainedModel, param_name="model")
        check_type(model_ckpt, str, param_name="model_ckpt")

        # initialization
        self.cfg = cfg
        self.logger = msmodelslim_logger
        self.original_model = copy.deepcopy(model)
        self.model_ckpt = get_valid_read_path(model_ckpt, size_max=MAX_READ_FILE_SIZE_20G)

        self.calib_data = calib_data if calib_data else []
        check_type(self.calib_data, list, param_name="calib_data")
        self.rollback_names = self.cfg.disable_names
        self.quantized_weight_dict = {}

        linear_weight_pattern = r"model\.layers\.\d+\.(attention[^_]|feed_forward|augs_attn\d+).*"
        self.linear_reg = re.compile(linear_weight_pattern)
        
        # quantize model
        try:
            self.model = self.quantize_model(model)
        except Exception as e:
            raise Exception("Quantize model Failed. Please check the model. ", e) from e

    
    def quantize_model(self, model, is_fake_quant=False):
        def _convert(modules, depth=0):
            # check recursive depth limit
            if depth > MAX_RECURSIVE_DEPTH:
                self.logger.error(
                    f"Recursive depth exceeds the maximum limit {MAX_RECURSIVE_DEPTH}, "
                    "possible circular reference or model structure too deep"
                )
                raise RecursionError(f"Recursive depth exceeds the maximum limit {MAX_RECURSIVE_DEPTH}")
            
            keys = list(modules.keys())
            for k in keys:
                if isinstance(modules[k], Linear):
                    name = modules[k].weight.name[:-7]
                    if name in self.rollback_names:
                        self.logger.info(f"name: {name} in disable_names will not be quantized.")
                        continue
                    modules[k] = LinearSparseQuantizer(cell=modules[k], is_fake_quant=is_fake_quant)
                else:
                    _convert(modules[k]._cells, depth + 1)

        self.logger.info("Start quantizing the model...")
        _convert(model._cells)
        model.update_parameters_name()
        self.logger.info("Quantize the model Finished.")

        return model


    def run(self):
        # 1. First Stage  -- weight quantization
        self.logger.info("Start weight quantization...")
        self.quantized_weight_dict = self.run_weight_quantization()
        self.logger.info("Weight quantization Finished. ")

        # 2. Second Stage -- activation quantization
        param_not_load = ms.load_param_into_net(self.model, self.quantized_weight_dict, strict_load=False)

        for data in tqdm(self.calib_data):
            try:
                self.model.generate(data, max_new_tokens=1, topk=1)
            except Exception as e:
                raise Exception("Run model.generate Failed. Please check the model. ", e) from e
        
        self.logger.info("Calibration End.")

    
    def fake_quantize_model(self):

        if not self.quantized_weight_dict:
            self.logger.warning("Weights are not quantized. It will be quantized automatically for further use.")
            self.quantized_weight_dict = self.run_weight_quantization()

        fake_quant_model = self.quantize_model(self.original_model, is_fake_quant=True)
        
        quantized_model_dict = {
            k: v
            for k, v in self.model.parameters_dict().items()
            if "key_cache" not in k and "value_cache" not in k
        }
        param_not_load = ms.load_param_into_net(fake_quant_model, quantized_model_dict, strict_load=False)

        return fake_quant_model

    def run_weight_quantization(self):

        quantized_param_dict = {}
        param_dict = ms.load_checkpoint(self.model_ckpt)

        for key in tqdm(param_dict):
            if self.linear_reg.search(key):
                if key[:-7] in self.rollback_names or key[:-5] in self.rollback_names:
                    quantized_param_dict[key] = param_dict[key]
                elif key.endswith("weight"):
                    recovered_weight, weight_scale, quant_weight, _ = quant_one_weight_by_outliers(param_dict[key],
                                                                                powerquant=False,
                                                                                fraction=self.cfg.fraction,
                                                                                num_bits=4,
                                                                                per_channel=True)
                    quantized_param_dict[f"{key[:-7]}.cell.weight"] = Parameter(recovered_weight, name="weight")
                    quantized_param_dict[f"{key[:-7]}.quant_weight"] = Parameter(quant_weight, name="quant_weight")
                    quantized_param_dict[f"{key[:-7]}.weight_scale"] = Parameter(weight_scale, name="weight_scale")
                elif key.endswith("bias"):
                    quantized_param_dict[f"{key[:-5]}.cell.bias"] = param_dict[key]
            else:
                quantized_param_dict[key] = param_dict[key]

        return quantized_param_dict


    def save(self, output_path):
        check_type(output_path, str, param_name="output_path")
        save_valid_path = get_valid_write_path(output_path, extensions=[".ckpt"])

        fp_ckpt_dict = ms.load_checkpoint(self.model_ckpt)
        quantized_weight_dict = self.quantized_weight_dict
        quantized_model_param_dict = self.model.parameters_dict()
        quantized_param_dict = dict()
        
        for key in tqdm(fp_ckpt_dict):
            if self.linear_reg.search(key):
                if key[:-7] in self.rollback_names or key[:-5] in self.rollback_names:
                    quantized_param_dict[key] = fp_ckpt_dict[key]
                    continue
                elif key.endswith("bias"):
                    cell_name = key[:-5]
                    fp_bias = fp_ckpt_dict[key]
                    bias_shape = fp_bias.shape
                else:
                    cell_name = key[:-7]
                    fp_bias = None
                    
                fp_weight = fp_ckpt_dict[f"{cell_name}.weight"]
                quant_weight = quantized_weight_dict.get(f"{cell_name}.quant_weight", None)
                weight_scale = quantized_weight_dict.get(f"{cell_name}.weight_scale", None)
                input_scale = quantized_model_param_dict[f"{cell_name}.quant_input.input_scale"].astype(ms.float16)
                input_offset = quantized_model_param_dict[f"{cell_name}.quant_input.input_offset"].astype(ms.int8)

                deq_scale = self.deqscale_process(input_scale, weight_scale).astype(ms.float32)
                correction = (quant_weight.astype(ms.float32).sum(axis=1) * input_offset.astype(ms.float32))
                fp_bias = self.change_bias(fp_weight, fp_bias)
                quant_bias = P.round(fp_bias / deq_scale - correction).astype(ms.int32)
                deq_scale = self.deqscale2int64(deq_scale).astype(ms.int64)

                quantized_param_dict[f"{cell_name}.weight"] = Parameter(quant_weight, name="weight")
                quantized_param_dict[f"{cell_name}.input_scale"] = Parameter(input_scale, name="input_scale")
                quantized_param_dict[f"{cell_name}.input_offset"] = Parameter(input_offset, name="input_offset")
                quantized_param_dict[f"{cell_name}.quant_bias"] = Parameter(quant_bias, name="quant_bias")
                quantized_param_dict[f"{cell_name}.deq_scale"] = Parameter(deq_scale, name="deq_scale")
            else:
                quantized_param_dict[key] = fp_ckpt_dict[key]

        with SafeWriteUmask(umask=0o377):
            ms.save_checkpoint(quantized_param_dict, save_valid_path)
        self.logger.info(f"Save sparse ckpt to `{save_valid_path}` Successfully !")


    def deqscale_process(self, input_scale, scale):
        deq_scale = input_scale * scale
        if deq_scale.ndim > 1:
            deq_scale = deq_scale.squeeze(1)
        return deq_scale


    def change_bias(self, fp_weight, bias=None):
        if isinstance(bias, Parameter):
            fp_bias = bias
        else:
            bias_shape = fp_weight.shape[0]
            fp_bias = P.zeros(bias_shape)
        return fp_bias


    def deqscale2int64(self, scale):
        scale = scale.numpy()
        scale = np.frombuffer(scale.tobytes(), dtype=np.int32).astype(np.int64)
        scale = ms.tensor(scale)
        return scale
