#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

from __future__ import absolute_import, division, print_function
import os
import copy

import torch
import torch.nn as nn
import onnx
import numpy as np
from tqdm import tqdm

from ascend_utils.common import security
from ascend_utils.common.security import check_type, get_valid_write_path, SafeWriteUmask, get_write_directory,  \
    get_valid_read_path, safe_delete_path_if_exists, json_safe_dump
from ascend_utils.common.security.pytorch import check_torch_module
from msmodelslim.pytorch.quant.ptq_tools.quant_modules import Quantizer, Conv2dQuantizer, LinearQuantizer
from msmodelslim.pytorch.quant.ptq_tools.quant_deploy import quantize_model_deploy, convert_linear_params
from msmodelslim.pytorch.quant.ptq_tools.quant_deploy import ConvertLinearParams, ModelDeployQuantParams
from msmodelslim.pytorch.quant.ptq_tools.ptq_kia.quant_funcs import amp_decision, fake_quantize  # squant algorithm api
from msmodelslim import logger
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig
from msmodelslim.pytorch.quant.ptq_tools.quant_modules import TensorQuantizer

WEIGHT_TYPE = "W8A8"
FLOAT_TYPE = "FLOAT"
CHECK_DTYPE = {'added_cond_kwargs': dict, 'return_dict': bool, 't_idx': int}


class Calibrator(object):
    """ Calibrator for post-training quantization."""

    def __init__(self, model, cfg, calib_data=None, fuse_module_call_back=None):
        check_type(cfg, QuantConfig, param_name="cfg")
        check_torch_module(model)
        self.cfg = cfg
        if calib_data is None:
            calib_data = []
        self.calib_data = self.get_calib_data(calib_data)
        if not isinstance(calib_data, list):
            raise ValueError("calib_data should be a list of tensor data")

        if fuse_module_call_back is not None and callable(fuse_module_call_back):
            fuse_module_call_back(model)
        else:
            fuse_module(model)

        self.fp_model = copy.deepcopy(model)

        self.ori_fp_weight = {}
        for key, value in self.fp_model.state_dict().items():
            if not isinstance(value, torch.Tensor):
                continue
            self.ori_fp_weight[key] = value

        self.model = self.quantize(model)
        self.set_quant()

    def quantize(self, model):
        return quantize_model(model, cfg=self.cfg)

    def set_quant(self):
        self.enable_quant()
        if not self.cfg.act_quant:
            self.disable_input_quant()

        if not self.cfg.disable_names:
            set_first_last_layer(self.model)
        else:
            disable_quantization(self.model, self.cfg.disable_names)

    def enable_quant(self):
        enable_quantization(self.model)

    def disable_quant(self):
        disable_quantization(self.model)

    def disable_input_quant(self):
        disable_input_quantization(self.model)

    def get_calib_data(self, calib_data):
        if calib_data:
            if not isinstance(calib_data, list):
                raise ValueError("calib_data should be list of tensors")
            self.check_calib_data(calib_data)
            return calib_data

        if self.cfg.input_shape:
            try:
                rand_input = torch.rand(
                    self.cfg.input_shape, dtype=torch.float,
                    requires_grad=False
                )
            except RuntimeError as ex:
                logger.error("calib_data init failed, please check input_shape. %s", str(ex))
                raise ex
            calib_data = [[rand_input]]
            return calib_data

        raise ValueError("The calib_data or input_shape"
                         " should be offered")

    def check_calib_data(self, calib_data):
        for i, calib_data_item in enumerate(calib_data):
            check_type(calib_data_item, (list, dict, tuple), param_name=f'calib_data[{i}]')
            if isinstance(calib_data_item, dict):
                for key, value in calib_data_item.items():
                    check_type(value, CHECK_DTYPE.get(key, torch.Tensor))
            else:
                for _, item in enumerate(calib_data_item):
                    if item is not None and not isinstance(item, (torch.Tensor, int)):
                        raise ValueError("Not all elements in calib_data are torch.Tensor, "
                                         "please make sure that the model can run with model(*(calib_data[0]))"
                                         "or with model(**(calib_data[0]))")

    def amp(self, calib_amp=10):
        # for quantized model
        try:
            model_quant = copy.deepcopy(self.model)
        except BaseException as exception:
            model_quant = copy.deepcopy(self.fp_model)
            model_quant = self.quantize(model_quant)
            model_quant.eval()
            with torch.no_grad():
                for data in self.calib_data:
                    model_quant(*data)

        model_quant.eval()
        layer_names = list()
        out_feats = list()

        def hook(module, fea_in, fea_out):
            out_feats.append(fea_out)

        for name, module in model_quant.named_modules():
            if isinstance(module, Conv2dQuantizer) or\
                    isinstance(module, LinearQuantizer):
                layer_names.append(name)
                module.register_forward_hook(hook=hook)

        # for fp model
        model_fp = copy.deepcopy(self.fp_model)
        model_fp.eval()
        out_feats_fp = list()

        def hook_fp(module, fea_in, fea_out):
            out_feats_fp.append(fea_out)

        for name, module in model_fp.named_modules():
            if hasattr(nn.modules.linear, "NonDynamicallyQuantizableLinear"):
                is_linear = isinstance(module, nn.Linear) or \
                            isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear)
            else:
                is_linear = isinstance(module, nn.Linear)

            if isinstance(module, nn.Conv2d) or is_linear:
                if name in layer_names:
                    module.register_forward_hook(hook=hook_fp)


        fallback_layers = amp_decision(model_quant, model_fp,
                                       out_feats, out_feats_fp,
                                       layer_names, self.calib_data[:calib_amp], self.cfg.amp_num)
        if fallback_layers:
            logger.info("Fallback start")
            disable_quantization(self.model, names=fallback_layers)
            logger.info("Fallback end!")

    def run(self):
        try:
            self._run()
        except Exception as ex:
            raise Exception("Please check your config, model and input!", ex) from ex

    def get_quant_params(self):
        self.model.eval()
        input_scale = {}
        input_offset = {}
        weight_scale = {}
        weight_offset = {}
        quant_weight = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (Conv2dQuantizer, LinearQuantizer)):
                if not module.quant_weight.is_enable:
                    continue
                input_scale[name] = module.quant_input.input_scale
                input_offset[name] = module.quant_input.input_offset
                weight_scale[name] = module.quant_weight.weight_scale
                weight_offset[name] = module.quant_weight.weight_offset
                quant_weight[name] = module.quant_weight.int_weight_tensor

        return input_scale, input_offset, \
               weight_scale, weight_offset, quant_weight

    def save_param(self, output_path, output_name, output_file):
        output_path = os.path.join(output_path, output_name)
        output_path = get_valid_write_path(output_path)
        with SafeWriteUmask(umask=0o377):
            np.save(output_path, output_file)

    def export_param(self, save_path):
        logger.info("Path of quant param is %r ", save_path)
        if not os.path.exists(save_path):
            save_path = get_valid_write_path(save_path)
            os.makedirs(save_path, mode=0o750, exist_ok=True)
        input_scale, input_offset, weight_scale, weight_offset, quant_weight = self.get_quant_params()
        self.save_param(save_path, "input_scale.npy", input_scale)
        self.save_param(save_path, "input_offset.npy", input_offset)
        self.save_param(save_path, "weight_scale.npy", weight_scale)
        self.save_param(save_path, "weight_offset.npy", weight_offset)
        self.save_param(save_path, "quant_weight.npy", quant_weight)
        logger.info("Save quant param success!")

    def export_onnx(self, model_arch, save_path, input_names):
        security.check_type(model_arch, str, param_name="model_arch")
        security.check_character(model_arch, param_name="model_arch")
        security.check_write_directory(save_path)
        security.check_element_type(input_names, str, list, param_name="input_names")
        security.check_character(input_names, param_name="input_names")

        if self.cfg.input_shape:
            input_shape = self.cfg.input_shape
            input_shape[0] = 1
            dummpy_input = torch.rand(
                input_shape, dtype=torch.float,
                requires_grad=False
            )
        else:
            dummpy_input = tuple(self.calib_data[0])
        self.fp_model.eval()

        if "yolov5" in model_arch.lower():
            self.fp_model.model[-1].export = True

        temp_fp_model_file = os.path.join(save_path, "{}_fp.onnx".format(model_arch))
        temp_fp_model_file = get_valid_write_path(temp_fp_model_file)

        with SafeWriteUmask():
            try:
                torch.onnx.export(self.fp_model, dummpy_input, temp_fp_model_file,
                                  input_names=input_names, verbose=True, opset_version=11)
            except RuntimeError as ex:
                logger.error("Export fp_model to onnx failed, please check model. %s", str(ex))
                raise ex

    def export_quant_onnx(self, model_arch, save_path, input_names=None, fuse_add=True, save_fp=False):
        security.check_type(model_arch, str, param_name="model_arch")
        security.check_character(model_arch, param_name="model_arch")
        security.check_write_directory(save_path)
        security.check_element_type(input_names, str, list, param_name="input_names")
        security.check_character(input_names, param_name="input_names")
        check_type(fuse_add, bool, param_name="fuse_add")
        check_type(save_fp, bool, param_name="save_fp")

        self.export_onnx(model_arch, save_path, input_names)
        onnx_path = os.path.join(save_path, "{}_fp.onnx".format(model_arch))
        onnx_path = get_valid_read_path(onnx_path)
        model = onnx.load(onnx_path)

        if "swinv2" in model_arch.lower() or "solov2" in model_arch.lower():
            from onnxsim import simplify
            model, check = simplify(model)

        graph = model.graph
        nodes = graph.node

        input_scale, input_offset, weight_scale, weight_offset, quant_weight = \
            self.get_quant_params()
        linear_params = ConvertLinearParams(
            onnx_model=model,
            input_scale=input_scale,
            input_offset=input_offset,
            weight_scale=weight_scale,
            weight_offset=weight_offset,
            quant_weight=quant_weight
        )
        input_scale, input_offset, weight_scale, weight_offset, quant_weight = \
            convert_linear_params(linear_params)
        quantized_weight_namd = []
        for item in nodes:
            if item.op_type == "Conv":
                weight_name = ".".join(item.input[1].split(".")[:-1])
                if weight_name in weight_scale.keys() and \
                        weight_scale.get(weight_name) is not None:
                    quantized_weight_namd.append(weight_name)
                    logger.info("Conv, item.name :%r, weight_name :%r ", item.name, weight_name)

            elif item.op_type == "MatMul":
                weight_name = item.input[1]
                if weight_name in weight_scale.keys() and \
                        weight_scale.get(weight_name) is not None:
                    quantized_weight_namd.append(weight_name)
                    logger.info("MatMul, item.name :%r, weight_name :%r ", item.name, weight_name)

        quantize_model_deploy_params = ModelDeployQuantParams(
            quantized_weight_name=quantized_weight_namd,
            quant_weight_dict=quant_weight,
            input_scale_dict=input_scale,
            input_offset_dict=input_offset,
            weight_scale_dict=weight_scale,
            weight_offset_dict=weight_offset,
            fuse_add=fuse_add
        )
        quantize_model_deploy(graph, quantize_model_deploy_params)

        temp_quant_model_file = os.path.join(save_path, "{}_quant.onnx".format(model_arch))
        temp_quant_model_file = get_valid_write_path(temp_quant_model_file)
        with SafeWriteUmask():
            onnx.save(model, temp_quant_model_file)
            logger.info("Quantification ended and onnx is stored in %r ", temp_quant_model_file)

        if not save_fp:
            save_fp_path = os.path.join(save_path, "{}_fp.onnx".format(model_arch))
            safe_delete_path_if_exists(save_fp_path)

    def get_quant_safetensor_params(self):
        quant_param_dict = {}
        quant_name_weight_list = []
        quant_name = []
        original_type = next(self.model.parameters()).dtype

        for name, module in self.model.named_modules():
            quant_name.append(name)
            fp_name = name.rsplit(".", 1)[0]
            if isinstance(module, TensorQuantizer) and module.input_scale is not None:
                quant_param_dict[fp_name + '.input_scale'] = module.input_scale.to(original_type).cpu()
                quant_param_dict[fp_name + '.input_offset'] = module.input_offset.to(original_type).cpu()
                input_offset = module.input_offset.cpu()
        
            if isinstance(module, TensorQuantizer) and module.int_weight_tensor is not None:
                quant_weight = module.int_weight_tensor.cpu()
                fp_weight_bias = self.ori_fp_weight.get(fp_name + '.bias').cpu()
                quant_param_dict[fp_name + '.weight'] = module.int_weight_tensor.round().to(torch.int8)
                quant_param_dict[fp_name + '.bias'] = fp_weight_bias
                quant_name_weight_list.append(fp_name + '.weight')

                if fp_name + '.input_scale' in quant_param_dict:
                    deq_scale = deqscale_process(
                        quant_param_dict[fp_name + '.input_scale'].cpu(),
                        module.weight_scale.cpu()
                    ).cpu()
                    correction = quant_weight.to(torch.float32).sum(dim=1) * input_offset.to(torch.float32).cpu()
                    quant_bias = torch.round(fp_weight_bias / deq_scale - correction).to(torch.int32)
                    quant_param_dict[fp_name + '.quant_bias'] = quant_bias.cpu()
                    deq_scale = deqscale2int64_by_dtype(deq_scale, original_type == torch.bfloat16)
                    quant_param_dict[fp_name + '.deq_scale'] = deq_scale.cpu()

        quant_model_description = {key: WEIGHT_TYPE for key in quant_param_dict.keys()}
        quant_model_description["model_quant_type"] = WEIGHT_TYPE

        for ori_model_state_dict_name, ori_model_state_dict in self.ori_fp_weight.items():
            if ori_model_state_dict_name not in quant_name_weight_list:
                quant_param_dict[ori_model_state_dict_name] = ori_model_state_dict
                quant_model_description[ori_model_state_dict_name] = FLOAT_TYPE
        return quant_param_dict, quant_model_description

 
    def export_quant_safetensor(self, output_path, safetensors_name=None, json_name=None):
        """
        基于浮点、量化两份独立权重，存储完整的量化、浮点混合权重，用户仅需加载一个混合权重即可
        """
        from safetensors.torch import save_file
        check_type(output_path, str, param_name="output_path")
        output_path = get_write_directory(output_path, write_mode=0o750)

        if not isinstance(safetensors_name, str):
            default_safetensors_name = f"quant_model_weight_{WEIGHT_TYPE.lower()}.safetensors"
            logger.warning(f"invalid `safetensors_name`, defaulting to `{default_safetensors_name}`")
            safetensors_name = default_safetensors_name
        if not isinstance(json_name, str):
            default_json_name = f"quant_model_description_{WEIGHT_TYPE.lower()}.json"
            logger.warning(f"invalid `json_name`, defaulting to `{default_json_name}`")
            json_name = default_json_name
        
        quant_model_weight_path = os.path.join(output_path, safetensors_name)
        quant_model_description_path = os.path.join(output_path, json_name)
        quant_model_weight_path = get_valid_write_path(quant_model_weight_path, extensions=[".safetensors"])
        quant_model_description_path = get_valid_write_path(quant_model_description_path, extensions=[".json"])
        
        safetensor_weight, quant_model_description = self.get_quant_safetensor_params()

        for key, item in safetensor_weight.items():
            safetensor_weight[key] = item.cpu().contiguous()
        logger.info("The directory path for the saved safetensors is %r", quant_model_weight_path)

        with SafeWriteUmask(umask=0o377):
            save_file(safetensor_weight, quant_model_weight_path)
        json_safe_dump(quant_model_description, quant_model_description_path, indent=2)
    
    def _calculate_quant_weight(self):
        for name, module in self.model.named_modules():
            fp_name = name.rsplit(".", 1)[0]
            if isinstance(module, TensorQuantizer) and \
                    module.weight_scale is not None and \
                    module.weight_offset is not None:
                module.int_weight_tensor, _ = fake_quantize(self.ori_fp_weight.get(fp_name + '.weight'),
                                                            module.weight_scale, 
                                                            module.weight_offset, 
                                                            module.bit)

    def _run(self, calib_amp=10):
        logger.info("Calibration start!")
        self.model.eval()
        check_device = True
        if isinstance(self.calib_data[0], dict):
            model_device = self.model.device.type
            data_deivce = next(iter(self.calib_data[0].values())).device.type
            check_device = model_device == data_deivce
        with torch.no_grad():
            for data in tqdm(self.calib_data):
                if isinstance(data, dict):
                    if check_device:
                        self.model(**data)
                    else:
                        item = {kk: vv.to(model_device) if isinstance(vv, torch.Tensor) else vv
                                for kk, vv in data.items()}
                        self.model(**item)
                        item = {kk: vv.to(data_deivce) if isinstance(vv, torch.Tensor) else vv
                                for kk, vv in data.items()}
                else:
                    self.model(*data)
        logger.info("Calibration end!")
        
        # calculate quant weight before saving
        self._calculate_quant_weight()

        if self.cfg.amp_num > 0:
            logger.info("AMP start!")
            self.amp(calib_amp)
            logger.info("AMP end!")



def quantize_model(model, cfg=None):
    def _set_module(model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for sub_token in sub_tokens:
            cur_mod = getattr(cur_mod, sub_token)
        setattr(cur_mod, tokens[-1], module)

    for name, mod in model.named_modules():
        logger.info("quantize_model, name :%r, type of mod :%s ", name, type(mod))
        if isinstance(mod, nn.Conv2d):
            quant_mod = Conv2dQuantizer(cfg=cfg)
            quant_mod.set_param(mod)
            _set_module(model, name, quant_mod)
            continue

        if hasattr(nn.modules.linear, "NonDynamicallyQuantizableLinear"):
            is_linear = isinstance(mod, nn.Linear) or isinstance(mod, nn.modules.linear.NonDynamicallyQuantizableLinear)
        else:
            is_linear = isinstance(mod, nn.Linear)

        if is_linear:
            quant_mod = LinearQuantizer(cfg=cfg)
            quant_mod.set_param(mod)
            _set_module(model, name, quant_mod)
    return model


def set_first_last_layer(model, last=True):
    module_list = []
    for name, mod in model.named_modules():
        if isinstance(mod, Conv2dQuantizer):
            logger.info("Quantized conv module:%r", name)
            module_list += [mod]
        if isinstance(mod, LinearQuantizer):
            logger.info("Quantized linear module:%r", name)
            module_list += [mod]
    module_list[0].quant_input.is_enable = False
    module_list[0].quant_weight.is_enable = False
    if last:
        module_list[-1].quant_weight.is_enable = False
        module_list[-1].quant_input.is_enable = False


def disable_input_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            logger.info("Disabling quantization of input quantizer:%r", name)
            module.disable_input_quantization()


def enable_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            logger.info("Enabling quantizer:%r", name)
            module.enable_quantization(name)


def set_disable_quantization(module, name):
    module.disable_quantization(name)
    logger.info("Disabling quantizer:%r", name)


def disable_quantization(model, names=None):
    quantizer_names = list()
    if names is None:
        names = []
    if names:
        for name in names:
            quantizer_names.append(name + '.quant_input')
            quantizer_names.append(name + '.quant_weight')

    for name, module in model.named_modules():
        if isinstance(module, Quantizer):
            if (not quantizer_names) or (name in quantizer_names):
                set_disable_quantization(module, name)


def fuse_set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for sub_token in sub_tokens:
        cur_mod = getattr(cur_mod, sub_token)
    setattr(cur_mod, tokens[-1], module)


def fuse(conv, batchnorm):
    weight = conv.weight
    mean = batchnorm.running_mean
    var_sqrt = torch.sqrt(batchnorm.running_var + batchnorm.eps)

    beta = batchnorm.weight
    gamma = batchnorm.bias

    if conv.bias is not None:
        bias = conv.bias
    else:
        bias = mean.new_zeros(mean.shape)

    if (var_sqrt == 0).all():
        raise ZeroDivisionError
    else:
        weight = weight * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
        bias = (bias - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           conv.kernel_size,
                           conv.stride,
                           conv.padding,
                           groups=conv.groups,
                           bias=True)
    fused_conv.weight = nn.Parameter(weight)
    fused_conv.bias = nn.Parameter(bias)
    return fused_conv


def fuse_conv_batchnorm_module(model, conv2d, conv2d_name, batchnorm, batchnorm_name):
    batchnorm_conv = fuse(conv2d, batchnorm)
    fuse_set_module(model, conv2d_name, batchnorm_conv)
    fuse_set_module(model, batchnorm_name, nn.Sequential())


def fuse_module(model):
    children = list(model.named_children())
    child_conv2d = None
    child_conv2d_name = None
    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            if not isinstance(child_conv2d, torch.nn.Conv2d):
                pass
            elif child_conv2d.out_channels != child.running_mean.shape[0]:
                pass
            else:
                fuse_conv_batchnorm_module(model, child_conv2d, child_conv2d_name, child, name)
                child_conv2d = None
                child_conv2d_name = None
        elif isinstance(child, nn.Conv2d):
            child_conv2d = child
            child_conv2d_name = name
        else:
            fuse_module(child)


def deqscale_process(input_scale, scale):
    deq_scale = input_scale * scale
    if deq_scale.ndim > 1:
        deq_scale = deq_scale.squeeze(1)
    deq_scale = deq_scale.cpu()
    return deq_scale


def deqscale2int64(scale):
    scale = scale.numpy()
    scale = np.frombuffer(scale.tobytes(), dtype=np.int32).astype(np.int64)
    scale = torch.tensor(scale)
    return scale


def deqscale2int64_by_dtype(scale, is_bf16):
    if is_bf16:
        return scale
    else:
        return deqscale2int64(scale)
