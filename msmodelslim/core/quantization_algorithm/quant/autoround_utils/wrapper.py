#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2023 Intel Corporation
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.functional import F

from msmodelslim.core.quantization_algorithm.quant.autoround_utils.utils import get_quant_func


def reshape_and_pad_tensor(v, group_size=-1):
    """Reshapes the tensor based on the group size.

    Args:
        v (torch.Tensor): The input tensor to be reshaped.
        group_size (int, optional): The number of elements to group together.

    Returns:
        torch.Tensor: The reshaped tensor. If padding is applied, the padded tensor is returned.
    """
    if group_size == 0:
        return v.reshape(1, -1)
    if group_size == -1 or v.shape[1] < group_size:
        return v
    if v.shape[1] % group_size == 0:
        v = v.reshape(-1, group_size)
    else:
        pad_len = (v.shape[1] + group_size - 1) // group_size * group_size - v.shape[1]
        v = torch.nn.functional.pad(v, (0, pad_len))
        v = v.reshape(-1, group_size)
    return v


def get_scale_shape(weight, group_size):
    """Computes the shape of the scale tensor for quantization based on the weight tensor and group size.

    Args:
      weight (torch.Tensor): The weight tensor of the layer.
      group_size (int): The size of the groups for quantization.

    Returns:
      The shape of the scale tensor to be used for quantization.
    """
    if group_size == 0:
        return 1
    elif group_size == -1 or weight.shape[1] < group_size:
        shape = weight.shape[0]
    else:
        shape = weight.shape[0] * ((weight.shape[1] + group_size - 1) // group_size)

    return shape


class WrapperLinear(torch.nn.Module):
    """A wrapper for linear/conv1d layers to enable quantization and tuning.

    This module wraps an existing linear or conv1d layer and provides additional functionality
    for quantization, parameter tuning, and activation/bias normalization.

    Args:
        orig_layer (torch.nn.Module): The original layer to be wrapped (linear or conv1d).
        enable_minmax_tuning (bool): Whether to enable min-max scale tuning.
    """

    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_round_tuning=True,
        enable_trainable_smooth=False,
        enable_norm_bias_tuning=False,
        config=None,
        **kwargs,
    ):
        """Initializes the WrapperLinear module.

        Args:
            orig_layer (torch.nn.Module): The original layer to wrap.
            enable_minmax_tuning (bool): Whether to enable min-max scale tuning.
        """
        super().__init__()
        self.orig_layer = orig_layer
        self.enable_minmax_tuning = enable_minmax_tuning
        self.enable_round_tuning = enable_round_tuning
        self.enable_norm_bias_tuning = enable_norm_bias_tuning
        self.enable_trainable_smooth = enable_trainable_smooth
        self.enable_act_quant = self.orig_layer.act_bits <= 8
        self.q_scale_thresh = 1e-5
        self._init_tuning_params_and_quant_func()
        self.orig_forward = self.linear_forward if isinstance(self.orig_layer, torch.nn.Linear) else self.conv1d_forward
        self.config = config

        self.train_with_act_quant = False  # quant activations during Autoround calibration

    def unwrapper(self, best_params):
        """Restores the original layer by applying the best tuning parameters.

        Args:
            best_params (dict): Dictionary containing the best tuning parameters.

        Returns:
            torch.nn.Module: The unwrapped and restored original layer.
        """
        best_params = best_params or {}
        v = best_params.get("value", torch.tensor(0.0))
        min_scale = best_params.get("min_scale", torch.tensor(1.0))
        max_scale = best_params.get("max_scale", torch.tensor(1.0))

        ##unwrapper weight
        if self.enable_trainable_smooth:
            smooth_scale = best_params.get(
                "smooth_scale", torch.ones((self.orig_layer.weight.shape[-1],), dtype=self.orig_layer.weight.dtype)
            )
            q_weight, scale, zp = self._qdq_weight(v, min_scale, max_scale, smooth_scale, output_qdq=False)
        else:
            q_weight, scale, zp = self._qdq_weight(v, min_scale, max_scale, output_qdq=False)

        self.orig_layer.weight.data.copy_(q_weight)
        self.orig_layer.weight.grad = None

        shape = q_weight.shape

        def _set_dict_attr(attr_dict, attr_name):
            for key in attr_dict.keys():
                if key == attr_name:
                    setattr(self.orig_layer, attr_name, attr_dict[key].reshape(shape[0], -1))
                else:
                    name = "w_" + key
                    setattr(self.orig_layer, name, attr_dict[key])

        if isinstance(scale, dict):
            _set_dict_attr(scale, "scale")
        else:
            self.orig_layer.scale = scale.reshape(shape[0], -1)

        if zp is not None:
            if isinstance(zp, dict):
                _set_dict_attr(zp, "zp")
            else:
                zp = zp.reshape(shape[0], -1)
                self.orig_layer.zp = zp if zp is not None else None
        else:
            self.orig_layer.zp = None

        self.unwrapper_bias(best_params)
        return self.orig_layer

    def unwrapper_bias(self, best_params):
        if self.enable_norm_bias_tuning and "bias_v" in best_params.keys():
            bias_v = best_params["bias_v"]
            bias = self.orig_layer.bias
            bias, _, _ = self._qdq_bias(bias, bias_v)
            self.orig_layer.bias.grad = None
            self.orig_layer.bias.data.copy_(bias)

    def linear_forward(self, x, weight, bias):
        """Performs the forward pass for a linear layer."""
        return F.linear(x, weight, bias)

    def conv1d_forward(self, x, weight, bias):
        """Performs the forward pass for a Conv1D layer."""
        size_out = x.size()[:-1] + (self.orig_layer.nf,)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        x = x.view(*size_out)
        return x

    def forward(self, x):
        """Executes the forward pass with quantized weights and optional bias/activation quantization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the wrapped layer.
        """

        w_corr = self.value

        if self.enable_trainable_smooth:
            weight_q, _, _ = self._qdq_weight(w_corr, self.min_scale, self.max_scale, self.act_smooth_scale)
        else:
            weight_q, _, _ = self._qdq_weight(w_corr, self.min_scale, self.max_scale)

        # Online ratation Hood
        if self.orig_layer._forward_pre_hooks:
            for hook in self.orig_layer._forward_pre_hooks.values():
                x = hook(self.orig_layer, (x,))[0]

        if self.enable_act_quant and self.train_with_act_quant:
            act_max = self.orig_layer.act_max if hasattr(self.orig_layer, "act_max") else None
            if self.enable_trainable_smooth:
                x, _, _ = self._qdq_act(
                    x, act_max_scale=self.act_max_scale, act_smooth_scale=self.act_smooth_scale, act_max=act_max
                )
            else:
                x, _, _ = self._qdq_act(x, act_max_scale=self.act_max_scale, act_max=act_max)

        bias = self.orig_layer.bias

        output = self.orig_forward(x, weight_q, bias)
        return output

    def _init_tuning_params_and_quant_func(self):
        """Initializes tuning parameters and quantization functions.

        This method sets up required parameters and functions for weight quantization,
        activation quantization, and bias/normalization.
        """
        self.params = {}
        p_dtype = torch.float32

        orig_layer = self.orig_layer
        orig_weight = getattr(orig_layer, "get_weight", lambda: orig_layer.weight)()
        weight_reshape = reshape_and_pad_tensor(orig_weight.data, orig_layer.group_size)
        self.weight_min = torch.clamp(weight_reshape.min(1)[0], max=0)
        self.weight_max = torch.clamp(weight_reshape.max(1)[0], min=0)
        self._init_params("value", p_dtype, weight_reshape.shape, 0, self.enable_round_tuning)
        # Min-max scale initialization
        shape = get_scale_shape(orig_weight, orig_layer.group_size)
        self._init_params("min_scale", p_dtype, shape, 1.0, self.enable_minmax_tuning)
        self._init_params("max_scale", p_dtype, shape, 1.0, self.enable_minmax_tuning)

        self.weight_quant_func, self.data_type = get_quant_func(orig_layer.data_type, orig_layer.bits, orig_layer.sym)

        if self.enable_act_quant:
            self.act_quant_func, self.act_data_type = get_quant_func(
                orig_layer.act_data_type, orig_layer.act_bits, orig_layer.act_sym
            )

            if self.enable_trainable_smooth and hasattr(orig_layer, "to_smooth"):
                self._init_params("act_max_scale", p_dtype, (1), 1.0, not orig_layer.act_dynamic)
                self._init_params("act_smooth_scale", torch.bfloat16, orig_layer.weight.shape[1], 1.0, True)
            elif self.enable_trainable_smooth:
                self._init_params("act_max_scale", p_dtype, (1), 1.0, not orig_layer.act_dynamic)
                self.enable_trainable_smooth = False
            else:
                self._init_params("act_max_scale", p_dtype, (1), 1.0, not orig_layer.act_dynamic)

    def _init_params(self, name, dtype, shape, value, tunable):
        """Initializes a parameter for tuning or uses a constant if tuning is disabled.

        Args:
            name (str): Name of the parameter.
            dtype (torch.dtype): Data type of the parameter.
            shape (tuple): Shape of the parameter.
            value (float): Initial value for the parameter.
            tunable (bool): Whether the parameter should be tunable.
        """
        if tunable:
            p = torch.nn.Parameter(torch.ones(shape, dtype=dtype) * value, requires_grad=True)
            self.params.update({name: p})
        else:
            p = torch.tensor(1.0 * value, dtype=dtype)

        setattr(self, name, p)

    def _qdq_weight(self, value, min_scale, max_scale, smooth_scale=None, output_qdq=True):
        """Quantizes and dequantizes weights with tuning parameters.

        Args:
            value (torch.Tensor): Value added for rounding for tuning.
            min_scale (torch.Tensor): Minimum scale for the min value of quantization.
            max_scale (torch.Tensor): Maximum scale for the max value of quantization.

        Returns:
            tuple: Quantized weight, scale, and zero point.
        """
        min_scale.data.clamp_(0, 1.0)
        max_scale.data.clamp_(0, 1.0)
        weight = self.orig_layer.weight

        quant_kwargs = {}
        if hasattr(self.orig_layer, "super_bits"):
            quant_kwargs["super_bits"] = self.orig_layer.super_bits
            quant_kwargs["super_group_size"] = self.orig_layer.super_group_size

        if smooth_scale is not None:
            if "o_proj" in self.orig_layer.name and self.config.num_key_value_heads != self.config.num_attention_heads:
                scale = smooth_scale.view(
                    self.config.num_key_value_heads,
                    self.config.num_attention_heads // self.config.num_key_value_heads,
                    -1,
                )
                scale_for_fusion = scale.mean(1).repeat_interleave(
                    self.config.num_attention_heads // self.config.num_key_value_heads, dim=0
                )
                scale_for_fusion = scale_for_fusion.view(-1).contiguous()
                weight = weight / scale_for_fusion.unsqueeze(0)
            else:
                weight = weight / smooth_scale.unsqueeze(0)

        weight_q, scale, zp = self.weight_quant_func(
            weight,
            bits=self.orig_layer.bits,
            group_size=self.orig_layer.group_size,
            w_corr=value,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_dtype=self.orig_layer.scale_dtype,
            tensor_min=self.weight_min,
            tensor_max=self.weight_max,
            data_type=self.data_type,
            q_scale_thresh=self.q_scale_thresh,
            output_qdq=output_qdq,
            **quant_kwargs,
        )
        weight_q = weight_q.to(weight.dtype)
        return weight_q, scale, zp

    def _qdq_act(self, x, act_max_scale, act_smooth_scale=None, act_max=None):
        """Quantizes and dequantizes activations.

        Args:
            x (torch.Tensor): Input activations.
            act_max_scale (torch.Tensor): Maximum scale for the act_max
            act_max (torch.Tensor, optional): Maximum value for activation quantization. Defaults to None.

        Returns:
            tuple: Quantized activation, scale, and zero point.
        """
        act_max_scale.data.clamp_(0, 1.0)
        if self.enable_trainable_smooth:
            if "o_proj" in self.orig_layer.name and self.config.num_key_value_heads != self.config.num_attention_heads:
                scale = act_smooth_scale.view(
                    self.config.num_key_value_heads,
                    self.config.num_attention_heads // self.config.num_key_value_heads,
                    -1,
                )
                scale = scale.mean(1).repeat_interleave(
                    self.config.num_attention_heads // self.config.num_key_value_heads, dim=0
                )
                scale = scale.view(-1).contiguous()
                x_smoothed = x * scale
            else:
                x_smoothed = x * act_smooth_scale
            x, scale, zp = self.act_quant_func(
                x_smoothed,
                bits=self.orig_layer.act_bits,
                group_size=self.orig_layer.act_group_size,
                scale_dtype=self.orig_layer.scale_dtype,
                q_scale_thresh=self.q_scale_thresh,
                data_type=self.act_data_type,
                max_scale=act_max_scale,
                tensor_max=act_max,
            )
        else:
            x, scale, zp = self.act_quant_func(
                x,
                bits=self.orig_layer.act_bits,
                group_size=self.orig_layer.act_group_size,
                scale_dtype=self.orig_layer.scale_dtype,
                q_scale_thresh=self.q_scale_thresh,
                data_type=self.act_data_type,
                max_scale=act_max_scale,
                tensor_max=act_max,
            )
        return x, scale, zp

    def _qdq_bias(self, bias, bias_v):
        """Quantizes and dequantizes bias.

        Args:
            bias (torch.Tensor): Bias tensor to be quantized.
            bias_v (torch.Tensor): Value added for rounding for tuning.

        Returns:
            tuple: Quantized bias, scale, and zero point.
        """
        bias_bits = 4
        bias_group_size = -1
        bias, scale, zp = self.bias_quant_func(
            bias,
            bits=bias_bits,
            group_size=bias_group_size,
            v=bias_v,
            q_scale_thresh=self.q_scale_thresh,
        )
        return bias, scale, zp
