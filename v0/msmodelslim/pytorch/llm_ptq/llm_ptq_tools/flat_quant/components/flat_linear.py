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

from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.components.quantizer import (
    WeightQuantizer, 
    ActivationQuantizer
)


class ForwardMode(Enum):
    """
    Forward mode enumeration for different computation modes.
    
    ORG: Original mode without quantization
    CALIB: Calibration mode for collecting statistics and training quantization parameters
    EVAL: Evaluation mode using fixed quantization parameters for inference
    """
    ORG = "org"
    CALIB = "calib"
    EVAL = "eval"


class FakeQuantizedLinearConfig:
    """
    Configuration class for fake quantized linear layers.
    """
    def __init__(self, w_bits=16,
                        a_bits=16,
                        w_asym=False,
                        a_asym=False,
                        lwc=False,
                        lac=False,
                        a_groupsize=-1,
                        a_per_tensor=False):
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.w_asym = w_asym
        self.a_asym = a_asym
        self.lwc = lwc
        self.lac = lac
        self.a_groupsize = a_groupsize
        self.a_per_tensor = a_per_tensor


class FakeQuantizedLinear(nn.Module):
    """
    Fake quantized linear layer that simulates quantization effects during training.
    
    This class implements a linear layer with fake quantization support, which can simulate
    quantization effects during training while maintaining floating-point computation for 
    gradient backpropagation. It supports both weight and activation quantization with 
    configurable parameters.
    """
    def __init__(self, config, linear: nn.Linear):
        """
        Initialize fake quantized linear layer.
        
        Args:
            config (FakeQuantizedLinearConfig): Quantization configuration
            linear (nn.Linear): Original linear layer to be quantized
        """
        super(FakeQuantizedLinear, self).__init__()
        self.config = config
        weight = linear.weight
        self.weight = weight
        if linear.bias is not None:
            bias = linear.bias
            self.bias = bias
        else:
            self.bias = None
            
        self.weight_quantizer = WeightQuantizer(bits=config.w_bits,
                                                in_size=linear.weight.shape[1], 
                                                out_size=linear.weight.shape[0], 
                                                perchannel=True, 
                                                sym=not (config.w_asym), 
                                                lwc=config.lwc
                                                )
        self.act_quantizer = ActivationQuantizer(bits=config.a_bits, 
                                                 sym=not (config.a_asym), 
                                                 lac=config.lac, 
                                                 groupsize=config.a_groupsize,
                                                 per_tensor=config.a_per_tensor)
        self._mode = ForwardMode.ORG

    def extra_repr(self):
        return f"weight shape: {tuple(self.weight.shape)}, bias={self.bias is not None}"

    def set_act_clip_factor(self, clip_factor):
        if self.act_quantizer.lac:
            self.act_quantizer.clip_factor = clip_factor
    
    def forward(self, hidden_states):
        weight = self.weight
        bias = self.bias
        if self._mode == ForwardMode.ORG:
            return self._ori_forward(hidden_states)
        else:
            return self._fake_quant_forward(hidden_states, weight, bias)
        
    def to_org_mode(self):
        """Switch to original mode without quantization."""
        self._mode = ForwardMode.ORG

    def to_calib_mode(self):
        """Switch to calibration mode for collecting statistics."""
        self._mode = ForwardMode.CALIB
   
    def reparameterize(self):
        """
        Reparameterize quantizers by converting learnable parameters to fixed buffers.
        """
        self.weight_quantizer.reparameterize()
        self.act_quantizer.reparameterize()

    def to_eval_mode(self):
        """Switch to evaluation mode with fixed quantization parameters."""
        if not self._mode == ForwardMode.EVAL:
            with torch.no_grad():
                self.reparameterize()
            self._mode = ForwardMode.EVAL

    def fake_quant_weight(self):
        """Apply fake quantization to weights in-place."""
        self.weight.data = self.weight_quantizer.get_fake_quant_weight(self.weight.data)

    def _ori_forward(self, hidden_states):
        """
        Original forward propagation without quantization.
        Used for collecting statistics without applying quantization effects.
        """
        hidden_states = self.act_quantizer(hidden_states, quantize=False)
        weight = self.weight_quantizer(self.weight, quantize=False)
        return F.linear(hidden_states, weight, self.bias)
    
    def _fake_quant_forward(self, hidden_states, weight, bias):
        """
        Fake quantized forward propagation.
        Applies quantization to both activations and weights.
        """
        hidden_states = self.act_quantizer(hidden_states)
        weight = self.weight_quantizer(weight)
        return F.linear(hidden_states, weight, bias)


class FlatQuantizedLinear(FakeQuantizedLinear):
    """
    FlatQuant quantized linear layer with transformation support.
    
    This class extends FakeQuantizedLinear by adding support for transformation matrices,
    implementing the quantization-aware transformation functionality in the FlatQuant algorithm.
    The transformations help reduce quantization error by adapting the weight and activation
    distributions before quantization.
    """
    def __init__(self, args, linear: nn.Linear):
        """
        Initialize FlatQuant quantized linear layer.
        """
        super(FlatQuantizedLinear, self).__init__(args, linear)
        # Initialize transformation matrices
        self.weight_in_trans = None   # Weight input transformation matrix
        self.weight_out_trans = None  # Weight output transformation matrix
        self.act_in_trans = None      # Activation input transformation matrix
        self.save_trans = None        # Saved transformation for checkpointing

    def set_trans(self, weight_in_trans=None, 
                  weight_out_trans=None, 
                  act_in_trans=None,
                  save_trans=None):
        """
        Set transformation matrices for quantization-aware transformations.
        """
        if weight_in_trans is not None:
            self.weight_in_trans = weight_in_trans
        if weight_out_trans is not None:
            self.weight_out_trans = weight_out_trans
        if act_in_trans is not None:
            self.act_in_trans = act_in_trans
        if save_trans is not None:
            self.save_trans = save_trans

    def forward(self, hidden_states):
        """
        Forward propagation with different computation paths based on mode.
        
        - ORG mode: Original computation without transformations
        - CALIB mode: Apply transformations during calibration
        - EVAL mode: Use pre-applied transformations for inference
        """
        weight_in_trans = self.weight_in_trans
        weight_out_trans = self.weight_out_trans
        act_in_trans = self.act_in_trans
        if self._mode == ForwardMode.ORG:
            return self._ori_forward(hidden_states)
        elif self._mode == ForwardMode.CALIB:
            return self._calib_forward(hidden_states, 
                                        weight_in_trans=weight_in_trans, 
                                        weight_out_trans=weight_out_trans, 
                                        act_in_trans=act_in_trans)
        else:
            return self._eval_forward(hidden_states, act_in_trans=act_in_trans)

    def to_org_mode(self):
        """Switch to original mode without transformations."""
        self._mode = ForwardMode.ORG

    def to_calib_mode(self):
        """Switch to calibration mode with active transformations."""
        self._mode = ForwardMode.CALIB

    def reparameterize(self):
        """
        Reparameterize by applying transformations to weights and biases.
        
        This method applies learned transformation matrices to weights and biases,
        then deletes the transformation matrices to save memory. This is typically
        called when switching from calibration to evaluation mode.
        
        The transformation order is:
        1. Apply weight input transformation (inverse transpose)
        2. Apply learnable weight clipping if enabled
        3. Apply weight output transformation
        4. Apply output transformation to bias if present
        """
        if not self._mode == ForwardMode.EVAL:
            weight_in_trans = self.weight_in_trans
            weight_out_trans = self.weight_out_trans
            weight = self.weight.data
            ori_dtype = weight.dtype
            weight = weight.to(torch.float64)
            
            # Apply quantization-adaptive transformations
            if weight_in_trans is not None:
                weight = weight_in_trans(weight, inv_t=True)
            if self.weight_quantizer.lwc:
                weight = self.weight_quantizer.apply_wclip(weight)
            if weight_out_trans is not None:
                weight = weight_out_trans(weight.T).T
            if weight_out_trans is not None and self.bias is not None:
                self.bias.data = weight_out_trans(self.bias.data)                
            self.weight.data = weight.to(ori_dtype)
            
            # Clean up transformation matrices to save memory
            del self.weight_in_trans
            del self.weight_out_trans
            self.weight_in_trans = None
            self.weight_out_trans = None
            
            # Reparameterize quantizers
            self.act_quantizer.reparameterize()
            self.weight_quantizer.reparameterize()
    
    def _calib_forward(self, hidden_states, weight_in_trans=None, weight_out_trans=None, act_in_trans=None):
        """
        Calibration mode forward propagation.
        
        In calibration mode, transformations are applied dynamically to weights and activations
        before quantization. This allows the quantization parameters to adapt to the transformed
        distributions.
        
        Args:
            hidden_states: Input activations
            weight_in_trans: Weight input transformation
            weight_out_trans: Weight output transformation  
            act_in_trans: Activation input transformation
        """
        if act_in_trans is not None:
            hidden_states = act_in_trans(hidden_states)
        weight = self.weight.data
        
        # Apply quantization-adaptive transformations to weights
        if weight_in_trans is not None:
            weight = weight_in_trans(weight, inv_t=True)
        if self.weight_quantizer.lwc:
            weight = self.weight_quantizer.apply_wclip(weight)
        # Apply learnable weight clipping 
        if weight_out_trans is not None:
            weight = weight_out_trans(weight.T).T
        if weight_out_trans is not None and self.bias is not None:
            bias = weight_out_trans(self.bias.data)
        else:
            bias = self.bias
            
        # Apply quantization to transformed weights and activations
        output = self._fake_quant_forward(hidden_states, weight, bias)
        return output

    def _eval_forward(self, hidden_states, act_in_trans=None):
        """
        Evaluation mode forward propagation.
        
        In evaluation mode, transformations have already been applied to weights during
        reparameterization, so only activation transformations and quantization are needed.
        
        Args:
            hidden_states: Input activations
            act_in_trans: Activation input transformation
        """
        if act_in_trans is not None:
            hidden_states = act_in_trans(hidden_states)
        weight = self.weight
        bias = self.bias
        # Apply quantization to activations and pre-transformed weights
        output = self._fake_quant_forward(hidden_states, weight, bias)
        return output


class FlatNormWrapper(nn.Module):
    """
    FlatQuant normalization layer wrapper with transformation support.
    
    This class wraps normalization layers (like LayerNorm) with transformation support,
    used to handle output transformations of normalization layers in the FlatQuant algorithm.
    The wrapper allows applying transformations to the output of normalization layers
    to improve quantization quality of subsequent layers.
    """
    def __init__(self, norm, trans=None):
        """
        Initialize normalization layer wrapper.
        
        Args:
            norm: Normalization layer (e.g., LayerNorm, RMSNorm)
            trans: Transformation matrix to apply to normalization output
        """
        super(FlatNormWrapper, self).__init__()
        self.norm = norm
        self.trans = trans
        self._mode = ForwardMode.ORG

        weight = norm.weight
        del self.norm.weight
        self.norm.register_buffer('weight', weight)
        if hasattr(self.norm, "bias") and self.norm.bias is not None:
            bias = norm.bias
            del self.norm.bias
            self.norm.register_buffer('bias', bias)

    @property
    def weight(self):
        return self.norm.weight

    @property
    def bias(self):
        return self.norm.bias

    def forward(self, hidden_states):
        if self._mode == ForwardMode.ORG:
            return self._ori_forward(hidden_states)
        else:
            return self._calib_eval_forward(hidden_states)
        
    def to_org_mode(self):
        self._mode = ForwardMode.ORG

    def to_calib_mode(self):
        self._mode = ForwardMode.CALIB
        
    def to_eval_mode(self):
        self.reparameterize()
        self._mode = ForwardMode.EVAL

    def reparameterize(self):
        if not self._mode == ForwardMode.EVAL:
            self._mode = ForwardMode.EVAL

    def _ori_forward(self, hidden_states):
        return self.norm(hidden_states)
    
    def _calib_eval_forward(self, hidden_states):
        """
        Calibration/evaluation mode forward propagation.
        
        First applies normalization, then applies transformation if present.
        This allows the transformation to adapt the normalized output distribution
        for better quantization in subsequent layers.
        """
        hidden_states = self.norm(hidden_states)
        if self.trans is not None:
            hidden_states = self.trans(hidden_states)
        return hidden_states