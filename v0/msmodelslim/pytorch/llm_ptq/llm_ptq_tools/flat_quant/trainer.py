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

import functools
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Any

import torch

from tqdm import tqdm
from accelerate import dispatch_model



from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.processors.flat_quant import (
    FlatQuantQuantizerConfig,
    quantize_model,
    get_trainable_parameters
)
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.flat_quant.models import (
    get_model_bridge,
    get_module_by_name, 
    RunnerStopExecution
)


npu_available = False
try:
    import torch_npu
except ImportError:
    pass
else:
    npu_available = True


@dataclass
class LayerTrainingData:
    """Layer training data container."""
    layer: torch.nn.Module
    layer_inputs: List[List[Any]]
    fp_outs: List[torch.Tensor]
    data_info: Dict[str, Any]
    device: torch.device
    layer_idx: int


def empty_cache():
    if npu_available:
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def get_device_str(device):
    if isinstance(device, int):
        device = str(device)
    if npu_available:
        if not device.startswith("npu"):
            return "npu:" + device
        else:
            return device
    else:
        if not device.startswith("cuda"):
            return "cuda:" + device
        else:
            return device


def convert_outputs_to_inputs(outputs):
    converted_inputs = []
    for output in outputs:
        converted_inputs.append([output])
    return converted_inputs


class TrainingConfig:
    """Training configuration class, encapsulating all training-related configurations."""
    def __init__(self, args):
        self.seed = args.seed
        self.epochs = args.epochs
        self.flat_lr = args.flat_lr
        self.warmup = args.warmup
        self.deactive_amp = args.deactive_amp
        self.amp_dtype = args.amp_dtype
        self.quant_by_quant = args.quant_by_quant
        
        # Set training data type and context
        if self.deactive_amp:
            self.dtype = torch.float32
            self.traincast = nullcontext
        else:
            if self.amp_dtype == "bfloat16":
                self.dtype = torch.bfloat16
            elif self.amp_dtype == "float16":
                self.dtype = torch.float16
            else:
                raise ValueError(f"Invalid AMP dtype: {self.amp_dtype}")
            
            if npu_available:
                self.traincast = functools.partial(torch.amp.autocast, device_type="npu", dtype=self.dtype)
            else:
                self.traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=self.dtype)


class ModelAdapter:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.default_device = model.device
        self.hf_device_map = getattr(model, "hf_device_map", None)
        self._prepare_model()
        self.model_bridge = self._create_bridge()
        # Merge the functionality of LayerRunner
        self._init_layers()

    def get_model_bridge(self):
        """Get the model bridge."""
        return self.model_bridge

    def prepare_first_layer_input(self, data_list):
        """Prepare the input for the first layer, capture arbitrary parameters, original LayerRunner functionality."""
        first_layer_data = {'inputs': [], 'kwargs_list': []}
        
        def hook_fn(module, args, kwargs):
            # Capture positional and keyword arguments
            first_layer_data['inputs'].append(args)
            first_layer_data['kwargs_list'].append(kwargs)
            raise RunnerStopExecution
        
        hook_handle = self.layers[0].register_forward_pre_hook(hook_fn, with_kwargs=True)
        
        try:
            for data in data_list:
                try:
                    self._run_calibration(data)
                except RunnerStopExecution:
                    pass
        finally:
            hook_handle.remove()
        self.model.cpu()
        return first_layer_data

    def get_layer_by_index(self, index):
        """Get the name of the specified layer."""
        return self.model_bridge.get_layer_by_index(index)

    def create_quantizer(self, model_bridge, layer_map):
        """Create a quantizer."""
        config = FlatQuantQuantizerConfig(
            w_bits=self.args.w_bits, 
            a_bits=self.args.a_bits, 
            w_asym=self.args.w_asym, 
            a_asym=self.args.a_asym, 
            lwc=self.args.lwc, 
            lac=self.args.lac, 
            a_groupsize=self.args.a_groupsize,
            add_diag=self.args.add_diag,
            diag_alpha=self.args.diag_alpha,
            diag_relu=self.args.diag_relu,
            tran_type="inv" if self.args.direct_inv else "svd"
        )
        
        quantizer = quantize_model(model_bridge, layer_map, config)
            
        return quantizer
        
    def get_layer_device(self, layer_idx):
        """Get the device of the specified layer."""
        if self.hf_device_map is not None:
            # Multi-device case, get from device_map
            layer_name = f"model.layers.{layer_idx}"
            if layer_name in self.hf_device_map:
                return self.hf_device_map[layer_name]
        return self.default_device

    def finalize_model(self, quantizer):
        """Finalize model setup: tie weights, set evaluation mode, dispatch model."""
        self.model.tie_weights()
        
        if hasattr(self.model, "hf_device_map"):
            dispatch_model(self.model, self.model.hf_device_map)
        quantizer.to_eval_mode()

    def _prepare_model(self):
        """Prepare the model: set to eval mode, disable gradients."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def _create_bridge(self):
        """Create a model bridge."""
        model_bridge = get_model_bridge(self.model)
        model_bridge.analyze_structure()
        return model_bridge
    
    def _init_layers(self):
        """Initialize layer-related attributes, original LayerRunner functionality."""
        layers_name = self.model_bridge.get_layers()
        self.layers = get_module_by_name(self.model, layers_name)
        self.num_layers = len(self.layers)

    def _run_calibration(self, input_data):
        """Run calibration forward propagation."""
        if isinstance(input_data, tuple) or isinstance(input_data, list):
            with torch.no_grad():
                self.model(*input_data)
        elif isinstance(input_data, dict):
            with torch.no_grad():
                self.model(**input_data)


class DataPreparer:
    """Data preparer, responsible for handling calibration data."""
    def __init__(self, model_adapter, training_config):
        self.model_adapter = model_adapter
        self.training_config = training_config
        
    def prepare_calibration_data(self, calib_data):
        """Prepare calibration data."""
        # Get the input of the first layer
        first_layer_data = self.model_adapter.prepare_first_layer_input(calib_data)
        
        layer_inputs = first_layer_data['inputs']
        layer_kwargs_list = first_layer_data['kwargs_list']
        nsamples = len(layer_inputs)
        
        # Data information
        data_info = {
            'layer_inputs': layer_inputs,
            'layer_kwargs_list': layer_kwargs_list,
            'nsamples': nsamples
        }

        return data_info


class LayerTrainer:
    """Single layer trainer, responsible for quantization training of a single layer."""
    def __init__(self, training_config):
        self.config = training_config
        self.loss_fn = torch.nn.MSELoss()
    
    @staticmethod
    def extract_fp_outputs(layer, layer_inputs, data_info, device):
        """Extract FP model outputs."""
        fp_outs = []
        with torch.no_grad():
            layer.float()
            for j in range(data_info['nsamples']):
                # Move positional arguments to device
                device_args = []
                for arg in layer_inputs[j]:
                    if isinstance(arg, torch.Tensor):
                        device_args.append(arg.to(device))
                    else:
                        device_args.append(arg)
                
                # Move keyword arguments to device
                device_kwargs = {}
                for key, value in data_info['layer_kwargs_list'][j].items():
                    if isinstance(value, torch.Tensor):
                        device_kwargs[key] = value.to(device)
                    else:
                        device_kwargs[key] = value
                
                fp_out = layer(*device_args, **device_kwargs)[0].cpu()
                fp_outs.append(fp_out)
        return fp_outs
        
    def setup_optimizer(self, trainable_params, data_info):
        """Set up optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(trainable_params)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.epochs * data_info['nsamples'], 
            eta_min=self.config.flat_lr * 1e-3
        )
        
        if self.config.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=16
            )
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
        else:
            scheduler = scheduler_main
            
        return optimizer, scheduler
        
    def train_layer(self, training_data: LayerTrainingData, logger):
        """Train a single layer."""
        params, trainable_params, need_train = get_trainable_parameters(training_data.layer, self.config.flat_lr)
        
        if not need_train:
            return convert_outputs_to_inputs(training_data.fp_outs)
            
        optimizer, scheduler = self.setup_optimizer(trainable_params, training_data.data_info)
        quant_outputs = []
        
        for epoch in range(self.config.epochs):
            mse = 0
            epoch_outputs = []
            
            for j in range(training_data.data_info['nsamples']):
                # Move positional arguments to device
                device_args = []
                for arg in training_data.layer_inputs[j]:
                    if isinstance(arg, torch.Tensor):
                        device_args.append(arg.to(training_data.device))
                    else:
                        device_args.append(arg)
                
                # Move keyword arguments to device
                device_kwargs = {}
                for key, value in training_data.data_info['layer_kwargs_list'][j].items():
                    if isinstance(value, torch.Tensor):
                        device_kwargs[key] = value.to(training_data.device)
                    else:
                        device_kwargs[key] = value
                
                with self.config.traincast():
                    quant_output = training_data.layer(*device_args, **device_kwargs)[0]
                    loss = self.loss_fn(quant_output, training_data.fp_outs[j].to(quant_output.device))
                    mse += loss.detach().cpu().item()
                    loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_outputs.append(quant_output.detach().cpu())
                    
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(f"layer {training_data.layer_idx} epoch {epoch}, lr {cur_lr:.8f}, MSE loss: {mse:.8f}")
            quant_outputs = epoch_outputs
            
        if self.config.quant_by_quant:
            return convert_outputs_to_inputs(quant_outputs)
        else:
            return convert_outputs_to_inputs(training_data.fp_outs)
    

class FlatQuantTrainer:
    """Main class for FlatQuant trainer."""
    def __init__(self, model, args, logger):
        self.model_adapter = ModelAdapter(model, args)
        self.training_config = TrainingConfig(args)
        self.layer_trainer = LayerTrainer(self.training_config)
        self.data_preparer = DataPreparer(self.model_adapter, self.training_config)
        self.logger = logger
        self.args = args
        
    def train(self, calib_data, layer_map):
        """Main training process."""
        
        # Prepare model
        model_bridge = self.model_adapter.get_model_bridge()
        
        # Prepare data
        data_info = self.data_preparer.prepare_calibration_data(calib_data)
        
        # Create quantizer
        quantizer = self.model_adapter.create_quantizer(model_bridge, layer_map)
        
        empty_cache()
        
        # Convert input data type
        float_layer_inputs = []
        quant_layer_inputs = []
        
        for args in data_info['layer_inputs']:
            float_args = []
            quant_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    float_args.append(arg.to(self.training_config.dtype))
                    quant_args.append(arg.to(self.training_config.dtype))
                else:
                    float_args.append(arg)
                    quant_args.append(arg)
            float_layer_inputs.append(float_args)
            quant_layer_inputs.append(quant_args)
        
        self.logger.info(self.model_adapter.model)
        
        # Train layer by layer
        for i in tqdm(range(self.model_adapter.num_layers)):
            device = self.model_adapter.get_layer_device(i)
            layer = self.model_adapter.layers[i].to(device)
            

            dtype_dict = {name: param.dtype for name, param in layer.named_parameters()}
            dtype_dict.update({name: buf.dtype for name, buf in layer.named_buffers()})

            quantizer.to_org_mode()
            fp_outs = self.layer_trainer.extract_fp_outputs(layer, float_layer_inputs, data_info, device)
            empty_cache()
            
            # Train quantized model
            quantizer.to_calib_mode(prefix=self.model_adapter.get_layer_by_index(i))
            training_data = LayerTrainingData(
                layer=layer,
                layer_inputs=quant_layer_inputs,
                fp_outs=fp_outs,
                data_info=data_info,
                device=device,
                layer_idx=i
            )
            output_data = self.layer_trainer.train_layer(training_data, self.logger)
            

            if self.training_config.quant_by_quant and output_data != fp_outs:
                quant_layer_inputs = output_data
                float_layer_inputs = output_data
            else:
                # If not trained or not using quant_by_quant, use FP output as input for the next layer
                fp_inputs = convert_outputs_to_inputs(fp_outs)
                if output_data != fp_outs:
                    quant_layer_inputs = fp_inputs
                float_layer_inputs = fp_inputs
                
            layer.cpu()
            for name, param in layer.named_parameters():
                param.requires_grad = False
                if name in dtype_dict:
                    param.data = param.to(dtype_dict[name])
            for name, buf in layer.named_buffers():
                if name in dtype_dict:
                    buf.data = buf.to(dtype_dict[name])
            empty_cache()
            
        self.model_adapter.finalize_model(quantizer)
        empty_cache()


def flat_quant_train(model, calib_data, layer_map, args, logger):
    """layer training enter point"""
    trainer = FlatQuantTrainer(model, args, logger)
    trainer.train(calib_data, layer_map)
