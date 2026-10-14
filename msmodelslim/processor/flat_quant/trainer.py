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
# pylint: disable=logging-fstring-interpolation

import torch
from msmodelslim.utils.logging import get_logger
from msmodelslim.processor.flat_quant.flat_quant_utils.utils import (
    get_trainable_parameters,
    move_tensors_to_device,
    convert_outputs_to_inputs,
)
from msmodelslim.utils.exception import UnexpectedError  # noqa: F401
from typing import List, Dict, Any, Literal, Tuple  # noqa: F401


class LayerTrainer:
    """单层训练器, 允许单层量化校准训练."""

    def __init__(self, config):
        """初始化训练器，设置配置和损失函数"""
        self.config = config
        self.loss_fn = torch.nn.MSELoss()

    def setup_optimizer(self, trainable_params, nsamples):
        """设置优化器和学习率调度器，支持warmup和余弦退火"""
        optimizer = torch.optim.AdamW(trainable_params)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs * nsamples, eta_min=self.config.flat_lr * 1e-3
        )
        if self.config.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=16)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
        else:
            scheduler = scheduler_main
        return optimizer, scheduler

    @torch.enable_grad()
    def train_layer(self, request, float_output):
        """训练单个层的量化参数，通过多次epoch优化损失"""
        nsamples = len(request.datas)
        device = next(request.module.parameters()).device.type
        params, trainable_params, need_train = get_trainable_parameters(request.module, self.config.flat_lr)
        if not need_train:
            return convert_outputs_to_inputs(float_output)
        optimizer, scheduler = self.setup_optimizer(trainable_params, nsamples)

        quant_outputs = []
        for epoch in range(self.config.epochs):
            mse = 0
            epoch_outputs = []
            for j in range(nsamples // self.config.cali_bsz):
                index = j * self.config.cali_bsz

                device_args = []
                device_kwargs = {}
                float_out = []
                for i in range(index, index + self.config.cali_bsz):
                    device_args.append(request.datas[i][0][0])
                    float_out.append(float_output[i])

                float_out = torch.cat(float_out, dim=0)
                device_args = move_tensors_to_device(torch.cat(device_args, dim=0), device)
                device_kwargs = move_tensors_to_device(request.datas[index][1], device)

                with self.config.traincast():
                    quant_output = request.module(device_args, **device_kwargs)[0]
                    loss = self.loss_fn(quant_output.to(device), float_out.to(device))
                    mse += loss.detach().cpu().item()
                    loss = loss / loss.clone().detach()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_outputs.append(quant_output.detach().cpu())

            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            get_logger().info(f"{request.name} epoch {epoch}, lr {cur_lr:.8f}, MSE loss: {mse:.8f}")
            quant_outputs = epoch_outputs  # noqa: F841
        return convert_outputs_to_inputs(float_output)
