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
import copy
from collections import deque, UserDict

import torch
from torch import autocast

from msmodelslim.core.quantization_algorithm.quant.autoround_utils.sign_sgd import SignSGD
from msmodelslim.utils.logging import get_logger


class BlockQuantTrainer:
    """
    用于训练模型块的类，实现量化参数优化
    """

    def __init__(
        self,
        batch_size: int = 1,
        iters: int = 2,
        enable_minmax_tuning: bool = False,
        enable_quanted_input: bool = True,
        enable_ema_loss: bool = True,
        lr: float = None,
        minmax_lr: float = None,
        sampler: str = 'rand',
        seqlen: int = 2048,
        gradient_accumulate_steps: int = 1,
        batch_dim: int = 0,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        not_use_best_mse: bool = False,
        dynamic_max_gap: int = -1,
        infer_bs_coeff: int = 1,
        shared_cache_keys: tuple = (),
        lr_scheduler=None,
    ):
        self.enable_minmax_tuning = enable_minmax_tuning
        self.enable_quanted_input = enable_quanted_input
        self.enable_ema_loss = enable_ema_loss
        self.iters = iters
        if self.iters < 0:
            self.iters = 200
            get_logger().info(
                "The number of iterations is invalid, using default value: %s",
                self.iters,
            )
        if iters == 0:
            self.lr = 5e-3
        else:
            self.lr = lr or (1.0 / self.iters)
        self.minmax_lr = minmax_lr or self.lr
        self.iters = iters
        self.batch_size = batch_size
        self.gradient_accumulate_steps = gradient_accumulate_steps
        self.sampler = sampler
        self.seqlen = seqlen
        self.batch_dim = batch_dim
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.not_use_best_mse = not_use_best_mse
        self.dynamic_max_gap = dynamic_max_gap
        self.infer_bs_coeff = infer_bs_coeff
        self.shared_cache_keys = shared_cache_keys
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = lr_scheduler
        self.lr_multiplier = 5

    @staticmethod
    def get_optimizer():
        return SignSGD

    @staticmethod
    def scale_loss_and_backward(loss):
        """Scales the loss and performs backward pass.

        Args:
        loss: The loss to be scaled.

        Returns:
        The scaled loss.
        """
        scale_loss = loss * 1000
        scale_loss.backward()
        return scale_loss

    @staticmethod
    def step(optimizer, lr_schedule):
        """Performs a step in the optimization process.

        Args:
        optimizer: The optimizer for the step.
        lr_schedule: The learning rate schedule.

        Returns:
        None
        """
        optimizer.step()
        optimizer.zero_grad()
        lr_schedule.step()

    @staticmethod
    def collect_best_params(block):
        params = {}
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                params[n] = {}
                for key in m.params.keys():
                    params[n][key] = copy.deepcopy(m.params[key].data)
        return params

    @torch.no_grad()
    def sampling_inputs(self, input_ids, input_others, indices, batch_dim=0, share_cache_keys=()):
        """Samples inputs based on the given indices and sequence length.

        Args:
        input_ids: The list of input tensor containing  input_ids.
        input_others: A dictionary containing other input data.
        indices: The indices to sample from the input.

        Returns:
        current_input_ids: The sampled input IDs.
        current_input_others: The sampled other input data.
        """
        current_input_ids = [input_ids[i] for i in indices]

        current_input_ids = torch.cat(current_input_ids, dim=batch_dim)

        current_input_others = {"positional_inputs": input_others["positional_inputs"]}
        for key in input_others.keys():
            if "positional_inputs" in key:
                continue
            if (key not in share_cache_keys or len(indices) == 1) and not isinstance(
                input_others[key], (str, bool, type(None))
            ):
                current_input_others[key] = None
                if input_others[key] is not None:
                    current_input_others[key] = [input_others[key][i] for i in indices]
                    if len(indices) == 1:
                        current_input_others[key] = current_input_others[key][0]
                    else:
                        try:
                            current_input_others[key] = torch.cat(current_input_others[key], dim=0)
                        except TypeError:
                            pass
            else:
                current_input_others[key] = input_others[key]

        return current_input_ids, current_input_others

    @torch.enable_grad()
    def train_block(self, block, input_ids, input_others, float_output, device):
        """训练模型块以优化量化参数

        Args:
            block: 要训练的模型块
            input_ids: 输入ID
            input_others: 其他输入数据
            float_output: 原始浮点输出
            device: 计算设备

        Returns:
            tuple: (量化输出, 原始输出)
        """

        # 收集需要优化的参数
        round_params = []
        minmax_params = []
        for _, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                for key in m.params.keys():
                    if "min" in key or "max" in key:
                        minmax_params.append(m.params[key])
                    else:
                        round_params.append(m.params[key])

        # 设置优化器
        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": self.minmax_lr * self.lr_multiplier}],
                lr=self.lr * self.lr_multiplier,
                weight_decay=0,
            )
        else:
            optimizer = self.optimizer(round_params, lr=self.lr * self.lr_multiplier, weight_decay=0)

        # 设置学习率调度器
        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        # 初始化训练参数
        nsamples = len(input_ids)
        pick_samples = self.batch_size * self.gradient_accumulate_steps
        pick_samples = min(nsamples, pick_samples)

        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        num_elm = 1
        mse_reduction = "mean"

        whole_indices = torch.randperm(nsamples)[:pick_samples]

        if self.gradient_accumulate_steps != 1:
            mse_reduction = "sum"

        mse_loss = torch.nn.L1Loss(reduction="none")
        init_loss = best_loss
        best_params = {}
        total_loss = 0

        # laos添加的参数
        ema_beta = 0.7
        window_size = 5
        loss_history = deque(maxlen=window_size)
        input_others["output_attentions"] = False

        # 开始训练迭代
        for i in range(self.iters):
            total_loss = 0
            if self.sampler == "rand":
                whole_indices = torch.randperm(nsamples)[:pick_samples]
                # 假设块输入和输出形状相同
                if self.gradient_accumulate_steps != 1:
                    current_input_ids = [input_ids[i] for i in whole_indices]
                    num_elm = sum(id.numel() for id in current_input_ids)

            for tmp_step in range(self.gradient_accumulate_steps):
                indices = whole_indices[tmp_step * self.batch_size : (tmp_step + 1) * self.batch_size]
                if len(indices) == 0:
                    indices = [0]
                current_input_ids, current_input_others = self.sampling_inputs(
                    input_ids,
                    input_others,
                    indices,
                    batch_dim=self.batch_dim,
                    share_cache_keys=self.shared_cache_keys,
                )

                current_output = [float_output[x] for x in indices]
                current_output = torch.cat(current_output, dim=self.batch_dim)
                current_output = self.to_device(current_output, device)

                if "attention_mask" in current_input_others and current_input_others["attention_mask"] is not None:
                    valid_mask = current_input_others["attention_mask"].squeeze(1)[:, -1] == 0
                else:
                    batch_size, seq_len, _ = current_input_ids.shape
                    valid_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

                # block_forward
                if current_input_ids.device != device:
                    current_input_ids = self.to_device(current_input_ids, device)
                    current_input_others = self.to_device(current_input_others, device)
                current_input_tuple = current_input_others.pop("positional_inputs", None)
                if self.amp:
                    with autocast(device_type=device.split(":")[0], dtype=self.amp_dtype):  # pragma: no cover
                        output_q = block(current_input_ids, *current_input_tuple, **current_input_others)
                else:
                    output_q = block(current_input_ids, *current_input_tuple, **current_input_others)
                if isinstance(output_q, (list, tuple)):
                    output_q = output_q[0]

                output_q = self.to_device(output_q, device)  # 不加这个在npu上报错

                if self.amp:
                    with autocast(device_type=device.split(":")[0], dtype=self.amp_dtype):
                        loss = mse_loss(output_q, current_output)
                else:
                    loss = mse_loss(output_q.to(torch.float32), current_output.to(torch.float32))

                get_logger().debug("Loss for iter %s: %s", i, total_loss)

                valid_mask = valid_mask.to(loss.device)

                emb_dim = loss.shape[-1]
                loss_mask, num_valid = (valid_mask, valid_mask.sum(-1))
                loss = (loss.sum(-1) * loss_mask).sum(-1)
                loss = loss.sum() if mse_reduction == 'sum' else (loss / num_valid).mean() / emb_dim
                self.scale_loss_and_backward(loss)
                total_loss += loss.item() / num_elm

            # 记录初始损失
            if i == 0:
                init_loss = total_loss
                best_params = self.collect_best_params(block)
                if self.enable_ema_loss:
                    best_loss = total_loss / max(self.gradient_accumulate_steps, 1)

            if self.enable_ema_loss:
                total_val = total_loss / self.gradient_accumulate_steps
                loss_history.append(total_val)
                avg_loss = sum(loss_history) / len(loss_history)
                new_best_loss = ema_beta * best_loss + (1 - ema_beta) * avg_loss
                if avg_loss < best_loss:
                    best_loss = new_best_loss
                    if not self.not_use_best_mse:
                        best_params = self.collect_best_params(block)
                        last_best_iter = i
            else:
                if total_loss < best_loss:
                    best_loss = total_loss
                    if not self.not_use_best_mse:
                        best_params = self.collect_best_params(block)
                        last_best_iter = i

            if self.not_use_best_mse and i == self.iters - 1:
                best_params = self.collect_best_params(block)

            # 检查是否提前停止
            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break

            self.step(optimizer, lr_schedule)

        # 获取最终结果
        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter

        # 打印logger信息
        get_logger().info(
            "Training completed for block, loss iter 0: %.6f -> iter %s: %.6f",
            init_loss,
            best_iter,
            last_loss,
        )

        return best_params

    def to_device(self, inputs, device=torch.device("cpu")):
        """Moves input data to the specified device.

        Args:
        input: The input data to be moved.
        device: The target device.

        Returns:
        The input data on the specified device.
        """
        if inputs is None:
            return None
        if isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        if isinstance(inputs, (dict, UserDict)):
            for inp in inputs.keys():
                inputs[inp] = self.to_device(inputs[inp], device)

        elif isinstance(inputs, (list, tuple)):
            if len(inputs) == 0:
                return inputs
            inputs_res = []
            for inp in inputs:
                inputs_res.append(self.to_device(inp, device))
            if isinstance(inputs, tuple):
                inputs_res = tuple(inputs_res)
            inputs = inputs_res

        return inputs
