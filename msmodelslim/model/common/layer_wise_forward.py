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
# pylint: disable=consider-merging-isinstance,reimported,unpacking-non-sequence,unsubscriptable-object

from collections.abc import Generator, Iterable
from typing import Tuple, Dict, Optional, Any

import torch.nn
from torch import nn, distributed as dist

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.utils.exception import InvalidModelError


class TransformersForwardBreak(Exception):
    """内部使用的异常类，用于中断前向传播并捕获输入"""

    pass


def generated_decoder_layer_visit_func_with_keyword(
    model: torch.nn.Module,
    keyword: str = "decoderlayer",
) -> Generator[ProcessRequest, Any, None]:
    transformer_blocks = [
        (name, module) for name, module in model.named_modules() if keyword in module.__class__.__name__.lower()
    ]
    return generated_decoder_layer_visit_func(model, transformer_blocks=transformer_blocks)


def generated_decoder_layer_visit_func(
    model: torch.nn.Module,
    transformer_blocks: Optional[Iterable[Tuple[str, torch.nn.Module]]] = None,
) -> Generator[ProcessRequest, Any, None]:
    if transformer_blocks is None:
        transformer_blocks = [
            (name, module)
            for name, module in model.named_modules()
            if 'decoderlayer' in module.__class__.__name__.lower()
        ]

    if dist.is_initialized():
        dist.barrier()

    for name, block in transformer_blocks:
        yield ProcessRequest(name, block, tuple(), {})


def transformers_generated_forward_func(
    model: torch.nn.Module,
    inputs: Any,
) -> Generator[ProcessRequest, Any, None]:
    transformer_blocks = [
        (name, module) for name, module in model.named_modules() if "decoderlayer" in module.__class__.__name__.lower()
    ]

    # 存储第一个transformer block的输入
    first_block_input = None

    def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
        nonlocal first_block_input
        first_block_input = (
            hook_args,
            hook_kwargs,
        )
        raise TransformersForwardBreak()

    hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)]

    # 执行一次前向传播以获取输入
    try:
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            model(*inputs)
        elif isinstance(inputs, dict):
            model(**inputs)
        else:
            model(inputs)
    except TransformersForwardBreak:
        pass
    except Exception as e:
        raise e
    finally:
        for hook in hooks:
            hook.remove()

    if first_block_input is None:
        raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

    # 循环处理每个transformer block
    current_inputs = first_block_input

    if dist.is_initialized():
        dist.barrier()

    for name, block in transformer_blocks:
        args, kwargs = current_inputs
        outputs = yield ProcessRequest(name, block, args, kwargs)
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        current_inputs = ((hidden_states,), current_inputs[1])
