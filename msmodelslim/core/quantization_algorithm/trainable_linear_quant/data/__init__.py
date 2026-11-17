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

# Block-level data contracts for trainable linear quant.

from .block_data import (
    BlockInput,
    BlockOutput,
    DefaultTLQBlockDataInterface,
    TLQBlockDataInterface,
    propagate_outputs_to_inputs,
    resolve_tlq_block_data_interface,
)

__all__ = [
    "BlockInput",
    "BlockOutput",
    "DefaultTLQBlockDataInterface",
    "TLQBlockDataInterface",
    "propagate_outputs_to_inputs",
    "resolve_tlq_block_data_interface",
]
