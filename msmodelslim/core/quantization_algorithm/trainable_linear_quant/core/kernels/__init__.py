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

# TLQ 数学内核：可监听伪量化流水线 + dtype 族注册。

from .base import FakeQuantDriver, FakeQuantListener, FakeQuantStage, TLQKernel
from .context import (
    ContextFactory,
    FakeQuantContext,
    FakeQuantPipelineContext,
    FakeQuantResult,
    KernelFamily,
    kernel_family_from_config,
)
from .int import (
    IntFakeQuantContext,
    build_int_context,
    create_int_kernel,
    int_fake_quant_driver,
    int_max_bound,
)
from .mxfp import (
    MxFakeQuantContext,
    build_mx_context,
    create_mxfp_kernel,
    mx_block_size,
    mxfp_fake_quant_driver,
)
from .registry import (
    TLQKernelFactory,
    create_tlq_kernel,
    ensure_tlq_kernel_registered,
    register_tlq_kernel,
    registered_tlq_dispatch_keys,
    tlq_kernel_dispatch_key,
)

__all__ = [
    "ContextFactory",
    "FakeQuantDriver",
    "FakeQuantContext",
    "FakeQuantListener",
    "FakeQuantPipelineContext",
    "FakeQuantResult",
    "FakeQuantStage",
    "IntFakeQuantContext",
    "KernelFamily",
    "kernel_family_from_config",
    "MxFakeQuantContext",
    "TLQKernel",
    "TLQKernelFactory",
    "build_int_context",
    "build_mx_context",
    "create_int_kernel",
    "create_mxfp_kernel",
    "create_tlq_kernel",
    "ensure_tlq_kernel_registered",
    "int_fake_quant_driver",
    "int_max_bound",
    "mx_block_size",
    "mxfp_fake_quant_driver",
    "register_tlq_kernel",
    "registered_tlq_dispatch_keys",
    "tlq_kernel_dispatch_key",
]
