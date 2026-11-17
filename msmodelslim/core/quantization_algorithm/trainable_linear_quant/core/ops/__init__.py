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

# 可训练 TLQ Op 包。

from .base import (
    LinearTLQOp,
    SubgraphTLQOp,
    TLQOp,
    TLQOpConfig,
    create_linear_tlq_op,
    create_subgraph_tlq_op,
    is_subgraph_op_config,
    load_tlq_op_class,
    operations_need_adapter_subgraph,
    registered_tlq_op_types,
)
from .minmax_tune import MinmaxTuneOpConfig, MinmaxTuneOp
from .round_tune import RoundTuneOpConfig, RoundTuneOp
from .trainable_smooth import (
    SMOOTH_SCALE_KEY,
    TrainableSmoothOpConfig,
    TrainableSmoothOp,
)

__all__ = [
    "MinmaxTuneOpConfig",
    "MinmaxTuneOp",
    "RoundTuneOpConfig",
    "RoundTuneOp",
    "TrainableSmoothOpConfig",
    "SMOOTH_SCALE_KEY",
    "LinearTLQOp",
    "SubgraphTLQOp",
    "TLQOp",
    "TLQOpConfig",
    "TrainableSmoothOp",
    "create_linear_tlq_op",
    "create_subgraph_tlq_op",
    "is_subgraph_op_config",
    "load_tlq_op_class",
    "operations_need_adapter_subgraph",
    "registered_tlq_op_types",
]
