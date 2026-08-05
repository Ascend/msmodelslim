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

from typing import List, Protocol

from msmodelslim.core.graph.adapter_types import AdapterConfig
from msmodelslim.processor.trainable_linear_quant.data import TLQBlockDataInterface


class TLQSubgraphAdapter(Protocol):
    """模型 adapter 的可选能力：为 subgraph TLQ op 提供子图拓扑。

    仅当 ``operations`` 含 subgraph op（如 ``trainable_smooth``）时由 Processor 调用；
    默认 ``[minmax_tune, round_tune]`` 无需实现此方法，adapter 也可为 ``None``。
    与 ``IterSmoothInterface.get_adapter_config_for_subgraph`` 签名一致。
    """

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]: ...


__all__ = [
    "TLQBlockDataInterface",
    "TLQSubgraphAdapter",
]
