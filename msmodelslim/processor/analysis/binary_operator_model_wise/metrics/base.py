"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You may use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from abc import abstractmethod
from typing import Any, List

from msmodelslim.processor.analysis.methods_base import LayerAnalysisMethod


class ModelWiseAnalysisMethod(LayerAnalysisMethod):
    """模型级敏感层分析方法基类：仅指标语义（``compute_score``）。

    **Processor**：双路径 batch 构造、路径合并、前向编排；block 输出对齐为张量由 Processor 内建逻辑完成。

    **各 metrics 模块**（如 ``mse.py``）：实现 ``compute_score``。

    扩展新指标：在 ``metrics`` 下新增 ``<name>.py`` 并在 ``factory`` 注册；若编排相同可复用当前 Processor。
    """

    @abstractmethod
    def compute_score(
        self,
        ref_outputs: List[Any],
        cand_outputs: List[Any],
    ) -> float:
        """根据参考输出与候选输出计算单层敏感分数。

        Processor 对每一层分别传入 ``base_count`` 条纯浮点参考与等量量化候选输出。
        """

    def get_hook(self) -> Any:
        """兼容 LayerAnalysisMethod；本分析类型不使用子模块 forward hook。"""
        return None
