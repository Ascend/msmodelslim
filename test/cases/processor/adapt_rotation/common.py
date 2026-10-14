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
# pylint: disable=pointless-string-statement

"""
adapt_rotation 测试共用的 mock 与工具。
"""
from msmodelslim.processor.adapt_rotation.interface import AdaptRotationInterface  # noqa: E402


class MockQuaRotAdapter(AdaptRotationInterface):
    """实现 AdaptRotationInterface 的 mock 适配器，供 adapt_rotation 相关单测共用。"""

    def __init__(self, hidden_dim: int = 4):
        self._hidden_dim = hidden_dim

    def get_hidden_dim(self):
        return self._hidden_dim

    def get_ln_fuse_map(self):
        return {}, {}

    def get_bake_names(self):
        return [], []

    def get_rotate_map(self, block_size: int):
        return [], []
