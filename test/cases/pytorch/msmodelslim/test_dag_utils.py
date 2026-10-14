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
# pylint: disable=ungrouped-imports

import unittest
from unittest.mock import MagicMock

from typing import Tuple  # noqa: F401
from unittest.mock import patch  # noqa: F401
from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.model_infos import ModuleType
from msmodelslim.pytorch.llm_ptq.anti_outlier.dag_utils.model_structure_process import StructureProcess
from ascend_utils.core.dag.dag_node import DagNode


class TestStructureProcess(unittest.TestCase):
    def test_is_ffn_matmul_2_layers_valid(self):
        """测试2层FFN结构的有效情况

        验证：
        1. 当输入2个matmul节点且维度匹配时返回True
        2. 正确检查输入/输出特征维度关系
        """
        # 创建模拟节点
        prev_mat = MagicMock(spec=DagNode)
        prev_mat.node.in_features = 512
        prev_mat.node.out_features = 1024

        after_mat = MagicMock(spec=DagNode)
        after_mat.node.in_features = 1024
        after_mat.node.out_features = 512

        # 测试有效情况
        self.assertTrue(StructureProcess.is_ffn_matmul([prev_mat, after_mat], 2))

    def test_is_ffn_matmul_2_layers_invalid(self):
        """测试2层FFN结构的无效情况

        验证：
        1. 当维度不匹配时返回False
        2. 正确处理维度检查逻辑
        """
        # 创建维度不匹配的模拟节点
        prev_mat = MagicMock(spec=DagNode)
        prev_mat.node.in_features = 512
        prev_mat.node.out_features = 1024

        after_mat = MagicMock(spec=DagNode)
        after_mat.node.in_features = 512  # 不匹配
        after_mat.node.out_features = 512

        # 测试无效情况
        self.assertFalse(StructureProcess.is_ffn_matmul([prev_mat, after_mat], 2))

    def test_is_ffn_matmul_3_layers_invalid(self):
        """测试3层FFN结构的无效情况

        验证：
        1. 当不满足Llama结构时返回False
        2. 正确处理无效结构的情况
        """
        # 创建模拟节点 (Llama结构)
        mat1 = MagicMock(spec=DagNode)
        mat1.node.in_features = 512
        mat1.node.out_features = 1024

        mat2 = MagicMock(spec=DagNode)
        mat2.node.in_features = 1024
        mat2.node.out_features = 2048

        mat3 = MagicMock(spec=DagNode)
        mat3.node.in_features = 2048
        mat3.node.out_features = 1024

        # 测试无效情况
        self.assertFalse(StructureProcess.is_ffn_matmul([mat1, mat2, mat3], 3))

    def test_is_ffn_matmul_3_layers_valid(self):
        """测试3层FFN结构(Llama风格)的有效情况

        验证：
        1. 当输入3个matmul节点且满足Llama结构时返回True
        2. 正确处理case1和case2条件判断
        """
        # 创建模拟节点
        mat1 = MagicMock(spec=DagNode)
        mat1.node.in_features = 512
        mat1.node.out_features = 512

        mat2 = MagicMock(spec=DagNode)
        mat2.node.in_features = 512
        mat2.node.out_features = 512

        mat3 = MagicMock(spec=DagNode)
        mat3.node.in_features = 512
        mat3.node.out_features = 512

        # 测试有效情况
        self.assertTrue(StructureProcess.is_ffn_matmul([mat1, mat2, mat3], 3))

    def test_mhsa_matmul_process_qkv_proj(self):
        """测试MHSA中QKV和Proj矩阵处理(标准情况)

        验证：
        1. 正确处理标准QKV结构(proj.out*3 == qkv.out)
        2. 正确填充qkv_list和proj_list
        """
        # 创建模拟节点
        proj_mat = MagicMock(spec=DagNode)
        proj_mat.node.out_features = 512
        proj_mat.name_in_network = "proj"

        qkv_mat = MagicMock(spec=DagNode)
        qkv_mat.node.out_features = 1536  # 512 * 3
        qkv_mat.name_in_network = "qkv"
        qkv_mat.inputs = [MagicMock()]

        qkv_list = []
        proj_list = []

        StructureProcess.mhsa_matmul_process([proj_mat, qkv_mat], qkv_list, proj_list)

        # 验证结果
        self.assertEqual(qkv_list, [["qkv"]])
        self.assertEqual(proj_list, ["proj"])

    def test_mhsa_matmul_process_separate_qkv(self):
        """测试MHSA中分离的QKV矩阵处理

        验证：
        1. 正确处理分离的QKV结构(proj.out == qkv.out)
        2. 正确识别3个线性层作为QKV
        3. 正确填充qkv_list和proj_list
        """
        # 创建模拟节点
        proj_mat = MagicMock(spec=DagNode)
        proj_mat.node.out_features = 512
        proj_mat.name_in_network = "proj"

        qkv_mat = MagicMock(spec=DagNode)
        qkv_mat.node.out_features = 512
        qkv_mat.name_in_network = "qkv_mat"

        # 创建3个线性层节点(Q/K/V)
        q_linear = MagicMock(spec=DagNode)
        q_linear.op_type = ModuleType.LINEAR
        q_linear.name_in_network = "q"
        k_linear = MagicMock(spec=DagNode)
        k_linear.op_type = ModuleType.LINEAR
        k_linear.name_in_network = "k"
        v_linear = MagicMock(spec=DagNode)
        v_linear.op_type = ModuleType.LINEAR
        v_linear.name_in_network = "v"

        # 设置输入节点关系
        input_node = MagicMock()
        input_node.output_nodes = [q_linear, k_linear, v_linear]
        qkv_mat.input_nodes = [input_node]
        qkv_mat.inputs = [input_node]  # 确保inputs也被设置

        qkv_list = []
        proj_list = []

        StructureProcess.mhsa_matmul_process([proj_mat, qkv_mat], qkv_list, proj_list)

        # 验证结果
        self.assertEqual(len(qkv_list), 1)  # 确保qkv_list有内容
        self.assertEqual(len(qkv_list[0]), 3)  # 确保有3个QKV矩阵
        self.assertEqual(sorted(qkv_list[0]), sorted(["q", "k", "v"]))
        self.assertEqual(proj_list, ["proj"])

    def test_mhsa_matmul_process_invalid_length(self):
        """测试MHSA处理中matmul数量不正确的情况

        验证：
        1. 当matmul数量不等于2时不做任何处理
        """
        qkv_list = []
        proj_list = []

        StructureProcess.mhsa_matmul_process([MagicMock()], qkv_list, proj_list)

        # 验证列表未被修改
        self.assertEqual(len(qkv_list), 0)
        self.assertEqual(len(proj_list), 0)

    def test_mhsa_matmul_ln_process_with_ln(self):
        """测试带LayerNorm的MHSA处理

        验证：
        1. 正确处理带LN的标准QKV结构
        2. 正确填充mhsa_ln_list
        """
        # 创建模拟节点
        proj_mat = MagicMock(spec=DagNode)
        proj_mat.node.out_features = 512
        proj_mat.name_in_network = "proj"

        qkv_mat = MagicMock(spec=DagNode)
        qkv_mat.node.out_features = 1536  # 512 * 3
        qkv_mat.name_in_network = "qkv"
        qkv_mat.inputs = [MagicMock()]

        ln_node = MagicMock(spec=DagNode)
        ln_node.name_in_network = "ln"

        qkv_list = []
        proj_list = []
        mhsa_ln_list = []

        StructureProcess.mhsa_matmul_ln_process([proj_mat, qkv_mat], [ln_node], qkv_list, proj_list, mhsa_ln_list)

        # 验证结果
        self.assertEqual(qkv_list, [["qkv"]])
        self.assertEqual(proj_list, ["proj"])
        self.assertEqual(mhsa_ln_list, ["ln"])

    def test_mhsa_matmul_ln_process_invalid_qkv(self):
        """测试无效QKV结构的MHSA处理

        验证：
        1. 当QKV结构无效时不填充任何列表
        """
        # 创建无效的模拟节点
        proj_mat = MagicMock(spec=DagNode)
        proj_mat.node.out_features = 512

        qkv_mat = MagicMock(spec=DagNode)
        qkv_mat.node.out_features = 1024  # 既不是512 * 3也不是有效分离结构
        qkv_mat.inputs = [MagicMock()]

        qkv_list = []
        proj_list = []
        mhsa_ln_list = []

        StructureProcess.mhsa_matmul_ln_process([proj_mat, qkv_mat], [MagicMock()], qkv_list, proj_list, mhsa_ln_list)

        # 验证列表未被修改
        self.assertEqual(len(qkv_list), 0)
        self.assertEqual(len(proj_list), 0)
        self.assertEqual(len(mhsa_ln_list), 0)


if __name__ == '__main__':
    unittest.main()
