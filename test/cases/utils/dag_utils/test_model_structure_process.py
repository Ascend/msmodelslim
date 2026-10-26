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

# msmodelslim.utils.dag_utils.model_structure_process 模块的单元测试

import unittest
from unittest.mock import Mock

from msmodelslim.utils.dag_utils.model_structure_process import StructureProcess


class MockDagNode:
    def __init__(self, name, op_type, in_features=0, out_features=0, inputs=None, input_nodes=None, output_nodes=None):
        self.name_in_network = name
        self.op_type = op_type
        self.node = Mock()
        self.node.in_features = in_features
        self.node.out_features = out_features
        self.inputs = inputs or []
        self.input_nodes = input_nodes or []
        self.output_nodes = output_nodes or []


class TestStructureProcess(unittest.TestCase):
    """测试StructureProcess类"""

    def test_is_ffn_matmul_with_ffn_matmul_num_2_true_case(self):
        """测试is_ffn_matmul函数，ffn_matmul_num=2，返回True的情况"""
        matmul1 = MockDagNode("matmul1", "Linear", in_features=128, out_features=256)
        matmul2 = MockDagNode("matmul2", "Linear", in_features=256, out_features=128)

        matmul_list = [matmul1, matmul2]
        result = StructureProcess.is_ffn_matmul(matmul_list, 2)
        self.assertTrue(result)

    def test_is_ffn_matmul_with_ffn_matmul_num_2_false_case(self):
        """测试is_ffn_matmul函数，ffn_matmul_num=2，返回False的情况"""
        matmul1 = MockDagNode("matmul1", "Linear", in_features=128, out_features=256)
        matmul2 = MockDagNode("matmul2", "Linear", in_features=128, out_features=64)

        matmul_list = [matmul1, matmul2]
        result = StructureProcess.is_ffn_matmul(matmul_list, 2)
        self.assertFalse(result)

    def test_is_ffn_matmul_with_wrong_length(self):
        """测试is_ffn_matmul函数，长度不匹配的情况"""
        matmul1 = MockDagNode("matmul1", "Linear", in_features=128, out_features=256)
        matmul_list = [matmul1]
        result = StructureProcess.is_ffn_matmul(matmul_list, 2)
        self.assertFalse(result)

    def test_is_ffn_matmul_with_ffn_matmul_num_3_true_case1(self):
        """测试is_ffn_matmul函数，ffn_matmul_num=3，返回True的情况1"""
        matmul1 = MockDagNode("matmul1", "Linear", in_features=128, out_features=256)
        matmul2 = MockDagNode("matmul2", "Linear", in_features=256, out_features=128)
        matmul3 = MockDagNode("matmul3", "Linear", in_features=64, out_features=32)

        matmul_list = [matmul1, matmul2, matmul3]
        result = StructureProcess.is_ffn_matmul(matmul_list, 3)
        self.assertTrue(result)

    def test_is_ffn_matmul_with_ffn_matmul_num_3_true_case2(self):
        """测试is_ffn_matmul函数，ffn_matmul_num=3，返回True的情况2"""
        matmul1 = MockDagNode("matmul1", "Linear", in_features=128, out_features=256)
        matmul2 = MockDagNode("matmul2", "Linear", in_features=256, out_features=128)
        matmul3 = MockDagNode("matmul3", "Linear", in_features=256, out_features=128)

        matmul_list = [matmul1, matmul2, matmul3]
        result = StructureProcess.is_ffn_matmul(matmul_list, 3)
        self.assertTrue(result)

    def test_is_ffn_matmul_with_ffn_matmul_num_3_false_case(self):
        """测试is_ffn_matmul函数，ffn_matmul_num=3，返回False的情况"""
        matmul1 = MockDagNode("matmul1", "Linear", in_features=128, out_features=256)
        matmul2 = MockDagNode("matmul2", "Linear", in_features=128, out_features=64)
        matmul3 = MockDagNode("matmul3", "Linear", in_features=32, out_features=16)

        matmul_list = [matmul1, matmul2, matmul3]
        result = StructureProcess.is_ffn_matmul(matmul_list, 3)
        self.assertFalse(result)

    def test_is_ffn_matmul_with_unsupported_ffn_matmul_num(self):
        """测试is_ffn_matmul函数，不支持的ffn_matmul_num"""
        matmul1 = MockDagNode("matmul1", "Linear", in_features=128, out_features=256)
        matmul2 = MockDagNode("matmul2", "Linear", in_features=256, out_features=128)
        matmul3 = MockDagNode("matmul3", "Linear", in_features=128, out_features=256)
        matmul4 = MockDagNode("matmul4", "Linear", in_features=256, out_features=128)

        matmul_list = [matmul1, matmul2, matmul3, matmul4]

        with self.assertRaises(Exception) as context:
            StructureProcess.is_ffn_matmul(matmul_list, 4)
        self.assertIn("unsupported ffn_matmul_num: 4", str(context.exception))

    def test_mhsa_matmul_process_with_proj_mat_out_features_times_3_equal_qkv_mat_out_features(self):
        """测试mhsa_matmul_process函数，proj_mat.out_features * 3 == qkv_mat.out_features的情况"""
        proj_mat = MockDagNode("proj", "Linear", in_features=256, out_features=128)
        qkv_mat = MockDagNode("qkv", "Linear", in_features=128, out_features=384)

        matmul_list = [proj_mat, qkv_mat]
        qkv_list = []
        proj_list = []

        StructureProcess.mhsa_matmul_process(matmul_list, qkv_list, proj_list)

        self.assertEqual(qkv_list, [["qkv"]])
        self.assertEqual(proj_list, ["proj"])

    def test_mhsa_matmul_process_with_proj_mat_out_features_equal_qkv_mat_out_features(self):
        """测试mhsa_matmul_process函数，proj_mat.out_features == qkv_mat.out_features的情况"""
        input_node = MockDagNode("input", "Add")
        qkv1 = MockDagNode("q", "Linear", in_features=128, out_features=256)
        qkv2 = MockDagNode("k", "Linear", in_features=128, out_features=256)
        qkv3 = MockDagNode("v", "Linear", in_features=128, out_features=256)

        input_node.output_nodes = [qkv1, qkv2, qkv3, MockDagNode("other", "ReLU")]

        qkv_mat = MockDagNode("qkv", "Linear", in_features=128, out_features=256)
        qkv_mat.input_nodes = [input_node]
        qkv_mat.inputs = [Mock()]

        proj_mat = MockDagNode("proj", "Linear", in_features=256, out_features=256)

        matmul_list = [proj_mat, qkv_mat]
        qkv_list = []
        proj_list = []

        StructureProcess.mhsa_matmul_process(matmul_list, qkv_list, proj_list)

        self.assertEqual(qkv_list, [["q", "k", "v"]])
        self.assertEqual(proj_list, ["proj"])

    def test_mhsa_matmul_process_with_wrong_length(self):
        """测试mhsa_matmul_process函数，matmul_list长度不正确"""
        matmul1 = MockDagNode("matmul1", "Linear", in_features=128, out_features=256)
        matmul_list = [matmul1]
        qkv_list = []
        proj_list = []

        StructureProcess.mhsa_matmul_process(matmul_list, qkv_list, proj_list)

        self.assertEqual(qkv_list, [])
        self.assertEqual(proj_list, [])

    def test_mhsa_matmul_process_with_qkv_mat_inputs_not_equal_1(self):
        """测试mhsa_matmul_process函数，qkv_mat.inputs长度不为1"""
        qkv_mat = MockDagNode("qkv", "Linear", in_features=128, out_features=256)
        qkv_mat.inputs = [Mock(), Mock()]

        proj_mat = MockDagNode("proj", "Linear", in_features=256, out_features=256)

        matmul_list = [proj_mat, qkv_mat]
        qkv_list = []
        proj_list = []

        StructureProcess.mhsa_matmul_process(matmul_list, qkv_list, proj_list)

        self.assertEqual(qkv_list, [])
        self.assertEqual(proj_list, [])

    def test_mhsa_matmul_process_with_wrong_qkv_mat_list_length(self):
        """测试mhsa_matmul_process函数，qkv_mat_list长度不为3"""
        input_node = MockDagNode("input", "Add")
        qkv1 = MockDagNode("q", "Linear", in_features=128, out_features=256)
        qkv2 = MockDagNode("k", "Linear", in_features=128, out_features=256)

        input_node.output_nodes = [qkv1, qkv2, MockDagNode("other", "ReLU")]

        qkv_mat = MockDagNode("qkv", "Linear", in_features=128, out_features=256)
        qkv_mat.input_nodes = [input_node]
        qkv_mat.inputs = [Mock()]

        proj_mat = MockDagNode("proj", "Linear", in_features=256, out_features=256)

        matmul_list = [proj_mat, qkv_mat]
        qkv_list = []
        proj_list = []

        StructureProcess.mhsa_matmul_process(matmul_list, qkv_list, proj_list)

        self.assertEqual(qkv_list, [])
        self.assertEqual(proj_list, [])

    def test_mhsa_matmul_ln_process_with_proj_mat_out_features_times_3_equal_qkv_mat_out_features(self):
        """测试mhsa_matmul_ln_process函数，proj_mat.out_features * 3 == qkv_mat.out_features的情况"""
        proj_mat = MockDagNode("proj", "Linear", in_features=256, out_features=128)
        qkv_mat = MockDagNode("qkv", "Linear", in_features=128, out_features=384)

        ln1 = MockDagNode("ln", "LayerNorm")

        matmul_list = [proj_mat, qkv_mat]
        ln_list = [ln1]
        qkv_list = []
        proj_list = []
        mhsa_ln_list = []

        StructureProcess.mhsa_matmul_ln_process(matmul_list, ln_list, qkv_list, proj_list, mhsa_ln_list)

        self.assertEqual(qkv_list, [["qkv"]])
        self.assertEqual(proj_list, ["proj"])
        self.assertEqual(mhsa_ln_list, ["ln"])

    def test_mhsa_matmul_ln_process_with_proj_mat_out_features_equal_qkv_mat_out_features(self):
        """测试mhsa_matmul_ln_process函数，proj_mat.out_features == qkv_mat.out_features的情况"""
        input_node = MockDagNode("input", "Add")
        qkv1 = MockDagNode("q", "Linear", in_features=128, out_features=256)
        qkv2 = MockDagNode("k", "Linear", in_features=128, out_features=256)
        qkv3 = MockDagNode("v", "Linear", in_features=128, out_features=256)

        input_node.output_nodes = [qkv1, qkv2, qkv3, MockDagNode("other", "ReLU")]

        qkv_mat = MockDagNode("qkv", "Linear", in_features=128, out_features=256)
        qkv_mat.input_nodes = [input_node]
        qkv_mat.inputs = [Mock()]

        proj_mat = MockDagNode("proj", "Linear", in_features=256, out_features=256)
        ln1 = MockDagNode("ln", "LayerNorm")

        matmul_list = [proj_mat, qkv_mat]
        ln_list = [ln1]
        qkv_list = []
        proj_list = []
        mhsa_ln_list = []

        StructureProcess.mhsa_matmul_ln_process(matmul_list, ln_list, qkv_list, proj_list, mhsa_ln_list)

        self.assertEqual(qkv_list, [["q", "k", "v"]])
        self.assertEqual(proj_list, ["proj"])
        self.assertEqual(mhsa_ln_list, ["ln"])

    def test_mhsa_matmul_ln_process_with_wrong_length(self):
        """测试mhsa_matmul_ln_process函数，matmul_list长度不正确"""
        matmul1 = MockDagNode("matmul1", "Linear", in_features=128, out_features=256)
        matmul_list = [matmul1]
        ln_list = [MockDagNode("ln", "LayerNorm")]
        qkv_list = []
        proj_list = []
        mhsa_ln_list = []

        StructureProcess.mhsa_matmul_ln_process(matmul_list, ln_list, qkv_list, proj_list, mhsa_ln_list)

        self.assertEqual(qkv_list, [])
        self.assertEqual(proj_list, [])
        self.assertEqual(mhsa_ln_list, [])

    def test_mhsa_matmul_ln_process_with_qkv_mat_inputs_not_equal_1(self):
        """测试mhsa_matmul_ln_process函数，qkv_mat.inputs长度不为1"""
        qkv_mat = MockDagNode("qkv", "Linear", in_features=128, out_features=256)
        qkv_mat.inputs = [Mock(), Mock()]

        proj_mat = MockDagNode("proj", "Linear", in_features=256, out_features=256)
        ln1 = MockDagNode("ln", "LayerNorm")

        matmul_list = [proj_mat, qkv_mat]
        ln_list = [ln1]
        qkv_list = []
        proj_list = []
        mhsa_ln_list = []

        StructureProcess.mhsa_matmul_ln_process(matmul_list, ln_list, qkv_list, proj_list, mhsa_ln_list)

        self.assertEqual(qkv_list, [])
        self.assertEqual(proj_list, [])
        self.assertEqual(mhsa_ln_list, [])

    def test_mhsa_matmul_ln_process_with_wrong_qkv_mat_list_length(self):
        """测试mhsa_matmul_ln_process函数，qkv_mat_list长度不为3"""
        input_node = MockDagNode("input", "Add")
        qkv1 = MockDagNode("q", "Linear", in_features=128, out_features=256)
        qkv2 = MockDagNode("k", "Linear", in_features=128, out_features=256)

        input_node.output_nodes = [qkv1, qkv2, MockDagNode("other", "ReLU")]

        qkv_mat = MockDagNode("qkv", "Linear", in_features=128, out_features=256)
        qkv_mat.input_nodes = [input_node]
        qkv_mat.inputs = [Mock()]

        proj_mat = MockDagNode("proj", "Linear", in_features=256, out_features=256)
        ln1 = MockDagNode("ln", "LayerNorm")

        matmul_list = [proj_mat, qkv_mat]
        ln_list = [ln1]
        qkv_list = []
        proj_list = []
        mhsa_ln_list = []

        StructureProcess.mhsa_matmul_ln_process(matmul_list, ln_list, qkv_list, proj_list, mhsa_ln_list)

        self.assertEqual(qkv_list, [])
        self.assertEqual(proj_list, [])
        self.assertEqual(mhsa_ln_list, [])


if __name__ == '__main__':
    unittest.main()
