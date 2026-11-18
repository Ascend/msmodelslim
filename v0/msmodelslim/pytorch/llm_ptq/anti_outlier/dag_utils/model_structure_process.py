#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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


from typing import Tuple

from ascend_utils.core.dag.dag_node import DagNode
from .model_infos import ModuleType


class StructureProcess:
    @staticmethod
    def is_ffn_matmul(matmul_list,
                        ffn_matmul_num):
        if len(matmul_list) != ffn_matmul_num:
            return False
        if ffn_matmul_num == 2:
            prev_mat, after_mat = matmul_list

            if prev_mat.node.in_features != after_mat.node.out_features \
                    or prev_mat.node.out_features != after_mat.node.in_features:
                return False
        elif ffn_matmul_num == 3:
            # LLAMA support
            case1 = (matmul_list[0].node.in_features == matmul_list[1].node.out_features) or \
                    (matmul_list[1].node.out_features == matmul_list[2].node.out_features)
            case2 = (matmul_list[0].node.out_features == matmul_list[1].node.in_features) or \
                    (matmul_list[1].node.in_features == matmul_list[2].node.in_features)
            if case1 and case2:
                return True
            else:
                return False
        else:
            raise Exception(f"unsupported ffn_matmul_num: {ffn_matmul_num}")
        return True

    @staticmethod
    def mhsa_matmul_process(matmul_list: Tuple[DagNode],
                            qkv_list,
                            proj_list):
        expected_matmul_lang = 2
        if len(matmul_list) != expected_matmul_lang:
            return

        proj_mat, qkv_mat = matmul_list
        if proj_mat.node.out_features * 3 == qkv_mat.node.out_features:
            qkv_list.append([qkv_mat.name_in_network])
            proj_list.append(proj_mat.name_in_network)
        elif proj_mat.node.out_features == qkv_mat.node.out_features:
            if len(qkv_mat.inputs) != 1:
                return
            input_nodes = [in_node for in_node in qkv_mat.input_nodes]
            qkv_mat_list = [item for item in input_nodes[0].output_nodes if item.op_type == ModuleType.LINEAR]
            if len(qkv_mat_list) != 3:
                return

            qkv_list.append([item.name_in_network for item in qkv_mat_list])
            proj_list.append(proj_mat.name_in_network)

    @staticmethod
    def mhsa_matmul_ln_process(matmul_list: Tuple[DagNode],
                               ln_list,
                               qkv_list,
                               proj_list,
                               mhsa_ln_list):
        expected_matmul_lang = 2
        if len(matmul_list) != expected_matmul_lang:
            return

        proj_mat, qkv_mat = matmul_list
        if proj_mat.node.out_features * 3 == qkv_mat.node.out_features:
            qkv_list.append([qkv_mat.name_in_network])
            proj_list.append(proj_mat.name_in_network)
            mhsa_ln_list.append(ln_list[0].name_in_network)
        elif proj_mat.node.out_features == qkv_mat.node.out_features:
            if len(qkv_mat.inputs) != 1:
                return
            input_nodes = [in_node for in_node in qkv_mat.input_nodes]
            qkv_mat_list = [item for item in input_nodes[0].output_nodes if item.op_type == ModuleType.LINEAR]
            if len(qkv_mat_list) != 3:
                return

            qkv_list.append([item.name_in_network for item in qkv_mat_list])
            proj_list.append(proj_mat.name_in_network)
            mhsa_ln_list.append(ln_list[0].name_in_network)