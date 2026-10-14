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
# pylint: disable=arguments-differ,function-redefined

import unittest

from msmodelslim.processor.flat_quant.flat_quant_interface import FlatQuantInterface


class MockFlatQuantAdapter(FlatQuantInterface):
    """用于测试的 FlatQuantInterface 模拟实现"""

    def get_flatquant_subgraph(self):
        """返回模拟的结构配置"""
        from msmodelslim.processor.flat_quant.flat_quant_utils.structure_pair import (
            AttnNormLinearPair,
            AttnLinearLinearPair,
            MLPNormLinearPair,
            MLPLinearLinearPair,
        )

        structure_configs = [
            {
                "source": "input_layernorm",
                "targets": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "pair_class": AttnNormLinearPair,
            },
            {
                "source": "self_attn.v_proj",
                "targets": ["self_attn.o_proj"],
                "pair_class": AttnLinearLinearPair,
                "extra_config": {'head_dim': 64, 'num_attention_heads': 8},
            },
            {
                "source": "post_attention_layernorm",
                "targets": ["mlp.gate_proj", "mlp.up_proj"],
                "pair_class": MLPNormLinearPair,
            },
            {"source": "mlp.up_proj", "targets": ["mlp.down_proj"], "pair_class": MLPLinearLinearPair},
        ]
        return structure_configs


class TestFlatQuantInterface(unittest.TestCase):
    """Unit tests for FlatQuantInterface."""

    def test_FlatQuantInterface_has_method_get_flatquant_subgraph(self):
        """FlatQuantInterface-检查-包含get_flatquant_subgraph方法"""
        self.assertTrue(hasattr(FlatQuantInterface, 'get_flatquant_subgraph'))

    def test_FlatQuantInterface_get_flatquant_subgraph_is_callable(self):
        """FlatQuantInterface-检查-get_flatquant_subgraph可调用"""
        self.assertTrue(callable(FlatQuantInterface.get_flatquant_subgraph))


class TestFlatQuantInterface(unittest.TestCase):  # noqa: F811
    """Unit tests for FlatQuantInterface."""

    def test_FlatQuantInterface_has_method_get_flatquant_subgraph(self):
        """FlatQuantInterface-检查-包含get_flatquant_subgraph方法"""
        self.assertTrue(hasattr(FlatQuantInterface, 'get_flatquant_subgraph'))

    def test_FlatQuantInterface_get_flatquant_subgraph_is_callable(self):
        """FlatQuantInterface-检查-get_flatquant_subgraph可调用"""
        self.assertTrue(callable(FlatQuantInterface.get_flatquant_subgraph))


class TestFlatQuantInterface(unittest.TestCase):  # noqa: F811
    """测试 FlatQuantInterface 类 - FlatQuant适配器接口"""

    def test_FlatQuantInterface_has_method_get_flatquant_subgraph(self):
        """FlatQuantInterface-检查-包含get_flatquant_subgraph方法"""
        self.assertTrue(hasattr(FlatQuantInterface, 'get_flatquant_subgraph'))

    def test_FlatQuantInterface_get_flatquant_subgraph_is_callable(self):
        """FlatQuantInterface-检查-get_flatquant_subgraph可调用"""
        self.assertTrue(callable(FlatQuantInterface.get_flatquant_subgraph))

    def test_MockFlatQuantAdapter_returns_list_when_get_flatquant_subgraph(self):
        """MockFlatQuantAdapter-调用get_flatquant_subgraph-返回列表"""
        adapter = MockFlatQuantAdapter()
        structure_configs = adapter.get_flatquant_subgraph()
        self.assertIsInstance(structure_configs, list)
        self.assertEqual(len(structure_configs), 4)

    def test_MockFlatQuantAdapter_config_has_required_keys_when_get_flatquant_subgraph(self):
        """MockFlatQuantAdapter-调用get_flatquant_subgraph-配置包含必需键"""
        adapter = MockFlatQuantAdapter()
        structure_configs = adapter.get_flatquant_subgraph()
        for config in structure_configs:
            self.assertIn("source", config)
            self.assertIn("targets", config)
            self.assertIn("pair_class", config)
            self.assertIsInstance(config["source"], str)
            self.assertIsInstance(config["targets"], list)


if __name__ == '__main__':
    unittest.main()
