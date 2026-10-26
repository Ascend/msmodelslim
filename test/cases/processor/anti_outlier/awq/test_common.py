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

Unit tests for anti_outlier.awq.common (context lookup, offload/onload, dataclasses).
"""

import unittest
from dataclasses import dataclass

import torch

from msmodelslim.core.context import ContextFactory, ContextManager
from msmodelslim.processor.anti_outlier.awq.common import (
    AWQConfig,
    AWQContext,
    get_global_awq_stats,
    offload,
    onload,
)


class DummySearcher:
    def search(self, linears2scale, context):  # pragma: no cover - not executed
        return None


class TestGlobalAwqStats(unittest.TestCase):
    """get_global_awq_stats reads the `awq` namespace from the active context."""

    def test_no_context_raises_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            get_global_awq_stats()
        self.assertIn("No global context found", str(ctx.exception))

    def test_with_context_returns_namespace_state(self):
        with ContextManager(ctx=ContextFactory().create(is_distributed=False)):
            stats = get_global_awq_stats()
            stats["act_mean"] = 1
            self.assertEqual(get_global_awq_stats()["act_mean"], 1)


class TestOffloadOnload(unittest.TestCase):
    """offload/onload recursively walk containers and dataclasses."""

    def test_offload_moves_tensor_to_cpu(self):
        tensor = torch.zeros(2, 3)
        result = offload(tensor, torch.device("cpu"))
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.equal(result, tensor))

    def test_offload_recurses_into_containers(self):
        tree = {
            "a": [torch.ones(2), torch.zeros(2)],
            "b": ({"c": torch.ones(2)},),
        }
        result = offload(tree, torch.device("cpu"))
        self.assertEqual(result["a"][0].device.type, "cpu")
        self.assertEqual(result["b"][0]["c"].device.type, "cpu")

    def test_offload_none_device_is_noop(self):
        tensor = torch.ones(2)
        self.assertIs(offload(tensor, None), tensor)

    def test_offload_dataclass_moves_fields(self):
        @dataclass
        class Wrap:
            data: torch.Tensor

        wrap = Wrap(data=torch.ones(3))
        result = offload(wrap, torch.device("cpu"))
        self.assertEqual(result.data.device.type, "cpu")

    def test_onload_none_device_keeps_objects(self):
        tensor = torch.ones(2)
        tree = [tensor, {"k": tensor}]
        result = onload(tree, None)
        self.assertIs(result[0], tensor)
        self.assertIs(result[1]["k"], tensor)

    def test_onload_non_tensor_returns_as_is(self):
        self.assertEqual(onload(5, None), 5)
        self.assertEqual(onload("text", None), "text")


class TestAwqDataclasses(unittest.TestCase):
    """AWQConfig and AWQContext keep their fields."""

    def test_awq_config_fields(self):
        searcher = DummySearcher()
        config = AWQConfig(version=1, awq_searcher=searcher)
        self.assertEqual(config.version, 1)
        self.assertIs(config.awq_searcher, searcher)

    def test_awq_context_fields(self):
        act_mean = torch.ones(4)
        context = AWQContext(act_mean=act_mean, inspect_module=None, inspect_module_args=[1, 2])
        self.assertIs(context.act_mean, act_mean)
        self.assertEqual(context.inspect_module_args, [1, 2])


if __name__ == "__main__":
    unittest.main()
