#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from typing import Literal

from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.processor.common.module_func import ModuleFuncProcessorConfig
from msmodelslim.processor.container.group import GroupProcessor, GroupProcessorConfig


class _FlagConfig(AutoProcessorConfig):
    type: Literal["group_flag_demo"] = "group_flag_demo"
    data_free: bool = False
    kv_cache: bool = False
    distributed: bool = False


@QABCRegistry.register(dispatch_key=_FlagConfig, abc_class=AutoSessionProcessor)
class _FlagProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter=None):
        super().__init__(model)
        self.config = config

    def is_data_free(self) -> bool:
        return self.config.data_free

    def need_kv_cache(self) -> bool:
        return self.config.kv_cache

    def support_distributed(self) -> bool:
        return self.config.distributed


def _module_func_cfg(name: str) -> ModuleFuncProcessorConfig:
    return ModuleFuncProcessorConfig(name=name, func=lambda n, m: None)


class TestGroupProcessor(unittest.TestCase):
    def test_repr_lists_child_processor_names(self):
        config = GroupProcessorConfig(type="group", configs=[_module_func_cfg("a"), _module_func_cfg("b")])
        processor = GroupProcessor(nn.Module(), config)
        self.assertEqual(repr(processor), "GroupProcessor(processors=['ModuleFuncProcessor', 'ModuleFuncProcessor'])")

    def test_is_data_free_is_true_only_when_all_children_are(self):
        free_config = GroupProcessorConfig(
            type="group",
            configs=[_FlagConfig(data_free=True), _module_func_cfg("a")],
        )
        self.assertTrue(GroupProcessor(nn.Module(), free_config).is_data_free())

        mixed_config = GroupProcessorConfig(
            type="group",
            configs=[_FlagConfig(data_free=False), _module_func_cfg("a")],
        )
        self.assertFalse(GroupProcessor(nn.Module(), mixed_config).is_data_free())

    def test_need_kv_cache_is_true_when_any_child_needs_it(self):
        config = GroupProcessorConfig(type="group", configs=[_FlagConfig(kv_cache=True), _module_func_cfg("a")])
        self.assertTrue(GroupProcessor(nn.Module(), config).need_kv_cache())

        none_config = GroupProcessorConfig(type="group", configs=[_FlagConfig(kv_cache=False), _module_func_cfg("a")])
        self.assertFalse(GroupProcessor(nn.Module(), none_config).need_kv_cache())

    def test_support_distributed_is_true_when_any_child_supports_it(self):
        config = GroupProcessorConfig(type="group", configs=[_FlagConfig(distributed=True), _module_func_cfg("a")])
        self.assertTrue(GroupProcessor(nn.Module(), config).support_distributed())

    def test_empty_group_uses_identity_defaults(self):
        processor = GroupProcessor(nn.Module(), GroupProcessorConfig(type="group", configs=[]))
        self.assertEqual(processor.processor_names, [])
        self.assertTrue(processor.is_data_free())
        self.assertFalse(processor.need_kv_cache())
        self.assertFalse(processor.support_distributed())

    def test_preprocess_delegates_to_each_child(self):
        calls = []

        def make_func(tag):
            def _record(name, module):
                calls.append((tag, name))

            return _record

        config = GroupProcessorConfig(
            type="group",
            configs=[
                ModuleFuncProcessorConfig(name="first", func=make_func("first")),
                ModuleFuncProcessorConfig(name="second", func=make_func("second")),
            ],
        )
        processor = GroupProcessor(nn.Module(), config)
        request = BatchProcessRequest(name="layers.3", module=nn.Module(), datas=None)

        processor.preprocess(request)

        self.assertEqual(calls, [("first", "layers.3"), ("second", "layers.3")])


if __name__ == "__main__":
    unittest.main()
