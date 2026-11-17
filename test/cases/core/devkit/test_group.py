#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from typing import List, Literal, Optional

from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.core.quantization_schema.qal.qregistry import QABCRegistry
from msmodelslim.core.quantization_algorithm.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.core.devkit.group import GroupProcessor, GroupProcessorConfig


class _FlagConfig(AutoProcessorConfig):
    type: Literal["group_flag_demo"] = "group_flag_demo"
    data_free: bool = True
    kv_cache: bool = False
    distributed: bool = False
    tag: Optional[str] = None


@QABCRegistry.register(dispatch_key=_FlagConfig, abc_class=AutoSessionProcessor)
class _FlagProcessor(AutoSessionProcessor):
    call_log: List[tuple] = []

    def __init__(self, model, config, adapter=None):
        super().__init__(model)
        self.config = config

    def __repr__(self) -> str:
        return f"_FlagProcessor(tag={self.config.tag})"

    def is_data_free(self) -> bool:
        return self.config.data_free

    def need_kv_cache(self) -> bool:
        return self.config.kv_cache

    def support_distributed(self) -> bool:
        return self.config.distributed

    def preprocess(self, request: BatchProcessRequest) -> None:
        if self.config.tag is not None:
            _FlagProcessor.call_log.append((self.config.tag, request.name))


class TestGroupProcessor(unittest.TestCase):
    def setUp(self):
        _FlagProcessor.call_log = []

    def test_repr_lists_child_processor_names(self):
        config = GroupProcessorConfig(
            type="group",
            configs=[_FlagConfig(tag="a"), _FlagConfig(tag="b")],
        )
        processor = GroupProcessor(nn.Module(), config)
        self.assertEqual(repr(processor), "GroupProcessor(processors=['_FlagProcessor', '_FlagProcessor'])")

    def test_is_data_free_is_true_only_when_all_children_are(self):
        free_config = GroupProcessorConfig(
            type="group",
            configs=[_FlagConfig(data_free=True), _FlagConfig(data_free=True)],
        )
        self.assertTrue(GroupProcessor(nn.Module(), free_config).is_data_free())

        mixed_config = GroupProcessorConfig(
            type="group",
            configs=[_FlagConfig(data_free=False), _FlagConfig(data_free=True)],
        )
        self.assertFalse(GroupProcessor(nn.Module(), mixed_config).is_data_free())

    def test_need_kv_cache_is_true_when_any_child_needs_it(self):
        config = GroupProcessorConfig(
            type="group",
            configs=[_FlagConfig(kv_cache=True), _FlagConfig(kv_cache=False)],
        )
        self.assertTrue(GroupProcessor(nn.Module(), config).need_kv_cache())

        none_config = GroupProcessorConfig(
            type="group",
            configs=[_FlagConfig(kv_cache=False), _FlagConfig(kv_cache=False)],
        )
        self.assertFalse(GroupProcessor(nn.Module(), none_config).need_kv_cache())

    def test_support_distributed_is_true_when_any_child_supports_it(self):
        config = GroupProcessorConfig(
            type="group",
            configs=[_FlagConfig(distributed=True), _FlagConfig(distributed=False)],
        )
        self.assertTrue(GroupProcessor(nn.Module(), config).support_distributed())

    def test_empty_group_uses_identity_defaults(self):
        processor = GroupProcessor(nn.Module(), GroupProcessorConfig(type="group", configs=[]))
        self.assertEqual(processor.processor_names, [])
        self.assertTrue(processor.is_data_free())
        self.assertFalse(processor.need_kv_cache())
        self.assertFalse(processor.support_distributed())

    def test_preprocess_delegates_to_each_child(self):
        config = GroupProcessorConfig(
            type="group",
            configs=[
                _FlagConfig(tag="first"),
                _FlagConfig(tag="second"),
            ],
        )
        processor = GroupProcessor(nn.Module(), config)
        request = BatchProcessRequest(name="layers.3", module=nn.Module(), datas=None)

        processor.preprocess(request)

        self.assertEqual(_FlagProcessor.call_log, [("first", "layers.3"), ("second", "layers.3")])


if __name__ == "__main__":
    unittest.main()
