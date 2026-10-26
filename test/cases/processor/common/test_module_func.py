#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest

from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.base import AutoSessionProcessor
from msmodelslim.utils.exception import SchemaValidateError
from msmodelslim.processor.common.module_func import (
    ModuleFuncProcessor,
    ModuleFuncProcessorConfig,
)


def _noop_func(name, module) -> None:
    _ = (name, module)


class TestModuleFuncProcessorConfig(unittest.TestCase):
    def test_type_defaults_to_module_func(self):
        config = ModuleFuncProcessorConfig(name="rename", func=_noop_func)
        self.assertEqual(config.type, "module_func")
        self.assertEqual(config.name, "rename")
        self.assertIs(config.func, _noop_func)

    def test_name_is_required(self):
        with self.assertRaises(SchemaValidateError):
            ModuleFuncProcessorConfig(func=_noop_func)

    def test_func_is_required(self):
        with self.assertRaises(SchemaValidateError):
            ModuleFuncProcessorConfig(name="rename")


class TestModuleFuncProcessor(unittest.TestCase):
    def _make(self, name="rename", func=_noop_func):
        config = ModuleFuncProcessorConfig(name=name, func=func)
        return ModuleFuncProcessor(nn.Module(), config)

    def test_repr_includes_config_name(self):
        processor = self._make(name="rename")
        self.assertEqual(repr(processor), "ModuleFuncProcessor(name=rename)")

    def test_is_data_free_is_true(self):
        self.assertTrue(self._make().is_data_free())

    def test_preprocess_invokes_callable_with_request_name_and_module(self):
        calls = []

        def record(name, module):
            calls.append((name, module))

        target = nn.Linear(3, 3)
        processor = self._make(func=record)
        request = BatchProcessRequest(name="layers.0.self_attn", module=target, datas=None)

        processor.preprocess(request)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "layers.0.self_attn")
        self.assertIs(calls[0][1], target)

    def test_from_config_creates_module_func_processor(self):
        config = ModuleFuncProcessorConfig(name="rename", func=_noop_func)
        processor = AutoSessionProcessor.from_config(nn.Module(), config, adapter=None)
        self.assertIsInstance(processor, ModuleFuncProcessor)


if __name__ == "__main__":
    unittest.main()
