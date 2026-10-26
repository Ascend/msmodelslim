#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from typing import Literal
from unittest.mock import patch

import pytest
from torch import nn

from msmodelslim.core.base.processor import BaseProcessor
from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.ir.qal.qregistry import QABCRegistry
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.exception import SchemaValidateError


class _DemoConfig(AutoProcessorConfig):
    type: Literal["base_demo"] = "base_demo"
    value: int = 1


class _DataFreeProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter=None):
        super().__init__(model)
        self.config = config

    def is_data_free(self) -> bool:
        return True


class _DataProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter=None):
        super().__init__(model)
        self.config = config


@QABCRegistry.register(dispatch_key=_DemoConfig, abc_class=AutoSessionProcessor)
class _RegisteredProcessor(AutoSessionProcessor):
    def __init__(self, model, config, adapter=None):
        super().__init__(model)
        self.config = config


class TestAutoProcessorConfig(unittest.TestCase):
    def test_subclass_uses_declared_type_and_default_value(self):
        config = _DemoConfig()
        self.assertEqual(config.type, "base_demo")
        self.assertEqual(config.value, 1)

    def test_type_field_is_literal_validated(self):
        with self.assertRaises(SchemaValidateError):
            _DemoConfig(type="not_a_known_type")

    def test_value_field_rejects_non_integer(self):
        with self.assertRaises(SchemaValidateError):
            _DemoConfig(value="not-an-int")

    def test_base_validate_dispatches_dict_to_registered_subclass(self):
        config = AutoProcessorConfig.model_validate({"type": "base_demo", "value": 7})
        self.assertIsInstance(config, _DemoConfig)
        self.assertEqual(config.value, 7)

    def test_base_validate_rejects_non_dict_item_with_diagnostics(self):
        with pytest.raises(SchemaValidateError) as ctx:
            AutoProcessorConfig.model_validate("bare-string")
        self.assertIn("Invalid config item", str(ctx.value))


class TestAutoSessionProcessor(unittest.TestCase):
    @staticmethod
    def _make(proc_cls):
        return proc_cls(nn.Linear(2, 2), _DemoConfig())

    def test_repr_returns_class_name(self):
        self.assertEqual(repr(self._make(_DataProcessor)), "_DataProcessor")

    def test_default_lifecycle_flags(self):
        proc = self._make(_DataProcessor)
        self.assertFalse(proc.support_distributed())
        self.assertFalse(proc.is_data_free())
        self.assertFalse(proc.need_kv_cache())

    def test_process_calls_base_forward_for_data_processor(self):
        proc = self._make(_DataProcessor)
        request = BatchProcessRequest(name="", module=nn.Linear(2, 2), datas=[])
        with patch.object(BaseProcessor, "process") as mock_process:
            proc.process(request)
        mock_process.assert_called_once_with(request)

    def test_process_skips_base_forward_when_data_free(self):
        proc = self._make(_DataFreeProcessor)
        request = BatchProcessRequest(name="", module=nn.Linear(2, 2), datas=[])
        with patch.object(BaseProcessor, "process") as mock_process:
            proc.process(request)
        mock_process.assert_not_called()

    def test_from_config_creates_registered_processor(self):
        model = nn.Linear(2, 2)
        proc = AutoSessionProcessor.from_config(model, _DemoConfig(), adapter=None)
        self.assertIsInstance(proc, _RegisteredProcessor)
        self.assertIs(proc.model, model)


if __name__ == "__main__":
    unittest.main()
