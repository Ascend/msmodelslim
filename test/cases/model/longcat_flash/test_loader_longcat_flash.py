#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from msmodelslim.model.longcat_flash import loader as target
from msmodelslim.model.longcat_flash.loader import LongCatFlashAdapterLoader


def test_should_inherit_base_loader_when_loader_class_inspected_given_longcat_flash_loader():
    # given
    from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader

    # when / then
    assert issubclass(LongCatFlashAdapterLoader, BaseModelAdapterLoader)


def test_should_define_adapter_class_path_when_loader_class_inspected_given_longcat_flash_loader():
    # given / when
    adapter_class_path = target.LongCatFlashAdapterLoader.ADAPTER_CLASS_PATH

    # then
    assert adapter_class_path == "msmodelslim.model.longcat_flash.model_adapter:LongCatFlashModelAdapter"


def test_should_split_module_and_class_name_when_class_path_inspected_given_longcat_flash_loader():
    # given
    adapter_class_path = LongCatFlashAdapterLoader.ADAPTER_CLASS_PATH

    # when
    module_path, class_name = adapter_class_path.split(":", 1)

    # then
    assert module_path == "msmodelslim.model.longcat_flash.model_adapter"
    assert class_name == "LongCatFlashModelAdapter"


def test_should_resolve_adapter_class_when_module_path_imported_given_longcat_flash_loader():
    # given
    module_path, class_name = LongCatFlashAdapterLoader.ADAPTER_CLASS_PATH.split(":", 1)

    # when
    adapter_module = importlib.import_module(module_path)

    # then
    assert hasattr(adapter_module, class_name)


def test_should_return_requirements_dict_when_get_loader_requirements_called_given_loader_instance():
    # given
    loader = LongCatFlashAdapterLoader()

    # when
    requirements = loader.get_loader_requirements()

    # then
    assert isinstance(requirements, dict)


def test_should_raise_unsupported_error_when_load_called_given_invalid_class_path():
    # given
    from msmodelslim.utils.exception import UnsupportedError

    class _BadLoader(LongCatFlashAdapterLoader):
        ADAPTER_CLASS_PATH = "invalid_no_colon"

    # when / then
    with pytest.raises(UnsupportedError):
        _BadLoader().load("longcat_flash", Path("."))


def test_should_import_adapter_class_when_load_called_given_loader_instance():
    # given
    from msmodelslim.model.plugin_factory import base_loader

    loader = LongCatFlashAdapterLoader()
    fake_adapter_class = type(
        "FakeAdapter",
        (),
        {
            "__init__": lambda self, **kwargs: setattr(self, "kwargs", kwargs),
        },
    )

    # when
    with (
        patch.object(base_loader, "get_require_packages", return_value={}),
        patch.object(
            base_loader,
            "import_module",
            return_value=SimpleNamespace(LongCatFlashModelAdapter=fake_adapter_class),
        ),
        patch.object(loader, "check_requirements") as mock_check,
    ):
        out = loader.load("longcat_flash", Path("/tmp/model"), trust_remote_code=True)

    # then
    assert isinstance(out, fake_adapter_class)
    assert out.kwargs == {
        "model_type": "longcat_flash",
        "model_path": Path("/tmp/model"),
        "trust_remote_code": True,
    }
    assert mock_check.called


def test_should_skip_path_validation_when_precheck_called_given_loader_instance(monkeypatch):
    # given
    loader = LongCatFlashAdapterLoader()
    monkeypatch.setattr(loader, "check_requirements", lambda **kwargs: None)

    # when
    loader.precheck("longcat_flash", Path("."))

    # then
    assert not loader._requirements
