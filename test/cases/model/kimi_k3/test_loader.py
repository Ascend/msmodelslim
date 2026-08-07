# -*- coding: UTF-8 -*-
from msmodelslim.model.kimi_k3.loader import KimiK3AdapterLoader
from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


def test_loader_path():
    assert issubclass(KimiK3AdapterLoader, BaseModelAdapterLoader)
    assert KimiK3AdapterLoader.ADAPTER_CLASS_PATH.endswith("KimiK3ModelAdapter")
