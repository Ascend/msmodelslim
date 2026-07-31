#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class MiniMaxM3AdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = "msmodelslim.model.minimax_m3.model_adapter:MiniMaxM3ModelAdapter"
