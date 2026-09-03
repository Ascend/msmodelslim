# -*- coding: UTF-8 -*-

from pathlib import Path

from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader

from .detect import LLM_ADAPTER_CLASS_PATH, resolve_adapter_class_path


class TransformersAdapterLoader(BaseModelAdapterLoader):
    ADAPTER_CLASS_PATH = LLM_ADAPTER_CLASS_PATH

    def load(
        self,
        model_type: str,
        model_path: Path,
        trust_remote_code: bool = False,
    ):
        self.ADAPTER_CLASS_PATH = resolve_adapter_class_path(model_path)
        return super().load(model_type, model_path, trust_remote_code)
