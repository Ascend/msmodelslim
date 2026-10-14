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
# pylint: disable=abstract-class-instantiated,use-implicit-booleaness-not-comparison,useless-parent-delegation

from pathlib import Path
from typing import List

import pytest

from msmodelslim.infra.dataset_loader.base_dataset_loader import BaseDatasetLoader
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample


class _DummyDatasetLoader(BaseDatasetLoader):
    def is_support(self, resolved_path: Path) -> bool:
        return resolved_path.name == "ok"

    def load_data(self, resolved_path: Path) -> List[VlmCalibSample]:
        return [VlmCalibSample(text="dummy")]


class _OnlyIsSupportLoader(BaseDatasetLoader):
    def is_support(self, resolved_path: Path) -> bool:
        return True

    def load_data(self, resolved_path: Path) -> List[VlmCalibSample]:
        return super().load_data(resolved_path)


class _OnlyLoadDataLoader(BaseDatasetLoader):
    def is_support(self, resolved_path: Path) -> bool:
        return super().is_support(resolved_path)

    def load_data(self, resolved_path: Path) -> List[VlmCalibSample]:
        return [VlmCalibSample(text="x")]


class TestBaseDatasetLoader:
    def test_init_set_empty_extensions_when_use_default_ctor(self):
        loader = _DummyDatasetLoader()
        assert loader.supported_image_extensions == {}
        assert loader.supported_audio_extensions == {}
        assert loader.supported_video_extensions == {}

    def test_init_set_default_text_when_use_default_ctor(self):
        loader = _DummyDatasetLoader()
        assert loader.default_text == "Describe this image in detail."

    def test_init_set_custom_text_when_pass_default_text(self):
        loader = _DummyDatasetLoader(default_text="custom")
        assert loader.default_text == "custom"

    def test_abstract_class_raise_type_error_when_direct_init(self):
        with pytest.raises(TypeError):
            BaseDatasetLoader()

    def test_abstract_is_support_raise_not_implemented_when_subclass_not_override(self, tmp_path: Path):
        loader = _OnlyLoadDataLoader()
        with pytest.raises(NotImplementedError):
            loader.is_support(tmp_path)

    def test_abstract_load_data_raise_not_implemented_when_subclass_not_override(self, tmp_path: Path):
        loader = _OnlyIsSupportLoader()
        with pytest.raises(NotImplementedError):
            loader.load_data(tmp_path)
