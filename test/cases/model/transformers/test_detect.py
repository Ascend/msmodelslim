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

import unittest
from pathlib import Path
from unittest.mock import patch

from msmodelslim.model.transformers.detect import (
    LLM_ADAPTER_CLASS_PATH,
    ModelClass,
    detect_transformers_kind,
    resolve_adapter_class_path,
)
from msmodelslim.utils.exception import SecurityError, UnsupportedError


class TestDetectTransformersKind(unittest.TestCase):
    def setUp(self):
        self.model_path = Path("/tmp/transformers-model")

    def test_detect_transformers_kind_returns_llm_when_config_has_no_hints(self):
        with patch("msmodelslim.model.transformers.detect.json_safe_load", return_value={"model_type": "llama"}):
            self.assertEqual(detect_transformers_kind(self.model_path), ModelClass.LLM)

    def test_detect_transformers_kind_returns_vlm_when_vision_config_present(self):
        with patch("msmodelslim.model.transformers.detect.json_safe_load", return_value={"vision_config": {}}):
            self.assertEqual(detect_transformers_kind(self.model_path), ModelClass.VLM)

    def test_detect_transformers_kind_returns_dit_when_diffusers_version_present(self):
        with patch(
            "msmodelslim.model.transformers.detect.json_safe_load",
            return_value={"_diffusers_version": "0.30.0"},
        ):
            self.assertEqual(detect_transformers_kind(self.model_path), ModelClass.DIT)

    def test_detect_transformers_kind_raises_unsupported_error_when_config_missing(self):
        with patch(
            "msmodelslim.model.transformers.detect.json_safe_load",
            side_effect=SecurityError("invalid path"),
        ):
            with self.assertRaises(UnsupportedError) as ctx:
                detect_transformers_kind(self.model_path)

        self.assertIn("config.json", str(ctx.exception))
        self.assertIn("transformers", str(ctx.exception))


class TestResolveAdapterClassPath(unittest.TestCase):
    def setUp(self):
        self.model_path = Path("/tmp/transformers-model")

    def test_resolve_adapter_class_path_returns_llm_path_when_llm_config(self):
        with patch("msmodelslim.model.transformers.detect.json_safe_load", return_value={}):
            self.assertEqual(resolve_adapter_class_path(self.model_path), LLM_ADAPTER_CLASS_PATH)

    def test_resolve_adapter_class_path_raises_unsupported_error_when_vlm_config(self):
        with patch("msmodelslim.model.transformers.detect.json_safe_load", return_value={"vision_config": {}}):
            with self.assertRaises(UnsupportedError) as ctx:
                resolve_adapter_class_path(self.model_path)

        self.assertIn("VLM", str(ctx.exception))

    def test_resolve_adapter_class_path_raises_unsupported_error_when_dit_config(self):
        with patch(
            "msmodelslim.model.transformers.detect.json_safe_load",
            return_value={"_diffusers_version": "0.30.0"},
        ):
            with self.assertRaises(UnsupportedError) as ctx:
                resolve_adapter_class_path(self.model_path)

        self.assertIn("DiT", str(ctx.exception))
