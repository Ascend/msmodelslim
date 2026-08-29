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
from unittest.mock import MagicMock, patch

from msmodelslim.model.glm5_next.loader import Glm5NextAdapterLoader
from msmodelslim.model.glm5_next.model_adapter import Glm5NextModelAdapter
from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class TestGlm5NextAdapterLoader(unittest.TestCase):
    def test_Glm5NextAdapterLoader_adapterClassPath_shouldBeGlm5NextModelAdapter_when_defined(self):
        self.assertEqual(
            Glm5NextAdapterLoader.ADAPTER_CLASS_PATH,
            "msmodelslim.model.glm5_next.model_adapter:Glm5NextModelAdapter",
        )

    def test_Glm5NextAdapterLoader_shouldInherit_when_fromBaseModelAdapterLoader(self):
        self.assertTrue(issubclass(Glm5NextAdapterLoader, BaseModelAdapterLoader))

    def test_Glm5NextModelAdapter_pedigree_shouldBeGlm5Next_when_created(self):
        self.assertEqual(Glm5NextModelAdapter.get_model_pedigree(None), "glm5_next")

    @patch("msmodelslim.model.glm5_next.loader.BaseModelAdapterLoader.precheck")
    def test_Glm5NextAdapterLoader_precheck_shouldCallParent_when_called(self, mock_parent_precheck):
        loader = Glm5NextAdapterLoader()
        model_type = "GLM-5.3-Flash"
        model_path = Path("/fake/path")

        loader.precheck(model_type, model_path)

        mock_parent_precheck.assert_called_once_with(model_type, model_path)

    @patch("msmodelslim.model.glm5_next.loader.BaseModelAdapterLoader.load")
    def test_Glm5NextAdapterLoader_load_shouldReturnAdapter_when_called(self, mock_parent_load):
        mock_adapter = MagicMock()
        mock_parent_load.return_value = mock_adapter

        loader = Glm5NextAdapterLoader()
        model_type = "GLM-5.3-Flash"
        model_path = Path("/fake/path")

        result = loader.load(model_type, model_path, trust_remote_code=False)

        mock_parent_load.assert_called_once_with(model_type, model_path, trust_remote_code=False)
        self.assertIs(result, mock_adapter)


if __name__ == '__main__':
    unittest.main()
