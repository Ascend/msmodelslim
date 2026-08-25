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
from unittest.mock import patch, MagicMock

from msmodelslim.model.glm_5_2.loader import Glm52AdapterLoader
from msmodelslim.model.plugin_factory.base_loader import BaseModelAdapterLoader


class TestGlm52AdapterLoader(unittest.TestCase):
    def test_Glm52AdapterLoader_adapterClassPath_shouldBeGML52ModelAdapter_when_defined(self):
        self.assertEqual(
            Glm52AdapterLoader.ADAPTER_CLASS_PATH, "msmodelslim.model.glm_5_2.model_adapter:GLM52ModelAdapter"
        )

    def test_Glm52AdapterLoader_shouldInherit_when_fromBaseModelAdapterLoader(self):
        self.assertTrue(issubclass(Glm52AdapterLoader, BaseModelAdapterLoader))

    @patch("msmodelslim.model.glm_5_2.loader.BaseModelAdapterLoader.precheck")
    def test_Glm52AdapterLoader_precheck_shouldCallParent_when_called(self, mock_parent_precheck):
        loader = Glm52AdapterLoader()
        model_type = "GLM-5.2"
        model_path = Path("/fake/path")

        loader.precheck(model_type, model_path)

        mock_parent_precheck.assert_called_once_with(model_type, model_path)

    @patch("msmodelslim.model.glm_5_2.loader.BaseModelAdapterLoader.load")
    def test_Glm52AdapterLoader_load_shouldReturnAdapter_when_called(self, mock_parent_load):
        mock_adapter = MagicMock()
        mock_parent_load.return_value = mock_adapter

        loader = Glm52AdapterLoader()
        model_type = "GLM-5.2"
        model_path = Path("/fake/path")

        result = loader.load(model_type, model_path, trust_remote_code=False)

        mock_parent_load.assert_called_once_with(model_type, model_path, trust_remote_code=False)
        self.assertIs(result, mock_adapter)


if __name__ == '__main__':
    unittest.main()
