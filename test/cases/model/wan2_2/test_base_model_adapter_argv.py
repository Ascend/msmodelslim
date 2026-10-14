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

# pylint: disable=use-implicit-booleaness-not-comparison

from msmodelslim.model.wan2_2.base_model_adapter import Wan2_2BaseModelAdapter


class TestNamespaceToArgv:
    @staticmethod
    def test_serializes_scalar_sample_guide_scale_when_float():
        argv = Wan2_2BaseModelAdapter._namespace_to_argv(
            {
                "sample_guide_scale": 4.5,
            }
        )
        assert "--sample_guide_scale" in argv
        assert "4.5" in argv

    @staticmethod
    def test_skips_tuple_sample_guide_scale_to_match_generate_cli():
        argv = Wan2_2BaseModelAdapter._namespace_to_argv(
            {
                "task": "t2v-A14B",
                "sample_guide_scale": (3.0, 4.0),
                "convert_model_dtype": True,
            }
        )
        joined = " ".join(argv)
        assert "sample_guide_scale" not in joined
        assert "--convert_model_dtype" in argv
        assert "--task" in argv and "t2v-A14B" in argv

    @staticmethod
    def test_skips_list_values():
        argv = Wan2_2BaseModelAdapter._namespace_to_argv(
            {
                "sample_guide_scale": [4.5],
            }
        )
        assert argv == []
