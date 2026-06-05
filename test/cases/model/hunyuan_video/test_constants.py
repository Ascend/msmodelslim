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

from msmodelslim.model.hunyuan_video.constants import (
    PLACEHOLDER_PROMPT,
    TASK_TYPE,
    DEFAULT_MODEL_RESOLUTION,
    DEFAULT_VIDEO_SIZE,
    HYVIDEO_CLI_LIST_FIELDS,
    DIT_WEIGHT_REL,
    VAE_PATH_REL,
    TEXT_ENCODER_PATH_REL,
    TEXT_ENCODER_2_PATH_REL,
)


class TestPlaceHolderPrompt:
    """测试PLACEHOLDER_PROMPT常量"""

    def test_placeholder_prompt_when_accessed_then_is_string(self):
        """PLACEHOLDER_PROMPT-访问-应为字符串"""
        assert isinstance(PLACEHOLDER_PROMPT, str)

    def test_placeholder_prompt_when_accessed_then_is_non_empty(self):
        """PLACEHOLDER_PROMPT-访问-应为非空字符串"""
        assert len(PLACEHOLDER_PROMPT) > 0


class TestTaskType:
    """测试TASK_TYPE常量"""

    def test_task_type_when_accessed_then_equals_hunyuanvideo(self):
        """TASK_TYPE-访问-应为hunyuanvideo"""
        assert TASK_TYPE == "hunyuanvideo"


class TestDefaultModelResolution:
    """测试DEFAULT_MODEL_RESOLUTION常量"""

    def test_default_model_resolution_when_accessed_then_is_valid_resolution(self):
        """DEFAULT_MODEL_RESOLUTION-访问-应为有效分辨率"""
        assert DEFAULT_MODEL_RESOLUTION in ("540p", "720p")

    def test_default_model_resolution_when_accessed_then_is_720p(self):
        """DEFAULT_MODEL_RESOLUTION-访问-应为720p"""
        assert DEFAULT_MODEL_RESOLUTION == "720p"


class TestDefaultVideoSize:
    """测试DEFAULT_VIDEO_SIZE常量"""

    def test_default_video_size_when_accessed_then_is_tuple_of_two(self):
        """DEFAULT_VIDEO_SIZE-访问-应为二元素元组"""
        assert isinstance(DEFAULT_VIDEO_SIZE, tuple)
        assert len(DEFAULT_VIDEO_SIZE) == 2

    def test_default_video_size_when_accessed_then_is_720_1280(self):
        """DEFAULT_VIDEO_SIZE-访问-应为(720, 1280)"""
        assert DEFAULT_VIDEO_SIZE == (720, 1280)

    def test_default_video_size_when_accessed_then_height_less_than_width(self):
        """DEFAULT_VIDEO_SIZE-访问-高度应小于宽度"""
        assert DEFAULT_VIDEO_SIZE[0] < DEFAULT_VIDEO_SIZE[1]


class TestHyvideoCliListFields:
    """测试HYVIDEO_CLI_LIST_FIELDS常量"""

    def test_hyvideo_cli_list_fields_when_accessed_then_is_frozenset(self):
        """HYVIDEO_CLI_LIST_FIELDS-访问-应为frozenset"""
        assert isinstance(HYVIDEO_CLI_LIST_FIELDS, frozenset)

    def test_hyvideo_cli_list_fields_when_accessed_then_contains_video_size(self):
        """HYVIDEO_CLI_LIST_FIELDS-访问-应包含video_size"""
        assert "video_size" in HYVIDEO_CLI_LIST_FIELDS


class TestPathConstants:
    """测试路径相关常量"""

    def test_dit_weight_rel_when_accessed_then_is_tuple_of_three(self):
        """DIT_WEIGHT_REL-访问-应为三元素元组"""
        assert isinstance(DIT_WEIGHT_REL, tuple)
        assert len(DIT_WEIGHT_REL) == 3

    def test_dit_weight_rel_when_accessed_then_end_with_pt(self):
        """DIT_WEIGHT_REL-访问-最后一个元素应以.pt结尾"""
        assert DIT_WEIGHT_REL[-1].endswith(".pt")

    def test_vae_path_rel_when_accessed_then_is_tuple(self):
        """VAE_PATH_REL-访问-应为元组"""
        assert isinstance(VAE_PATH_REL, tuple)
        assert len(VAE_PATH_REL) > 0

    def test_text_encoder_path_rel_when_accessed_then_is_tuple(self):
        """TEXT_ENCODER_PATH_REL-访问-应为元组"""
        assert isinstance(TEXT_ENCODER_PATH_REL, tuple)
        assert len(TEXT_ENCODER_PATH_REL) > 0

    def test_text_encoder_2_path_rel_when_accessed_then_is_tuple(self):
        """TEXT_ENCODER_2_PATH_REL-访问-应为元组"""
        assert isinstance(TEXT_ENCODER_2_PATH_REL, tuple)
        assert len(TEXT_ENCODER_2_PATH_REL) > 0

    def test_dit_weight_rel_when_accessed_then_contains_720p(self):
        """DIT_WEIGHT_REL-访问-应包含720p相关路径"""
        assert "720p" in DIT_WEIGHT_REL[0]

    def test_vae_path_rel_when_accessed_then_contains_720p(self):
        """VAE_PATH_REL-访问-应包含720p相关路径"""
        assert "720p" in VAE_PATH_REL[0]
