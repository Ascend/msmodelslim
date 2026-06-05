#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=use-implicit-booleaness-not-comparison

from msmodelslim.model.hunyuan_video.model_adapter import HunyuanVideoModelAdapter


class TestNamespaceToArgv:
    @staticmethod
    def test_expands_video_size():
        argv = HunyuanVideoModelAdapter._namespace_to_argv(
            {
                "video_size": (720, 1280),
                "infer_steps": 50,
            }
        )
        assert argv == ["--video-size", "720", "1280", "--infer-steps", "50"]

    @staticmethod
    def test_skips_dict_values():
        argv = HunyuanVideoModelAdapter._namespace_to_argv(
            {
                "video_size": [480, 832],
                "extra": {"a": 1},
            }
        )
        assert "--extra" not in argv
        assert argv[:4] == ["--video-size", "480", "832"]

    @staticmethod
    def test_skips_list_values_not_registered_for_cli():
        argv = HunyuanVideoModelAdapter._namespace_to_argv(
            {
                "cfg_scale": [1.0, 2.0],
            }
        )
        assert argv == []

    @staticmethod
    def test_skips_none_values():
        """None值应被跳过"""
        argv = HunyuanVideoModelAdapter._namespace_to_argv({"infer_steps": None, "seed": 42})
        assert "--infer-steps" not in argv
        assert argv == ["--seed", "42"]

    @staticmethod
    def test_bool_true_appends_flag():
        """True布尔值应只添加flag，不带值"""
        argv = HunyuanVideoModelAdapter._namespace_to_argv({"use_cache": True})
        assert argv == ["--use-cache"]

    @staticmethod
    def test_bool_false_skipped():
        """False布尔值应被跳过"""
        argv = HunyuanVideoModelAdapter._namespace_to_argv({"use_cache": False})
        assert argv == []

    @staticmethod
    def test_empty_dict_returns_empty():
        """空字典返回空列表"""
        assert HunyuanVideoModelAdapter._namespace_to_argv({}) == []
