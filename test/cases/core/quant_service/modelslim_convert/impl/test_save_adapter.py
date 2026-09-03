#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter 模块的单元测试
"""

from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter import (
    SaveProcessorAdapter,
)
from msmodelslim.processor.save.processor import QuantSaveProcessor


def _context(dst_format: str, part_file_size: int = 4) -> ConvertContext:
    config = ConvertConfig(
        model_path="/m",
        save_path="/out",
        dst_format=dst_format,
        part_file_size=part_file_size,
    )
    context = ConvertContext(config=config)
    context.reader = MagicMock()
    return context


class TestSaveProcessorAdapter:
    """测试 SaveProcessorAdapter 整树 save 与流式生命周期"""

    def test_save_call_ascendv1_when_dst_format_ascendv1(self):
        """场景：dst_format=ascendv1 整树保存。
        预期：构造 AscendV1Saver，调用 pre_run / post_run，不按 named_modules 做 postprocess。
        """
        tree = nn.Module()
        context = _context("ascendv1")
        with (
            patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.AscendV1Saver") as mock_cls,
            patch(
                "msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter._lazy_init_unsaved_modules",
            ),
        ):
            mock_saver = MagicMock()
            mock_cls.return_value = mock_saver
            SaveProcessorAdapter().save(context, tree)
            mock_cls.assert_called_once()
            mock_saver.pre_run.assert_called_once()
            mock_saver.post_run.assert_called_once()
            mock_saver.postprocess.assert_not_called()

    def test_save_call_compressed_tensors_when_dst_format_hf(self):
        """场景：dst_format=huggingface 整树保存。
        预期：构造 QuantSaveProcessor，对 named_modules 调用 postprocess。
        """
        tree = nn.Module()
        tree.add_module("linear", nn.Linear(2, 2, bias=False))
        context = _context("huggingface")
        with (
            patch.object(QuantSaveProcessor, "pre_run"),
            patch.object(QuantSaveProcessor, "postprocess") as mock_post,
            patch.object(QuantSaveProcessor, "post_run"),
            patch(
                "msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter.QuantSaveProcessor",
            ) as mock_cls,
        ):
            mock_cls.return_value = MagicMock()
            mock_cls.return_value.postprocess = mock_post
            SaveProcessorAdapter().save(context, tree)
            mock_cls.assert_called_once()

    def test_save_raise_error_when_dst_format_unsupported(self):
        """场景：未知 dst_format。
        预期：begin 与 save 均抛 ValueError。
        """
        tree = nn.Module()
        context = _context("unknown_fmt")
        with pytest.raises(ValueError, match="Unsupported dst_format"):
            SaveProcessorAdapter().save(context, tree)
        with pytest.raises(ValueError, match="Unsupported dst_format"):
            SaveProcessorAdapter().begin(context, tree)

    def test_save_compressed_tensors_builds_valid_quant_save_config(self):
        """场景：HF 保存默认 part_file_size。
        预期：format 配置为 CompressedTensorsQuantFormatConfig 且 part_file_size=4。
        """
        from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
            CompressedTensorsQuantFormatConfig,
        )

        tree = nn.Module()
        context = _context("huggingface")
        with (
            patch.object(QuantSaveProcessor, "pre_run"),
            patch.object(QuantSaveProcessor, "postprocess"),
            patch.object(QuantSaveProcessor, "post_run"),
            patch(
                "msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter.QuantSaveProcessor",
            ) as mock_cls,
        ):
            SaveProcessorAdapter().save(context, tree)
            cfg = mock_cls.call_args[0][1]
            assert isinstance(cfg.format, CompressedTensorsQuantFormatConfig)
            assert cfg.format.part_file_size == 4

    def test_save_compressed_tensors_use_part_file_size_when_config_set(self):
        """场景：HF 保存 part_file_size=0。
        预期：透传到 format 配置。
        """
        from msmodelslim.format.compressed_tensors_format.compressed_tensors import (
            CompressedTensorsQuantFormatConfig,
        )

        tree = nn.Module()
        context = _context("huggingface", part_file_size=0)
        with (
            patch.object(QuantSaveProcessor, "pre_run"),
            patch.object(QuantSaveProcessor, "postprocess"),
            patch.object(QuantSaveProcessor, "post_run"),
            patch(
                "msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter.QuantSaveProcessor",
            ) as mock_cls,
        ):
            SaveProcessorAdapter().save(context, tree)
            cfg = mock_cls.call_args[0][1]
            assert isinstance(cfg.format, CompressedTensorsQuantFormatConfig)
            assert cfg.format.part_file_size == 0

    def test_save_ascendv1_use_part_file_size_when_config_set(self):
        """场景：AscendV1 保存 part_file_size=8。
        预期：透传到 AscendV1Config。
        """
        tree = nn.Module()
        context = _context("ascendv1", part_file_size=8)
        with (
            patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.AscendV1Saver") as mock_saver_cls,
            patch(
                "msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter._lazy_init_unsaved_modules",
            ),
        ):
            mock_saver = MagicMock()
            mock_saver_cls.return_value = mock_saver
            SaveProcessorAdapter().save(context, tree)
            saver_cfg = mock_saver_cls.call_args.kwargs["config"]
            assert saver_cfg.part_file_size == 8
            mock_saver.pre_run.assert_called_once()
            mock_saver.post_run.assert_called_once()

    def test_begin_accept_finalize_when_streaming_ascendv1(self):
        """场景：流式保存 AscendV1，accept 一个模块后 finalize。
        预期：pre_run → postprocess(该路径) → lazy_init → post_run。
        """
        tree = nn.Module()
        module = nn.Linear(2, 2, bias=False)
        context = _context("ascendv1")
        adapter = SaveProcessorAdapter()
        with (
            patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.AscendV1Saver") as mock_cls,
            patch(
                "msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter._lazy_init_unsaved_modules",
            ) as mock_lazy,
        ):
            mock_saver = MagicMock()
            mock_cls.return_value = mock_saver
            adapter.begin(context, tree)
            adapter.accept("layers.0.q_proj", module)
            adapter.finalize()
            mock_saver.pre_run.assert_called_once()
            mock_saver.postprocess.assert_called_once()
            assert mock_saver.postprocess.call_args[0][0].name == "layers.0.q_proj"
            mock_lazy.assert_called_once()
            mock_saver.post_run.assert_called_once()

    def test_accept_clear_processed_modules_when_streaming(self):
        """场景：流式 accept 一个模块。
        预期：写完立即清空 processed_modules，不把一次性模块握到扫完树。
        """
        tree = nn.Module()
        module = nn.Linear(2, 2, bias=False)
        context = _context("ascendv1")
        adapter = SaveProcessorAdapter()
        with (
            patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.AscendV1Saver") as mock_cls,
            patch("msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter._lazy_init_unsaved_modules"),
        ):
            mock_saver = MagicMock()
            mock_saver.processed_modules = {module}
            mock_cls.return_value = mock_saver
            adapter.begin(context, tree)
            adapter.accept("layers.0.q_proj", module)
            assert mock_saver.processed_modules == set()
            adapter.finalize()

    def test_begin_write_to_main_staging_when_npu_multi_ascendv1(self):
        """场景：npu_multi + ascendv1 流式 begin。
        预期：主进程 saver 写到 .direct_main staging，供收尾 merge。
        """
        tree = nn.Module()
        context = _context("ascendv1")
        context.parallel_mode = "npu_multi"
        adapter = SaveProcessorAdapter()
        with patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.AscendV1Saver") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter.begin(context, tree)
            cfg = mock_cls.call_args.kwargs["config"]
            assert cfg.save_directory.endswith(".direct_main")
            adapter.abort()

    def test_finalize_merge_staged_output_when_direct_write_metas_set(self):
        """场景：npu_multi 流式 finalize 且已注入 worker_metas。
        预期：merge 收到 worker_metas 与 main_meta。
        """
        tree = nn.Module()
        context = _context("ascendv1")
        context.parallel_mode = "npu_multi"
        adapter = SaveProcessorAdapter()
        with (
            patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.AscendV1Saver") as mock_cls,
            patch(
                "msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter._lazy_init_unsaved_modules",
            ),
            patch(
                "msmodelslim.core.quant_service.modelslim_convert.impl.direct_save.merge_staged_output",
            ) as mock_merge,
        ):
            mock_saver = MagicMock()
            mock_saver.safetensors_writer.saved_keys_map = {"a": "f.safetensors"}
            mock_saver.json_writer.value_map = {"a": "FLOAT"}
            mock_cls.return_value = mock_saver
            adapter.begin(context, tree)
            adapter.set_direct_worker_metas([(".direct_0", {"b": "g.safetensors"}, {"b": "W8A8_MXFP8"})])
            adapter.finalize()
            mock_merge.assert_called_once()
            args = mock_merge.call_args[0]
            assert args[2] == [(".direct_0", {"b": "g.safetensors"}, {"b": "W8A8_MXFP8"})]
            assert args[3] == ({"a": "f.safetensors"}, {"a": "FLOAT"})

    def test_abort_skip_post_run_when_session_open(self):
        """场景：begin 后 abort。
        预期：不调用 post_run，不写索引；再次 abort 为空操作。
        """
        tree = nn.Module()
        context = _context("ascendv1")
        adapter = SaveProcessorAdapter()
        with patch("msmodelslim.core.quant_service.modelslim_v1.save.ascendv1.AscendV1Saver") as mock_cls:
            mock_saver = MagicMock()
            mock_cls.return_value = mock_saver
            adapter.begin(context, tree)
            adapter.abort()
            mock_saver.post_run.assert_not_called()
            adapter.abort()
            mock_saver.post_run.assert_not_called()
