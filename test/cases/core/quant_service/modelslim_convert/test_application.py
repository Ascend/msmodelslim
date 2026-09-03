#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
msmodelslim.core.quant_service.modelslim_convert.application 模块的单元测试
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from msmodelslim.core.convert.catalog import (
    PreprocessResult,
    TensorCatalog,
    TensorEntry,
)
from msmodelslim.core.convert.config import ConvertConfig, ConvertRule, ModuleRule
from msmodelslim.core.convert.types import IRKind
from msmodelslim.core.quant_service.modelslim_convert.application import (
    ConvertApplication,
)


class TestConvertApplication:
    """测试 ConvertApplication 类"""

    def test_run_execute_convert_and_save_when_config_valid(self, tmp_path: Path):
        catalog = TensorCatalog()
        catalog.add(TensorEntry(key="layers.0.q_proj.weight", shard="s0", dtype="bf16", shape=(2, 2)))
        preprocess_result = PreprocessResult(catalog=catalog)

        reader = MagicMock()
        reader.read_catalog.return_value = catalog

        preprocess = MagicMock()
        preprocess.run.return_value = preprocess_result

        from torch import nn

        tree = nn.Module()
        tree_builder = MagicMock()
        tree_builder.build.return_value = tree

        from msmodelslim.core.convert.tasks import IRTask
        from msmodelslim.core.convert.types import SourceIR, TensorRef

        task = IRTask(
            module_path="layers.0.q_proj",
            source_ir=SourceIR(kind=IRKind.FLOAT),
            target_ir=IRKind.FLOAT,
            tensor_bindings={
                "weight": TensorRef("weight", "layers.0.q_proj.weight", "s0", "bf16", (2, 2)),
            },
            inverse_weight_map={"s0": ["layers.0.q_proj.weight"]},
        )
        task_builder = MagicMock()
        task_builder.build.return_value = [task]

        executor = MagicMock()
        executor.run.return_value = iter([])
        save_adapter = MagicMock()

        from msmodelslim.core.convert.router import IRRouter

        router = IRRouter.default()
        app = ConvertApplication(
            checkpoint_reader_factory=lambda _: reader,
            preprocess_executor=preprocess,
            tree_builder=tree_builder,
            task_builder=task_builder,
            executor=executor,
            save_adapter=save_adapter,
            router=router,
        )
        save_path = tmp_path / "out"
        save_path.mkdir()
        config = ConvertConfig(
            model_path=str(tmp_path / "model"),
            save_path=str(save_path),
            module_rules=[ModuleRule(match="layers.*.q_proj", source_format="bf16")],
            convert_rules=[ConvertRule(match="layers.*.q_proj", target_ir=IRKind.FLOAT)],
        )
        app.run(config)
        executor.run.assert_called_once()
        save_adapter.begin.assert_called_once()
        save_adapter.finalize.assert_called_once()
        save_adapter.abort.assert_not_called()
        save_adapter.save.assert_not_called()
        assert save_adapter.begin.call_args[0][1] is tree

    def test_run_call_abort_when_executor_raises(self, tmp_path: Path):
        """场景：executor.run 抛异常。
        预期：调用 abort，不调用 finalize。
        """
        catalog = TensorCatalog()
        preprocess_result = PreprocessResult(catalog=catalog)
        reader = MagicMock()
        reader.read_catalog.return_value = catalog
        preprocess = MagicMock()
        preprocess.run.return_value = preprocess_result
        from torch import nn

        tree = nn.Module()
        tree_builder = MagicMock()
        tree_builder.build.return_value = tree
        task_builder = MagicMock()
        task_builder.build.return_value = []
        executor = MagicMock()
        executor.run.side_effect = RuntimeError("convert failed")
        save_adapter = MagicMock()
        from msmodelslim.core.convert.router import IRRouter

        app = ConvertApplication(
            checkpoint_reader_factory=lambda _: reader,
            preprocess_executor=preprocess,
            tree_builder=tree_builder,
            task_builder=task_builder,
            executor=executor,
            save_adapter=save_adapter,
            router=IRRouter.default(),
        )
        save_path = tmp_path / "out"
        save_path.mkdir()
        config = ConvertConfig(model_path=str(tmp_path / "model"), save_path=str(save_path))
        with pytest.raises(RuntimeError, match="convert failed"):
            app.run(config)
        save_adapter.begin.assert_called_once()
        save_adapter.abort.assert_called_once()
        save_adapter.finalize.assert_not_called()
