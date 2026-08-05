#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from torch import nn

from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.anti_outlier.common.subgraph_type import (
    LinearLinearSubgraph,
    NonFusionSubgraph,
    NormLinearSubgraph,
    OVSubgraph,
    UpDownSubgraph,
)
from msmodelslim.processor.trainable_linear_quant.core.ops.trainable_smooth import (
    TrainableSmoothOp,
    TrainableSmoothOpConfig,
)
from msmodelslim.processor.trainable_linear_quant.core.subgraph import (
    _iter_resolved_subgraphs,
    _iter_subgraph_target_modules,
    _resolve_wrapped_subgraph,
    _subgraph_dedup_key,
    _wrappers_from_subgraph,
    resolve_subgraphs_for_op,
)
from msmodelslim.processor.trainable_linear_quant.core.wrapper import (
    TrainableLinearQuantWrapper,
)
from msmodelslim.utils.exception import UnsupportedError


def _qconfig() -> LinearQConfig:
    return LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
    )


def _wrap(module: nn.Module, path: str) -> TrainableLinearQuantWrapper:
    wrapper = TrainableLinearQuantWrapper(module, linear_qconfig=_qconfig())
    wrapper.layer_path = path
    return wrapper


class TestSubgraphHelpers(unittest.TestCase):
    def test_iter_targets_for_common_subgraph_types(self):
        wrapped = _wrap(nn.Linear(4, 4), "blk.fc")

        norm_sg = NormLinearSubgraph(norm=nn.LayerNorm(4), linears=[wrapped], linear_names=["blk.fc"])
        self.assertEqual(list(_iter_subgraph_target_modules(norm_sg)), [("blk.fc", wrapped)])

        ll = LinearLinearSubgraph(linear1=nn.Linear(4, 4), linear2=wrapped, linear2_name="blk.fc2")
        self.assertEqual(list(_iter_subgraph_target_modules(ll)), [("blk.fc2", wrapped)])

        ov = OVSubgraph(
            o_proj=wrapped,
            v_proj=nn.Linear(4, 4),
            num_attention_heads=4,
            key_value_heads=4,
            o_proj_name="blk.o",
        )
        self.assertEqual(list(_iter_subgraph_target_modules(ov)), [("blk.o", wrapped)])

        ud = UpDownSubgraph(
            up_proj=nn.Linear(4, 8),
            down_proj=wrapped,
            gate_proj=nn.Linear(4, 8),
            down_proj_name="blk.down",
        )
        self.assertEqual(list(_iter_subgraph_target_modules(ud)), [("blk.down", wrapped)])

    def test_subgraph_dedup_key(self):
        cfg = AdapterConfig(
            subgraph_type="norm-linear",
            mapping=MappingConfig(source="blk.norm", targets=["blk.fc"]),
        )
        self.assertIn("norm-linear", _subgraph_dedup_key(cfg))
        self.assertIn("blk.fc", _subgraph_dedup_key(cfg))

    def test_unsupported_subgraph_type(self):
        with self.assertRaises(UnsupportedError):
            list(_iter_subgraph_target_modules(MagicMock()))


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = TinyBlock()
        self.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=4)


class TestWrappersEdgeCases(unittest.TestCase):
    def test_empty_path_and_empty_targets(self):
        wrapped = _wrap(nn.Linear(4, 4), "")
        sg = NormLinearSubgraph(norm=nn.LayerNorm(4), linears=[wrapped], linear_names=[""])
        with self.assertRaises(UnsupportedError):
            _wrappers_from_subgraph(sg)

        empty = NonFusionSubgraph(linears=[], linear_names=[])
        with self.assertRaises(UnsupportedError):
            _wrappers_from_subgraph(empty)


class TestResolveWrappedSubgraph(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyModel()
        self.wrapper = _wrap(self.model.block0.o_proj, "block0.o_proj")
        self.model.block0.o_proj = self.wrapper
        self.cfg = AdapterConfig(
            subgraph_type="ov",
            mapping=MappingConfig(source="block0.v_proj", targets=["block0.o_proj"]),
        )

    def test_resolve_success(self):
        subgraph, targets = _resolve_wrapped_subgraph(self.model, self.cfg, {"block0.o_proj"})
        self.assertIsInstance(subgraph, OVSubgraph)
        self.assertIn("block0.o_proj", targets)

    def test_targets_not_wrapped_raises(self):
        with self.assertRaises(UnsupportedError):
            _resolve_wrapped_subgraph(self.model, self.cfg, {"other.path"})

    def test_build_none_raises(self):
        with patch(
            "msmodelslim.processor.trainable_linear_quant.core.subgraph.build_subgraph_from_adapter",
            return_value=None,
        ):
            with self.assertRaises(UnsupportedError):
                _resolve_wrapped_subgraph(self.model, self.cfg, {"block0.o_proj"})

    def test_extra_targets_raises(self):
        built = MagicMock()
        built.subgraph = NormLinearSubgraph(
            norm=nn.LayerNorm(4),
            linears=[self.wrapper],
            linear_names=["block0.o_proj"],
        )
        with patch(
            "msmodelslim.processor.trainable_linear_quant.core.subgraph.build_subgraph_from_adapter",
            return_value=built,
        ):
            # wrapped_paths missing the resolved path
            with self.assertRaises(UnsupportedError):
                _resolve_wrapped_subgraph(self.model, self.cfg, set())


class TestIterResolvedAndResolveForOp(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyModel()
        self.wrapper = _wrap(self.model.block0.o_proj, "block0.o_proj")
        self.model.block0.o_proj = self.wrapper
        self.cfg = AdapterConfig(
            subgraph_type="ov",
            mapping=MappingConfig(source="block0.v_proj", targets=["block0.o_proj"]),
        )

    def test_iter_resolved_skips_and_yields(self):
        items = list(
            _iter_resolved_subgraphs(
                "ov",
                model=self.model,
                wrapped_paths={"block0.o_proj"},
                adapter_configs=[
                    self.cfg,
                    AdapterConfig(
                        subgraph_type="norm-linear",
                        mapping=MappingConfig(source="block0.v_proj", targets=["block0.o_proj"]),
                    ),
                    AdapterConfig(
                        subgraph_type="ov",
                        mapping=MappingConfig(source="block0.v_proj", targets=["missing"]),
                    ),
                ],
            )
        )
        self.assertEqual(len(items), 1)

    def test_iter_resolved_skips_unsupported(self):
        with patch(
            "msmodelslim.processor.trainable_linear_quant.core.subgraph._resolve_wrapped_subgraph",
            side_effect=UnsupportedError("boom"),
        ):
            items = list(
                _iter_resolved_subgraphs(
                    "ov",
                    model=self.model,
                    wrapped_paths={"block0.o_proj"},
                    adapter_configs=[self.cfg],
                )
            )
        self.assertEqual(items, [])

    def test_resolve_subgraphs_for_op(self):
        bindings = resolve_subgraphs_for_op(
            TrainableSmoothOp,
            TrainableSmoothOpConfig(),
            model=self.model,
            block_name="block0",
            wrappers_by_path={"block0.o_proj": self.wrapper},
            global_adapter_configs=[self.cfg, self.cfg],  # duplicate dedup
            subgraph_types=["ov"],
        )
        self.assertEqual(len(bindings), 1)

    def test_resolve_without_enable_subgraph_type(self):
        op_config = MagicMock(spec=["type"])
        op_config.type = "minmax_tune"
        bindings = resolve_subgraphs_for_op(
            TrainableSmoothOp,
            op_config,
            model=self.model,
            block_name="block0",
            wrappers_by_path={"block0.o_proj": self.wrapper},
            global_adapter_configs=[self.cfg],
            subgraph_types=["ov"],
        )
        self.assertEqual(bindings, [])


if __name__ == "__main__":
    unittest.main()
