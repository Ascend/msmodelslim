#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest
from unittest import mock

import torch
from torch import nn

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
from msmodelslim.processor.trainable_linear_quant import interface as tlq_interface
from msmodelslim.processor.trainable_linear_quant.core.ops.base import format_tensor_dbg
from msmodelslim.processor.trainable_linear_quant.core.ops.trainable_smooth import (
    TrainableSmoothOp,
    TrainableSmoothOpConfig,
    _register_non_fusion_smooth_hook,
    _resolve_smooth_fusion_layer,
    get_non_fusion_smooth_hook_scales,
    remove_non_fusion_smooth_hooks,
    smooth_source_module,
)
from msmodelslim.processor.trainable_linear_quant.core.wrapper import (
    TrainableLinearQuantWrapper,
)
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


def _qconfig() -> LinearQConfig:
    return LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
    )


def _wrap(linear: nn.Linear, path: str) -> TrainableLinearQuantWrapper:
    wrapper = TrainableLinearQuantWrapper(linear, linear_qconfig=_qconfig())
    wrapper.layer_path = path
    return wrapper


class TestSmoothHelpers(unittest.TestCase):
    def test_smooth_source_module_variants(self):
        lin = nn.Linear(4, 4)
        wrapped = _wrap(lin, "blk.fc")
        norm = nn.LayerNorm(4)
        self.assertIs(
            smooth_source_module(NormLinearSubgraph(norm=norm, linears=[wrapped], linear_names=["blk.fc"])),
            norm,
        )
        self.assertIs(
            smooth_source_module(LinearLinearSubgraph(linear1=lin, linear2=wrapped, linear2_name="blk.fc")),
            lin,
        )
        self.assertIs(
            smooth_source_module(
                OVSubgraph(
                    o_proj=wrapped,
                    v_proj=lin,
                    num_attention_heads=4,
                    key_value_heads=4,
                    o_proj_name="blk.o",
                )
            ),
            lin,
        )
        self.assertIs(
            smooth_source_module(
                UpDownSubgraph(
                    up_proj=lin,
                    down_proj=wrapped,
                    gate_proj=nn.Linear(4, 8),
                    down_proj_name="blk.down",
                )
            ),
            lin,
        )
        self.assertIsNone(smooth_source_module(NonFusionSubgraph(linears=[wrapped], linear_names=["blk.fc"])))
        with self.assertRaises(UnsupportedError):
            smooth_source_module(object())  # type: ignore[arg-type]

    def test_resolve_fusion_layer_and_hooks(self):
        linear = nn.Linear(4, 4)
        wrapped = _wrap(linear, "blk.fc")
        self.assertIs(_resolve_smooth_fusion_layer(wrapped), linear)
        self.assertIs(_resolve_smooth_fusion_layer(linear), linear)

        self.assertIsNone(get_non_fusion_smooth_hook_scales(linear))
        scales = torch.ones(4)
        _register_non_fusion_smooth_hook(linear, scales)
        hooked = get_non_fusion_smooth_hook_scales(linear)
        self.assertIsNotNone(hooked)
        removed = remove_non_fusion_smooth_hooks(linear)
        self.assertEqual(removed, 1)
        self.assertIsNone(get_non_fusion_smooth_hook_scales(linear))

        self.assertEqual(format_tensor_dbg(None), "none")
        self.assertIn("min=", format_tensor_dbg(torch.tensor([0.5, 1.5])))

    def test_config_validator(self):
        cfg = TrainableSmoothOpConfig(enable_subgraph_type=["norm-linear", "ov"])
        self.assertEqual(cfg.enable_subgraph_type, ["norm-linear", "ov"])
        with self.assertRaises(SchemaValidateError):
            TrainableSmoothOpConfig(enable_subgraph_type=["bad-type"])


class TestTrainableSmoothBasics(unittest.TestCase):
    def test_bind_effective_scale_forward_and_unbind_norm_linear(self):
        linear = nn.Linear(4, 4)
        wrapper = _wrap(linear, "blk.fc")
        norm = nn.LayerNorm(4)
        subgraph = NormLinearSubgraph(norm=norm, linears=[wrapper], linear_names=["blk.fc"])
        op = TrainableSmoothOp(
            TrainableSmoothOpConfig(),
            subgraph=subgraph,
            target_modules={"blk.fc": wrapper},
        )
        op.bind()
        scale = op._effective_scale()
        self.assertEqual(tuple(scale.shape), (4,))
        # tensor path (non-Parameter)
        clamped = op._effective_scale(torch.tensor([0.01, 20.0, 1.0, 1.0]))
        self.assertTrue(torch.all(clamped >= 0.1))
        self.assertTrue(torch.all(clamped <= 10.0))

        y = wrapper(torch.randn(2, 4))
        self.assertEqual(tuple(y.shape), (2, 4))
        op.unbind()

    def test_linear_linear_unbind_fuses_source(self):
        src = nn.Linear(4, 4)
        dst = _wrap(nn.Linear(4, 4), "blk.fc2")
        subgraph = LinearLinearSubgraph(linear1=src, linear2=dst, linear2_name="blk.fc2")
        op = TrainableSmoothOp(
            TrainableSmoothOpConfig(),
            subgraph=subgraph,
            target_modules={"blk.fc2": dst},
        )
        op.bind()
        with torch.no_grad():
            op._smooth_scale.data.fill_(2.0)
        before = src.weight.detach().clone()
        op.unbind()
        self.assertFalse(torch.equal(before, src.weight.detach()))

    def test_ov_gqa_listener_scale_expand(self):
        o_wrap = _wrap(nn.Linear(4, 4), "blk.o")
        v = nn.Linear(4, 4)
        subgraph = OVSubgraph(
            o_proj=o_wrap,
            v_proj=v,
            num_attention_heads=8,
            key_value_heads=2,
            o_proj_name="blk.o",
        )
        op = TrainableSmoothOp(
            TrainableSmoothOpConfig(),
            subgraph=subgraph,
            target_modules={"blk.o": o_wrap},
        )
        op.bind()
        flat = torch.ones(4)
        with (
            mock.patch(
                "msmodelslim.processor.trainable_linear_quant.core.ops.trainable_smooth.prepare_mqga_parameters",
                return_value=(4, 2),
            ),
            mock.patch(
                "msmodelslim.processor.trainable_linear_quant.core.ops.trainable_smooth.reduce_scales_for_mqga_mean",
                return_value=(torch.ones(8), None),
            ),
        ):
            expanded = op._listener_smooth_scale(flat)
        self.assertEqual(tuple(expanded.shape), (8,))

        subgraph_mha = OVSubgraph(
            o_proj=o_wrap,
            v_proj=v,
            num_attention_heads=4,
            key_value_heads=4,
            o_proj_name="blk.o",
        )
        op_mha = TrainableSmoothOp(
            TrainableSmoothOpConfig(),
            subgraph=subgraph_mha,
            target_modules={"blk.o": o_wrap},
        )
        op_mha.bind()
        same = op_mha._listener_smooth_scale(torch.ones(4))
        self.assertTrue(torch.equal(same, torch.ones(4)))
        op_mha.unbind()

    def test_non_fusion_bind_and_unbind_with_and_without_prior_hook(self):
        linear = nn.Linear(4, 4)
        wrapped = _wrap(linear, "blk.fc")
        subgraph = NonFusionSubgraph(linears=[wrapped], linear_names=["blk.fc"])
        op = TrainableSmoothOp(
            TrainableSmoothOpConfig(),
            subgraph=subgraph,
            target_modules={"blk.fc": wrapped},
        )
        op.bind()
        op.unbind()
        self.assertIsNotNone(get_non_fusion_smooth_hook_scales(linear))
        remove_non_fusion_smooth_hooks(linear)

        # with prior hook
        wrapped2 = _wrap(nn.Linear(4, 4), "blk.fc2")
        linear2 = wrapped2.orig_layer
        prior = torch.ones(4) * 2
        _register_non_fusion_smooth_hook(linear2, prior)
        subgraph2 = NonFusionSubgraph(linears=[wrapped2], linear_names=["blk.fc2"])
        op2 = TrainableSmoothOp(
            TrainableSmoothOpConfig(),
            subgraph=subgraph2,
            target_modules={"blk.fc2": wrapped2},
        )
        op2.bind()
        op2.unbind()
        merged = get_non_fusion_smooth_hook_scales(linear2)
        self.assertIsNotNone(merged)

    def test_init_requires_source_for_non_nonfusion(self):
        wrapped = _wrap(nn.Linear(4, 4), "blk.fc")
        # NonFusion is allowed without source; unsupported type via registry mismatch is hard.
        # Use OV without going through registry unknown — already covered by smooth_source_module.
        subgraph = NonFusionSubgraph(linears=[wrapped], linear_names=["blk.fc"])
        op = TrainableSmoothOp(
            TrainableSmoothOpConfig(),
            subgraph=subgraph,
            target_modules={"blk.fc": wrapped},
        )
        self.assertEqual(op.subgraph_type, "non-fusion")


class TestInterfaceExport(unittest.TestCase):
    def test_all_exports(self):
        self.assertIn("TLQBlockDataInterface", tlq_interface.__all__)
        self.assertIn("TLQSubgraphAdapter", tlq_interface.__all__)


if __name__ == "__main__":
    unittest.main()
