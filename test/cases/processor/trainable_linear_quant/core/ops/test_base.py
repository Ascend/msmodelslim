#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for trainable_linear_quant core ops base: config, lifecycle and factories."""

import unittest
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import torch
from torch import nn

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.core.ops import (
    LinearTLQOp,
    MinmaxTuneOp,
    MinmaxTuneOpConfig,
    TLQOp,
    TLQOpConfig,
    create_linear_tlq_op,
    is_subgraph_op_config,
    load_tlq_op_class,
    operations_need_adapter_subgraph,
    registered_tlq_op_types,
)
from msmodelslim.processor.trainable_linear_quant.core.ops.base import format_tensor_dbg
from msmodelslim.processor.trainable_linear_quant.core.wrapper import (
    TrainableLinearQuantWrapper,
)
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


class _ProbeOpConfig(TLQOpConfig):
    """Concrete config subclass so the base-class lr constraint can be exercised."""

    type: Literal["probe_op"] = "probe_op"


class _DummyOp(TLQOp):
    def __init__(self) -> None:
        super().__init__()
        self.op_id = "dummy_op"
        self._param = nn.Parameter(torch.tensor([1.0, 2.0]))

    def bind(self) -> None:
        return None

    def unbind(self) -> None:
        return None

    @property
    def train_params(self) -> dict:
        return {"p": self._param}


class _DummyLinearOp(LinearTLQOp):
    def bind(self) -> None:
        return None

    def unbind(self) -> None:
        return None

    @property
    def train_params(self) -> dict:
        return {}


def _int8_wrapper() -> TrainableLinearQuantWrapper:
    qconfig = LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
    )
    wrapper = TrainableLinearQuantWrapper(nn.Linear(4, 4), qconfig)
    wrapper.layer_path = "block0.fc"
    return wrapper


class TestTLQOpConfig(unittest.TestCase):
    def test_lr_is_optional_and_positive(self):
        self.assertIsNone(_ProbeOpConfig().lr)
        self.assertEqual(_ProbeOpConfig(lr=0.5).lr, 0.5)

    def test_non_positive_lr_is_rejected(self):
        for bad_lr in (0.0, -0.1):
            with self.assertRaises(SchemaValidateError):
                _ProbeOpConfig(lr=bad_lr)


class TestTLQOpLifecycle(unittest.TestCase):
    def test_best_params_roundtrip_and_reset(self):
        op = _DummyOp()
        self.assertIsNone(op.best_params)
        with self.assertRaises(UnsupportedError):
            op.load_best_params()

        op.save_best_params()
        self.assertTrue(torch.equal(op.best_params["p"], torch.tensor([1.0, 2.0])))
        self.assertIsNot(op.best_params["p"], op.train_params["p"])

        op.train_params["p"].data.fill_(9.0)
        op.load_best_params()
        self.assertTrue(torch.equal(op.train_params["p"], torch.tensor([1.0, 2.0])))

        op.reset_params({"p": torch.tensor([3.0, 4.0])})
        self.assertTrue(torch.equal(op.train_params["p"], torch.tensor([3.0, 4.0])))

    def test_reset_params_skips_absent_keys(self):
        op = _DummyOp()
        op.reset_params({})
        self.assertTrue(torch.equal(op.train_params["p"], torch.tensor([1.0, 2.0])))

    def test_release_cached_params_drops_snapshot(self):
        op = _DummyOp()
        op.save_best_params()
        self.assertIsNotNone(op.best_params)

        op.release_cached_params()

        self.assertIsNone(op.best_params)
        with self.assertRaises(UnsupportedError):
            op.load_best_params()


class TestLinearTLQOp(unittest.TestCase):
    def test_op_id_and_target_modules(self):
        wrapper = MagicMock()
        op = _DummyLinearOp(_ProbeOpConfig(), "block0.fc", wrapper)

        self.assertEqual(op.op_id, "block0.fc.probe_op")
        self.assertEqual(list(op.target_modules), ["block0.fc"])
        self.assertIs(op.target_modules["block0.fc"], wrapper)


class TestOpRegistryAndFactories(unittest.TestCase):
    def test_registered_types_include_builtin_ops(self):
        types = registered_tlq_op_types()
        self.assertIn("minmax_tune", types)
        self.assertIn("round_tune", types)

    def test_load_tlq_op_class_resolves_registered_plugin(self):
        self.assertIs(load_tlq_op_class(MinmaxTuneOpConfig()), MinmaxTuneOp)

    def test_create_linear_tlq_op_binds_layer_and_wrapper(self):
        wrapper = _int8_wrapper()
        op = create_linear_tlq_op(MinmaxTuneOpConfig(), "block0.fc", wrapper)

        self.assertIsInstance(op, MinmaxTuneOp)
        self.assertEqual(op.op_id, "block0.fc.minmax_tune")
        self.assertIs(op.target_modules["block0.fc"], wrapper)


class TestSubgraphConfigDetection(unittest.TestCase):
    def test_is_subgraph_op_config_follows_enable_subgraph_type(self):
        self.assertFalse(is_subgraph_op_config(MinmaxTuneOpConfig()))
        self.assertTrue(is_subgraph_op_config(SimpleNamespace(enable_subgraph_type="smooth")))

    def test_operations_need_adapter_subgraph(self):
        self.assertFalse(operations_need_adapter_subgraph([MinmaxTuneOpConfig()]))
        self.assertTrue(
            operations_need_adapter_subgraph([MinmaxTuneOpConfig(), SimpleNamespace(enable_subgraph_type="smooth")])
        )


class TestFormatTensorDbg(unittest.TestCase):
    def test_none_is_reported_explicitly(self):
        self.assertEqual(format_tensor_dbg(None), "none")

    def test_base_fields(self):
        self.assertEqual(format_tensor_dbg(torch.tensor([-2.0, 0.0, 1.0])), "min=-2,max=1,mean=-0.3333")

    def test_optional_fields(self):
        text = format_tensor_dbg(
            torch.tensor([-2.0, 0.0, 1.0]),
            include_abs_mean=True,
            include_nnz=True,
            include_grad=True,
        )
        self.assertEqual(text, "min=-2,max=1,mean=-0.3333,abs_mean=1,nnz=2/3,grad=None")

    def test_grad_fields(self):
        t = torch.tensor([-2.0, 0.0, 1.0], requires_grad=True)
        t.grad = torch.tensor([0.0, 2.0, 0.0])
        self.assertEqual(
            format_tensor_dbg(t, include_grad=True),
            "min=-2,max=1,mean=-0.3333,grad_abs_mean=0.6667,grad_nnz=1/3",
        )


if __name__ == "__main__":
    unittest.main()
