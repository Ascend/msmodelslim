#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tests for block forward capture helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.core.base.protocol import BatchProcessRequest
from msmodelslim.processor.trainable_linear_quant.pipeline.runtime import (
    capture_float_teacher,
    capture_quant_propagation,
    log_npu_mem,
)
from msmodelslim.processor.trainable_linear_quant.pipeline.runtime.capture import _output_to_cpu


class TinyBlock(nn.Module):
    def forward(self, x):
        return x * 2


def _request(datas):
    return BatchProcessRequest(name="block0", module=TinyBlock(), datas=datas)


class TestOutputToCpu(unittest.TestCase):
    def test_tensor_is_returned_on_cpu(self):
        out = _output_to_cpu(torch.zeros(2))

        self.assertEqual(out.device.type, "cpu")
        self.assertTrue(torch.equal(out, torch.zeros(2)))

    def test_sequence_container_type_and_non_tensor_entries_are_preserved(self):
        out = _output_to_cpu([torch.zeros(1), "meta"])

        self.assertIsInstance(out, list)
        self.assertEqual(out[1], "meta")
        self.assertEqual(out[0].device.type, "cpu")

    def test_non_tensor_passes_through(self):
        self.assertEqual(_output_to_cpu("plain"), "plain")


class TestCaptureFloatTeacher(unittest.TestCase):
    def test_moves_outputs_to_cpu_and_clears_request_outputs(self):
        x = torch.randn(2, 3)
        request = _request([((x,), {})])

        outputs = capture_float_teacher(request)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].device.type, "cpu")
        self.assertTrue(torch.equal(outputs[0], x * 2))
        self.assertIsNone(request.outputs)

    def test_skips_empty_sample(self):
        request = _request([((), {})])

        self.assertEqual(capture_float_teacher(request), [])


class TestCaptureQuantPropagation(unittest.TestCase):
    def test_keeps_outputs_and_sets_request_outputs(self):
        x = torch.randn(2, 3)
        request = _request([((x,), {})])

        outputs = capture_quant_propagation(request)

        self.assertIs(request.outputs, outputs)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].device.type, "cpu")
        self.assertTrue(torch.equal(outputs[0], x * 2))


class TestLogNpuMem(unittest.TestCase):
    def test_logs_allocated_and_reserved_when_npu_available(self):
        fake_npu = SimpleNamespace(
            memory_allocated=lambda: 2 * 1024**3,
            memory_reserved=lambda: 4 * 1024**3,
        )

        with patch.object(torch, "npu", fake_npu, create=True):
            with self.assertLogs("msmodelslim", level="INFO") as logs:
                log_npu_mem("UNIT")

        self.assertTrue(any("[MEM UNIT] allocated=2.00GB, reserved=4.00GB" in message for message in logs.output))

    def test_no_log_when_npu_is_absent(self):
        sentinel = object()
        saved = getattr(torch, "npu", sentinel)
        if saved is not sentinel:
            delattr(torch, "npu")
        try:
            with self.assertNoLogs("msmodelslim", level="INFO"):
                log_npu_mem("NO_NPU")
        finally:
            if saved is not sentinel:
                torch.npu = saved


if __name__ == "__main__":
    unittest.main()
