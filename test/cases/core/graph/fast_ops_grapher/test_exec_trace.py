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

# pylint: disable=protected-access
import pytest
import torch

from test.cases.core.graph.fast_ops_grapher.conftest import _make_operator_record
from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace import (
    TensorInfo,
    OperatorRecord,
    ExecutionTrace,
    _ExecutionTracer,
    trace_module,
)


class TestTensorInfo:
    def test_tensor_info_creation_sets_all_attributes(self):
        info = TensorInfo(id=42, varname="args[0]", dtype=torch.float32, shape=torch.Size([1, 10]))
        assert info.id == 42
        assert info.varname == "args[0]"
        assert info.dtype == torch.float32
        assert info.shape == torch.Size([1, 10])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64, torch.bool])
    def test_tensor_info_supports_various_data_types(self, dtype):
        info = TensorInfo(id=0, varname="x", dtype=dtype, shape=torch.Size([2, 3]))
        assert info.dtype == dtype

    @pytest.mark.parametrize("shape", [torch.Size([]), torch.Size([1]), torch.Size([2, 3, 4])])
    def test_tensor_info_supports_various_shapes(self, shape):
        info = TensorInfo(id=0, varname="x", dtype=torch.float32, shape=shape)
        assert info.shape == shape


class TestOperatorRecord:
    def test_operator_record_default_init_has_empty_values(self):
        record = _make_operator_record()
        assert record.op_name == ""
        assert not record.inputs
        assert not record.outputs
        assert isinstance(record.traceback, list)

    def test_operator_record_init_sets_custom_op_name(self):
        record = _make_operator_record(op_name="addmm.default")
        assert record.op_name == "addmm.default"

    def test_operator_record_init_sets_inputs_and_outputs(self, sample_tensor_info):
        output_info = TensorInfo(id=2, varname="output", dtype=torch.float32, shape=torch.Size([1, 20]))
        record = _make_operator_record(
            op_name="addmm.default",
            inputs=[sample_tensor_info],
            outputs=[output_info],
        )
        assert len(record.inputs) == 1
        assert len(record.outputs) == 1
        assert record.inputs[0].id == 1
        assert record.outputs[0].id == 2

    def test_operator_record_traceback_is_always_a_list(self):
        record = _make_operator_record(op_name="test_op")
        assert isinstance(record.traceback, list)

    def test_operator_record_supports_multiple_input_tensors(self):
        inputs = [TensorInfo(id=i, varname=f"args[{i}]", dtype=torch.float32, shape=torch.Size([1])) for i in range(5)]
        record = _make_operator_record(op_name="cat.default", inputs=inputs)
        assert len(record.inputs) == 5


class TestExecutionTrace:
    def test_execution_trace_default_init_has_empty_operators_list(self):
        trace = ExecutionTrace()
        assert not trace.operators

    def test_execution_trace_init_with_operators_populates_list(self, sample_operator_record):
        trace = ExecutionTrace(operators=[sample_operator_record])
        assert len(trace.operators) == 1

    def test_execution_trace_supports_appending_operators(self, sample_operator_record):
        trace = ExecutionTrace()
        trace.operators.append(sample_operator_record)
        assert len(trace.operators) == 1


class TestExecutionTracer:
    def test_accessing_execution_trace_before_trace_raises_error(self):
        tracer = _ExecutionTracer()
        with pytest.raises(RuntimeError, match="Execution trace is not available"):
            _ = tracer.execution_trace

    def test_trace_method_creates_empty_execution_trace(self):
        tracer = _ExecutionTracer()
        _ = tracer.trace()
        assert tracer.execution_trace is not None
        assert isinstance(tracer.execution_trace, ExecutionTrace)
        assert not tracer.execution_trace.operators

    def test_record_operator_adds_operator_to_execution_trace(self, sample_operator_record):
        tracer = _ExecutionTracer()
        _ = tracer.trace()
        tracer.record_operator(sample_operator_record)
        assert len(tracer.execution_trace.operators) == 1
        assert tracer.execution_trace.operators[0].op_name == "addmm.default"

    def test_record_multiple_operators_adds_all_to_trace(self):
        tracer = _ExecutionTracer()
        _ = tracer.trace()
        for i in range(3):
            op = _make_operator_record(op_name=f"op_{i}")
            tracer.record_operator(op)
        assert len(tracer.execution_trace.operators) == 3


class TestTraceModule:
    def test_trace_module_returns_execution_trace_for_simple_model(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,))
        assert isinstance(trace, ExecutionTrace)
        assert len(trace.operators) > 0
        op_names = [op.op_name for op in trace.operators]
        assert any("addmm" in name for name in op_names)

    def test_trace_module_returns_trace_for_single_op_model(self, single_op_model, small_input):
        trace = trace_module(single_op_model, args=(small_input,))
        assert len(trace.operators) > 0

    def test_trace_module_works_with_keyword_arguments(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,), kwargs={})
        assert isinstance(trace, ExecutionTrace)
        assert len(trace.operators) > 0

    def test_operator_records_in_trace_have_valid_tensor_info(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,))
        for op in trace.operators:
            assert isinstance(op, OperatorRecord)
            assert isinstance(op.op_name, str)
            for t in op.inputs:
                assert isinstance(t, TensorInfo)
            for t in op.outputs:
                assert isinstance(t, TensorInfo)

    def test_operator_records_in_trace_contain_traceback_info(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,))
        for op in trace.operators:
            assert isinstance(op.traceback, list)

    def test_trace_module_handles_kwargs_none_gracefully(self, simple_model, sample_input):
        trace = trace_module(simple_model, args=(sample_input,), kwargs=None)
        assert isinstance(trace, ExecutionTrace)
