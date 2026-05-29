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

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Optional
import inspect

import torch

_EXEC_TRACE_MODULE_PREFIX = 'msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_trace'
_EXEC_TRACE_START_API = 'trace_module'


@dataclass
class TensorInfo:
    """Tensor 元信息数据类。"""

    id: int
    varname: str
    dtype: torch.dtype
    shape: torch.Size


@dataclass
class OperatorRecord:
    """算子执行记录数据类。"""

    op_name: str
    inputs: List[TensorInfo]
    outputs: List[TensorInfo]
    traceback: List[Dict[str, Any]]

    def __init__(self, op_name="", inputs=None, outputs=None):
        self.op_name = op_name
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if outputs is not None else []
        self.traceback = self.__capture_traceback()

    def __capture_traceback(self) -> List[Dict[str, Any]]:
        frames = inspect.stack()
        filtered_frames = []
        trace_started = False

        for frame_info in frames[::-1]:
            frame = frame_info.frame
            frame_module = inspect.getmodule(frame)
            if frame_module is None:
                continue
            frame_module_name = frame_module.__name__

            if frame_module_name.startswith(_EXEC_TRACE_MODULE_PREFIX):
                if frame.f_code.co_name == _EXEC_TRACE_START_API:
                    trace_started = True
                continue
            if not trace_started:
                continue

            if frame_module_name.startswith('torch'):
                continue

            traceback_info = {
                "filename": frame_info.filename,
                "lineno": frame_info.lineno,
                "module": frame_module_name,
                "function": frame_info.function,
                "code_context": ';'.join(line.strip() for line in (frame_info.code_context or [])),
            }

            filtered_frames.append(traceback_info)
        return filtered_frames


@dataclass
class ExecutionTrace:
    """执行轨迹数据类。"""

    operators: List[OperatorRecord]

    def __init__(self, operators=None):
        self.operators = operators if operators is not None else []


class _ExecutionTracer:
    """算子执行追踪器，线性记录算子名称和输入输出张量的元信息。"""

    def __init__(self):
        self._execution_trace = None

    @property
    def execution_trace(self) -> ExecutionTrace:
        if self._execution_trace is None:
            raise RuntimeError("Execution trace is not available. Please run trace first.")
        return self._execution_trace

    def record_operator(self, op: OperatorRecord):
        if self._execution_trace is None:
            raise RuntimeError("Execution trace is not available. Please run trace first.")
        self._execution_trace.operators.append(op)

    def trace(self) -> _ExecutionTraceDispatch:
        self._execution_trace = ExecutionTrace()
        return self._ExecutionTraceDispatch(self)

    class _ExecutionTraceDispatch(torch.utils._python_dispatch.TorchDispatchMode):
        def __init__(self, execution_tracer: _ExecutionTracer):
            super().__init__()
            self.__execution_tracer = execution_tracer

        def _extract_tensors(self, tensors: Any, prefix: str = "") -> dict[str, torch.Tensor]:
            # 此函数递归地从任意嵌套结构中提取 torch.Tensor 对象，暂未设计递归深度上限，因为模型网络计算中的嵌套结构通常很简单。如果真的出现了极端情况导致递归溢出，则此函数需要重新设计，确保提取过程中所有的tensor都能被正确记录，而不是简单地抛出异常或跳过。
            if isinstance(tensors, torch.Tensor):
                return {prefix: tensors}
            if isinstance(tensors, (list, tuple)):
                result = {}
                for i, t in enumerate(tensors):
                    result.update(self._extract_tensors(t, f"{prefix}[{i}]"))
                return result
            if isinstance(tensors, dict):
                result = {}
                for k, t in tensors.items():
                    result.update(self._extract_tensors(t, f"{prefix}['{k}']"))
                return result
            return {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs if kwargs is not None else {}
            output = func(*args, **kwargs)

            input_tensors = self._extract_tensors(args, "args") | self._extract_tensors(kwargs, "kwargs")
            output_tensors = self._extract_tensors(output, "output")

            op_record = OperatorRecord()
            op_record.op_name = func.__name__
            op_record.inputs = [
                TensorInfo(
                    id=id(v),
                    varname=k,
                    dtype=v.dtype,
                    shape=v.shape,
                )
                for k, v in input_tensors.items()
            ]
            op_record.outputs = [
                TensorInfo(
                    id=id(v),
                    varname=k,
                    dtype=v.dtype,
                    shape=v.shape,
                )
                for k, v in output_tensors.items()
            ]

            self.__execution_tracer.record_operator(op_record)
            return output


def trace_module(
    model: torch.nn.Module,
    *,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
) -> ExecutionTrace:
    kwargs = kwargs if kwargs is not None else {}
    tracer = _ExecutionTracer()
    with tracer.trace():
        model(*args, **kwargs)
    return tracer.execution_trace
