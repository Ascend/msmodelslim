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

from typing import Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import ComputationGraph

_formatters: Dict[str, Callable[['ComputationGraph'], str]] = {}


def register_formatter(format_preset: str) -> Callable:
    def decorator(func: Callable[['ComputationGraph'], str]) -> Callable[['ComputationGraph'], str]:
        _formatters[format_preset] = func
        return func

    return decorator


def format_graph(format_preset: str, graph: 'ComputationGraph') -> str:
    if format_preset not in _formatters:
        raise KeyError(f"Formatter '{format_preset}' not found. Available formatters: {list(_formatters.keys())}")
    return _formatters[format_preset](graph)


def list_formatters() -> list[str]:
    return list(_formatters.keys())
