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

from typing import TYPE_CHECKING, Tuple

from msmodelslim.core.graph.fast_ops_grapher.formatters.formatter_mng import register_formatter

if TYPE_CHECKING:
    from msmodelslim.core.graph.fast_ops_grapher.exec_observer.exec_dag import ComputationGraph


DOT_ESCAPE_CHARS = {'"': '\\"', '\n': '\\n', '\\': '\\\\'}


def _generate_node_names(op_name: str, node_id: int) -> Tuple[str, str]:
    simple_op_name = op_name.rsplit('.', 1)[0] if '.' in op_name else op_name
    node_name = f"{simple_op_name}_{node_id}"
    return simple_op_name, node_name


@register_formatter("dot")
def dot_formatter(graph: 'ComputationGraph') -> str:
    lines = ["digraph fx_graph {", "    rankdir=TB;"]

    for node in graph.iter_nodes():
        node_info = node.format_info()
        simple_op_name, node_name = _generate_node_names(node_info['op_name'], node.id)
        attr = {
            'label': simple_op_name,
            'op_name': node_info['op_name'],
            'call_stack': node_info['call_stack'].translate(str.maketrans(DOT_ESCAPE_CHARS)),
        }
        attr_str = ",".join(f'{k}="{v}"' for k, v in attr.items())
        lines.append(f'    {node_name} [{attr_str}];')

    for edge in graph.iter_edges():
        source_node = graph.get_node(edge.source_node_id)
        target_node = graph.get_node(edge.target_node_id)
        if source_node and target_node:
            _, source_name = _generate_node_names(source_node.format_info()['op_name'], source_node.id)
            _, target_name = _generate_node_names(target_node.format_info()['op_name'], target_node.id)
            edge_info = edge.format_info()
            edge_attr = {'label': edge_info['varname'], 'dtype': edge_info['dtype'], 'shape': edge_info['shape']}
            edge_attr_str = ",".join(f'{k}="{v}"' for k, v in edge_attr.items())
            lines.append(f"    {source_name} -> {target_name} [{edge_attr_str}];")

    lines.append("}")
    return "\n".join(lines)
