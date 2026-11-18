#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
import os
import shutil
import subprocess
from typing import Iterable, Optional
import numpy as np
import onnx
from onnx import TensorProto, helper, mapping, NodeProto, ModelProto, AttributeProto, GraphProto
from ascend_utils.common.security import get_valid_read_path, safe_delete_path_if_exists, get_valid_write_path


def create_empty_folder(folder_name):
    """
    Create folder
    """
    safe_delete_path_if_exists(folder_name)
    os.mkdir(folder_name, mode=0o750)


def encode_vector(vector):
    """
    Encode vector to string format for easy read (RLE-like)
    """
    vector_encoded = []
    count = 0
    for i in range(1, len(vector)):
        count += 1
        if vector[i - 1] != vector[i]:
            vector_encoded.append(f"{count} {vector[i - 1]}")
            count = 0
    vector_encoded.append(f" {count + 1} {vector[i]}")
    vector_encoded_str = ", ".join(vector_encoded)
    return vector_encoded_str


def decode_vector(vector):
    """
    Decode string of RLE format to list
    """
    vector = vector.split(', ')
    vector_decoded = []
    for pair in vector:
        pair = pair.split()
        times, fill = int(pair[0]), int(pair[1])
        for _ in range(times):
            vector_decoded.append(fill)
    return vector_decoded


def get_output_shape(model):
    """
    Check the output shape of the ONNX model
    """
    total_output_shape = 0
    outputs = [output for output in model.graph.output]
    for output in outputs:
        shape = np.prod([shape.dim_value for shape in output.type.tensor_type.shape.dim])
        total_output_shape += shape

    return total_output_shape


def create_adjacency_matrix(model):
    """
    Creating an adjacency matrix by model nodes
    """

    nodes = model.graph.node

    adj_matrix_undir = np.zeros([len(nodes), len(nodes)], dtype=int)
    adj_matrix_dir = np.zeros([len(nodes), len(nodes)], dtype=int)

    for ind_a, node_a in enumerate(nodes):
        input_a = set(list(node_a.input))
        output_a = set(list(node_a.output))
        for ind_b, node_b in enumerate(nodes):
            input_b = set(list(node_b.input))
            output_b = set(list(node_b.output))
            if (input_a | output_a) & (input_b | output_b):
                adj_matrix_undir[ind_a, ind_b] = 1
            if output_a & input_b:
                adj_matrix_dir[ind_a, ind_b] = 1

    return adj_matrix_undir, adj_matrix_dir


def get_model_info(model_path: str, batch_size: int) -> tuple:
    """
    Collecting output nodes and input shapes to model
    """
    model = onnx.load(model_path)
    output_nodes = []
    for node in model.graph.node:
        for out_name in [x.name for x in model.graph.output]:
            node_output = list(node.output)
            if out_name in node_output:
                output_nodes += [node.name + ":" + str(node_output.index(out_name))]
    output_nodes_str = '"' + ';'.join(output_nodes) + '"'

    input_info_list = []
    for graph_input in model.graph.input:
        input_name = graph_input.name
        input_unwrapped = [el.dim_value for el in graph_input.type.tensor_type.shape.dim]
        input_shape = ['batch'] + input_unwrapped[1:]
        input_shape_str = ','.join([str(dim) for dim in input_shape])
        input_info_list.append(f'{input_name}:{input_shape_str}')
    input_shape = ';'.join(input_info_list)
    batch_dummy = (input_shape.split(':')[1]).split(',')[0]
    input_shape = input_shape.replace(batch_dummy, str(batch_size))

    return output_nodes_str, input_shape


def onnx2om(
            model_path: str,
            batch_size: int,
            soc_version: str,
            device_id: int,
            om_method: str) -> None:
    _, input_shape = get_model_info(model_path, batch_size)
    model_folder, filename = os.path.split(model_path)
    model_name, ext = os.path.splitext(filename)

    om_folder = model_folder
    om_name = model_name
    input_onnx = os.path.join(model_folder, model_name+'.onnx')
    input_onnx = get_valid_read_path(input_onnx)
    output = os.path.join(om_folder, om_name)
    output = get_valid_write_path(output, is_dir=False)

    if om_method == 'atc':
        command = "atc --model %s --framework 5 --output %s --soc_version %s --input_shape %s --op_select_implmode " \
                  "high_performance_for_all --optypelist_for_implmode \'Gelu\' --device %d" % \
                  (input_onnx, output, soc_version, input_shape, device_id)
    elif om_method == 'aoe':
        command = "aoe --model %s --framework 5 --output %s --input_shape %s --job_type 1 --op_select_implmode " \
                  "high_performance --optypelist_for_implmode \'Gelu\' --device %d" % \
                  (input_onnx, output, input_shape, device_id)
    else:
        raise ValueError('om_method should be atc or aoe.')
    subprocess.run(command.split())
    return


def generate_model_inputs_for_onnxruntime(model: ModelProto) -> {np.ndarray}:
    inputs = model.graph.input
    ort_inputs = {}
    for inp in inputs:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        if inp.type.tensor_type.elem_type == TensorProto.INT64:
            ort_input = np.ones(shape, dtype='int64')
        elif inp.type.tensor_type.elem_type == TensorProto.INT32:
            ort_input = np.ones(shape, dtype='int32')
        elif inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            ort_input = np.random.random(shape).astype(np.float32)
        elif inp.type.tensor_type.elem_type == TensorProto.DOUBLE:
            ort_input = np.random.random(shape)
        else:
            raise ValueError(f'Unsupported type: {inp.type.tensor_type.elem_type}')
        ort_inputs.update({inp.name: ort_input})
    return ort_inputs


def __np_type_to_tf_type(it: np.dtype) -> int:
    if it in [np.float16, np.float32]:
        return TensorProto.FLOAT
    elif it == np.int32:
        return TensorProto.INT32
    elif it == np.float64:
        return TensorProto.DOUBLE
    elif it == np.int64:
        return TensorProto.INT64
    raise ValueError(f'Unsupported type: {it}')


def _get_bytearray(init_input, np_dtype):
    if init_input.raw_data != b'':
        return bytearray(init_input.raw_data)
    else:
        if np_dtype == np.int64:
            return init_input.int64_data
        elif np_dtype == np.int32:
            return init_input.int32_data
        elif np_dtype == np.int16:
            return init_input.int16_data
        elif np_dtype == np.float32 or np_dtype == np.float16 or np_dtype == np.float64:
            return init_input.float_data
        elif np_dtype == np.uint64:
            return init_input.uint64_data
        else:
            raise TypeError(" Unsupported data type of input! ")


def get_init(graph: GraphProto, tuple_input: tuple) -> tuple:
    list_input_value, values = [], []

    def __append_list_input_value(inp) -> None:
        np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[inp.data_type]
        dims = list(inp.dims)
        value_bin = _get_bytearray(inp, np_dtype)
        if np.array(value_bin).dtype == np.int32 and np_dtype in [np.int64, np.uint64] or \
                np.array(value_bin).dtype == np.float64:
            list_input_value.append(np.array(value_bin).astype(np_dtype).reshape(dims))
        else:
            list_input_value.append(np.array(value_bin).view(dtype=np_dtype).reshape(dims))
        values.append(inp)

    for init_name in tuple_input:
        init_input = next((init for init in graph.initializer if init.name == init_name), None)
        if init_input:
            __append_list_input_value(init_input)
            continue
        const_value = None
        node_const = next((n for n in graph.node if n.op_type == 'Constant' and init_name in n.output), None)
        if node_const is not None:
            value_attr = next(attr for attr in node_const.attribute if attr.name == 'value')
            if value_attr is not None and value_attr.type == AttributeProto.TENSOR:
                const_value = value_attr.t
        if const_value is not None:
            __append_list_input_value(const_value)
        else:
            list_input_value.append(None)
            values.append(None)
    return list_input_value, values


def clear_node_inputs(node: NodeProto) -> None:
    for _ in range(len(node.input)):
        node.input.pop(0)


def replace_node_inputs(node: NodeProto, inp: Iterable[str]) -> None:
    clear_node_inputs(node)
    node.input.extend(inp)


def clear_node_outputs(node: NodeProto) -> None:
    for _ in range(len(node.output)):
        node.output.pop(0)


def replace_node_outputs(node: NodeProto, out: Iterable[str]) -> None:
    clear_node_outputs(node)
    node.output.extend(out)


def clean_constant_nodes(graph: GraphProto) -> None:
    all_node_inputs = set()
    for node_const in [n for n in graph.node if n.op_type == 'Constant']:
        if len(all_node_inputs) == 0:
            for n in graph.node:
                all_node_inputs.update(n.input)
        if all(out not in all_node_inputs for out in node_const.output):
            graph.node.remove(node_const)


def clean_initializer(graph: GraphProto) -> None:
    inits = graph.initializer
    nodes = graph.node
    init_idx = 0
    while init_idx < len(inits):
        init = inits[init_idx]
        if any(init.name in n.input for n in nodes):
            init_idx += 1
        else:
            inits.remove(init)


def fix_model_outputs(model,
                      op_version: int,
                      ir_version: int,
                      logger) -> ModelProto:
    logger.info('Started: Updating model outputs')
    ort_inputs = generate_model_inputs_for_onnxruntime(model)

    import onnxruntime
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    outputs = ort_session.run(None, ort_inputs)
    model_outputs = []
    idx = 0
    while idx < len(outputs):
        out_name = model.graph.output[idx].name if idx < len(model.graph.output) else f'Unk_out:{idx}'
        out_value_info = helper.make_tensor_value_info(
            out_name,
            __np_type_to_tf_type(outputs[idx].dtype),
            outputs[idx].shape
        )
        model_outputs.append(out_value_info)
        idx += 1
    graph_updated = onnx.helper.make_graph(
        model.graph.node, 'opt_graph', model.graph.input,
        model_outputs, initializer=model.graph.initializer
    )
    model_updated = onnx.helper.make_model(graph_updated, producer_name='opt_model')
    model_updated.opset_import[0].version = op_version
    model_updated.ir_version = ir_version

    logger.info('Finished: Successfully updated outputs')
    return model_updated


def check_topology_sorting(graph: GraphProto, logger) -> bool:
    confirmed_inputs = set([inp.name for inp in graph.input])
    for n in graph.node:
        init_inputs, _ = get_init(graph, tuple(n.input))
        unconfirmed_input = next(
            (inp for idx, inp in enumerate(n.input) if init_inputs[idx] is None and inp not in confirmed_inputs),
            None
        )
        if unconfirmed_input is not None:
            logger.error(
                f"ERROR: topology is broken: the input '{unconfirmed_input}' of node '{n.name}' "
                f"(op_type = '{n.op_type}') is neither a model input nor an output of any of the previous nodes"
            )
            return False
        confirmed_inputs.update(n.output)
    logger.info('Topology sorting is OK')
    return True


def check_and_fix_topology_sorting(model: ModelProto, logger) -> ModelProto:
    graph = model.graph
    confirmed_inputs = set([inp.name for inp in graph.input])
    topology_ok = True
    i = 0
    while i < len(graph.node):
        node = graph.node[i]
        init_inputs, _ = get_init(graph, tuple(node.input))
        unconfirmed_inputs = [
            inp
            for idx, inp in enumerate(node.input)
            if init_inputs[idx] is None and inp not in confirmed_inputs
        ]
        if len(unconfirmed_inputs) > 0:
            if topology_ok:
                logger.error(
                    f"ERROR: topology is broken: the inputs '{'; '.join(unconfirmed_inputs)}' of node '{node.name}' "
                    f"(op_type = '{node.op_type}') are neither model inputs nor outputs of any of the previous nodes"
                )
                topology_ok = False
                logger.info('Updating topology sorting...')
            graph_nodes = list(enumerate(graph.node))[:-1]
            node_idx = next(idx for idx, n in graph_nodes if output_has_bad_inp(n, unconfirmed_inputs))
            graph.node.remove(node)
            graph.node.insert(node_idx, node)
        else:
            confirmed_inputs.update(node.output)
            i += 1

    if topology_ok:
        logger.info('Topology sorting was checked')
        return model

    graph_sorted = onnx.helper.make_graph(
        graph.node, graph.name, graph.input, graph.output, initializer=graph.initializer
    )
    model_sorted = onnx.helper.make_model(graph_sorted, producer_name=graph.name)
    model_sorted.opset_import[0].version = model.opset_import[0].version
    model_sorted.ir_version = model.ir_version
    logger.info("Topology was sorted")
    return model_sorted


def output_has_bad_inp(node, unconfirmed_inputs):
    return any(bad_inp in node.output for bad_inp in unconfirmed_inputs)


def convert_const_to_init(model,
                          op_version: int,
                          ir_version: int,
                          logger) -> ModelProto:
    logger.info('Started: Converting Constant nodes into initializer inputs')
    graph = model.graph
    nodes = graph.node
    init = graph.initializer
    old_node_count = len(nodes)
    for node_idx in range(old_node_count)[::-1]:
        node = nodes[node_idx]
        if node.op_type == 'Constant':

            value_attr = next(attr for attr in node.attribute if attr.name == 'value')
            if value_attr is None:
                logger.info(f"Warning: Constant node f{node.name} will not be converted, because only 'value' "
                             f"attribute is currently supported, and the node does not have the 'value' attribute.")
            else:
                if value_attr.type == AttributeProto.TENSOR:
                    t = value_attr.t
                    t.name = node.output[0]
                    init.extend([t])
                    nodes.remove(node)
                else:
                    logger.info(f'Warning: Constant node {node.name} will not be converted, because '
                                 f'its attribute type is {value_attr.type} which is currently unsupported')
    graph_updated = onnx.helper.make_graph(
        nodes, 'opt_graph', model.graph.input,
        model.graph.output, initializer=model.graph.initializer
    )
    model_updated = onnx.helper.make_model(graph_updated, producer_name='opt_model')
    model_updated.opset_import[0].version = op_version
    model_updated.ir_version = ir_version

    logger.info(f'Finished: Converted {old_node_count - len(nodes)} Constant nodes into initializer')
    return model_updated


def rename_nodes(model,
                 op_version: int,
                 ir_version: int,
                 logger) -> ModelProto:
    logger.info('Started: Renaming nodes in ONNX model')
    graph = model.graph
    nodes = graph.node

    for num, node in enumerate(nodes):
        node.name = f"{node.op_type}_{num}"

    graph_updated = onnx.helper.make_graph(
        nodes, 'opt_graph', model.graph.input,
        model.graph.output, initializer=model.graph.initializer
    )
    model_updated = onnx.helper.make_model(graph_updated, producer_name='opt_model')
    model_updated.opset_import[0].version = op_version
    model_updated.ir_version = ir_version

    logger.info(f'Finished: Nodes were renamed')
    return model_updated


def simplify_model(model,
                   op_version: int,
                   ir_version: int,
                   logger) -> ModelProto:
    logger.info('Started: Simplifying ONNX model')
    graph = model.graph
    input_shapes = {}
    for tensor in graph.input:
        input_name = tensor.name
        # 安全提取输入形状，处理动态维度情况
        input_shape = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.HasField('dim_value') and dim.dim_value > 0:
                input_shape.append(dim.dim_value)
            else:
                # 动态维度或无效维度，使用默认值或跳过
                logger.warning(f"Input '{input_name}' has dynamic or invalid dimension: {dim}")
                # 对于动态维度，我们可以使用一个合理的默认值
                # 或者跳过该输入的形状定义
                input_shape.append(1)  # 使用默认值1
        input_shapes[input_name] = input_shape
    from onnxsim import simplify
    model_simp, _ = simplify(model=model, test_input_shapes=input_shapes)
    model_simp.opset_import[0].version = op_version
    model_simp.ir_version = ir_version

    logger.info(f'Finished: ONNX model were simplified')
    return model_simp


def rebatch_model(model: ModelProto,
                  batch_size: int,
                  op_version: int,
                  ir_version: int,
                  logger) -> ModelProto:
    logger.info('Started: Rebatching ONNX model')

    inputs = [
        helper.make_tensor_value_info(
            inp.name, inp.type.tensor_type.elem_type,
            [batch_size] + [d.dim_value for d in inp.type.tensor_type.shape.dim[1:]]
        )
        for inp in model.graph.input
    ]
    outputs = [
        helper.make_tensor_value_info(
            out.name, out.type.tensor_type.elem_type,
            [batch_size] + [d.dim_value for d in out.type.tensor_type.shape.dim[1:]]
        )
        for out in model.graph.output
    ]
    graph = helper.make_graph(
        model.graph.node,
        f'{model.graph.name}_bs{batch_size}',
        inputs,
        outputs
    )
    graph.initializer.extend(model.graph.initializer)
    model_rebatched = helper.make_model(graph, producer_name=f'{model.producer_name}_bs{batch_size}')
    model_rebatched.opset_import[0].version = op_version
    model_rebatched.ir_version = ir_version

    logger.info(f'Finished: ONNX model rebatched to batch size {batch_size}')
    return model_rebatched


def define_batch_size(model: ModelProto, default_batch_size: Optional[int] = None) -> Optional[int]:
    batch_size = None
    for t in list(model.graph.input) + list(model.graph.output):
        bs = t.type.tensor_type.shape.dim[0].dim_value
        if bs == 0:
            continue
        elif batch_size is None:
            batch_size = bs
        elif batch_size != bs:
            raise ValueError('The model has an ambiguously defined batch size')
    return default_batch_size if batch_size is None else batch_size


def load_model(folder_path: str, model_name: str, logger) -> ModelProto:
    model_path = os.path.join(folder_path, f'{model_name}.onnx')
    model_path = get_valid_read_path(model_path, is_dir=False)
    if not os.path.exists(model_path):
        logger.error('Error: ONNX version of model is not available!')
        raise FileNotFoundError()
    logger.info('ONNX version of model was found.')
    model = onnx.load(model_path)
    return model


def is_model_quantized(model: ModelProto) -> bool:
    return any(n.op_type in ['AscendQuant', 'AscendDequant'] for n in model.graph.node)
