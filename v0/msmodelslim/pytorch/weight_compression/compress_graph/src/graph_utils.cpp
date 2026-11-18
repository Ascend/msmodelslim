/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */

#include "graph_utils.h"

using namespace std;

namespace GraphUtils {
int CheckShape(vector<int64_t> &shape)
{
    for (auto const &val : shape) {
        if (val > 50000) {
            return GraphUtils::FAILED;
        }
    }
    return GraphUtils::SUCCESS;
}

void GetDataSizeFromShape(vector<int64_t> shape, int64_t &size)
{
    for (const auto &val : shape) {
        size *= val;
    }
}

bool GetDataFromBin(std::string input_path, std::vector<int64_t> shapes, uint8_t *&data, int data_type_size)
{
    std::ifstream inFile(input_path, std::ios::binary);
    if (!inFile.is_open()) {
        std::cout << "Failed to open: " << input_path << std::endl;
        return false;
    }

    inFile.seekg(0, std::ios::end);
    auto fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    int64_t size = 1;
    GetDataSizeFromShape(shapes, size);
    uint64_t dataLen = static_cast<uint64_t>(size) * data_type_size;

    if (dataLen != static_cast<uint64_t>(fileSize)) {
        std::cout << "Invalid param: expected len=" << dataLen
                  << ", but file size=" << fileSize << std::endl;
        return false;
    }

    uint8_t* heapData = new (std::nothrow) uint8_t[dataLen];
    if (!heapData) {
        std::cout << "Failed to allocate memory." << std::endl;
        return false;
    }

    inFile.read(reinterpret_cast<char*>(heapData), dataLen);
    inFile.close();

    if (inFile.fail()) {
        delete[] heapData;
        heapData = nullptr;
        std::cout << "Read file failed." << std::endl;
        return false;
    }

    data = heapData;
    return true;
}

int32_t BuildCompressFcGraph(Graph &graph, uint8_t *data, vector<int64_t> &shape, vector<int64_t> &compressParameters)
{
    /*
      weight
        |
    CompressFc
    /        \
  compress  compress_index
  */
    auto shape_weight = shape;
    int64_t compress_size = 1;
    GetDataSizeFromShape(shape_weight, compress_size);

    TensorDesc desc_weight(ge::Shape(shape_weight), FORMAT_FRACTAL_NZ, DT_INT8);
    Tensor weight_tensor(desc_weight);
    weight_tensor.SetData(data, compress_size);
    auto weight = op::Const("weight_compress").set_attr_value(weight_tensor);

    auto dimK = shape_weight[0];
    auto dimN = shape_weight[1];
    auto tileNumN = (dimN + 8 - 1) / 8;
    auto tileNumK = (dimK + 8 - 1) / 8;
    int64_t index_size = 8 * tileNumN * tileNumK;
    std::cout << "compress_size: " << compress_size << " index_size: " << index_size << std::endl;
    auto compressFc =
        op::CompressFcOp("compress_fc").set_input_weight(weight).set_attr_compress_parameters(compressParameters);
    compressFc.SetAttr("alg", "weight_unzip");

    TensorDesc desc_weight_compress(ge::Shape({compressParameters[0]}), FORMAT_ND, DT_INT8);

    auto shape_index = vector<int64_t>({8 * tileNumN * tileNumK});
    std::cout << "tileNumN: " << tileNumN << " tileNumK: " << tileNumK << std::endl;
    TensorDesc desc_index(ge::Shape(shape_index), FORMAT_ND, DT_UINT8);
    TensorDesc desc_tiling_info(ge::Shape({3}), FORMAT_ND, DT_UINT32);

    compressFc.update_input_desc_weight(desc_weight);
    compressFc.update_output_desc_weight_compress(desc_weight_compress);
    compressFc.update_output_desc_compress_index(desc_index);
    compressFc.update_output_desc_compress_info(desc_tiling_info);

    vector<Operator> inputs = {weight};
    vector<Operator> outputs = {compressFc};

    graph.SetInputs(inputs).SetOutputs(outputs);

    return GraphUtils::SUCCESS;
}
}
