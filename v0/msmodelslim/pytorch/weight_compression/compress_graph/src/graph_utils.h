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

#ifndef DAVINCI_GRAPH_UTILS_H
#define DAVINCI_GRAPH_UTILS_H
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "all_ops.h"

using namespace ge;

namespace GraphUtils {
constexpr int FAILED = -1;
constexpr int SUCCESS = 0;

int CheckShape(std::vector<int64_t> &shape);

void GetDataSizeFromShape(std::vector<int64_t> shape, int64_t &size);

bool GetDataFromBin(std::string input_path, std::vector<int64_t> shapes, uint8_t *&data, int data_type_size);

int32_t BuildCompressFcGraph(Graph &graph, uint8_t* data, std::vector<int64_t> &shape, std::vector<int64_t> &compressParameters);
}
#endif // DAVINCI_GRAPH_UTILS_H
