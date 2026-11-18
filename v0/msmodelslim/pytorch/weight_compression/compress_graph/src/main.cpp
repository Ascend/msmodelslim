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
#include "ge_ir_build.h"
#include "File.h"

constexpr int NUMBER_12 = 12;

/**
 * @brief Write data to file with security check
 * @param filePath The path of the file to write
 * @param data The data to write
 * @param count The number of elements to write
 * @return GraphUtils::SUCCESS if write successfully, GraphUtils::FAILED otherwise
 */
template <typename T>
int WriteDataToFile(const char *filePath, const T *data, size_t count)
{
    // Check if input file path is valid
    if (filePath == nullptr) {
        std::cout << "Error: file path is null." << std::endl;
        return GraphUtils::FAILED;
    }

    // Check if input data is valid
    if (data == nullptr) {
        std::cout << "Error: input data is null." << std::endl;
        return GraphUtils::FAILED;
    }

    // Check if count is valid
    if (count == 0) {
        std::cout << "Error: count cannot be 0." << std::endl;
        return GraphUtils::FAILED;
    }

    if (!File::CheckFileBeforeCreateOrWrite(filePath, true)) {
        return GraphUtils::FAILED;
    }

    FILE *fp = fopen(filePath, "w+");
    if (fp == nullptr) {
        std::cout << "Error: open file failed." << std::endl;
        return GraphUtils::FAILED;
    }

    bool success = true;
    if (fwrite(data, sizeof(T), count, fp) != count) {
        std::cout << "Error: write file failed." << std::endl;
        success = false;
    }

    if (fclose(fp) != 0) {
        std::cout << "Error: close file failed." << std::endl;
        success = false;
    }

    return success ? GraphUtils::SUCCESS : GraphUtils::FAILED;
}

static int RunCompressGraph(ge::Session *session, uint8_t *data, vector<int64_t> &shape, vector<int64_t> &compressParameters,
    vector<string> paths)
{
    uint32_t compressFc_graph_id = 1;
    Graph compressFcGraph("compressFc Graph");
    // Build graph
    int ret = GraphUtils::BuildCompressFcGraph(compressFcGraph, data, shape, compressParameters);
    if (ret != GraphUtils::SUCCESS) {
        std::cout << "Generate compressFc Graph failed." << std::endl;
        return GraphUtils::FAILED;
    }

    // Add graph
    ret = session->AddGraph(compressFc_graph_id, compressFcGraph);
    if (ret != GraphUtils::SUCCESS) {
        std::cout << "Session add compressFc Graph failed." << std::endl;
        return GraphUtils::FAILED;
    }
    std::cout << "Session add compressFc Graph success." << std::endl;

    std::vector<Tensor> input_mm;
    std::vector<Tensor> output_mm;
    ret = session->RunGraph(compressFc_graph_id, input_mm, output_mm);
    if (ret != GraphUtils::SUCCESS) {
        std::cout << "Session run compressFc graph failed." << std::endl;
        return GraphUtils::FAILED;
    }
    if (output_mm.empty()) {
        std::cout << "Error: output_mm is empty!" << std::endl;
        return GraphUtils::FAILED;
    }
    if (output_mm.size() <= 2) {
        std::cout << "Error: output_mm size is too small (expected >= 3, got " << output_mm.size() << ")!" << std::endl;
        return GraphUtils::FAILED;
    }
    auto infoData = reinterpret_cast<uint32_t *>(output_mm[2].GetData());

    constexpr uint8_t OUTPUT_WEIGHT_PATH_INDEX = 0;
    constexpr uint8_t INDEX_PATH_INDEX = 1;
    constexpr uint8_t COMPRESS_INFO_PATH_INDEX = 2;

    // Write output weight to file
    int outputWeightSize = infoData[2];
    ret = WriteDataToFile<int8_t>(paths[OUTPUT_WEIGHT_PATH_INDEX].c_str(),
                                  reinterpret_cast<int8_t *>(output_mm[OUTPUT_WEIGHT_PATH_INDEX].GetData()),
                                  outputWeightSize);
    if (ret != GraphUtils::SUCCESS) {
        return GraphUtils::FAILED;
    }

    // Write index to file
    ret = WriteDataToFile<uint8_t>(paths[INDEX_PATH_INDEX].c_str(),
                                   reinterpret_cast<uint8_t *>(output_mm[INDEX_PATH_INDEX].GetData()),
                                   output_mm[1].GetSize());
    if (ret != GraphUtils::SUCCESS) {
        return GraphUtils::FAILED;
    }

    // Write compress info to file
    int compressInfoSize = 3;
    ret = WriteDataToFile<uint32_t>(paths[COMPRESS_INFO_PATH_INDEX].c_str(),
                                    reinterpret_cast<uint32_t *>(output_mm[COMPRESS_INFO_PATH_INDEX].GetData()),
                                    compressInfoSize);
    if (ret != GraphUtils::SUCCESS) {
        return GraphUtils::FAILED;
    }

    return GraphUtils::SUCCESS;
}

static int RunSession(uint8_t *data, vector<int64_t> &shape, vector<string> paths, vector<int64_t> &compressParameters)
{
    // System init
    std::map<AscendString, AscendString> config = {
        {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "0"}, {"ge.exec.precision_mode", "allow_fp32_to_fp16"}};

    Status ret = ge::GEInitialize(config);
    if (ret != GraphUtils::SUCCESS) {
        std::cout << "GE initialize failed.\n";
        return GraphUtils::FAILED;
    }
    std::cout << "Initialize ge success." << std::endl;

    // creat session
    std::map<ge::AscendString, ge::AscendString> options;
    ge::Session *session = new (std::nothrow) Session(options);
    if (session == nullptr) {
        std::cout << "Create session failed." << std::endl;
        return GraphUtils::FAILED;
    }
    std::cout << "Create session success." << std::endl;

    RunCompressGraph(session, data, shape, compressParameters, paths);

    // destroy session
    delete session;
    session = nullptr;

    // system finalize
    ret = ge::GEFinalize();
    if (ret != GraphUtils::SUCCESS) {
        std::cout << "Finalize ge failed." << std::endl;
        return GraphUtils::FAILED;
    }
    std::cout << "Finalize ge success." << std::endl;
    return GraphUtils::SUCCESS;
}

static bool CheckInputsStollValid(int argc, char *argv[])
{
    if (argc != NUMBER_12) {
        std::cout << "Please check your input params count is 11." << std::endl;
        return false;
    }

    try {
        const int inputStollCount = 7;
        for (int i = 1; i <= inputStollCount; i++) {
            try {
                std::stoll(argv[i]);
            } catch (const std::invalid_argument &e) {
                std::cout << "Invalid argument for param " << i << ". Please check your input params." << std::endl;
                return false;
            } catch (const std::out_of_range &e) {
                std::cout << "Out of range for param " << i << ". Please check your input params." << std::endl;
                return false;
            }
        }
    } catch (...) {
        std::cout << "An unknown error occurred." << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char *argv[])
{
    if (!CheckInputsStollValid(argc, argv)) {
        return GraphUtils::FAILED;
    }

    const int64_t dimK = std::stoll(argv[1]);
    const int64_t dimN = std::stoll(argv[2]);
    const int64_t isTight = std::stoll(argv[3]);
    const int64_t k_value = std::stoll(argv[4]);
    const int64_t n_value = std::stoll(argv[5]);
    const int64_t compressType = std::stoll(argv[6]);
    const int64_t isTiling = std::stoll(argv[7]);
    const string inputWeightPath = argv[8];
    const string outputWeightPath = argv[9];
    const string indexPath = argv[10];
    const string compressInfoPath = argv[11];

    vector<string> paths = {outputWeightPath, indexPath, compressInfoPath};
    vector<int64_t> inputWeightShape = {dimK, dimN, 16, 32};
    if (GraphUtils::CheckShape(inputWeightShape) == GraphUtils::FAILED) {
        std::cout << "Invalid shape. Please check your input dimK(" << dimK << ") dimN(" << dimN << ")." << std::endl;
    }
    uint8_t *data = nullptr;
    int64_t compress_size = 1;
    GraphUtils::GetDataSizeFromShape(inputWeightShape, compress_size);
    int64_t tileNumN = (inputWeightShape[1] + 8 - 1) / 8;
    int64_t tileNumK = (inputWeightShape[0] + 8 - 1) / 8;
    int64_t indexBaseSize = 2;
    if (isTight != 0) {
        indexBaseSize = 8;
    }
    int64_t index_size = indexBaseSize * tileNumK * tileNumN;
    /*
   (0)compressed_size,
   (1)index_size,
   (2)isTight: 1
   (3)enginnum: 4
   (4)channel: 2
   (5)ratios: 64
   (6)k_value: 1
   (7)n_value: 1
   (8)block_dim: 1
   (9)CompressType: 0
   (10)isTiling: 0
  */
    vector<int64_t> compressParameters = {compress_size, index_size, isTight, 4, 2, 64, k_value,
                                          n_value, 1, compressType, isTiling};
    try {
        if (!GraphUtils::GetDataFromBin(inputWeightPath, inputWeightShape, data, sizeof(int8_t))) {
            delete[] data;
            data = nullptr;
            std::cout << "read file failed.\n";
            return GraphUtils::FAILED;
        }

        int ret = RunSession(data, inputWeightShape, paths, compressParameters);
        delete[] data;
        data = nullptr;

        if (ret != GraphUtils::SUCCESS) {
            std::cout << "run session failed.\n";
            return GraphUtils::FAILED;
        }
    } catch (const std::exception &e) {
        delete[] data;
        data = nullptr;
        std::cout << "read file or run session failed.\n";
        return GraphUtils::FAILED;
    }
    return GraphUtils::SUCCESS;
}
