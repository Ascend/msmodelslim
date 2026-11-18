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

#ifndef MS_SAFE_CHECK_BASE_FILE_H
#define MS_SAFE_CHECK_BASE_FILE_H

#include <string>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

constexpr int DIR_CHECK_MODE = R_OK | W_OK | X_OK;
constexpr const char PATH_SEPARATOR = '/';
constexpr const char* FILE_VALID_PATTERN = "^[a-zA-Z0-9_./-]+$";

constexpr const uint32_t FULL_PATH_LENGTH_MAX = 4096;
constexpr const uint32_t FILE_NAME_LENGTH_MAX = 255;
constexpr const uint32_t PATH_DEPTH_MAX = 32;

constexpr size_t MAX_NUMPY_SIZE = 10ULL * 1024 * 1024 * 1024;
constexpr size_t MAX_JSON_SIZE = 1024ULL * 1024 * 1024;
constexpr size_t MAX_CSV_SIZE = 1024ULL * 1024 * 1024;
constexpr size_t MAX_OM_SIZE = 64ULL * 1024 * 1024 * 1024;
constexpr size_t MAX_FILE_SIZE_DEFAULT = 10ULL * 1024 * 1024 * 1024;

constexpr mode_t NORMAL_FILE_MODE_DEFAULT = 0640;
constexpr mode_t READONLY_FILE_MODE_DEFAULT = 0440;
constexpr mode_t SCRIPT_FILE_MODE_DEFAULT = 0550;
constexpr mode_t NORMAL_DIR_MODE_DEFAULT = 0750;
constexpr mode_t MAX_PERMISSION = 0777;
constexpr mode_t READ_FILE_NOT_PERMITTED = S_IWGRP | S_IWOTH;
constexpr mode_t WRITE_FILE_NOT_PERMITTED = S_IWGRP | S_IWOTH | S_IROTH | S_IXOTH;
constexpr mode_t CREATE_FILE_MODE_DEFAULT = O_EXCL | O_CREAT;

// File 类主要处理文件相关操作
class File {
public:
    File() = default;
    virtual ~File() = default;
    // 文件夹校验：包括路径长度，文件存在性，软链接，属组，权限
    static bool CheckDir(const std::string &path);
    // 文件权限校验
    static bool IsFileWritable(const std::string& path);
    // 文件存在性校验
    static bool IsPathExist(const std::string& path);
    static bool IsOtherWritable(const std::string& path);
    // 软链接校验
    static bool IsSoftLink(const std::string &path);
    // 校验是否是文件夹
    static bool IsDir(const std::string& path);
    // 校验文件属组
    static bool CheckOwner(const std::string &path);
    // 获取文件父目录
    static std::string GetParentDir(const std::string& path);
    // 获取绝对路径
    static std::string GetFullPath(const std::string &originPath);
    static std::string GetAbsPath(const std::string &originPath);
    // 路径长度校验
    static bool IsPathLengthLegal(const std::string& path);
    // 路径字符校验
    static bool IsPathCharactersValid(const std::string& path);
    // 路径深度校验
    static bool IsPathDepthValid(const std::string& path);
    // 常规文件校验
    static bool IsRegularFile(const std::string& path);
    // 获取文件权限
    static mode_t GetFilePermissions(const std::string& path);
    // 写文件前的校验
    static bool CheckFileBeforeCreateOrWrite(const std::string &path, bool overwrite = false);
};

#endif // MS_SAFE_CHECK_BASE_FILE_H