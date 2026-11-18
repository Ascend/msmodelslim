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
#include <map>
#include <iostream>
#include <cstring>
#include <regex>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <unordered_map>
#include "Log.h"
#include "File.h"


std::string File::GetFullPath(const std::string &originPath)
{
    if (originPath.empty()) {
        return "";
    }
    if (originPath[0] == PATH_SEPARATOR) {
        return originPath;
    }

    char* cwd = nullptr;
    char* cwdBuf = nullptr;
    try {
        cwdBuf = new char[PATH_MAX];
    } catch (const std::bad_alloc& e) {
        ERROR_LOG("create buffer failed: %s", e.what());
        throw std::runtime_error("No memory.");
    }
    cwd = getcwd(cwdBuf, PATH_MAX);
    if (cwd == nullptr) {
        delete[] cwdBuf;
        return "";
    }

    std::string fullPath = std::move(std::string(cwd) + PATH_SEPARATOR + originPath);
    delete[] cwdBuf;
    cwdBuf = nullptr;

    return fullPath;
}

static std::vector<std::string> SplitPath(const std::string &path)
{
    std::vector<std::string> tokens;
    size_t len = path.length();
    size_t start = 0;

    while (start < len) {
        size_t end = path.find(PATH_SEPARATOR, start);
        if (end == std::string::npos) {
            end = len;
        }
        if (start != end) {
            tokens.push_back(path.substr(start, end - start));
        }
        start = end + 1;
    }
    return tokens;
}

std::string File::GetAbsPath(const std::string &originPath)
{
    std::string fullPath = GetFullPath(originPath);
    if (fullPath.empty()) {
        return "";
    }

    std::vector<std::string> tokens = SplitPath(fullPath);
    std::vector<std::string> tokensRefined;

    for (std::string& token : tokens) {
        if (token.empty() || token == ".") {
            continue;
        } else if (token == "..") {
            if (tokensRefined.empty()) {
                return "";
            }
            tokensRefined.pop_back();
        } else {
            tokensRefined.emplace_back(token);
        }
    }

    if (tokensRefined.empty()) {
        return "/";
    }
    std::string resolvedPath("");
    for (std::string& token : tokensRefined) {
        resolvedPath.append("/").append(token);
    }
    return resolvedPath;
}

bool File::IsFileWritable(const std::string& path)
{
    return access(path.c_str(), W_OK) == 0;
}

bool File::IsOtherWritable(const std::string& path)
{
    return ((GetFilePermissions(path) & READ_FILE_NOT_PERMITTED) > 0);
}

bool File::IsPathExist(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

bool File::IsPathLengthLegal(const std::string& path)
{
    if (path.length() > FULL_PATH_LENGTH_MAX || path.length() == 0) {
        return false;
    }
    std::vector<std::string> tokens = SplitPath(path);
    for (std::string& token : tokens) {
        if (token.length() > FILE_NAME_LENGTH_MAX) {
            return false;
        }
    }
    return true;
}

bool File::IsPathCharactersValid(const std::string& path)
{
    return std::regex_match(path, std::regex(FILE_VALID_PATTERN));
}

bool File::IsPathDepthValid(const std::string& path)
{
    return std::count(path.begin(), path.end(), PATH_SEPARATOR) <= PATH_DEPTH_MAX;
}

bool File::IsRegularFile(const std::string& path)
{
    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) == 0) {
        return S_ISREG(path_stat.st_mode);
    }
    return false;
}

bool File::IsDir(const std::string& path)
{
    struct stat buffer;
    if (stat(path.c_str(), &buffer) == 0) {
        return (buffer.st_mode & S_IFDIR) != 0;
    }
    return false;
}

std::string File::GetParentDir(const std::string& path)
{
    size_t found = path.find_last_of('/');
    if (found != std::string::npos) {
        return path.substr(0, found);
    }
    return ".";
}

mode_t File::GetFilePermissions(const std::string& path)
{
    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) != 0) {
        ERROR_LOG("file not exists");
        return MAX_PERMISSION;
    }
    mode_t permissions = path_stat.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
    return permissions;
}

bool File::IsSoftLink(const std::string &path)
{
    std::string absPath = GetAbsPath(path);
    struct stat fileStat;
    if (lstat(absPath.c_str(), &fileStat) != 0) {
        ERROR_LOG("the file lstat failed");
        return false;
    }
    return S_ISLNK(fileStat.st_mode);
}

/****************************** 通用检查函数 ********************************/
bool File::CheckOwner(const std::string &path)
{
    std::string absPath = GetAbsPath(path);
    struct stat buf;
    if (bool(stat(absPath.c_str(), &buf))) {
        ERROR_LOG("get file stat failed");
        return false;
    }
    if (buf.st_uid != getuid()) {
        ERROR_LOG("file owner is not process usr");
        return false;
    }
    return true;
}

bool File::CheckDir(const std::string &path)
{
    std::string absPath = GetAbsPath(path);
    if (absPath.empty()) {
        ERROR_LOG("path is empty");
        return false;
    }
    if (!IsPathLengthLegal(absPath)) {
        ERROR_LOG("path length illegal");
        return false;
    }
    if (!IsPathCharactersValid(absPath)) {
        ERROR_LOG("path characters invalid");
        return false;
    }
    if (!IsPathDepthValid(absPath)) {
        ERROR_LOG("path depth invalid");
        return false;
    }
    if (!IsPathExist(absPath)) {
        ERROR_LOG("path not exist");
        return false;
    }
    if (IsSoftLink(absPath)) {
        ERROR_LOG("path is soft link");
        return false;
    }
    if (!CheckOwner(absPath)) {
        return false;
    }
    if (!IsDir(absPath)) {
        ERROR_LOG("path is not dir");
        return false;
    }
    if (IsOtherWritable(absPath)) {
        ERROR_LOG("dir permission should not be over 0o755(rwxr-xr-x)");
        return false;
    }
    return true;
}

bool File::CheckFileBeforeCreateOrWrite(const std::string &path, bool overwrite)
{
    std::string absPath = GetAbsPath(path);
    if (absPath.empty()) {
        ERROR_LOG("path is empty");
        return false;
    }
    if (!IsPathLengthLegal(absPath)) {
        ERROR_LOG("path length illegal");
        return false;
    }
    if (!IsPathCharactersValid(absPath)) {
        ERROR_LOG("path characters invalid");
        return false;
    }
    if (!IsPathDepthValid(absPath)) {
        ERROR_LOG("path depth invalid");
        return false;
    }
    if (IsPathExist(absPath)) {
        if (!overwrite) {
            ERROR_LOG("path already exist and not allow to overwrite");
            return false;
        }
        if (!IsRegularFile(absPath)) {
            ERROR_LOG("path is not regular file");
            return false;
        }
        if (IsSoftLink(absPath)) {
            ERROR_LOG("path is soft link");
            return false;
        }
        if ((GetFilePermissions(absPath) & WRITE_FILE_NOT_PERMITTED) > 0) {
            ERROR_LOG("path permission should not be over 0o750(rwxr-x---)");
            return false;
        }
        /* 默认不允许覆盖其他用户创建的文件，若有特殊需求（如多用户通信管道等）由业务自行校验 */
        if (!IsFileWritable(absPath) || !CheckOwner(absPath)) {
            ERROR_LOG("path already create by other owner");
            return false;
        }
    }
    return CheckDir(GetParentDir(absPath));
}
