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

#ifndef CORE_LOG_H
#define CORE_LOG_H

#include <string>
#include <map>
#include <memory>
#include <ostream>
#include <iostream>

#define FILELINE __FILE__, __FUNCTION__, __LINE__

constexpr int LOG_DEBUG_LEVEL = 1;
constexpr int LOG_INFO_LEVEL = 2;
constexpr int LOG_WARNING_LEVEL = 3;
constexpr int LOG_ERROR_LEVEL = 4;

namespace Base {
class LogManager {
public:
    static LogManager& GetInstance() {
        static LogManager instance;
        return instance;
    }

    void SetLogLevel(int level) {
        logLevel_ = level;
    }

    int GetLogLevel() const {
        return logLevel_;
    }

private:
    LogManager() : logLevel_(LOG_INFO_LEVEL) {}
    LogManager(const LogManager&) = delete;
    LogManager& operator=(const LogManager&) = delete;

    int logLevel_;
};

void SETLOGLEVEL(int level);
}


#define DEBUG_LOG(fmt, args...)  do { if (Base::LogManager::GetInstance().GetLogLevel() <= LOG_DEBUG_LEVEL) \
    { printf("[DEBUG] " fmt "\n", ##args); fflush(stdout); } } while (0)
#define INFO_LOG(fmt, args...)  do { if (Base::LogManager::GetInstance().GetLogLevel() <= LOG_INFO_LEVEL) \
    { printf("[INFO] " fmt "\n", ##args); fflush(stdout); } } while (0)
#define WARN_LOG(fmt, args...)  do { if (Base::LogManager::GetInstance().GetLogLevel() <= LOG_WARNING_LEVEL) \
    { printf("[WARN] " fmt "\n", ##args); fflush(stdout); } } while (0)
#define ERROR_LOG(fmt, args...)  do { if (Base::LogManager::GetInstance().GetLogLevel() <= LOG_ERROR_LEVEL) \
    { printf("[ERROR] " fmt "\n", ##args); fflush(stdout); } } while (0)
#define ACLERR_LOG(ErrMsg) printf("[ACL ERROR] %s\n", ErrMsg)
#define PROMPT_MSG(fmt, args...) printf(fmt, ##args)

#endif  // CORE_LOG_H