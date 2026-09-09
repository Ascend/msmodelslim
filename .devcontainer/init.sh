#!/usr/bin/env bash
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
# postCreateCommand 启动入口：按脚本自身路径定位仓库根（不依赖 cwd/挂载路径），
# 剥离 post-create.sh 行尾 CR 后执行，规避 CRLF 检出与挂载路径差异导致的路径无法找到。
# -------------------------------------------------------------------------
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)
proj=$(dirname "${SCRIPT_DIR}")

if [ ! -f "${SCRIPT_DIR}/post-create.sh" ]; then
    echo "[init] ERROR: ${SCRIPT_DIR}/post-create.sh not found." >&2
    ls -la "${SCRIPT_DIR}" 2>/dev/null | head -30 >&2
    exit 127
fi

echo "[init] project root: ${proj}"
export MSMODELSLIM_PROJ="${proj}"
sed 's/\r$//' "${SCRIPT_DIR}/post-create.sh" | bash
