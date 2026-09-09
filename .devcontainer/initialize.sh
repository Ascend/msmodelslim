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
# 宿主侧、容器创建前执行（devcontainer.json initializeCommand）：
#   生成 .host-gitconfig 身份快照；预建 pip/pre-commit 缓存目录作为 bind source
#   （bind source 必须先于容器存在）。幂等可重复执行。
# -------------------------------------------------------------------------
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
HOST_GITCONFIG="${SCRIPT_DIR}/.host-gitconfig"

umask 077
: > "${HOST_GITCONFIG}"
chmod 600 "${HOST_GITCONFIG}"

name=$(git config --get user.name 2>/dev/null || true)
email=$(git config --get user.email 2>/dev/null || true)
[ -n "${name}" ]  && git config --file "${HOST_GITCONFIG}" user.name  "${name}"
[ -n "${email}" ] && git config --file "${HOST_GITCONFIG}" user.email "${email}"

mkdir -p "${HOME}/.cache/pip" "${HOME}/.cache/pre-commit"

echo "[initialize] .host-gitconfig = ${HOST_GITCONFIG}"
echo "[initialize] cache bind sources ready: ${HOME}/.cache/{pip,pre-commit}"
