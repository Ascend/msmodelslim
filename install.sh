#!/usr/bin/env bash
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

# 清理构建缓存，确保重新构建包
echo "cleaning build cache to ensure fresh build"
rm -rf build/ dist/ *.egg-info/

# 先卸载再安装，确保完全清理已移除的文件
echo "uninstalling existing msmodelslim package"
pip uninstall msmodelslim -y

# 重新安装包（不使用缓存）
echo "installing msmodelslim package without cache"
umask 027 && pip install . --no-cache-dir
