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

# 公共工具函数库

# 获取 msmodelslim 路径函数
# 参数1: 自定义路径（可选，如果提供则使用，否则自动获取）
get_msmodelslim_path() {
  local custom_path=$1

  if [ -n "$custom_path" ] && [ -d "$custom_path" ]; then
    echo "$custom_path"
    return 0
  fi

  # 默认获取方式
  local path=$(python -c "import msmodelslim; print(msmodelslim.__path__[0])" 2>/dev/null) # "
  if [ $? -eq 0 ] && [ -n "$path" ]; then
    echo "$path"
  else
    echo "错误：无法获取 msmodelslim 路径" >&2
    return 1
  fi
}

# 校验文件存在性函数
# 参数1: 目标路径
# 参数2-N: 文件列表
check_files_exist() {
  local target_path=$1
  shift
  local files=("$@")

  # 检查路径是否存在
  if [ ! -d "$target_path" ]; then
    echo "错误：目标路径 '$target_path' 不存在" >&2
    return 1
  fi

  # 初始化结果标志
  local all_exist=true

  # 遍历检查每个文件
  for file in "${files[@]}"; do
    full_path="$target_path/$file"
    if [ ! -e "$full_path" ]; then
      echo "错误：文件 '$file' 不存在" >&2
      all_exist=false
    fi
  done

  # 检查 quant_model_weights.safetensors 或其分片文件
  local quant_model_single="$target_path/quant_model_weights.safetensors"
  local quant_model_pattern="$target_path/quant_model_weights-*.safetensors"

  # 使用 nullglob 避免没有匹配文件时返回模式本身
  shopt -s nullglob
  local quant_model_files=($quant_model_pattern)
  shopt -u nullglob

  if [ ! -e "$quant_model_single" ] && [ ${#quant_model_files[@]} -eq 0 ]; then
    echo "错误：quant_model_weights.safetensors 或其分片文件不存在" >&2
    all_exist=false
  else
    if [ ! -e "$quant_model_single" ]; then
      for quant_file in "${quant_model_files[@]}"; do
        if [ ! -e "$quant_file" ]; then
          echo "错误：文件 '$quant_file' 不存在" >&2
          all_exist=false
        fi
      done
    fi
  fi

  # 返回结果
  if [ "$all_exist" = true ]; then
    echo "提示：所有文件均存在于目标路径"
    return 0
  else
    return 1
  fi
}
