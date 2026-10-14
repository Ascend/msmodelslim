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

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

# 用例名（用于打屏和指定输出路径）
CASE_NAME=modelslim_v1_qwen3_moe_w4a8_per_channel

# 无论在哪个目录执行脚本，SCRIPT_DIR 始终指向当前 .sh 文件所在的绝对目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
OUTPUT_DIR="${SCRIPT_DIR}/output_${CASE_NAME}"
# 直接拼接「脚本目录 + 配置文件名」，确保无论在哪执行都能找到配置文件
CONFIG_FILE="${SCRIPT_DIR}/moe-w4a8-v1.yaml"

# 引入公共模块
source ${SCRIPT_DIR}/../../utils/common_utils.sh

# 安装依赖
pip install -r ${SCRIPT_DIR}/requirements.txt

# 执行量化任务
msmodelslim quant \
  --model_path ${MODEL_RESOURCE_PATH}/Qwen3-30B-A3B \
  --save_path ${OUTPUT_DIR} \
  --device npu \
  --model_type Qwen3-30B \
  --config_path ${CONFIG_FILE} \
  --trust_remote_code True

# 配置待检查文件列表
FILES=(
  "config.json"
  "generation_config.json"
  "quant_model_description.json"
  "tokenizer_config.json"
  "tokenizer.json"
  # 可在此处添加更多文件，每行一个
)

if check_files_exist "$OUTPUT_DIR" "${FILES[@]}"; then
  echo "$CASE_NAME: Success"
else
  echo "$CASE_NAME: Failed"
  run_ok=$ret_failed
fi

exit $run_ok
