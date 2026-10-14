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

source /opt/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate multimodal_testcase

pip install qwen_vl_utils
pip install transformers==4.49.0

rm -rf $PROJECT_PATH/output/mllm_ptq_qwen2_5vl_m2
python run.py --anti_method m2 --save_directory $PROJECT_PATH/output/mllm_ptq_qwen2_5vl_m2

if [ $? -eq 0 ]
then
    echo mllm_ptq_qwen2_5vl_m2: Success
else
    echo mllm_ptq_qwen2_5vl_m2: Failed
    run_ok=$ret_failed
fi

rm -rf $PROJECT_PATH/output/mllm_ptq_qwen2_5vl_m2

rm -rf $PROJECT_PATH/output/mllm_ptq_qwen2_5vl_m4
python run.py --anti_method m4 --save_directory $PROJECT_PATH/output/mllm_ptq_qwen2_5vl_m4

if [ $? -eq 0 ]
then
    echo mllm_ptq_qwen2_5vl_m4: Success
else
    echo mllm_ptq_qwen2_5vl_m4: Failed
    run_ok=$ret_failed
fi

conda activate smoke_test_modelslim_0104

# 清理output
rm -rf $PROJECT_PATH/output/mllm_ptq_qwen2_5vl_m4

exit $run_ok
