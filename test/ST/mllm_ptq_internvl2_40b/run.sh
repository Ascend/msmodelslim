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

pip install timm
pip install fastchat
pip install transformers==4.46.0

rm -rf $PROJECT_PATH/output/mllm_ptq_internvl2_40b
MSMODELSIM_SOURCE_DIR=${MSMODELSIM_SOURCE_DIR:-"$PROJECT_PATH/resource/msit/msmodelslim"}
export PYTHONPATH=${MSMODELSIM_SOURCE_DIR}:$PYTHONPATH
python run.py

if [ $? -eq 0 ]
then
    echo mllm_ptq_internvl2_40b: Success
else
    echo mllm_ptq_internvl2_40b: Failed
    run_ok=$ret_failed
fi

conda activate smoke_test_modelslim_0104

# 清理output
rm -rf $PROJECT_PATH/output/mllm_ptq_internvl2_40b

exit $run_ok
