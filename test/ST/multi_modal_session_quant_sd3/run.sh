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

pip install -U accelerate
pip install -U diffusers

MSMODELSLIM_SOOURCE_DIR=${MSMODELSLIM_SOOURCE_DIR:-"$PROJECT_PATH/resource/msit/msmodelslim"}
export PYTHONPATH=${MSMODELSLIM_SOOURCE_DIR}:$PYTHONPATH

rm -rf $PROJECT_PATH/output/multi_modal_session_quant_sd3
python run.py \
    --sd3_model_path "$PROJECT_PATH/resource/multi_modal/sd3_project/stable-diffusion-3-medium-diffusers" \
    --prompt_path "$PROJECT_PATH/resource/multi_modal/sd3_project/calib_prompts.txt" \
    --width 1024 \
    --height 1024 \
    --infer_steps 1 \
    --seed 42 \
    --device "npu" \
    --save_path "$PROJECT_PATH/output/multi_modal_session_quant_sd3/images" \
    --do_quant \
    --quant_weight_save_folder "$PROJECT_PATH/output/multi_modal_session_quant_sd3/safetensors" \
    --quant_dump_calib_folder "$PROJECT_PATH/output/multi_modal_session_quant_sd3/cache" \
    --quant_type "w8a8"

if [ $? -eq 0 ]
then
    echo multi_modal_session_quant_sd3: Success
else
    echo multi_modal_session_quant_sd3: Failed
    run_ok=$ret_failed
fi

conda activate smoke_test_modelslim_0104

# 清理output
rm -rf $PROJECT_PATH/output/multi_modal_session_quant_sd3

exit $run_ok
