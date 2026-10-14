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

# 清理output路径
rm -rf $PROJECT_PATH/output/multi_modal_quant_anti_flux/m6/*

# ----------------配置环境变量----------------
source $CANN_PATH
MSMODELSLIM_SOURCE_DIR=${MSMODELSLIM_SOURCE_DIR:-"$PROJECT_PATH/resource/msit/msmodelslim"}
FLUX_REPO_DIR=${FLUX_REPO_DIR:-"$PROJECT_PATH/resource/multi_modal/FLUX.1-dev-code"}
MODEL_PATH=$PROJECT_PATH/resource/multi_modal/FLUX.1-dev
# ----------------配置环境变量----------------

export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=2
# add msmodelslim package path to python path
export PYTHONPATH=${MSMODELSLIM_SOURCE_DIR}:$PYTHONPATH

cd $FLUX_REPO_DIR
# 单卡32G Flux 等价优化推理
python inference_flux.py \
    --path ${MODEL_PATH} \
    --save_path "$PROJECT_PATH/output/multi_modal_quant_anti_flux/m6/results/quant/img" \
    --device_id 0 \
    --device "npu" \
    --prompt_path "$MSMODELSLIM_SOURCE_DIR/example/multimodal_sd/Flux/calib_prompts.txt" \
    --width 1024 \
    --height 1024 \
    --infer_steps 50 \
    --seed 42 \
    --use_cache \
    --device_type "A2-32g-single" \
    --batch_size 1 \
    --max_num_prompt 0 \
    --do_quant \
    --quant_weight_save_folder "$PROJECT_PATH/output/multi_modal_quant_anti_flux/m6/results/quant/safetensors" \
    --quant_dump_calib_folder "$PROJECT_PATH/resource/multi_modal_quant_anti_flux/results/quant/cache" \
    --quant_type "w8a8_dynamic" \
    --anti_method "m6"
