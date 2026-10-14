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

rm -rf $PROJECT_PATH/output/ptq-tools/*
ASCEND_RT_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node 1 --master_port 29999 \
-m run  \
--model_path $PROJECT_PATH/resource/multi_modal/opensoraplan_project/Open-Sora-Plan-v1.2.0/93x720p/ \
--num_frames 32 \
--height 720 \
--width 1280 \
--cache_dir $PROJECT_PATH/output/ptq-tools/cache_dir \
--text_encoder_name $PROJECT_PATH/resource/multi_modal/opensoraplan_project/mt5-xxl/ \
--text_prompt $PROJECT_PATH/resource/multi_modal/opensoraplan_project/prompt_list_0.txt \
--ae CausalVAEModel_D4_4x8x8 \
--ae_path $PROJECT_PATH/resource/multi_modal/opensoraplan_project/Open-Sora-Plan-v1.2.0/vae/ \
--save_img_path $PROJECT_PATH/output/ptq-tools/sample_videos/ \
--fps 8 \
--guidance_scale 7.5 \
--num_sampling_steps 100 \
--enable_tiling \
--tile_overlap_factor 0.125 \
--save_memory \
--max_sequence_length 512 \
--sample_method EulerAncestralDiscrete \
--model_type "dit"

if [ $? -eq 0 ]
then
    echo quant_opensora_plan_1_2: Success
else
    echo quant_opensora_plan_1_2: Failed
    run_ok=$ret_failed
fi

rm -rf $PROJECT_PATH/output/ptq-tools/*

exit $run_ok
