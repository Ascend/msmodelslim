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

export ASCEND_LAUNCH_BLOCKING=1
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:256"

# 获取脚本路径
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# add msmodelslim package path to python path
MSMODELSLIM_SOURCE_DIR=${MSMODELSLIM_SOURCE_DIR:-"$PROJECT_PATH/resource/msit/msmodelslim"}
export PYTHONPATH=${MSMODELSLIM_SOURCE_DIR}:$PYTHONPATH

# 设置参数
# 输入路径
MODEL_PATH=$PROJECT_PATH/resource/multi_modal/opensoraplan_project
TEXT_PROMPT=$PROJECT_PATH/resource/ditcache-t2v-prompt/prompt_list_5.txt
FLOAT_CALIB_VIDEO_DIR=$PROJECT_PATH/resource/multi_modal/opensoraplan_project/sample-optimize-calibration-videos-29x480p
CALIB_VIDEO_PATH=$PROJECT_PATH/output/multi_modal_sample_optimization_osp1.2/calib_vids

# 输出路径
TIMESTEP_SAVE_DIR=$PROJECT_PATH/output/multi_modal_sample_optimization_osp1.2/results
SAVE_IMG_PATH=$PROJECT_PATH/output/multi_modal_sample_optimization_osp1.2/results/generated_vids_using_new_timestep

NUM_STEPS=10

# 设置环境变量
if [ -n "${DEVICES}" ]; then
    echo "检测到 DEVICES 环境变量已设置"
else
    # 继续检查 ASCEND_RT_VISIBLE_DEVICES
    echo "检测到 DEVICES 环境变量未设置，继续检查 ASCEND_RT_VISIBLE_DEVICES "
    if [ -n "${ASCEND_RT_VISIBLE_DEVICES}" ]; then
        # 检测到 ASCEND_RT_VISIBLE_DEVICES 环境变量
        echo "检测到 ASCEND_RT_VISIBLE_DEVICES 环境变量已设置"
        DEVICES=${ASCEND_RT_VISIBLE_DEVICES}
    else
        # 默认使用前四卡
        echo "默认使用0，1，2，3卡"
        DEVICES="0,1,2,3"
    fi
fi

IFS=',' read -r -a devices <<< "$DEVICES"
device_count=${#devices[@]}


## run sample_optimization timestep search
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
(
torchrun --nnodes=1 --nproc_per_node=$device_count --master_port ${PORT:-29503} \
    -m example.osp1_2.search_t2v_sp \
    --model_path ${MODEL_PATH}/Open-Sora-Plan-v1.2.0/29x480p/ \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir $PROJECT_PATH/output/multi_modal_sample_optimization_osp1.2/cache_dir \
    --text_encoder_name ${MODEL_PATH}/mt5-xxl-original/ \
    --text_prompt ${TEXT_PROMPT} \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path ${MODEL_PATH}/Open-Sora-Plan-v1.2.0/vae/ \
    --save_img_path $PROJECT_PATH/output/multi_modal_sample_optimization_osp1.2/results/sample_t2v_sp_single_prompt \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps ${NUM_STEPS} \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --save_memory \
    --search_type "restep" \
    --videos_path $CALIB_VIDEO_PATH \
    --save_dir ${TIMESTEP_SAVE_DIR}

) || { echo "Search failed with exit status $?"; exit 1; }

## run inference with sample_optimization
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
(
torchrun --nnodes=1 --nproc_per_node=$device_count --master_port 29503 \
    -m example.osp1_2.sample_t2v_sp \
    --model_path ${MODEL_PATH}/Open-Sora-Plan-v1.2.0/29x480p/ \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir $PROJECT_PATH/output/multi_modal_ditcache_osp1.2/cache_dir \
    --text_encoder_name ${MODEL_PATH}/mt5-xxl-original/ \
    --text_prompt ${TEXT_PROMPT} \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path ${MODEL_PATH}/Open-Sora-Plan-v1.2.0/vae/ \
    --save_img_path ${SAVE_IMG_PATH} \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps ${NUM_STEPS} \
    --save_memory \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --schedule_timestep ${TIMESTEP_SAVE_DIR}

) || { echo "Inference failed with exit status $?"; exit 1; }
