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

pip install -r $PROJECT_PATH/resource/multi_modal/opensoraplan_project/open_sora_planv1_2/requirements.txt
pip install huggingface_hub==0.25.2
pip install -U accelerate

MSMODELSLIM_SOOURCE_DIR=${MSMODELSLIM_SOOURCE_DIR:-"$PROJECT_PATH/resource/msit/msmodelslim"}
export PYTHONPATH=${MSMODELSLIM_SOOURCE_DIR}:$PYTHONPATH
export PYTHONPATH=${PROJECT_PATH}/resource/multi_modal/opensoraplan_project/open_sora_planv1_2/:$PYTHONPATH
source $PROJECT_PATH/resource/multi_modal/opensoraplan_project/mindie/set_env.sh

rm -rf $PROJECT_PATH/output/multi_modal_session_quant_osp1_2
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"
export TASK_QUEUE_ENABLE=2
export HCCL_OP_EXPANSION_MODE="AIV"
torchrun --nnodes=1 --nproc_per_node 2  --master_port 29503 \
    run.py \
    --model_path "$PROJECT_PATH/resource/multi_modal/opensoraplan_project/Open-Sora-Plan-v1.2.0/93x720p" \
    --num_frames 93 \
    --height 720 \
    --width 1280 \
    --cache_dir "$PROJECT_PATH/output/multi_modal_session_quant_osp1_2/cache_dir" \
    --text_encoder_name $PROJECT_PATH/resource/multi_modal/opensoraplan_project/mt5-xxl \
    --text_prompt "$PROJECT_PATH/resource/multi_modal/opensoraplan_project/calib_prompts.txt" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "$PROJECT_PATH/resource/multi_modal/opensoraplan_project/Open-Sora-Plan-v1.2.0/vae" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 1 \
    --tile_overlap_factor 0.125 \
    --max_sequence_length 512 \
    --dtype bf16 \
    --use_cfg_parallel \
    --algorithm "dit_cache" \
    --save_img_path "$PROJECT_PATH/output/multi_modal_session_quant_osp1_2/images" \
    --do_quant \
    --quant_weight_save_folder "$PROJECT_PATH/output/multi_modal_session_quant_osp1_2/safetensors" \
    --quant_dump_calib_folder "$PROJECT_PATH/output/multi_modal_session_quant_osp1_2/cache" \
    --quant_type "w8a8"

if [ $? -eq 0 ]
then
    echo multi_modal_session_quant_osp1_2: Success
else
    echo multi_modal_session_quant_osp1_2: Failed
    run_ok=$ret_failed
fi

conda activate smoke_test_modelslim_0104

# 清理output
rm -rf $PROJECT_PATH/output/multi_modal_session_quant_osp1_2

exit $run_ok
