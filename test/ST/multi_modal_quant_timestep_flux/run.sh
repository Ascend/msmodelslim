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

chmod 640 $MSMODELSLIM_SOURCE_DIR/example/multimodal_sd/Flux/calib_prompts.txt

docker start smoke_flux

# 执行m3异常值抑制用例
docker exec -i smoke_flux bash -c "ASCEND_RT_VISIBLE_DEVICES='$ASCEND_RT_VISIBLE_DEVICES' CANN_PATH='$CANN_PATH' PROJECT_PATH='$PROJECT_PATH' MSMODELSLIM_SOURCE_DIR='$MSMODELSLIM_SOURCE_DIR' $PROJECT_PATH/test-case/multi_modal_quant_timestep_flux/quant_timestep.sh"

if [ $? -eq 0 ]
then
    echo multi_modal_quant_timestep_flux: Success
else
    echo multi_modal_quant_timestep_flux: Failed
    run_ok=$ret_failed
fi

docker stop smoke_flux

exit $run_ok
