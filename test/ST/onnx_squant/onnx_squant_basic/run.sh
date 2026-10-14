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

rm -rf $PROJECT_PATH/output/onnx_squant/resnet50_quant.onnx


python3 -m ais_bench --model $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/resnet50_bs64.om \
                     --input $PROJECT_PATH/resource/onnx_squant/prep_dataset \
                     --output $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/ \
                     --output_dirname result \
                     --outfmt TXT

python3 $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/vision_metric_ImageNet.py \
        $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/result \
        $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/val_label.txt \
        $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/ \
        result.json

python run.py
if [ $? -eq 0 ]
then
    echo onnx_squant_basic: Success
else
    echo onnx_squant_basic: Failed
    run_ok=$ret_failed
fi

atc --model=$PROJECT_PATH/output/onnx_squant/resnet50_quant.onnx \
    --framework=5 \
    --output=$PROJECT_PATH/resource/onnx_squant/resnet50_quant_bs64 \
    --input_format=NCHW \
    --input_shape="actual_input_1:64,3,224,224" \
    --enable_small_channel=1 \
    --log=error \
    --soc_version=Ascend310P3 \
    --insert_op_conf=$PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/aipp_resnet50.aippconfig

python3 -m ais_bench --model $PROJECT_PATH/resource/onnx_squant/resnet50_quant_bs64.om \
                     --input $PROJECT_PATH/resource/onnx_squant/prep_dataset \
                     --output $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/ \
                     --output_dirname result_quant \
                     --outfmt TXT

python3 $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/vision_metric_ImageNet.py \
        $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/result_quant \
        $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/val_label.txt \
        $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/ \
        result_quant.json

cat $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/result.json | python -m json.tool
cat $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/result_quant.json | python -m json.tool

rm -rf $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/result.json
rm -rf $PROJECT_PATH/resource/onnx_squant/Resnet50_Pytorch_Infer/result_quant.json

exit $run_ok
