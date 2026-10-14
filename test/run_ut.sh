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
set -e
script=$(readlink -f "$0")
route=$(dirname "$script")
rootdir=$(dirname "$route")
kia_dir=$(dirname "$rootdir")/automl_kia

ALL_CASES="pytorch app core ir mindspore common onnx processor quant utils model infra format"
MODELSlim_V1_CASES="app core ir infra processor utils format"

run_tests() {
    local cases="$1"
    local oldIFS="$IFS"
    IFS=' '
    for case_dir in $cases; do
        local junitxml="${route}/report/final_${case_dir}.xml"
        python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/cases/${case_dir} --junitxml="${junitxml}" || ret=1
    done
    IFS="$oldIFS"
}

run_smoke_test() {
    python3 -m coverage run --branch --source=${code_dir} -p -m pytest ${route}/smoke --junitxml="${route}/report/final_smoke.xml" || ret=1
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --modelslim_v1     Run only modelslim_v1 related test cases (app, core, ir, infra, processor, utils)"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                Run all test cases (default)"
    echo "  $0 --modelslim_v1 Run only modelslim_v1 related test cases"
}

if [ -e $kia_dir ]; then
    mkdir ${rootdir}/msmodelslim/pytorch/quant/ptq_tools/ptq_kia
    mkdir ${rootdir}/msmodelslim/onnx/squant_ptq/onnx_ptq_kia

    cp ${kia_dir}/quant_funcs.* ${rootdir}/msmodelslim/pytorch/quant/ptq_tools/ptq_kia -rf
    cp ${kia_dir}/weight_transform.* ${rootdir}/msmodelslim/pytorch/quant/ptq_tools/ptq_kia -rf

    cp ${kia_dir}/quant_funcs_onnx.* ${rootdir}/msmodelslim/onnx/squant_ptq/onnx_ptq_kia -rf
    cp ${kia_dir}/weight_transform_onnx.* ${rootdir}/msmodelslim/onnx/squant_ptq/onnx_ptq_kia -rf
fi

export PYTHONPATH="${rootdir}":$PYTHONPATH
export DEVICE_ID=0
echo "PYTHONPATH is ${PYTHONPATH}"

rm -rf ${route}/.coverage ${route}/report
mkdir -p ${route}/report
chmod o= ${route}/resources -R  # Others no permission
chmod g-w ${route}/resources -R  # Group not writable

ret=0
code_dir=${rootdir}/msmodelslim,${rootdir}/ascend_utils
cp ${rootdir}/lab_calib     ${rootdir}/msmodelslim/ -rf
cp ${rootdir}/lab_practice  ${rootdir}/msmodelslim/ -rf
cp ${rootdir}/config        ${rootdir}/msmodelslim/ -rf
chmod 640 ${rootdir}/msmodelslim/config/config.ini ${rootdir}/msmodelslim/config/__init__.py

case "${1:-}" in
    --modelslim_v1)
        echo "Running modelslim_v1 related test cases..."
        run_tests "$MODELSlim_V1_CASES"
        python3 -m coverage combine
        python3 -m coverage xml -o ${route}/report/coverage.xml
        cat ${route}/report/coverage.xml | grep line-rate | grep coverage
        exit ${ret}
        ;;
    --help)
        usage
        exit 0
        ;;
    "")
        echo "Running all test cases..."
        run_tests "$ALL_CASES"
        run_smoke_test
        python3 -m coverage combine
        python3 -m coverage xml -o ${route}/report/coverage.xml
        cat ${route}/report/coverage.xml | grep line-rate | grep coverage
        exit ${ret}
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac
