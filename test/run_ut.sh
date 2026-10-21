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

ALL_CASES="pytorch app core ir mindspore common onnx processor quant utils model infra format cli"
MODELSlim_V1_CASES="app core ir infra processor utils format"

run_tests() {
    local cases="$1"
    local oldIFS="$IFS"
    IFS=' '
    for case_dir in $cases; do
        local junitxml="${route}/report/final_${case_dir}.xml"
        python3 -m coverage run --branch --source=${code_dir} --omit="${coverage_omit}" -p -m pytest ${route}/cases/${case_dir} --junitxml="${junitxml}" || ret=1
    done
    IFS="$oldIFS"
}

run_smoke_test() {
    python3 -m coverage run --branch --source=${code_dir} --omit="${coverage_omit}" -p -m pytest ${route}/smoke --junitxml="${route}/report/final_smoke.xml" || ret=1
}

# ---------------------------------------------------------------------------
# Coverage gate
#   LINE_COVERAGE_THRESHOLD: line-coverage gate in percent (default 80).
#   Override from the environment, e.g. LINE_COVERAGE_THRESHOLD=85 bash run_ut.sh
#   Set to 0 to only report coverage without enforcing the gate.
#
# NOTE: `coverage run --branch` makes the TOTAL/Cover value of
# `coverage report` a *combined* line+branch ratio, but the gate is defined on
# *line* coverage. We therefore read the `line-rate` attribute of the root
# element in coverage.xml, which is the pure line (statement) coverage ratio.
# ---------------------------------------------------------------------------
LINE_COVERAGE_THRESHOLD="${LINE_COVERAGE_THRESHOLD:-80}"

report_coverage() {
    python3 -m coverage combine
    python3 -m coverage xml -o ${route}/report/coverage.xml
    cat ${route}/report/coverage.xml | grep line-rate | grep coverage

    if [ -z "${LINE_COVERAGE_THRESHOLD}" ] || [ "${LINE_COVERAGE_THRESHOLD}" = "0" ]; then
        echo "[coverage] line coverage gate disabled (LINE_COVERAGE_THRESHOLD=${LINE_COVERAGE_THRESHOLD})"
        return 0
    fi

    if python3 - "${route}/report/coverage.xml" "${LINE_COVERAGE_THRESHOLD}" <<'PY'
import sys
import xml.etree.ElementTree as ET

xml_path = sys.argv[1]
threshold = float(sys.argv[2])
line_rate = float(ET.parse(xml_path).getroot().get("line-rate", 0.0)) * 100
print("[coverage] line coverage: {:.2f}% (gate: {:g}%)".format(line_rate, threshold))
sys.exit(0 if line_rate >= threshold else 1)
PY
    then
        echo "[coverage] PASS: line coverage is not lower than ${LINE_COVERAGE_THRESHOLD}%"
    else
        echo "[coverage] FAIL: line coverage is lower than ${LINE_COVERAGE_THRESHOLD}%, UT gate failed"
        ret=1
    fi
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --modelslim_v1     Run only modelslim_v1 related test cases (app, core, ir, infra, processor, utils)"
    echo "  --help            Show this help message"
    echo ""
    echo "Environment:"
    echo "  LINE_COVERAGE_THRESHOLD  Line-coverage gate in percent (default: 80, 0 disables the gate)"
    echo ""
    echo "Coverage scope:"
    echo "  Excluded: the per-model directories under msmodelslim/model"
    echo "  Included: all other modules under msmodelslim and ascend_utils, including"
    echo "            msmodelslim/pytorch, msmodelslim/onnx, msmodelslim/mindspore and the"
    echo "            msmodelslim/model framework layer"
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
# ---------------------------------------------------------------------------
# Coverage scope
#   Only the per-model directories under msmodelslim/model are kept out of
#   the line-coverage denominator: they merely wire a model to the framework
#   interfaces and are frozen once the model has landed.
#   Everything else is measured, including the v0 implementations
#   (msmodelslim/pytorch, msmodelslim/onnx, msmodelslim/mindspore) and the
#   msmodelslim/model framework layer (MODEL_FRAMEWORK_DIRS).
#   Keep MODEL_FRAMEWORK_DIRS in sync when a framework subdirectory is added.
# ---------------------------------------------------------------------------
MODEL_FRAMEWORK_DIRS="common default plugin_factory transformers"
coverage_omit=""
for model_dir in "${rootdir}"/msmodelslim/model/*/; do
    model_name=$(basename "${model_dir}")
    case " ${MODEL_FRAMEWORK_DIRS} " in
        *" ${model_name} "*) ;;
        *) coverage_omit="${coverage_omit}${coverage_omit:+,}*/msmodelslim/model/${model_name}/*" ;;
    esac
done
echo "[coverage] excluded from line coverage: ${coverage_omit:-<none>}"
cp ${rootdir}/lab_calib     ${rootdir}/msmodelslim/ -rf
cp ${rootdir}/lab_practice  ${rootdir}/msmodelslim/ -rf
cp ${rootdir}/config        ${rootdir}/msmodelslim/ -rf
chmod 640 ${rootdir}/msmodelslim/config/config.ini ${rootdir}/msmodelslim/config/__init__.py

case "${1:-}" in
    --modelslim_v1)
        echo "Running modelslim_v1 related test cases..."
        run_tests "$MODELSlim_V1_CASES"
        report_coverage
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
        report_coverage
        exit ${ret}
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac
