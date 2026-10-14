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

ATHERIS_RUNS=1000
COVERAGE_SAVE_PATH="automl_fuzz_coverage"

RUN_FUZZ_SHELL_PATH=$(dirname `readlink -f "$0"`)
ABS_DIR_PATH=`readlink -f "${1:-$RUN_FUZZ_SHELL_PATH}"`  # If got 1 parameter, run tests under it

AUTOML_PATH=$(dirname `dirname $RUN_FUZZ_SHELL_PATH`)
echo ">>>> ABS_DIR_PATH: $ABS_DIR_PATH, AUTOML_PATH: $AUTOML_PATHW"

SOURCE_CODE=${AUTOML_PATH}/msmodelslim,${AUTOML_PATH}/ascend_utils
CASES=()
RESULTS=()
for FUZZ_PY_FILE in `find ${ABS_DIR_PATH} -name fuzz_test.py`; do
    TEST_PATH=`dirname $FUZZ_PY_FILE`
    echo ""
    echo ">>>> TEST_PATH: $TEST_PATH"
    SAMPLE_PATH=${TEST_PATH}/samples/
    if [ ! -e $SAMPLE_PATH ]; then
        SAMPLE_PATH=""  # sample path not exists, skip specifying
    fi
    PYTHONPATH="${AUTOML_PATH}:$PYTHONPATH" python3 -m coverage run --source=${SOURCE_CODE} -p ${TEST_PATH}/fuzz_test.py $SAMPLE_PATH -atheris_runs=$ATHERIS_RUNS
    RESULT=$?

    TEST_PATH_SPLIT=(${TEST_PATH[@]/fuzz\// })
    CASE_NAME=${TEST_PATH_SPLIT[1]}

    CASES[${#CASES[@]}]=$CASE_NAME
    RESULTS[${#RESULTS[@]}]=$RESULT
done

echo ""
echo ">>>> Fuzz results:"
for ((i=0; $i<${#CASES[@]}; i=$i+1)); do
    if [ ${RESULTS[i]} -eq 0 ]; then
        printf "     ${CASES[i]}: \033[32mPASSED\n\033[m"  # Green
    else
        printf "     ${CASES[i]}: \033[31mFAILED\n\033[m"  # Red
    fi
done

echo ""
echo ">>>> Done fuzz, generating coverage result..."
python3 -m coverage combine
python3 -m coverage html -d $COVERAGE_SAVE_PATH -i

# Clean temp samples if file is under `samples` and length is exactly 41
echo ">>>> clean samples"
find ${ABS_DIR_PATH} -wholename '*/samples/*' | xargs -I {} sh -c 'if [ `basename {} | wc -m` -eq 41 ]; then rm {}; fi'
