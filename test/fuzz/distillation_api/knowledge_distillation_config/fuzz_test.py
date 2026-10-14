#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import sys
import logging
import os
from random import choice
from test.fuzz.common.utils import random_change_dict_value
from resources.sample_net_mindspore import MsStudentModel
from resources.sample_net_mindspore import MsTeacherModel
from resources.sample_net_torch import TorchStudentModel
from resources.sample_net_torch import TorchTeacherModel
import atheris
import atheris.instrument_bytecode

from msmodelslim.common.knowledge_distill.knowledge_distill import KnowledgeDistillConfig
from msmodelslim.common.knowledge_distill.knowledge_distill import get_distill_model


@atheris.instrument_func
def fuzz_test(input_bytes):
    fuzz_value = input_bytes.decode('utf-8', 'ignore').strip()
    distill_soft_label = {
        "t_module": "teacher_fc",
        "s_module": "student_fc",
        "t_output_idx": 0,
        "s_output_idx": 0,
        "loss_func": [{"func_name": "KDCrossEntropy", "func_weight": 1}],
        "shape": [1],
    }
    random_change_dict_value(distill_soft_label, fuzz_value)
    config = KnowledgeDistillConfig()

    model_list = [
        TorchTeacherModel,
        TorchStudentModel,
        MsTeacherModel,
        MsStudentModel,
    ]
    model_t = choice(model_list)()
    model_s = choice(model_list)()

    try:
        config.add_inter_soft_label(distill_soft_label)
        get_distill_model(model_t, model_s, config)
    except ValueError as value_error:
        logging.error(value_error)
    except TypeError as type_error:
        logging.error(type_error)
    except Exception as ee:
        if not isinstance(ee.args[-1], (TypeError, ValueError)):
            logging.error(ee)


if __name__ == '__main__':
    TEST_SAVE_PATH = "msmodelslim_fuzz_test_save_path"
    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.chdir(TEST_SAVE_PATH)

    atheris.Setup(sys.argv, fuzz_test)
    atheris.Fuzz()
