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

from vega.common import Config
from vega.common import ClassFactory, ClassType  # noqa: F401


def run_train(model=None, vega_config=None, callback=None):
    config = Config()
    if vega_config is not None:
        vega_config.merge_to_config(config)
        vega_config.merge_to_dict(config)

    if callback is not None:
        callback.init_trainer()
        callback.before_train()
        callback.before_epoch(1)
        callback.before_train_step(1)
        callback.after_train_step(1)
        callback.after_epoch(1)
        callback.after_train()


def run_eval(model=None, vega_config=None, callback=None):
    config = Config()
    if vega_config is not None:
        vega_config.merge_to_config(config)
        vega_config.merge_to_dict(config)

    if callback is not None:
        callback.before_valid()
        callback.before_valid_step(1)
        callback.after_valid_step(1)
        callback.after_valid()

    return config.get("back", 1)
