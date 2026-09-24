#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

from msmodelslim.app.auto_tuning import TuningPlanManagerInfra, TuningPlanConfig
from msmodelslim.utils.security import yaml_safe_load, check_read_permission


class YamlTuningPlanManager(TuningPlanManagerInfra):
    def get_plan_by_id(self, plan_id: str) -> TuningPlanConfig:
        check_read_permission(plan_id)
        content = yaml_safe_load(plan_id)
        return TuningPlanConfig.model_validate(content)
