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


from ascend_utils.common.security import check_type
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes import BaseConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_config.quant_config_classes.config_utils import \
    check_and_generate_config_param


class KVQuantConfig(BaseConfig):
    """
    KV Cache量化的Config
    """

    def __init__(self, last_config: BaseConfig,
                       kv_sym: bool = True):
        super().__init__()
        # 获取上一个config的所有属性，利用上一个config实例完成初始化，并进行参数的修改
        self._init_attribute_by_config(last_config)

        # 更新当前特有的属性
        self.use_kvcache_quant = True
        self.kv_sym = kv_sym

        # 校验参数
        check_and_generate_config_param(self)
        check_type(self.kv_sym, bool, param_name="kv_sym")
